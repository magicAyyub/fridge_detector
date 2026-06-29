"""
Region Proposal Network (RPN), built from scratch.

For each spatial location of each FPN level, the RPN produces:
  - K objectness scores (one per anchor)  "is there an object here?"
  - K box offsets (4 per anchor)          "how to adjust the anchor"

Where K = num anchors per location.

Training:
  Each anchor is matched to ground truth based on IoU:
    - IoU >= pos_thresh (0.7) → positive (foreground)
    - IoU <  neg_thresh (0.3) → negative (background)
    - else                     → ignored

  Loss = binary CE on objectness (sampled to balance pos/neg)
       + smooth L1 on box offsets (positive anchors only)

Inference:
  For each level, take top-K anchors by objectness, decode to boxes,
  filter small boxes, apply NMS, then merge across levels and take
  the top-N proposals overall.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import box_iou, clip_boxes_to_image, remove_small_boxes, nms

from src.utils.box_ops import encode_boxes, decode_boxes

class RPNHead(nn.Module):
    """The shared head: 3x3 conv → two sibling 1x1 convs (objectness + boxes)."""

    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.objectness = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        # Initialize: small weights for stable early training
        for layer in [self.conv, self.objectness, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features: list) -> tuple:
        """
        Args:
            features: list of (B, C, H_i, W_i) tensors, one per FPN level

        Returns:
            objectness_logits: list of (B, K, H_i, W_i)
            bbox_offsets:      list of (B, K*4, H_i, W_i)
        """
        objectness, bbox = [], []
        for f in features:
            t = F.relu(self.conv(f))
            objectness.append(self.objectness(t))
            bbox.append(self.bbox_pred(t))
        return objectness, bbox


class RegionProposalNetwork(nn.Module):
    """The full RPN: head + anchor matching + loss + proposal generation."""

    def __init__(self, anchor_generator, in_channels: int = 256,
                 # Matching thresholds
                 pos_iou_thresh: float = 0.7,
                 neg_iou_thresh: float = 0.3,
                 # Sampling
                 batch_size_per_image: int = 256,
                 positive_fraction: float = 0.5,
                 # Inference
                 pre_nms_top_n_train: int = 2000,
                 pre_nms_top_n_test: int = 1000,
                 post_nms_top_n_train: int = 1000,
                 post_nms_top_n_test: int = 300,
                 nms_threshold: float = 0.7,
                 min_box_size: float = 1.0):
        super().__init__()
        self.anchor_generator = anchor_generator
        num_anchors = anchor_generator.num_anchors_per_location()[0]
        self.head = RPNHead(in_channels, num_anchors)

        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pre_nms_top_n_train = pre_nms_top_n_train
        self.pre_nms_top_n_test = pre_nms_top_n_test
        self.post_nms_top_n_train = post_nms_top_n_train
        self.post_nms_top_n_test = post_nms_top_n_test
        self.nms_threshold = nms_threshold
        self.min_box_size = min_box_size

    @staticmethod
    def _flatten_predictions(objectness, bbox_offsets):
        """
        Flatten predictions across spatial locations & levels.

        objectness[i]: (B, K, H_i, W_i) → (B, H_i*W_i*K)
        bbox_offsets[i]: (B, K*4, H_i, W_i) → (B, H_i*W_i*K, 4)

        Returns concatenated tensors over all levels.
        """
        flat_obj, flat_box = [], []
        for obj, box in zip(objectness, bbox_offsets):
            B, K, H, W = obj.shape
            # objectness: (B, K, H, W) → (B, H, W, K) → (B, H*W*K)
            obj = obj.permute(0, 2, 3, 1).reshape(B, -1)
            # bbox: (B, K*4, H, W) → (B, H, W, K, 4) → (B, H*W*K, 4)
            box = box.view(B, K, 4, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, 4)
            flat_obj.append(obj)
            flat_box.append(box)
        return torch.cat(flat_obj, dim=1), torch.cat(flat_box, dim=1)

    def assign_targets_to_anchors(self, anchors: torch.Tensor,
                                   gt_boxes: torch.Tensor) -> tuple:
        """
        Assign each anchor a label and a regression target.

        Args:
            anchors:  (N, 4) all anchors for one image
            gt_boxes: (M, 4) ground truth boxes

        Returns:
            labels:    (N,)  1 (positive), 0 (negative), -1 (ignore)
            matched_gt: (N, 4) for each anchor, the GT it's matched to (zero for negatives)
        """
        if gt_boxes.numel() == 0:
            # No GT in this image → all anchors are negative
            labels = torch.zeros(anchors.shape[0], dtype=torch.float32, device=anchors.device)
            matched_gt = torch.zeros_like(anchors)
            return labels, matched_gt

        ious = box_iou(anchors, gt_boxes)              # (N, M)
        max_iou_per_anchor, gt_idx_per_anchor = ious.max(dim=1)  # (N,)

        labels = torch.full((anchors.shape[0],), -1.0, device=anchors.device)
        labels[max_iou_per_anchor >= self.pos_iou_thresh] = 1.0
        labels[max_iou_per_anchor < self.neg_iou_thresh] = 0.0

        # Also mark as positive the best anchor for each GT (so every GT has at least one positive).
        # This handles cases where no anchor reaches the 0.7 threshold for a particular GT.
        max_iou_per_gt, _ = ious.max(dim=0)             # (M,)
        # Find anchors whose IoU equals the best for their assigned GT
        for gt_i in range(gt_boxes.shape[0]):
            best_anchors_for_gt = (ious[:, gt_i] == max_iou_per_gt[gt_i]).nonzero(as_tuple=True)[0]
            labels[best_anchors_for_gt] = 1.0

        matched_gt = gt_boxes[gt_idx_per_anchor]        # (N, 4)
        return labels, matched_gt

    def sample_anchors(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Sample a balanced mini-batch from the labeled anchors.
        Returns indices of sampled anchors (mix of pos and neg).
        """
        positive = (labels == 1).nonzero(as_tuple=True)[0]
        negative = (labels == 0).nonzero(as_tuple=True)[0]

        num_pos = int(self.batch_size_per_image * self.positive_fraction)
        num_pos = min(num_pos, positive.numel())
        num_neg = self.batch_size_per_image - num_pos
        num_neg = min(num_neg, negative.numel())

        # Random sampling without replacement
        perm_pos = torch.randperm(positive.numel(), device=labels.device)[:num_pos]
        perm_neg = torch.randperm(negative.numel(), device=labels.device)[:num_neg]
        pos_idx = positive[perm_pos]
        neg_idx = negative[perm_neg]
        return pos_idx, neg_idx

    def compute_loss(self, objectness: torch.Tensor, bbox_offsets: torch.Tensor,
                     anchors: torch.Tensor, targets: list) -> tuple:
        """
        Compute RPN losses for the whole batch.

        Args:
            objectness:   (B, N) flattened objectness logits
            bbox_offsets: (B, N, 4) flattened predicted offsets
            anchors:      (N, 4) same anchors for every image in the batch
            targets:      list of dicts, each with 'boxes' (M_i, 4)

        Returns:
            (objectness_loss, box_reg_loss) scalars
        """
        B = objectness.shape[0]
        all_obj_pred, all_obj_label = [], []
        all_box_pred, all_box_target = [], []

        for i in range(B):
            gt_boxes = targets[i]['boxes']
            labels, matched_gt = self.assign_targets_to_anchors(anchors, gt_boxes)
            pos_idx, neg_idx = self.sample_anchors(labels)
            sampled = torch.cat([pos_idx, neg_idx])

            # Objectness: sampled anchors only
            all_obj_pred.append(objectness[i, sampled])
            sampled_labels = torch.cat([
                torch.ones_like(pos_idx, dtype=torch.float32),
                torch.zeros_like(neg_idx, dtype=torch.float32),
            ])
            all_obj_label.append(sampled_labels)

            # Box regression: positive anchors only
            if pos_idx.numel() > 0:
                target_offsets = encode_boxes(matched_gt[pos_idx], anchors[pos_idx])
                all_box_pred.append(bbox_offsets[i, pos_idx])
                all_box_target.append(target_offsets)

        obj_pred = torch.cat(all_obj_pred)
        obj_label = torch.cat(all_obj_label)
        objectness_loss = F.binary_cross_entropy_with_logits(obj_pred, obj_label)

        if all_box_pred:
            box_pred = torch.cat(all_box_pred)
            box_target = torch.cat(all_box_target)
            # Smooth L1 (Huber). It's less sensitive to outliers than MSE
            box_reg_loss = F.smooth_l1_loss(box_pred, box_target, beta=1.0/9, reduction='sum') / max(B, 1) / self.batch_size_per_image
        else:
            box_reg_loss = objectness_loss.new_zeros(())

        return objectness_loss, box_reg_loss

    def filter_proposals(self, objectness: torch.Tensor, bbox_offsets: torch.Tensor,
                         anchors: torch.Tensor, image_size: tuple,
                         num_anchors_per_level: list) -> list:
        """
        Generate final proposals for each image.

        Strategy (per image):
          1. For each FPN level, take top-K anchors by objectness.
          2. Decode → image-space boxes.
          3. Clip to image, filter tiny boxes.
          4. Apply NMS per level, keep top-N overall by score.
        """
        B = objectness.shape[0]
        device = objectness.device

        pre_nms_top_n = self.pre_nms_top_n_train if self.training else self.pre_nms_top_n_test
        post_nms_top_n = self.post_nms_top_n_train if self.training else self.post_nms_top_n_test

        # Compute level offsets so we can split flat tensors back into levels
        level_offsets = [0]
        for n in num_anchors_per_level:
            level_offsets.append(level_offsets[-1] + n)

        proposals_per_image = []
        for b in range(B):
            level_proposals = []
            level_scores = []
            level_classes = []

            for lvl, n_lvl in enumerate(num_anchors_per_level):
                start, end = level_offsets[lvl], level_offsets[lvl + 1]
                lvl_obj = objectness[b, start:end]              # (n_lvl,)
                lvl_box = bbox_offsets[b, start:end]            # (n_lvl, 4)
                lvl_anchors = anchors[start:end]                 # (n_lvl, 4)

                # Top-K by objectness for this level
                top_k = min(pre_nms_top_n, n_lvl)
                top_scores, top_idx = lvl_obj.topk(top_k)
                top_box_offsets = lvl_box[top_idx]
                top_anchors = lvl_anchors[top_idx]

                # Decode and clean up
                proposals = decode_boxes(top_box_offsets, top_anchors)
                proposals = clip_boxes_to_image(proposals, image_size)
                keep = remove_small_boxes(proposals, self.min_box_size)
                proposals = proposals[keep]
                top_scores = torch.sigmoid(top_scores[keep])

                level_proposals.append(proposals)
                level_scores.append(top_scores)
                level_classes.append(torch.full_like(top_scores, lvl, dtype=torch.long))

            all_props = torch.cat(level_proposals)
            all_scores = torch.cat(level_scores)
            all_levels = torch.cat(level_classes)

            # Per-level NMS using the batched_nms trick (treat level as class)
            from torchvision.ops import batched_nms
            keep = batched_nms(all_props, all_scores, all_levels, self.nms_threshold)
            keep = keep[:post_nms_top_n]

            proposals_per_image.append({
                'boxes': all_props[keep],
                'scores': all_scores[keep],
            })

        return proposals_per_image

    def forward(self, features: list, image_size: tuple, targets: list = None) -> tuple:
        """
        Args:
            features: list of FPN feature maps [P2, P3, P4, P5]
            image_size: (H, W) of input image
            targets: list of dicts (one per image) with 'boxes'

        Returns:
            proposals: list of dicts per image, with 'boxes' and 'scores'
            losses: dict with 'rpn_obj_loss' and 'rpn_box_loss' (empty dict if no targets)
        """
        # 1. Run head
        objectness, bbox_offsets = self.head(features)

        # 2. Generate anchors for these feature maps
        anchors_per_level = self.anchor_generator(image_size, features)
        num_anchors_per_level = [a.shape[0] for a in anchors_per_level]
        anchors = torch.cat(anchors_per_level, dim=0)  # (N_total, 4)

        # 3. Flatten predictions
        objectness_flat, bbox_offsets_flat = self._flatten_predictions(objectness, bbox_offsets)

        # 4. Compute losses if training
        losses = {}
        if self.training and targets is not None:
            obj_loss, box_loss = self.compute_loss(
                objectness_flat, bbox_offsets_flat, anchors, targets
            )
            losses = {'rpn_obj_loss': obj_loss, 'rpn_box_loss': box_loss}

        # 5. Generate proposals (used in both training and inference,
        #    because the second-stage head needs them)
        # Detach so the second-stage gradients don't flow back through the proposal generation step
        proposals = self.filter_proposals(
            objectness_flat.detach(), bbox_offsets_flat.detach(),
            anchors, image_size, num_anchors_per_level
        )

        return proposals, losses
