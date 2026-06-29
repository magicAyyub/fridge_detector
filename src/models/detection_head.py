"""
Detection head (second stage).

Takes RoI-aligned features and produces, for each region:
  - Class scores over (num_classes + 1) — the +1 is "background"
  - Box refinement offsets per class (class-specific regression)

This is a standard two-FC-layer head (Fast R-CNN style).

Training:
  Sample a balanced batch of proposals (positives matched to GT, negatives
  randomly chosen). Compute classification CE loss (over all sampled
  proposals) + box regression smooth L1 loss (positives only).

Inference:
  Apply per-class score threshold and NMS, keep top-N detections per image.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import box_iou, clip_boxes_to_image, remove_small_boxes, batched_nms, RoIAlign

from src.utils.box_ops import encode_boxes, decode_boxes


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int = 256, roi_output_size: int = 7,
                 hidden_dim: int = 1024, num_classes: int = 25):
        super().__init__()
        self.num_classes = num_classes  # excluding background
        # +1 because we add a background class internally
        self.num_classes_plus_bg = num_classes + 1

        flat_dim = in_channels * roi_output_size * roi_output_size
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.cls_score = nn.Linear(hidden_dim, self.num_classes_plus_bg)
        # Per-class box regression: 4 offsets × num_classes (background has no box)
        self.bbox_pred = nn.Linear(hidden_dim, num_classes * 4)

        # Init
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(m.weight, a=1)
            nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, pooled_features: torch.Tensor) -> tuple:
        """
        Args:
            pooled_features: (N, C, H, W) — output of RoI Align

        Returns:
            class_logits: (N, num_classes + 1)
            box_offsets:  (N, num_classes, 4)
        """
        x = pooled_features.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.cls_score(x), self.bbox_pred(x).view(-1, self.num_classes, 4)


class RoIHeadsModule(nn.Module):
    """
    Wraps the detection head with proposal sampling, RoI Align, loss
    computation, and inference post-processing.

    Why "RoI Heads"? In Faster R-CNN terminology, this stage is called the
    "RoI head" because it operates on Regions of Interest from the RPN.
    """

    def __init__(self, fpn_channels: int = 256, num_classes: int = 25,
                 # RoI Align config
                 roi_output_size: int = 7,
                 # Per-FPN-level spatial scales (1 / stride)
                 fpn_strides: dict = None,
                 # Sampling
                 batch_size_per_image: int = 128,
                 positive_fraction: float = 0.25,
                 fg_iou_thresh: float = 0.5,
                 bg_iou_thresh: float = 0.5,
                 # Inference
                 score_thresh: float = 0.05,
                 nms_thresh: float = 0.5,
                 detections_per_image: int = 100):
        super().__init__()
        self.num_classes = num_classes

        if fpn_strides is None:
            fpn_strides = {'P2': 4, 'P3': 8, 'P4': 16, 'P5': 32}
        self.fpn_strides = fpn_strides
        self.fpn_levels = list(fpn_strides.keys())

        # One RoI Align per FPN level (different spatial scales)
        self.roi_aligners = nn.ModuleDict({
            lvl: RoIAlign(roi_output_size, spatial_scale=1.0/stride, sampling_ratio=2)
            for lvl, stride in fpn_strides.items()
        })

        self.head = DetectionHead(fpn_channels, roi_output_size, num_classes=num_classes)

        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_image = detections_per_image

    def assign_proposals_to_targets(self, proposals: torch.Tensor,
                                     gt_boxes: torch.Tensor,
                                     gt_labels: torch.Tensor) -> tuple:
        """
        For each proposal, decide if it's a positive sample (matched to a GT)
        or a negative (background).

        Returns:
            matched_labels: (N,) — class label (0 = background, 1..C = real class)
            matched_boxes:  (N, 4) — corresponding GT box (used for regression target)
        """
        if gt_boxes.numel() == 0:
            # No GT → all proposals are background
            matched_labels = torch.zeros(proposals.shape[0], dtype=torch.int64,
                                         device=proposals.device)
            matched_boxes = torch.zeros_like(proposals)
            return matched_labels, matched_boxes

        ious = box_iou(proposals, gt_boxes)             # (N, M)
        max_iou, gt_idx = ious.max(dim=1)               # (N,)

        # Foreground if IoU >= fg_iou_thresh
        # Background if IoU < bg_iou_thresh
        # (between → ignore, but here we use a single threshold for simplicity)
        matched_labels = torch.full((proposals.shape[0],), -1,
                                     dtype=torch.int64, device=proposals.device)
        matched_labels[max_iou >= self.fg_iou_thresh] = gt_labels[gt_idx[max_iou >= self.fg_iou_thresh]]
        matched_labels[max_iou < self.bg_iou_thresh] = 0  # 0 = background

        matched_boxes = gt_boxes[gt_idx]
        return matched_labels, matched_boxes

    def sample_proposals(self, matched_labels: torch.Tensor) -> torch.Tensor:
        """Balanced positive/negative sampling. Returns indices to keep."""
        positive = (matched_labels > 0).nonzero(as_tuple=True)[0]
        negative = (matched_labels == 0).nonzero(as_tuple=True)[0]

        num_pos = int(self.batch_size_per_image * self.positive_fraction)
        num_pos = min(num_pos, positive.numel())
        num_neg = self.batch_size_per_image - num_pos
        num_neg = min(num_neg, negative.numel())

        perm_pos = torch.randperm(positive.numel(), device=matched_labels.device)[:num_pos]
        perm_neg = torch.randperm(negative.numel(), device=matched_labels.device)[:num_neg]

        return torch.cat([positive[perm_pos], negative[perm_neg]])

    def assign_proposals_to_fpn_levels(self, proposals: torch.Tensor) -> torch.Tensor:
        """
        Decide which FPN level each proposal should be cropped from.

        Following the FPN paper: smaller proposals → finer levels (P2),
        larger proposals → coarser levels (P5).

        Formula (eq. 1 in FPN paper):
            level = floor(k0 + log2(sqrt(w*h) / 224))
        with k0 = 4 (i.e., a 224x224 box maps to P4).

        Returns: (N,) integer level index in [0, 3] for [P2, P3, P4, P5].
        """
        w = (proposals[:, 2] - proposals[:, 0]).clamp(min=1)
        h = (proposals[:, 3] - proposals[:, 1]).clamp(min=1)
        scale = torch.sqrt(w * h)
        # k0 = 4 means a 224x224 box maps to P4 (index 2)
        k = torch.floor(4 + torch.log2(scale / 224.0))
        # Clamp to [2, 5] (i.e. P2 = idx 0, ... P5 = idx 3)
        k = k.clamp(min=2, max=5)
        return (k - 2).long()  # idx 0..3

    def roi_align_per_level(self, fpn_features: dict, proposals_per_image: list) -> tuple:
        """
        Apply RoI Align — but route each proposal to the right FPN level.

        Returns:
            pooled_features: (N_total, C, H, W) — concatenated across images & levels
            roi_to_image: (N_total,) which image each pooled feature belongs to
        """
        # Flatten all proposals into a single (N, 5) tensor with batch indices
        all_rois = []
        for img_idx, proposals in enumerate(proposals_per_image):
            if proposals.numel() == 0:
                continue
            batch_col = torch.full((proposals.shape[0], 1), img_idx,
                                    dtype=proposals.dtype, device=proposals.device)
            all_rois.append(torch.cat([batch_col, proposals], dim=1))
        if not all_rois:
            empty = next(iter(fpn_features.values()))
            return empty.new_zeros((0, empty.shape[1], 7, 7)), torch.empty(0, dtype=torch.long)
        all_rois = torch.cat(all_rois, dim=0)  # (N_total, 5)

        # Decide which FPN level each RoI goes to
        levels = self.assign_proposals_to_fpn_levels(all_rois[:, 1:])

        # Pool from each level, then reassemble in original order
        N = all_rois.shape[0]
        out_channels = fpn_features[self.fpn_levels[0]].shape[1]
        pooled = all_rois.new_zeros((N, out_channels, 7, 7))
        for lvl_idx, lvl_name in enumerate(self.fpn_levels):
            mask = levels == lvl_idx
            if not mask.any():
                continue
            lvl_rois = all_rois[mask]
            pooled_lvl = self.roi_aligners[lvl_name](fpn_features[lvl_name], lvl_rois)
            pooled[mask] = pooled_lvl

        return pooled, all_rois[:, 0].long()

    def compute_loss(self, class_logits: torch.Tensor, box_offsets: torch.Tensor,
                     labels: torch.Tensor, regression_targets: torch.Tensor) -> dict:
        """
        Args:
            class_logits: (N, C+1)
            box_offsets:  (N, C, 4)
            labels:       (N,) — 0 (bg) or 1..C
            regression_targets: (N, 4) — only meaningful for positives
        """
        cls_loss = F.cross_entropy(class_logits, labels)

        # Box loss only on positives, only the predicted offsets for the *true class*
        positive = (labels > 0).nonzero(as_tuple=True)[0]
        if positive.numel() > 0:
            # box_offsets[positive] has shape (P, C, 4); pick the correct class per row
            cls_indices = labels[positive] - 1  # 1..C → 0..C-1
            pos_offsets = box_offsets[positive, cls_indices]  # (P, 4)
            box_loss = F.smooth_l1_loss(pos_offsets, regression_targets[positive],
                                         beta=1.0, reduction='sum') / max(labels.numel(), 1)
        else:
            box_loss = cls_loss.new_zeros(())

        return {'cls_loss': cls_loss, 'box_loss': box_loss}

    def postprocess_detections(self, class_logits: torch.Tensor, box_offsets: torch.Tensor,
                                proposals_per_image: list, image_size: tuple,
                                roi_to_image: torch.Tensor) -> list:
        """Convert raw network outputs into final per-image detections."""
        device = class_logits.device
        scores_all = F.softmax(class_logits, dim=-1)  # (N, C+1)
        # Discard background column
        scores_all = scores_all[:, 1:]                # (N, C)

        # Decode boxes per class
        # box_offsets: (N, C, 4); we need anchors per RoI to decode → use proposals as the anchors
        all_proposals = torch.cat(proposals_per_image)  # (N, 4)
        # Repeat each proposal C times so we can decode per class
        N = all_proposals.shape[0]
        C = self.num_classes
        proposals_repeated = all_proposals.unsqueeze(1).expand(N, C, 4).reshape(-1, 4)
        offsets_flat = box_offsets.reshape(-1, 4)
        decoded_flat = decode_boxes(offsets_flat, proposals_repeated)
        decoded = decoded_flat.view(N, C, 4)

        results = []
        B = len(proposals_per_image)
        for b in range(B):
            mask = roi_to_image == b
            if not mask.any():
                results.append({
                    'boxes': torch.empty((0, 4), device=device),
                    'scores': torch.empty(0, device=device),
                    'labels': torch.empty(0, dtype=torch.int64, device=device),
                })
                continue
            img_boxes = decoded[mask]                # (n_b, C, 4)
            img_scores = scores_all[mask]            # (n_b, C)

            # Flatten across classes
            n_b = img_boxes.shape[0]
            flat_boxes = img_boxes.reshape(-1, 4)
            flat_scores = img_scores.reshape(-1)
            # Class label for each (proposal, class) pair: 1..C
            flat_labels = (torch.arange(C, device=device) + 1).repeat(n_b)

            flat_boxes = clip_boxes_to_image(flat_boxes, image_size)

            keep = remove_small_boxes(flat_boxes, 1.0)
            flat_boxes = flat_boxes[keep]
            flat_scores = flat_scores[keep]
            flat_labels = flat_labels[keep]

            keep = (flat_scores >= self.score_thresh).nonzero(as_tuple=True)[0]
            flat_boxes = flat_boxes[keep]
            flat_scores = flat_scores[keep]
            flat_labels = flat_labels[keep]

            keep = batched_nms(flat_boxes, flat_scores, flat_labels, self.nms_thresh)
            keep = keep[:self.detections_per_image]

            results.append({
                'boxes': flat_boxes[keep],
                'scores': flat_scores[keep],
                'labels': flat_labels[keep],
            })
        return results

    def forward(self, fpn_features: dict, proposals_per_image: list,
                image_size: tuple, targets: list = None) -> tuple:
        """
        Args:
            fpn_features: dict 'P2'..'P5' → (B, C, H, W)
            proposals_per_image: list of (N_b, 4) tensors — RPN proposals per image
            image_size: (H, W) of input
            targets: list of {'boxes', 'labels'} — required for training

        Returns:
            detections: list per image (None if training)
            losses: dict (empty if not training)
        """
        # If training, augment proposals with the GT boxes themselves
        # (this guarantees positive samples even when RPN proposals are weak)
        # and also do positive/negative sampling.
        losses = {}
        if self.training and targets is not None:
            sampled_props = []
            sampled_labels_list = []
            sampled_targets_list = []
            for i, (props, tgt) in enumerate(zip(proposals_per_image, targets)):
                gt_boxes = tgt['boxes']
                gt_labels = tgt['labels']
                # Add GT to proposals
                augmented = torch.cat([props, gt_boxes], dim=0) if gt_boxes.numel() else props

                matched_labels, matched_boxes = self.assign_proposals_to_targets(
                    augmented, gt_boxes, gt_labels
                )
                sampled = self.sample_proposals(matched_labels)
                sampled_props.append(augmented[sampled])
                sampled_labels_list.append(matched_labels[sampled])

                # Compute regression targets (proposals → GT)
                reg_targets = encode_boxes(matched_boxes[sampled], augmented[sampled])
                sampled_targets_list.append(reg_targets)

            proposals_for_head = sampled_props
            all_labels = torch.cat(sampled_labels_list)
            all_reg_targets = torch.cat(sampled_targets_list)
        else:
            proposals_for_head = proposals_per_image

        # RoI Align across FPN levels
        pooled, roi_to_image = self.roi_align_per_level(fpn_features, proposals_for_head)
        if pooled.shape[0] == 0:
            # No proposals (unusual) — return empty
            if self.training:
                return None, {'cls_loss': pooled.new_zeros(()), 'box_loss': pooled.new_zeros(())}
            return [{'boxes': pooled.new_zeros((0, 4)),
                     'scores': pooled.new_zeros(0),
                     'labels': pooled.new_zeros(0, dtype=torch.int64)} for _ in proposals_per_image], {}

        class_logits, box_offsets = self.head(pooled)

        if self.training:
            losses = self.compute_loss(class_logits, box_offsets, all_labels, all_reg_targets)
            return None, losses
        else:
            detections = self.postprocess_detections(
                class_logits, box_offsets, proposals_for_head, image_size, roi_to_image
            )
            return detections, {}
