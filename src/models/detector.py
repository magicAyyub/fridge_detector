"""
Full Fridge Detector — wires together backbone, FPN, RPN, and detection head.

Pipeline:
  Image → Backbone → C2..C5
        → FPN      → P2..P5
        → RPN      → proposals (per image)
        → RoI Heads → final detections (per image)

Loss = rpn_obj_loss + rpn_box_loss + cls_loss + box_loss
"""
import torch
import torch.nn as nn

from .backbone import ResNetBackbone
from .fpn import FeaturePyramidNetwork
from .rpn import RegionProposalNetwork
from .detection_head import RoIHeadsModule
from src.utils.anchors import AnchorGenerator


# ImageNet normalization (the pretrained backbone expects this)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FridgeDetector(nn.Module):
    def __init__(self, num_classes: int = 25, fpn_channels: int = 256,
                 backbone_arch: str = 'resnet50',
                 pretrained_backbone: bool = True):
        """
        Args:
            num_classes: number of ingredient classes (excluding background)
            backbone_arch: 'resnet18' (lightweight) or 'resnet50' (default)
        """
        super().__init__()
        self.num_classes = num_classes

        # 1. Backbone
        self.backbone = ResNetBackbone(arch=backbone_arch, pretrained=pretrained_backbone,
                                        freeze_early_layers=True)

        # 2. FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_dict=self.backbone.out_channels,
            out_channels=fpn_channels,
        )

        # 3. Anchor generator — different sizes per FPN level
        # Following standard FPN-detector convention:
        # P2 (stride 4)  → small objects   (~32 px)
        # P3 (stride 8)  → medium-small    (~64 px)
        # P4 (stride 16) → medium          (~128 px)
        # P5 (stride 32) → large           (~256 px)
        anchor_sizes = ((32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * 4   # 3 per level → 3 anchors / cell
        self.anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        # 4. RPN
        self.rpn = RegionProposalNetwork(
            anchor_generator=self.anchor_generator,
            in_channels=fpn_channels,
        )

        # 5. RoI heads (second stage)
        self.roi_heads = RoIHeadsModule(
            fpn_channels=fpn_channels,
            num_classes=num_classes,
            fpn_strides={'P2': 4, 'P3': 8, 'P4': 16, 'P5': 32},
        )

        # Register normalization constants as buffers
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std

    def forward(self, images: torch.Tensor, targets: list = None) -> tuple:
        """
        Args:
            images: (B, 3, H, W) — float in [0, 1]
            targets: list of dicts with keys 'boxes' (M, 4) and 'labels' (M,)
                     — required during training

        Returns:
            If training: (None, losses_dict)
            If eval:     (detections_list, {})
        """
        if self.training and targets is None:
            raise ValueError("Targets required when training")

        x = self.normalize(images)
        image_size = images.shape[-2:]

        # 1. Backbone → C2..C5
        backbone_feats = self.backbone(x)

        # 2. FPN → P2..P5
        fpn_feats = self.fpn(backbone_feats)
        fpn_levels = ['P2', 'P3', 'P4', 'P5']
        fpn_feat_list = [fpn_feats[lvl] for lvl in fpn_levels]

        # 3. RPN
        proposals, rpn_losses = self.rpn(fpn_feat_list, image_size, targets)
        proposal_boxes = [p['boxes'] for p in proposals]

        # 4. RoI heads
        detections, head_losses = self.roi_heads(
            fpn_feats, proposal_boxes, image_size, targets
        )

        if self.training:
            losses = {**rpn_losses, **head_losses}
            return None, losses
        return detections, {}
