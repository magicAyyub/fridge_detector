"""
Feature Pyramid Network (FPN), built from scratch.

The backbone gives us features at strides 4, 8, 16, 32 (C2, C3, C4, C5).
Higher levels have stronger semantics but lower resolution.

FPN combines them via:
  1. A 1x1 conv on each Cx to project to a common channel count (256).
  2. A top-down pathway: upsample the higher level by 2x, add to the
     lateral projection of the lower level.
  3. A 3x3 conv on each merged feature to clean up upsampling aliasing.

Result: feature maps P2, P3, P4, P5. same channels (256) at every level,
with strong semantics throughout. Small objects use P2, large objects use P5.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_dict: dict, out_channels: int = 256):
        """
        Args:
            in_channels_dict: e.g. {'C2': 256, 'C3': 512, 'C4': 1024, 'C5': 2048}
            out_channels: channel count of every output level (typically 256)
        """
        super().__init__()
        self.levels = ['C2', 'C3', 'C4', 'C5']

        # 1x1 lateral convolutions to unify channel count
        self.lateral_convs = nn.ModuleDict({
            lvl: nn.Conv2d(in_channels_dict[lvl], out_channels, kernel_size=1)
            for lvl in self.levels
        })

        # 3x3 output convolutions, one per level, to smooth out merged features
        self.output_convs = nn.ModuleDict({
            lvl: nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for lvl in self.levels
        })

        # Initialize with Xavier (helps training stability)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.out_channels = out_channels

    def forward(self, features: dict) -> dict:
        """
        Args:
            features: dict with 'C2', 'C3', 'C4', 'C5' from backbone

        Returns:
            dict with 'P2', 'P3', 'P4', 'P5',  all with `out_channels` channels
        """
        # Top-down pass, start from C5 (deepest)
        c5 = features['C5']
        c4 = features['C4']
        c3 = features['C3']
        c2 = features['C2']

        # Project each level to common channels
        p5 = self.lateral_convs['C5'](c5)
        p4 = self.lateral_convs['C4'](c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_convs['C3'](c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_convs['C2'](c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')

        # 3x3 conv to clean up
        p5 = self.output_convs['C5'](p5)
        p4 = self.output_convs['C4'](p4)
        p3 = self.output_convs['C3'](p3)
        p2 = self.output_convs['C2'](p2)

        return {'P2': p2, 'P3': p3, 'P4': p4, 'P5': p5}
