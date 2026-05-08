"""
Backbone: a pre-trained ResNet50 with the classification head removed.

We allow the use of a pre-trained backbone (this is the only "borrowed" piece).
Everything that turns features → detections is built from scratch.

Output: a feature map at stride 16 (after layer3) with 1024 channels.
For multi-scale detection (FPN), see detector.py for how multiple levels are used.
"""
import torch
import torch.nn as nn
from torchvision.models import (resnet50, ResNet50_Weights,
                                  resnet18, ResNet18_Weights)


# Channel counts for the supported backbones, indexed by stage (C2, C3, C4, C5)
_BACKBONE_CHANNELS = {
    'resnet18': {'C2': 64, 'C3': 128, 'C4': 256, 'C5': 512},
    'resnet50': {'C2': 256, 'C3': 512, 'C4': 1024, 'C5': 2048},
}


class ResNetBackbone(nn.Module):
    """
    Truncated ResNet (18 or 50).

    Layout (ResNet50 / ResNet18 channel counts):
        conv1 + bn + relu + maxpool   →   stride 4,   64 channels
        layer1                         →   stride 4,  256 / 64 channels
        layer2                         →   stride 8,  512 / 128 channels
        layer3                         →   stride 16, 1024 / 256 channels
        layer4                         →   stride 32, 2048 / 512 channels
        avgpool + fc (REMOVED)

    We expose the outputs of layer1..layer4 so an FPN can use them.
    """

    def __init__(self, arch: str = 'resnet50', pretrained: bool = True,
                 freeze_early_layers: bool = True):
        super().__init__()
        if arch == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = resnet50(weights=weights)
        elif arch == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        # Stem: conv1 → bn1 → relu → maxpool
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1   # stride 4,  output 256 channels
        self.layer2 = backbone.layer2   # stride 8,  output 512 channels
        self.layer3 = backbone.layer3   # stride 16, output 1024 channels
        self.layer4 = backbone.layer4   # stride 32, output 2048 channels

        # Output channel count per stage (used by FPN)
        self.out_channels = dict(_BACKBONE_CHANNELS[arch])
        # Stride per stage (used by anchor generator)
        self.strides = {'C2': 4, 'C3': 8, 'C4': 16, 'C5': 32}

        # Freezing the stem and layer1 is standard for fine-tuning detection
        # on top of an ImageNet-pretrained backbone. Their features are very
        # general (edges/textures) and small batch sizes make BN unstable.
        if freeze_early_layers:
            for p in self.stem.parameters():
                p.requires_grad = False
            for p in self.layer1.parameters():
                p.requires_grad = False
            # Also freeze BN running stats throughout (common practice)
            self._freeze_bn()

    def _freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False

    def train(self, mode: bool = True):
        # Override .train() so BatchNorm stays in eval mode even after .train()
        super().train(mode)
        self._freeze_bn()
        return self

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 3, H, W) image tensor (normalized with ImageNet stats)

        Returns:
            dict with keys 'C2', 'C3', 'C4', 'C5' — feature maps at increasing strides
        """
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {'C2': c2, 'C3': c3, 'C4': c4, 'C5': c5}
