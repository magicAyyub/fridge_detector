"""
Anchor box generator.

Anchors are pre-defined reference boxes placed at every spatial location
of a feature map. The network learns to predict offsets that adjust each
anchor to fit a real object.

For a feature map of size (H_feat, W_feat) extracted from an image of size
(H_img, W_img), the stride is roughly H_img / H_feat. Each cell on the
feature map corresponds to a `stride x stride` patch in the original image.
At each cell we place K anchors (K = num_scales × num_aspect_ratios).
"""
import torch
import torch.nn as nn


class AnchorGenerator(nn.Module):
    """
    Generates anchor boxes for one or more feature maps (FPN-ready).

    For each feature map level, anchors are tiled at every spatial position.
    A single anchor is parameterized by (size, aspect_ratio):
        w = size * sqrt(aspect_ratio)
        h = size / sqrt(aspect_ratio)

    Args:
        sizes: tuple of tuples; sizes[i] = anchor sizes (in image pixels) for feature level i
        aspect_ratios: tuple of tuples; aspect_ratios[i] for feature level i
                       (same for all levels by default)

    Example:
        sizes = ((32,), (64,), (128,), (256,), (512,))   # one size per FPN level
        aspect_ratios = ((0.5, 1.0, 2.0),) * 5            # 3 aspect ratios per level
        → 3 anchors per feature-map cell at each level
    """

    def __init__(self, sizes=((32, 64, 128, 256, 512),),
                 aspect_ratios=((0.5, 1.0, 2.0),)):
        super().__init__()
        # Allow either a single tuple (single feature map) or a tuple-of-tuples (FPN)
        if not isinstance(sizes[0], (list, tuple)):
            sizes = (sizes,)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,)
        # Broadcast aspect_ratios if single set provided
        if len(aspect_ratios) == 1 and len(sizes) > 1:
            aspect_ratios = aspect_ratios * len(sizes)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

        # Pre-compute the K reference anchor templates per level (centered at origin)
        self.cell_anchors = [
            self._generate_cell_anchors(s, a)
            for s, a in zip(sizes, aspect_ratios)
        ]

    @staticmethod
    def _generate_cell_anchors(sizes, aspect_ratios) -> torch.Tensor:
        """
        Generate the K base anchors centered at (0, 0) for one feature level.

        Returns: (K, 4) tensor in xyxy format, K = len(sizes) * len(aspect_ratios)
        """
        anchors = []
        for size in sizes:
            for ratio in aspect_ratios:
                w = size * (ratio ** 0.5)
                h = size / (ratio ** 0.5)
                # Centered at origin: x1 = -w/2, y1 = -h/2, x2 = +w/2, y2 = +h/2
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(anchors, dtype=torch.float32)

    def num_anchors_per_location(self) -> list:
        """Return K (anchors per cell) for each feature map level."""
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def forward(self, image_size: tuple, feature_maps: list) -> list:
        """
        Generate anchors tiled across each feature map.

        Args:
            image_size: (H_img, W_img) original image dimensions
            feature_maps: list of tensors, each (B, C, H_feat, W_feat)

        Returns:
            list of (N_i, 4) tensors. One per feature level. N_i = H_i * W_i * K_i.
            Anchors are in image-space xyxy coordinates.
        """
        anchors_per_level = []
        H_img, W_img = image_size

        for feat_map, base_anchors in zip(feature_maps, self.cell_anchors):
            _, _, H_feat, W_feat = feat_map.shape
            stride_y = H_img / H_feat
            stride_x = W_img / W_feat
            device = feat_map.device

            # Generate grid centers in image-space coordinates.
            # Cell (i, j) corresponds to image center ((j + 0.5) * stride_x, (i + 0.5) * stride_y).
            shifts_x = (torch.arange(W_feat, device=device, dtype=torch.float32) + 0.5) * stride_x
            shifts_y = (torch.arange(H_feat, device=device, dtype=torch.float32) + 0.5) * stride_y
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            shifts = torch.stack([shift_x.flatten(), shift_y.flatten(),
                                  shift_x.flatten(), shift_y.flatten()], dim=1)  # (H*W, 4)

            # Add base anchors (K, 4) to each shift (H*W, 4) → (H*W, K, 4)
            base = base_anchors.to(device)
            anchors = (shifts[:, None, :] + base[None, :, :]).reshape(-1, 4)
            anchors_per_level.append(anchors)

        return anchors_per_level
