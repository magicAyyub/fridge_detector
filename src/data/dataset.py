"""Dataset loader.

Supports:
  1. Roboflow/YOLO-style datasets with ``data.yaml`` and ``labels/*.txt``
  2. A small synthetic toy dataset for smoke-testing the training loop

Each sample yields:
  - image: float tensor (3, H, W) in [0, 1]
  - target: dict with
      'boxes':  float (M, 4) in xyxy
      'labels': long  (M,)  — class IDs in [1, num_classes] (0 reserved for background)
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import yaml


SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')


def _normalize_class_names(names) -> list[str]:
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    raise ValueError(f'Unsupported names field in data.yaml: {type(names)!r}')


def _resolve_split_path(root: Path, split_value: str) -> Path:
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path.resolve()

    candidates = [(root / split_path).resolve()]

    trimmed_parts = list(split_path.parts)
    while trimmed_parts and trimmed_parts[0] == '..':
        trimmed_parts = trimmed_parts[1:]
        if trimmed_parts:
            candidates.append((root / Path(*trimmed_parts)).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def load_roboflow_data_config(data_dir: str) -> tuple[list[str], dict[str, Path]]:
    """Read Roboflow ``data.yaml`` and resolve split image dirs."""
    root = Path(data_dir).expanduser().resolve()
    data_yaml = root / 'data.yaml'
    if not data_yaml.exists():
        raise FileNotFoundError(f'Expected Roboflow data.yaml at {data_yaml}')

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f) or {}

    class_names = _normalize_class_names(cfg.get('names', []))
    if not class_names:
        raise ValueError(f'No class names found in {data_yaml}')

    split_dirs: dict[str, Path] = {}
    for key in ('train', 'val', 'valid', 'test'):
        value = cfg.get(key)
        if value:
            split_dirs[key] = _resolve_split_path(root, value)

    if 'train' not in split_dirs:
        raise ValueError(f'No train split defined in {data_yaml}')
    return class_names, split_dirs


class RoboflowDetectionDataset(Dataset):
    """YOLO txt annotations exported by Roboflow.

    Expected layout:
      dataset/
        data.yaml
        train/images/*.jpg
        train/labels/*.txt
        valid/images/*.jpg
        valid/labels/*.txt
    """

    def __init__(self, image_dir: str, label_dir: str, class_names: list[str],
                 image_size: int = 512, augment: bool = False):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.class_names = class_names
        self.image_size = image_size
        self.augment = augment

        if not self.image_dir.exists():
            raise FileNotFoundError(f'Image dir not found: {self.image_dir}')
        if not self.label_dir.exists():
            raise FileNotFoundError(f'Label dir not found: {self.label_dir}')

        self.samples: list[tuple[Path, Path]] = []
        for image_path in sorted(self.image_dir.rglob('*')):
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            label_path = self.label_dir / f'{image_path.stem}.txt'
            self.samples.append((image_path, label_path))

        if not self.samples:
            raise ValueError(f'No images found in {self.image_dir}')

    def __len__(self):
        return len(self.samples)

    def _parse_yolo_label(self, label_path: Path, orig_w: int, orig_h: int) -> tuple[list, list]:
        boxes, labels = [], []
        if not label_path.exists():
            return boxes, labels

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls_id = int(float(parts[0]))
                if cls_id < 0 or cls_id >= len(self.class_names):
                    continue

                cx, cy, bw, bh = (float(value) for value in parts[1:])
                x1 = (cx - bw / 2.0) * orig_w
                y1 = (cy - bh / 2.0) * orig_h
                x2 = (cx + bw / 2.0) * orig_w
                y2 = (cy + bh / 2.0) * orig_h

                x1 = max(0.0, min(float(orig_w), x1))
                y1 = max(0.0, min(float(orig_h), y1))
                x2 = max(0.0, min(float(orig_w), x2))
                y2 = max(0.0, min(float(orig_h), y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id + 1)

        return boxes, labels

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        boxes, labels = self._parse_yolo_label(label_path, orig_w, orig_h)

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        boxes = [[b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
                 for b in boxes]

        if self.augment and torch.rand(1).item() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            new_boxes = []
            for b in boxes:
                x1, y1, x2, y2 = b
                new_boxes.append([self.image_size - x2, y1, self.image_size - x1, y2])
            boxes = new_boxes

        image = TF.to_tensor(image)
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes
                     else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels
                      else torch.zeros((0,), dtype=torch.int64),
        }
        return image, target


class SyntheticFridgeDataset(Dataset):
    """Small synthetic dataset for end-to-end smoke tests."""

    CLASS_COLORS = {
        1: (220, 30, 30),
        2: (60, 180, 60),
        3: (250, 230, 80),
        4: (245, 245, 245),
    }
    CLASS_NAMES = ['tomato', 'lettuce', 'butter', 'milk']

    def __init__(self, length: int = 200, image_size: int = 256, max_objects: int = 4,
                 seed: int = 42):
        self.length = length
        self.image_size = image_size
        self.max_objects = max_objects
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed + idx)

        S = self.image_size
        image = torch.full((3, S, S), 0.6) + 0.05 * torch.randn(3, S, S, generator=g)

        n_objects = int(torch.randint(1, self.max_objects + 1, (1,), generator=g).item())
        boxes, labels = [], []
        for _ in range(n_objects):
            cls = int(torch.randint(1, len(self.CLASS_COLORS) + 1, (1,), generator=g).item())
            color = torch.tensor(self.CLASS_COLORS[cls], dtype=torch.float32) / 255.0

            min_size, max_size = S // 8, S // 3
            w = int(torch.randint(min_size, max_size, (1,), generator=g).item())
            h = int(torch.randint(min_size, max_size, (1,), generator=g).item())
            x1 = int(torch.randint(0, S - w, (1,), generator=g).item())
            y1 = int(torch.randint(0, S - h, (1,), generator=g).item())
            x2, y2 = x1 + w, y1 + h

            image[:, y1:y2, x1:x2] = color.view(3, 1, 1)
            image[:, y1:y2, x1:x2] += 0.03 * torch.randn(3, h, w, generator=g)

            boxes.append([x1, y1, x2, y2])
            labels.append(cls)

        image = image.clamp(0, 1)
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    """Custom collate — images stack, targets stay variable-length."""
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets