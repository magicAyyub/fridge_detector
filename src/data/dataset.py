"""
Dataset loader.

Supports:
  1. Pascal VOC-style annotations (XML files) — for the Freiburg Groceries
     extended dataset.
  2. A small synthetic toy dataset — for smoke-testing the pipeline without
     downloading anything.

Each sample yields:
  - image: float tensor (3, H, W) in [0, 1]
  - target: dict with
      'boxes':  float (M, 4) in xyxy
      'labels': long  (M,)  — class IDs in [1, num_classes] (0 reserved for background)
"""
import os
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class VOCDetectionDataset(Dataset):
    """Pascal-VOC-style dataset (XML annotations)."""

    def __init__(self, image_dir: str, annot_dir: str, class_names: list,
                 image_size: int = 512, augment: bool = False):
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.class_names = class_names
        self.class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}  # 0 = bg
        self.image_size = image_size
        self.augment = augment

        # Index all valid (image, annot) pairs
        self.samples = []
        for fname in sorted(os.listdir(annot_dir)):
            if not fname.endswith('.xml'):
                continue
            stem = fname[:-4]
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                img_path = os.path.join(image_dir, stem + ext)
                if os.path.exists(img_path):
                    self.samples.append((img_path, os.path.join(annot_dir, fname)))
                    break

    def __len__(self):
        return len(self.samples)

    def _parse_voc_xml(self, xml_path: str) -> tuple:
        root = ET.parse(xml_path).getroot()
        boxes, labels = [], []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.class_to_idx:
                continue
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_idx[name])
        return boxes, labels

    def __getitem__(self, idx):
        img_path, xml_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        boxes, labels = self._parse_voc_xml(xml_path)

        # Resize image and scale boxes
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        boxes = [[b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
                 for b in boxes]

        # Optional augmentation: horizontal flip
        if self.augment and torch.rand(1).item() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            new_boxes = []
            for b in boxes:
                x1, y1, x2, y2 = b
                new_boxes.append([self.image_size - x2, y1, self.image_size - x1, y2])
            boxes = new_boxes

        image = TF.to_tensor(image)  # (3, H, W) in [0, 1]
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes
                     else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels
                      else torch.zeros((0,), dtype=torch.int64),
        }
        return image, target


class SyntheticFridgeDataset(Dataset):
    """
    A toy dataset that draws colored rectangles at known positions.
    Used to smoke-test the training loop end-to-end without real data.

    Class colors:
      1: red    (tomato)
      2: green  (lettuce)
      3: yellow (butter)
      4: white  (milk)
    """
    CLASS_COLORS = {
        1: (220, 30, 30),     # tomato
        2: (60, 180, 60),     # lettuce
        3: (250, 230, 80),    # butter
        4: (245, 245, 245),   # milk
    }
    CLASS_NAMES = ['tomato', 'lettuce', 'butter', 'milk']

    def __init__(self, length: int = 200, image_size: int = 256, max_objects: int = 4,
                 seed: int = 42):
        self.length = length
        self.image_size = image_size
        self.max_objects = max_objects
        # Deterministic but per-sample
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Per-sample RNG so results are reproducible
        g = torch.Generator().manual_seed(self.seed + idx)

        S = self.image_size
        # Background — soft gray with slight noise
        image = torch.full((3, S, S), 0.6) + 0.05 * torch.randn(3, S, S, generator=g)

        n_objects = int(torch.randint(1, self.max_objects + 1, (1,), generator=g).item())
        boxes, labels = [], []
        for _ in range(n_objects):
            cls = int(torch.randint(1, len(self.CLASS_COLORS) + 1, (1,), generator=g).item())
            color = torch.tensor(self.CLASS_COLORS[cls], dtype=torch.float32) / 255.0

            # Random box, with some minimum size
            min_size, max_size = S // 8, S // 3
            w = int(torch.randint(min_size, max_size, (1,), generator=g).item())
            h = int(torch.randint(min_size, max_size, (1,), generator=g).item())
            x1 = int(torch.randint(0, S - w, (1,), generator=g).item())
            y1 = int(torch.randint(0, S - h, (1,), generator=g).item())
            x2, y2 = x1 + w, y1 + h

            # Paint rectangle (with a bit of edge softening so it looks less synthetic)
            image[:, y1:y2, x1:x2] = color.view(3, 1, 1)
            # Add small noise inside so the model learns texture too, not just color
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
    """Custom collate — images can stack, but targets stay as a list (variable-length)."""
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets
