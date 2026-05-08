# Fridge Ingredient Detector

A custom two-stage object detector for ingredient recognition, built from scratch in PyTorch.

The only "borrowed" component is the pre-trained ImageNet backbone (ResNet50 or ResNet18). Everything else — Feature Pyramid Network, Region Proposal Network, RoI Align, anchor generation, NMS, detection head, training loop — is implemented from first principles to demonstrate understanding of the full detection stack.

---

## Project Structure

```
fridge_detector/
├── models/
│   ├── backbone.py            # Pre-trained ResNet (the only borrowed piece)
│   ├── fpn.py                 # Feature Pyramid Network — built from scratch
│   ├── rpn.py                 # Region Proposal Network — built from scratch
│   ├── detection_head.py      # Second-stage classifier + box regressor
│   └── detector.py            # Top-level model wiring it all together
├── utils/
│   ├── box_ops.py             # IoU, encoding/decoding offsets, format conversions
│   ├── anchors.py             # Anchor generator (multi-scale, multi-aspect-ratio)
│   ├── nms.py                 # Non-Maximum Suppression — built from scratch
│   └── roi_align.py           # RoI Align with bilinear sampling
├── data/
│   └── dataset.py             # VOC-format loader + synthetic dataset for testing
├── scripts/
│   ├── test_utils.py          # Unit tests for the utilities
│   ├── smoke_test.py          # End-to-end forward+backward pass test
│   ├── train.py               # Training loop with logging & checkpointing
│   └── predict.py             # Inference + visualization
└── README.md
```

---

## Architecture

```
                  ┌──────────────────────┐
   Image (3,H,W)→ │  ResNet Backbone     │ → C2, C3, C4, C5 (feature maps)
                  │  (pretrained, frozen │
                  │   early layers)      │
                  └──────────────────────┘
                            │
                            ▼
                  ┌──────────────────────┐
                  │  Feature Pyramid     │ → P2, P3, P4, P5 (256 channels each)
                  │  Network (FPN)       │
                  └──────────────────────┘
                            │
                            ▼
                  ┌──────────────────────┐
                  │  Region Proposal     │ → ~300 candidate boxes per image
                  │  Network (RPN)       │   ("is there an object?" + "where?")
                  │  - 9 anchors / cell  │
                  │  - 4 FPN levels      │
                  └──────────────────────┘
                            │
                            ▼
                  ┌──────────────────────┐
                  │  RoI Align           │ → 7×7×256 features per proposal
                  │  (level-routed)      │   (bilinear interp, no quantization)
                  └──────────────────────┘
                            │
                            ▼
                  ┌──────────────────────┐
                  │  Detection Head      │ → class probabilities + box offsets
                  │  FC → FC → cls + box │
                  └──────────────────────┘
                            │
                            ▼
                  ┌──────────────────────┐
                  │  NMS + thresholding  │ → final detections (boxes, classes, scores)
                  └──────────────────────┘
```

---

## How Each Component Works

### 1. Backbone (`backbone.py`) — borrowed
A pre-trained ResNet with the classification head chopped off. Outputs four feature maps at strides 4, 8, 16, 32. We freeze the stem and `layer1` and BatchNorm running statistics — standard practice for fine-tuning detection on top of an ImageNet model.

### 2. FPN (`fpn.py`) — from scratch
Implements the classic top-down FPN. Lateral 1×1 convolutions project each backbone level to 256 channels, then we add 2× upsampled higher levels and apply a 3×3 conv to clean up. The result: every output level (P2..P5) has the same channel count and rich semantics — small objects use P2, large ones use P5.

### 3. Anchor Generator (`utils/anchors.py`) — from scratch
For each FPN level, generates anchor templates at the configured sizes and aspect ratios, then tiles them across the spatial grid in image coordinates. With 4 levels × 1 size × 3 aspect ratios = 3 anchors per cell per level. For a 192×192 image, that's roughly 9,500 anchors total.

### 4. RPN (`rpn.py`) — from scratch
The first stage. A small head (3×3 conv → two 1×1 sibling convs) predicts:
- An **objectness score** per anchor (binary: foreground or background)
- **Box offsets** (tx, ty, tw, th) refining each anchor

Training: each anchor is matched to a ground-truth box by IoU (≥0.7 = positive, <0.3 = negative), then we sample a balanced mini-batch of 256 anchors per image. Loss = BCE on objectness + smooth-L1 on box offsets (positives only).

Inference: for each level, take the top-K anchors by objectness, decode to image-space boxes, clip, drop tiny boxes, run NMS.

### 5. RoI Align (`utils/roi_align.py`) — from scratch
The classic Mask R-CNN trick that fixes RoI Pool's quantization error. For each proposal, we build a sampling grid (`sampling_ratio × sampling_ratio` points per output bin) in feature-map coordinates, run bilinear interpolation via `grid_sample`, then average across each bin. Output: a fixed-size 7×7×C feature volume per proposal, regardless of the proposal's original size.

### 6. Detection Head (`detection_head.py`) — from scratch
Two FC layers + two output heads (class scores over `C+1`, per-class box offsets `C×4`). Includes:
- **Per-FPN-level RoI assignment**: small proposals → P2, large → P5 (via the FPN paper's formula `level = floor(4 + log2(√(w·h)/224))`).
- **Proposal sampling**: 128 per image, 25% positive (IoU ≥ 0.5 with a GT) + 75% background.
- **GT augmentation**: GT boxes are added to the proposal set during training to guarantee positive samples.

### 7. NMS (`utils/nms.py`) — from scratch
Sort by score → keep best → suppress overlapping (IoU > threshold) → repeat. Includes a `batched_nms` variant that handles multi-class NMS by shifting boxes per class so they can't suppress each other across classes.

---

## How to Run

### Quick start (synthetic data — no downloads needed)

```bash
# 1. Test the utilities (IoU, anchors, NMS, RoI Align)
python scripts/test_utils.py

# 2. End-to-end smoke test (forward + backward)
python scripts/smoke_test.py

# 3. Train on the synthetic toy dataset (verifies everything learns)
python scripts/train.py --synthetic --epochs 5 --batch-size 4 \
    --backbone resnet18 --fpn-channels 64 --image-size 192

# 4. Run inference and visualize
python scripts/predict.py --checkpoint checkpoints/best.pt \
    --backbone resnet18 --fpn-channels 64 --image-size 192 \
    --output prediction.png
```

### Real training on Freiburg Groceries (with bounding boxes)

1. Download the dataset:
   ```bash
   git clone https://github.com/aleksandar-aleksandrov/groceries-object-detection-dataset
   ```
2. Organize it into `images/` and `annotations/` (Pascal VOC XML format).
3. Train:
   ```bash
   python scripts/train.py \
       --data-dir /path/to/dataset \
       --class-names beans cake candy cereal chips chocolate coffee corn fish \
                     flour honey jam juice milk nuts oil pasta rice soda \
                     spices sugar tea tomato_sauce vinegar water \
       --backbone resnet50 --fpn-channels 256 \
       --image-size 512 --batch-size 8 --epochs 30 \
       --lr 1e-4
   ```

---

## Loss Components

The total loss is a sum of four:

| Loss | What it trains | Loss type |
|---|---|---|
| `rpn_obj_loss` | Is this anchor an object? | Binary cross-entropy |
| `rpn_box_loss` | How to refine the anchor's coordinates | Smooth L1 |
| `cls_loss` | What class is this proposal? | Cross-entropy over `C+1` classes |
| `box_loss` | Refine proposal coordinates per class | Smooth L1 (positives only) |

Watch them all decrease during training. If `rpn_obj_loss` stays high, your anchor sizes probably don't match your data. If `cls_loss` stays high but RPN losses drop, the second stage isn't getting good proposals.

---

## Results from the Smoke Run

After 3 epochs on the synthetic dataset (CPU, ResNet18 + FPN64):

| Epoch | Total loss | Detections / image | Mean confidence |
|---:|---:|---:|---:|
| 1 | 1.08 | 82.8 | 0.25 |
| 2 | 0.39 | 17.6 | 0.48 |
| 3 | 0.25 | 9.3 | 0.58 |

The model is clearly learning: total loss drops 4×, confidence doubles, and the model stops spamming detections (it's learning what's *not* an object).

---

## Mobile Deployment Notes

For the actual mobile app:
- Switch backbone to MobileNetV3 (depthwise separable convolutions, ~5× fewer params than ResNet50).
- Reduce `fpn_channels` to 64 or 128.
- Export to ONNX, then convert to TFLite (Android) or Core ML (iOS).
- Quantize to INT8 — Faster R-CNN-style detectors can usually take 4-bit / 8-bit weights with minimal accuracy loss after post-training quantization.
- Consider replacing the two-stage architecture with a single-stage anchor-based head (RetinaNet-style) on top of the same FPN — same backbone, much faster inference.

---

## What This Demonstrates

| Component | Built? | Skill demonstrated |
|---|:---:|---|
| ResNet backbone | borrowed (allowed) | Understanding which layers to use, how to truncate, how to handle frozen BN |
| FPN | ✅ from scratch | Multi-scale feature fusion, lateral connections, top-down pathway |
| Anchor generator | ✅ from scratch | Coordinate systems, stride math, anchor parameterization |
| RPN | ✅ from scratch | Anchor matching, sample balancing, objectness + regression heads |
| RoI Align | ✅ from scratch | Bilinear sampling, sub-pixel alignment, `grid_sample` mechanics |
| Detection head | ✅ from scratch | Multi-task heads, per-class regression, FPN level routing |
| NMS | ✅ from scratch | Greedy suppression, IoU-based filtering, multi-class variant |
| Training loop | ✅ from scratch | Multi-task loss, sampling, gradient clipping, LR scheduling |

Every component can be inspected, tested in isolation, and modified — there are no opaque black boxes.
