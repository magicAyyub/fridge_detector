# Fridge Ingredient Detector

A custom two-stage object detector for ingredient recognition, built from scratch in PyTorch.

The only "borrowed" component is the pre-trained ImageNet backbone (ResNet50 or ResNet18). Everything else : Feature Pyramid Network, Region Proposal Network, RoI Align, anchor generation, NMS, detection head and training loop are implemented from scratch.

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

### Backbone (`backbone.py`) 
A pre-trained ResNet with the classification head chopped off. Outputs four feature maps at strides 4, 8, 16, 32. We freeze the stem and `layer1` and BatchNorm running statistics — standard practice for fine-tuning detection on top of an ImageNet model.

### FPN (`fpn.py`) 
Implements the classic top-down FPN. Lateral 1×1 convolutions project each backbone level to 256 channels, then we add 2× upsampled higher levels and apply a 3×3 conv to clean up. The result: every output level (P2..P5) has the same channel count and rich semantics — small objects use P2, large ones use P5.

### Anchor Generator (`utils/anchors.py`)
For each FPN level, generates anchor templates at the configured sizes and aspect ratios, then tiles them across the spatial grid in image coordinates. With 4 levels × 1 size × 3 aspect ratios = 3 anchors per cell per level. For a 192×192 image, that's roughly 9,500 anchors total.

### RPN (`rpn.py`) 
The first stage. A small head (3×3 conv → two 1×1 sibling convs) predicts:
- An **objectness score** per anchor (binary: foreground or background)
- **Box offsets** (tx, ty, tw, th) refining each anchor

Training: each anchor is matched to a ground-truth box by IoU (≥0.7 = positive, <0.3 = negative), then we sample a balanced mini-batch of 256 anchors per image. Loss = BCE on objectness + smooth-L1 on box offsets (positives only).

Inference: for each level, take the top-K anchors by objectness, decode to image-space boxes, clip, drop tiny boxes, run NMS.

### RoI Align (`utils/roi_align.py`)
The classic Mask R-CNN trick that fixes RoI Pool's quantization error. For each proposal, we build a sampling grid (`sampling_ratio × sampling_ratio` points per output bin) in feature-map coordinates, run bilinear interpolation via `grid_sample`, then average across each bin. Output: a fixed-size 7×7×C feature volume per proposal, regardless of the proposal's original size.

### Detection Head (`detection_head.py`)
Two FC layers + two output heads (class scores over `C+1`, per-class box offsets `C×4`). Includes:
- **Per-FPN-level RoI assignment**: small proposals → P2, large → P5 (via the FPN paper's formula `level = floor(4 + log2(√(w·h)/224))`).
- **Proposal sampling**: 128 per image, 25% positive (IoU ≥ 0.5 with a GT) + 75% background.
- **GT augmentation**: GT boxes are added to the proposal set during training to guarantee positive samples.

### NMS (`utils/nms.py`) 
Sort by score → keep best → suppress overlapping (IoU > threshold) → repeat. Includes a `batched_nms` variant that handles multi-class NMS by shifting boxes per class so they can't suppress each other across classes.

## Loss Components

The total loss is a sum of four:

| Loss | What it trains | Loss type |
|---|---|---|
| `rpn_obj_loss` | Is this anchor an object? | Binary cross-entropy |
| `rpn_box_loss` | How to refine the anchor's coordinates | Smooth L1 |
| `cls_loss` | What class is this proposal? | Cross-entropy over `C+1` classes |
| `box_loss` | Refine proposal coordinates per class | Smooth L1 (positives only) |

Watch them all decrease during training. If `rpn_obj_loss` stays high, your anchor sizes probably don't match your data. If `cls_loss` stays high but RPN losses drop, the second stage isn't getting good proposals.