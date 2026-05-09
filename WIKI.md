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

---

## Quantity Estimation with SAM

Detection answers "what is present". Quantity needs an additional stage.

### Why add SAM

- Bounding boxes are coarse and include background.
- Quantity proxies are stronger when based on segmented object pixels.
- SAM can be prompted with detector boxes, so we can reuse the current model.

### Proposed Hybrid Pipeline

1. Run current detector (class + score + box).
2. For each accepted detection, prompt SAM with the box.
3. Keep SAM mask with highest stability score for that box.
4. Compute mask-derived features:
     - mask area (pixels)
     - relative area (mask area / image area)
     - shape cues (elongation, compactness)
5. Convert features to quantity using class-specific rules.

### Quantity Rules (V1)

- **Countable items** (`apple`, `banana`, `tomato`, `eggs`):
    quantity = number of filtered instances.
- **Package-like items** (`milk`, `yogurt`, `cheese`, `butter`):
    quantity = estimated packs using instance count and confidence.
- **Bulk items** (`rice`, `lentil`, `beans`):
    quantity = relative amount (low/medium/high) from segmented area.

### Important Constraint

With a single RGB image, exact grams/ml are generally not reliable without
calibration, known container size, or depth. In V1, return practical units
(`count`, `pack`, `level`) instead of fake precision.

### API Output Extension

Keep existing fields and add quantity metadata per ingredient:

- `estimated_quantity`: numeric value
- `unit`: `count | pack | level`
- `method`: `instance_count | area_proxy | rule_based`
- `quantity_confidence`: 0..1

### Model Choice Notes

- Start with SAM 2 image predictor on backend for quality.
- If latency becomes an issue, switch to a lighter variant (for example
    MobileSAM/FastSAM) while keeping the same API contract.

### Rollout Plan

1. **V1 (fast win)**: detector instance counting + class-to-unit rules.
2. **V2**: SAM masks for area-based quantity proxy.
3. **V3**: class-specific estimators (for example egg-carton occupancy,
     bottle fill level).
4. **V4**: temporal smoothing across scans for stable household inventory.