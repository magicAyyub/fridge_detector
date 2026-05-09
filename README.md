# Fridge Detector, WhatIEat vision backend


A PyTorch Implementation of Faster R-CNN for Grocery Item Detection in Fridge Images, built from scratch with a custom backbone and FPN.

---

## Project Structure

```
fridge_detector/
├── models/
│   ├── backbone.py            # Pre-trained ResNet (the only borrowed piece)
│   ├── fpn.py                 # Feature Pyramid Network 
│   ├── rpn.py                 # Region Proposal Network 
│   ├── detection_head.py      # Second-stage classifier + box regressor
│   └── detector.py            # Top-level model wiring it all together
├── utils/
│   ├── box_ops.py             # IoU, encoding/decoding offsets, format conversions
│   ├── anchors.py             # Anchor generator (multi-scale, multi-aspect-ratio)
├── data/
│   └── dataset.py             # VOC-format loader + synthetic dataset for testing
├── scripts/
│   ├── test_utils.py          # Unit tests for the utilities
│   ├── smoke_test.py          # End-to-end forward+backward pass test
│   ├── train.py               # Training loop with logging & checkpointing
│   └── predict.py             # Inference + visualization
└── README.md
```

## How to Run

### Quick start (synthetic data)

```bash
# Test the utilities (IoU, anchors, NMS, RoI Align)
python scripts/test_utils.py

# End-to-end smoke test (forward + backward)
python scripts/smoke_test.py

# Train on the synthetic toy dataset (verifies everything learns)
python scripts/train.py --synthetic --epochs 5 --batch-size 4 \
    --backbone resnet18 --fpn-channels 64 --image-size 192

# Run inference and visualize
python scripts/predict.py --checkpoint checkpoints/best.pt \
    --backbone resnet18 --fpn-channels 64 --image-size 192 \
    --output prediction.png
```

## Kaggle Training

Note : Change `magicayyub` to your Kaggle username in the paths below.
The notebook is in the root of the project as fridge_detector.ipynb.

`Before running, attach these 3 datasets to this notebook`:

1. **Source code**, zip and upload your project:
```bash
zip -r fridge_detector_src.zip src/ scripts/ configs/ pyproject.toml -x '*.pyc' -x '__pycache__/*'
```
   Upload on kaggle.com: Datasets → New Dataset → `fridge_detector_src.zip`
   → Kaggle mounts at: `/kaggle/input/datasets/magicayyub/fridge-detector/`

2. **Images**, use the existing public dataset (images only):
   https://www.kaggle.com/datasets/mayarmohamedswilam/freiburg-groceries
   → Kaggle mounts at: `/kaggle/input/datasets/mayarmohamedswilam/freiburg-groceries/images/`

3. **Annotations**, use the existing public dataset (XML files, 19 MB):
   https://www.kaggle.com/datasets/magicayyub/freiburg-groceries-annotations
   → Kaggle mounts at: `/kaggle/input/datasets/magicayyub/freiburg-groceries-annotations/annotations/`

## API Serving

After training on kaggle, download the two output files and place them in the `checkpoints/` directory, then run the inference API:

```bash
uv run python scripts/serve_api.py --checkpoint checkpoints/best.pt --host 0.0.0.0 --port 8000
```

## Future work

add more classe detection throught new dataset and add variation of images to make the model more robust and accurate.
