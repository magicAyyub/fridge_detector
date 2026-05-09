# Fridge Detector, WhatIEat vision backend

A PyTorch Faster R-CNN implementation for grocery detection in real fridge scenes, trained from a Roboflow dataset export and served to the WhatIEat mobile app.

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
│   └── dataset.py             # Roboflow/YOLO loader + synthetic dataset for testing
├── scripts/
│   ├── test_utils.py          # Unit tests for the utilities
│   ├── smoke_test.py          # End-to-end forward+backward pass test
│   ├── train.py               # Training loop with logging & checkpointing
│   ├── predict.py             # Inference + visualization
│   ├── serve_api.py           # FastAPI inference backend for WhatIEat
│   └── download_roboflow.py   # Download Roboflow export with your API key
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

## Roboflow Dataset Workflow

This project now uses a single real-fridge dataset workflow based on a Roboflow export in YOLO format.

1. Export your API key:
```bash
export ROBOFLOW_API_KEY=your_api_key_here
```

2. Download the dataset:
```bash
uv run python scripts/download_roboflow.py \
   --workspace practicum-ziryz \
   --project fridge-dataset-oi7ld \
   --version 1
```

This downloads a dataset root like:

```text
data/roboflow/
   data.yaml
   train/images/
   train/labels/
   valid/images/
   valid/labels/
```

3. Train locally:
```bash
uv run python scripts/train.py \
   --config configs/local.yaml \
   --data-dir data/roboflow
```

`data.yaml` is the source of truth for class names and split paths. There is no separate XML/VOC path anymore.

## Kaggle Training

Use `fridge-detector.ipynb` after uploading only the source-code dataset for this repo.

Before running the notebook on Kaggle:

1. Enable Internet in the notebook session.
2. Add a Kaggle secret named `ROBOFLOW_API_KEY`.
3. Upload this repo as a Kaggle dataset so it is mounted at a path like `/kaggle/input/fridge-detector`.

Then set the source path in notebook cell 1. The dataset path stays in `/kaggle/working` because the notebook downloads it directly from Roboflow:

```python
SRC_DIR = '/kaggle/input/fridge-detector'
DATA_DIR = '/kaggle/working/data/roboflow'
```

The notebook will first run:

```bash
python /kaggle/input/fridge-detector/scripts/download_roboflow.py \
   --workspace practicum-ziryz \
   --project fridge-dataset-oi7ld \
   --version 1 \
   --output-dir /kaggle/working/data/roboflow
```

Then the training cell will run:

```bash
python /kaggle/input/fridge-detector/scripts/train.py \
   --config /kaggle/input/fridge-detector/configs/kaggle.yaml \
   --data-dir /kaggle/working/data/roboflow
```

Checkpoints are written to `/kaggle/working/checkpoints`.

## API Serving

After training, place the chosen checkpoint in `checkpoints/` and run the inference API:

```bash
uv run python scripts/serve_api.py --checkpoint checkpoints/best.pt --host 0.0.0.0 --port 8000
```

## Future work

Improve robustness with more fridge scenes, more lighting variation, and a larger product vocabulary.
