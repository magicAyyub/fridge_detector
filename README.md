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

## API Serving & Environment Profiles

The API backend is a unified FastAPI service supporting different environments (development and production) via the `APP_ENV` environment variable.

### 1. Environment Configurations
The server uses environment profiles to load configurations:
- `.env.development`: Used for local development. Copy `.env.development.example` to `.env.development` to start. S3 downloads are skipped and local checkpoints are used directly.
- `.env.production.example`: A template for production. Copy this file to `.env.production` (do not commit it!) and specify your S3 bucket and database details.

In production, system environment variables (e.g. set via ECS or App Runner) automatically override `.env` files.

### 2. S3 Weight Downloading
If `BUCKET_NAME` is defined in the active environment, the startup script will automatically check and download weights from AWS S3:
```bash
# Verify download behavior manually:
APP_ENV=production uv run python scripts/download_weights.py
```

### 3. Run Locally (Development)
To run the server in development mode using local weights:
```bash
# Start server with development profile
APP_ENV=development PYTHONPATH=src:. uv run python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000/docs` to view the Swagger API documentation.

### 4. Running with Docker & ECS (Production)
You can build the image, push to ECR, and run in ECS:

```bash
# Build the Docker image
docker build -t fridge-detector .

# Run the container locally (simulating development via volume mounting)
docker run -p 8000:8000 \
  -e APP_ENV=development \
  -e PORT=8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/data:/app/data \
  fridge-detector

# Run in production (downloads weights from S3 using IAM role/credentials)
docker run -p 8000:8000 \
  -e APP_ENV=production \
  -e PORT=8000 \
  -e BUCKET_NAME=my-prod-weights-bucket \
  -e DATABASE_URL="postgresql://user:pass@ep-xxx.neon.tech/dbname?sslmode=require" \
  fridge-detector
```

### 5. Connecting Mobile App (Physical Device vs Simulator)

When testing the Expo mobile application with a local backend:
- **Simulator (iOS or Android)**: You can connect using `localhost` or `http://127.0.0.1:8000`.
- **Physical Phone**: The phone and the development computer must be on the same Wi-Fi network (or connection sharing). Since the phone cannot resolve `localhost`, you must configure the backend to use your computer's local network IP address.

To set up a physical phone:
1. Find your computer's local IP address. On macOS:
   ```bash
   ipconfig getifaddr en0
   # (If using connection sharing, you can check system preferences or ifconfig)
   ```
2. Start the Expo app by overriding the API URL with your computer's local IP (e.g. `172.20.10.10`):
   ```bash
   EXPO_PUBLIC_API_URL=http://172.20.10.10:8000 npm run start
   ```
   This overrides the `apiBaseUrl` configuration dynamically without needing to edit `runtime-config.json` inside the repository.

## Future work

Improve robustness with more fridge scenes, more lighting variation, and a larger product vocabulary.
