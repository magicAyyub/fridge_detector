#!/bin/bash
# =========================================================================
# Start Fridge Detector FastAPI Backend in Development Mode
# =========================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Resolve script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== [INFO] Starting FastAPI Backend in Development Mode ==="

# Check checkpoints
BEST_CKPT="checkpoints/best.pt"
SAM_CKPT="checkpoints/sam2.1_hiera_tiny.pt"

if [ ! -f "$BEST_CKPT" ]; then
    echo "=== [WARNING] FRCNN checkpoint not found at: $BEST_CKPT ==="
    echo "Local object detection inference might fail."
fi

if [ ! -f "$SAM_CKPT" ]; then
    echo "=== [WARNING] SAM 2 checkpoint not found at: $SAM_CKPT ==="
    echo "Local image segmentation might fail."
fi

# Run the backend
echo "=== [INFO] Running server on http://localhost:8000 ==="
APP_ENV=development PYTHONPATH=src:. uv run python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
