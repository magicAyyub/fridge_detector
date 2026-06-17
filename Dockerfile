FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition file
COPY pyproject.toml ./

# Install dependencies directly into system Python, forcing CPU-only wheels
RUN uv pip install --system --no-cache -r pyproject.toml --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# Copy configs
COPY configs/ ./configs/

# Copy codebase
COPY src/ ./src/
COPY scripts/ ./scripts/

# Set environment variables for SAM 2
ENV SAM_CHECKPOINT=/app/checkpoints/sam2.1_hiera_tiny.pt
ENV SAM_MODEL_TYPE=tiny
ENV PORT=8000
ENV PYTHONPATH=/app:/app/src

EXPOSE 8000
CMD ["sh", "-c", "python scripts/download_weights.py && python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"]
