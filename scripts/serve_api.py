"""Serve Fridge Detector inference as a small FastAPI backend.

Run:
  uv run python scripts/serve_api.py --checkpoint checkpoints/best.pt --host 0.0.0.0 --port 8000

Endpoint:
  POST /vision/scan (multipart form-data)
    field: file (image)
    optional query: score_threshold=0.35
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms.functional as TF
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from models.detector import FridgeDetector
from utils import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=str)
    p.add_argument('--host', default='0.0.0.0', type=str)
    p.add_argument('--port', default=8000, type=int)
    p.add_argument('--image-size', default=512, type=int)
    p.add_argument('--backbone', default='resnet50', choices=['resnet18', 'resnet50'])
    p.add_argument('--fpn-channels', default=256, type=int)
    return p.parse_args()


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[FridgeDetector, list[str]]:
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt.get('class_names')
    if not class_names:
        num_classes = ckpt.get('num_classes', 1)
        class_names = [f'class_{idx + 1}' for idx in range(num_classes)]
    num_classes = ckpt.get('num_classes', len(class_names))

    model = FridgeDetector(
        num_classes=num_classes,
        fpn_channels=args.fpn_channels,
        backbone_arch=args.backbone,
        pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, class_names


def image_to_tensor(data: bytes, image_size: int, device: torch.device) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(data)).convert('RGB')
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Invalid image file: {exc}') from exc

    image = image.resize((image_size, image_size), Image.BILINEAR)
    return TF.to_tensor(image).unsqueeze(0).to(device)


def app_factory(args: argparse.Namespace) -> FastAPI:
    device = get_device()
    model, class_names = build_model(args, device)

    app = FastAPI(title='Fridge Detector API', version='1.0.0')

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @app.get('/health')
    async def health() -> dict:
        return {
            'status': 'ok',
            'device': str(device),
            'checkpoint': os.path.abspath(args.checkpoint),
        }

    @app.post('/vision/scan')
    async def vision_scan(
        file: UploadFile = File(...),
        score_threshold: float = Query(0.35, ge=0.0, le=1.0),
        target_class: str | None = Query(None),
    ) -> dict:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail='Empty file upload')

        image_tensor = image_to_tensor(content, args.image_size, device)

        with torch.no_grad():
            detections, _ = model(image_tensor)
        det = detections[0]

        boxes = det['boxes'].cpu()
        scores = det['scores'].cpu()
        labels = det['labels'].cpu()

        target = target_class.lower().strip() if target_class else None
        best_by_name: dict[str, float] = {}
        detections_payload: list[dict] = []
        for box, score, label in zip(boxes, scores, labels):
            s = float(score.item())
            if s < score_threshold:
                continue
            idx = int(label.item()) - 1
            if idx < 0 or idx >= len(class_names):
                continue
            name = class_names[idx]
            if target and name != target:
                continue

            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            detections_payload.append(
                {
                    'name': name,
                    'score': s,
                    'box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                }
            )

            prev = best_by_name.get(name, 0.0)
            if s > prev:
                best_by_name[name] = s

        ingredients = [
            {
                'id': name,
                'name': name,
                'quantity': f'{conf:.2f}',
            }
            for name, conf in sorted(best_by_name.items(), key=lambda kv: kv[1], reverse=True)
        ]

        confidence = max(best_by_name.values()) if best_by_name else 0.0
        return {
            'ingredients': ingredients,
            'confidence': confidence,
            'detections': sorted(detections_payload, key=lambda d: d['score'], reverse=True),
        }

    return app


def main() -> None:
    args = parse_args()
    app = app_factory(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
