import io
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, Depends, Request
from PIL import Image
import torchvision.transforms.functional as TF

from src.api.dependencies import verify_api_key

router = APIRouter(prefix="/vision", tags=["vision"])

# Class categories for quantity estimation
COUNTABLE_CLASSES = {
    'apple', 'banana', 'cabbage', 'carrot', 'cucumber', 'date', 'eggplant',
    'eggs', 'garlic', 'lemon', 'lettuce', 'okra', 'onion', 'orange', 'potato',
    'tomato'
}

PACKAGE_CLASSES = {
    'butter', 'cheese', 'milk', 'yogurt', 'bread'
}

BULK_CLASSES = {
    'beans', 'beef', 'bulgur', 'chicken', 'chickpea', 'fish', 'lamb',
    'lentil', 'rice', 'spinach'
}


class SamBoxSegmenter:
    """SAM 2 segmenter prompted by FRCNN bounding boxes."""

    # Maps SAM_MODEL_TYPE env value → SAM 2.1 config path
    _SAM2_CONFIGS = {
        'tiny':      'configs/sam2.1/sam2.1_hiera_t.yaml',
        'small':     'configs/sam2.1/sam2.1_hiera_s.yaml',
        'base_plus': 'configs/sam2.1/sam2.1_hiera_b+.yaml',
        'large':     'configs/sam2.1/sam2.1_hiera_l.yaml',
    }

    def __init__(self) -> None:
        self.enabled = False
        self.status = 'disabled'
        self._predictor = None

        ckpt = os.environ.get('SAM_CHECKPOINT')
        model_type = os.environ.get('SAM_MODEL_TYPE', 'tiny')
        if not ckpt:
            self.status = 'SAM_CHECKPOINT not set'
            return
        ckpt_path = Path(ckpt).expanduser().resolve()
        if not ckpt_path.exists():
            self.status = f'SAM_CHECKPOINT not found: {ckpt_path}'
            return

        config_name = self._SAM2_CONFIGS.get(model_type)
        if config_name is None:
            self.status = f'Unknown SAM_MODEL_TYPE: {model_type}. Choose from {list(self._SAM2_CONFIGS)}'
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            self.status = f'sam2 import failed: {exc}'
            return

        try:
            sam_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            sam2_model = build_sam2(config_name, str(ckpt_path), device=sam_device)
            self._predictor = SAM2ImagePredictor(sam2_model)
            self.enabled = True
            self.status = f'SAM 2 enabled ({model_type} on {sam_device})'
        except Exception as exc:
            self.status = f'init failed: {exc}'

    @staticmethod
    def _mask_to_polygon(mask: np.ndarray) -> list[list[float]] | None:
        contour_data = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_data[0] if len(contour_data) == 2 else contour_data[1]
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 8:
            return None

        epsilon = max(1.5, 0.01 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if approx.shape[0] < 3:
            return None

        points = [[float(p[0][0]), float(p[0][1])] for p in approx]
        return points

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(bool)
        b = b.astype(bool)
        inter = int((a & b).sum())
        union = int((a | b).sum())
        return inter / max(1, union)

    def segment_all_classes(
        self, image_rgb: np.ndarray, by_class: dict[str, list[dict[str, Any]]]
    ) -> dict[str, tuple[int, list[dict[str, Any]]]]:
        if not self.enabled:
            return {name: (len(dets), list(dets)) for name, dets in by_class.items()}

        assert self._predictor is not None
        self._predictor.set_image(image_rgb)

        results: dict[str, tuple[int, list[dict[str, Any]]]] = {}
        for class_name, class_dets in by_class.items():
            best_score = max(d['score'] for d in class_dets)

            raw: list[tuple[np.ndarray, float]] = []
            for det in class_dets:
                b = det['box']
                box = np.array([b['x1'], b['y1'], b['x2'], b['y2']], dtype=np.float32)
                try:
                    masks, iou_preds, _ = self._predictor.predict(
                        box=box[None],
                        multimask_output=True,
                    )
                except Exception:
                    continue
                best_idx = int(np.argmax(iou_preds))
                if iou_preds[best_idx] < 0.7:
                    continue
                raw.append((masks[best_idx], float(iou_preds[best_idx])))

            if not raw:
                results[class_name] = (len(class_dets), list(class_dets))
                continue

            raw.sort(key=lambda t: t[1], reverse=True)
            kept_masks: list[np.ndarray] = []
            for mask, _ in raw:
                if any(self._mask_iou(mask, km) > 0.5 for km in kept_masks):
                    continue
                kept_masks.append(mask)

            instance_dets: list[dict[str, Any]] = []
            for mask in kept_masks:
                polygon = self._mask_to_polygon(mask)
                if polygon is None:
                    continue
                rows = np.where(np.any(mask, axis=1))[0]
                cols = np.where(np.any(mask, axis=0))[0]
                if rows.size == 0 or cols.size == 0:
                    continue
                box_out = {
                    'x1': float(cols[0]), 'y1': float(rows[0]),
                    'x2': float(cols[-1]), 'y2': float(rows[-1]),
                }
                instance_dets.append({
                    'name': class_name,
                    'score': best_score,
                    'box': box_out,
                    'mask': {'polygon': polygon, 'area': int(mask.sum()), 'source': 'sam-box'},
                })

            if not instance_dets:
                results[class_name] = (len(class_dets), list(class_dets))
                continue

            results[class_name] = (len(instance_dets), instance_dets)

        return results


def image_to_tensor(data: bytes, image_size: int, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    try:
        image = Image.open(io.BytesIO(data)).convert('RGB')
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Invalid image file: {exc}') from exc

    image = image.resize((image_size, image_size), Image.BILINEAR)
    image_np = np.asarray(image, dtype=np.uint8)
    return TF.to_tensor(image).unsqueeze(0).to(device), image_np


@router.post('/scan', dependencies=[Depends(verify_api_key)])
async def vision_scan(
    request: Request,
    file: UploadFile = File(...),
    score_threshold: float = Query(0.35, ge=0.0, le=1.0),
    target_class: str | None = Query(None),
    include_masks: bool = Query(True),
) -> dict:
    # Retrieve models and configs from app state
    model = request.app.state.detector
    device = request.app.state.device
    class_names = request.app.state.class_names
    sam_segmenter = request.app.state.sam_segmenter
    image_size = request.app.state.image_size

    if model is None:
        raise HTTPException(status_code=503, detail="Detector model not loaded.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail='Empty file upload')

    image_tensor, image_np = image_to_tensor(content, image_size, device)

    with torch.no_grad():
        detections, _ = model(image_tensor)
    det = detections[0]

    boxes = det['boxes'].cpu()
    scores = det['scores'].cpu()
    labels = det['labels'].cpu()

    target = target_class.lower().strip() if target_class else None

    # Step 1: collect all FRCNN detections above threshold, grouped by class
    by_class: dict[str, list[dict]] = {}
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
        by_class.setdefault(name, []).append(
            {'name': name, 'score': s, 'box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}}
        )

    # Step 2: per class — FRCNN gives the zone, SAM counts instances inside
    final_detections: list[dict] = []
    ingredients: list[dict] = []
    best_score_by_name: dict[str, float] = {
        name: max(d['score'] for d in dets) for name, dets in by_class.items()
    }

    if include_masks and sam_segmenter and sam_segmenter.enabled:
        class_results = sam_segmenter.segment_all_classes(image_np, by_class)
    else:
        class_results = {name: (len(dets), list(dets)) for name, dets in by_class.items()}

    for name, (count, instance_dets) in sorted(
        class_results.items(), key=lambda kv: best_score_by_name[kv[0]], reverse=True
    ):
        final_detections.extend(instance_dets)
        
        # Determine class category, unit, and method
        if name in COUNTABLE_CLASSES:
            unit = 'count'
            quantity_val = str(len(instance_dets))
            method = 'instance_count' if (include_masks and sam_segmenter and sam_segmenter.enabled) else 'box_count'
        elif name in PACKAGE_CLASSES:
            unit = 'pack'
            quantity_val = str(len(instance_dets))
            method = 'instance_count' if (include_masks and sam_segmenter and sam_segmenter.enabled) else 'box_count'
        elif name in BULK_CLASSES:
            unit = 'level'
            method = 'area_proxy'
            has_masks = any('mask' in d for d in instance_dets)
            if has_masks:
                total_area = sum(d['mask']['area'] for d in instance_dets if 'mask' in d)
            else:
                total_area = sum((d['box']['x2'] - d['box']['x1']) * (d['box']['y2'] - d['box']['y1']) for d in instance_dets)
            
            rel_area = total_area / (image_size * image_size)
            if rel_area < 0.05:
                quantity_val = 'low'
            elif rel_area < 0.15:
                quantity_val = 'medium'
            else:
                quantity_val = 'high'
        else:
            unit = 'count'
            quantity_val = str(len(instance_dets))
            method = 'instance_count' if (include_masks and sam_segmenter and sam_segmenter.enabled) else 'box_count'

        mean_conf = float(np.mean([d['score'] for d in instance_dets])) if instance_dets else 0.0

        ingredients.append({
            'id': name,
            'name': name,
            'quantity': quantity_val,
            'unit': unit,
            'method': method,
            'quantity_confidence': round(mean_conf, 4)
        })

    confidence = max(best_score_by_name.values()) if best_score_by_name else 0.0
    return {
        'ingredients': ingredients,
        'confidence': confidence,
        'detections': sorted(final_detections, key=lambda d: d['score'], reverse=True),
    }
