import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.utils import get_device
from src.models.detector import FridgeDetector
from src.models.recommender import RecipeRecommender
from src.api.routes import vision, recommend, feedback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup device and defaults
    device = get_device()
    app.state.device = device
    app.state.image_size = int(os.environ.get("DETECTOR_IMAGE_SIZE", "512"))
    
    # 1. Load Fridge Detector (FRCNN)
    app.state.detector = None
    app.state.class_names = []
    checkpoint_path = os.environ.get("DETECTOR_CHECKPOINT", "checkpoints/best.pt")
    if checkpoint_path:
        ckpt_path = Path(checkpoint_path).expanduser().resolve()
        if not ckpt_path.exists():
            logger.error(f"❌ Detector checkpoint not found: {ckpt_path}")
        else:
            try:
                logger.info(f"Loading Fridge Detector model from {ckpt_path} on {device}...")
                ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
                class_names = ckpt.get('class_names')
                if not class_names:
                    num_classes = ckpt.get('num_classes', 1)
                    class_names = [f'class_{idx + 1}' for idx in range(num_classes)]
                num_classes = ckpt.get('num_classes', len(class_names))

                backbone = os.environ.get("DETECTOR_BACKBONE", "resnet50")
                fpn_channels = int(os.environ.get("DETECTOR_FPN_CHANNELS", "256"))

                model = FridgeDetector(
                    num_classes=num_classes,
                    fpn_channels=fpn_channels,
                    backbone_arch=backbone,
                    pretrained_backbone=False,
                ).to(device)
                model.load_state_dict(ckpt['model'])
                model.eval()
                
                app.state.detector = model
                app.state.class_names = class_names
                logger.info("✅ Fridge Detector loaded successfully.")
            except Exception as e:
                logger.error(f"❌ Error loading Fridge Detector: {e}", exc_info=True)

    # 2. Load SAM 2 Segmenter
    app.state.sam_segmenter = None
    try:
        logger.info("Initializing SAM 2 Segmenter...")
        from src.api.routes.vision import SamBoxSegmenter
        sam_segmenter = SamBoxSegmenter()
        app.state.sam_segmenter = sam_segmenter
        logger.info(f"SAM 2 Segmenter status: {sam_segmenter.status}")
    except Exception as e:
        logger.error(f"❌ Error initializing SAM 2 Segmenter: {e}", exc_info=True)

    # 3. Load Recipe Recommender
    app.state.recommender = None
    recipes_path = os.environ.get("RECIPES_PATH", "data/recipes.json")
    if recipes_path:
        r_path = Path(recipes_path).expanduser().resolve()
        if not r_path.exists():
            logger.error(f"❌ Recipes dataset not found: {r_path}")
        else:
            try:
                logger.info(f"Loading Recipe Recommender dataset from {r_path}...")
                recommender = RecipeRecommender(dataset_path=str(r_path))
                recommender.fit()
                app.state.recommender = recommender
                logger.info(f"✅ Recipe Recommender loaded: {recommender.n_recipes} recipes indexed.")
            except Exception as e:
                logger.error(f"❌ Error loading Recipe Recommender: {e}", exc_info=True)

    yield


app = FastAPI(
    title="WhatIEat API",
    description="Inference pour la détection d'ingrédients de frigo (FRCNN + SAM 2) et recommandation de recettes (Two-Tower).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(vision.router)
app.include_router(recommend.router)
app.include_router(feedback.router)


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "message": "WhatIEat Unified Backend API"}


@app.get('/health')
async def health() -> dict:
    device = getattr(app.state, "device", "unknown")
    sam_segmenter = getattr(app.state, "sam_segmenter", None)
    recommender = getattr(app.state, "recommender", None)
    return {
        'status': 'ok',
        'device': str(device),
        'checkpoint': os.path.abspath(os.environ.get("DETECTOR_CHECKPOINT", "checkpoints/best.pt")),
        'sam': {
            'enabled': sam_segmenter.enabled if sam_segmenter else False,
            'status': sam_segmenter.status if sam_segmenter else "not_initialized"
        },
        'recommender': {
            'recipes_loaded': recommender.n_recipes if recommender else 0,
            'version': '4.0.0'
        }
    }
