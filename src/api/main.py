"""
src/api/main.py
────────────────
WhatIEat Unified Backend — v5
Détection FRCNN + SAM 2 + Two-Tower + NeonDB + Auth JWT
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.utils import get_device
from src.models.detector import FridgeDetector
from src.models.recommender import RecipeRecommender
from src.utils.feedback_store import get_feedback_store
from src.utils.model_updater import ModelUpdater
from src.utils.database import get_pool, close_pool

# Routes existantes
from src.api.routes import vision, recommend

# Nouvelles routes
from src.api.routes.auth     import router as auth_router
from src.api.routes.users    import router as users_router
from src.api.routes.recipes  import router as recipes_router
from src.api.routes.fridge   import router as fridge_router
from src.api.routes.feedback import router as feedback_router

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_device()
    app.state.device     = device
    app.state.image_size = int(os.environ.get("DETECTOR_IMAGE_SIZE", "512"))

    # ── 1. NeonDB ─────────────────────────────────────────────────────
    app.state.db_pool = None
    try:
        pool = await get_pool()
        app.state.db_pool = pool
        get_feedback_store().set_pool(pool)
        logger.info("✅ NeonDB connecté.")
    except Exception as e:
        logger.error(f"❌ NeonDB connexion échouée : {e}")

    # ── 2. Détecteur FRCNN ────────────────────────────────────────────
    app.state.detector    = None
    app.state.class_names = []
    ckpt_path = Path(os.environ.get("DETECTOR_CHECKPOINT", "checkpoints/best.pt")).resolve()
    if ckpt_path.exists():
        try:
            ckpt        = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            class_names = ckpt.get("class_names") or [f"class_{i+1}" for i in range(ckpt.get("num_classes", 1))]
            model = FridgeDetector(
                num_classes      = ckpt.get("num_classes", len(class_names)),
                fpn_channels     = int(os.environ.get("DETECTOR_FPN_CHANNELS", "256")),
                backbone_arch    = os.environ.get("DETECTOR_BACKBONE", "resnet50"),
                pretrained_backbone = False,
            ).to(device)
            model.load_state_dict(ckpt["model"])
            model.eval()
            app.state.detector    = model
            app.state.class_names = class_names
            logger.info("✅ Détecteur FRCNN chargé.")
        except Exception as e:
            logger.error(f"❌ Détecteur FRCNN : {e}", exc_info=True)
    else:
        logger.warning(f"⚠️  Checkpoint FRCNN introuvable : {ckpt_path}")

    # ── 3. SAM 2 ─────────────────────────────────────────────────────
    app.state.sam_segmenter = None
    try:
        from src.api.routes.vision import SamBoxSegmenter
        sam = SamBoxSegmenter()
        app.state.sam_segmenter = sam
        logger.info(f"SAM 2 status : {sam.status}")
    except Exception as e:
        logger.error(f"❌ SAM 2 : {e}", exc_info=True)

    # ── 4. Two-Tower Recommender ──────────────────────────────────────
    app.state.recommender    = None
    app.state.model_updater  = None
    recipes_path = os.environ.get("RECIPES_PATH", "data/recipes.json")
    r_path = Path(recipes_path).resolve()
    if r_path.exists():
        try:
            recommender = RecipeRecommender(dataset_path=str(r_path))
            recommender.fit()
            app.state.recommender = recommender
            logger.info(f"✅ Two-Tower prêt — {recommender.n_recipes} recettes.")

            # 5. ModelUpdater — watcher + re-fit périodique
            updater = ModelUpdater(
                recommender  = recommender,
                pool         = app.state.db_pool,
                recipes_path = recipes_path,
            )
            app.state.model_updater = updater
            asyncio.create_task(updater.watch_recipes_file(check_interval_seconds=60))
            asyncio.create_task(updater.periodic_refit(interval_hours=24))
            logger.info("✅ ModelUpdater démarré.")
        except Exception as e:
            logger.error(f"❌ Two-Tower : {e}", exc_info=True)
    else:
        logger.error(f"❌ Dataset recettes introuvable : {r_path}")

    yield  # ── Application en marche ──────────────────────────────────

    await close_pool()


app = FastAPI(
    title       = "WhatIEat API",
    description = "FRCNN + SAM 2 + Two-Tower + NeonDB + Auth JWT",
    version     = "5.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────
app.include_router(auth_router)       # /auth/register, /auth/login
app.include_router(users_router)      # /users/me
app.include_router(recipes_router)    # /users/me/recipes
app.include_router(fridge_router)     # /users/me/fridge
app.include_router(feedback_router)   # /feedback, /substitute, /admin/*
app.include_router(vision.router)     # /vision/*
app.include_router(recommend.router)  # /recommend/*


@app.get("/")
async def root():
    return {"status": "ok", "message": "WhatIEat API v5", "docs": "/docs"}


@app.get("/health")
async def health():
    pool       = getattr(app.state, "db_pool", None)
    sam        = getattr(app.state, "sam_segmenter", None)
    recommender= getattr(app.state, "recommender", None)
    updater    = getattr(app.state, "model_updater", None)

    db_ok = False
    if pool:
        try:
            async with pool.acquire() as conn:
                db_ok = await conn.fetchval("SELECT 1") == 1
        except Exception:
            pass

    return {
        "status":  "ok",
        "version": "5.0.0",
        "db":      "connected" if db_ok else "disconnected",
        "detector":    {"loaded": app.state.detector is not None},
        "sam":         {"status": sam.status if sam else "not_initialized"},
        "recommender": {
            "recipes_loaded": recommender.n_recipes if recommender else 0,
            "refit_count":    updater._refit_count  if updater     else 0,
        },
    }