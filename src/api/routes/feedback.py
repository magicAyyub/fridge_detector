"""
src/api/routes/feedback.py
───────────────────────────
Routes feedback utilisateur + substituts.

Routes :
  POST /feedback                      → enregistre like/dislike/missing
  GET  /users/me/feedback             → historique des feedbacks
  POST /substitute                    → substituts pour un ingrédient
  POST /admin/refit                   → re-fit Two-Tower manuel
  POST /admin/recipes/add             → ajout recettes à chaud
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src.utils.schemas import (
    FeedbackInput,
    FeedbackResponse,
    SubstituteRequest,
    SubstituteResult,
)
from src.utils.feedback_store import get_feedback_store
from src.utils.substitutor import get_substitutes_for_missing, find_substitutes
from src.utils.auth_utils import get_current_user_id, get_optional_user_id
from src.api.dependencies import verify_api_key
from fastapi import Depends as FDepends

logger = logging.getLogger(__name__)
router = APIRouter(tags=["feedback"])
feedback_store = get_feedback_store()


# ── Feedback ───────────────────────────────────────────────────────────────

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    payload: FeedbackInput,
    request: Request,
    user_id: Optional[int] = FDepends(get_optional_user_id),
):
    """
    Enregistre le feedback (like/dislike/missing/cooked).
    - En mémoire : immédiat, influence le prochain /recommend
    - En DB : persisté si l'utilisateur est authentifié
    """
    # 1. Enregistrement mémoire (Two-Tower)
    await feedback_store.record_async(
        session_id          = payload.session_id,
        recipe_title        = payload.recipe_title,
        liked               = payload.liked,
        missing_ingredients = payload.missing_ingredients,
        cooked              = payload.cooked,
        user_id             = user_id,
    )

    # 2. Persistance DB si user connecté
    pool = getattr(request.app.state, "db_pool", None)
    if pool and user_id:
        try:
            async with pool.acquire() as conn:
                # Récupère recipe_id si la recette existe
                recipe_row = await conn.fetchrow(
                    "SELECT id FROM recipes WHERE title = $1", payload.recipe_title
                )
                if recipe_row:
                    recipe_id = recipe_row["id"]
                    await conn.execute(
                        """
                        INSERT INTO recipe_feedback (user_id, recipe_id, liked, cooked)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (user_id, recipe_id) DO UPDATE SET
                            liked      = EXCLUDED.liked,
                            cooked     = EXCLUDED.cooked,
                            updated_at = NOW()
                        """,
                        user_id, recipe_id, payload.liked, payload.cooked,
                    )
                    # Ingrédients manquants
                    if payload.missing_ingredients:
                        fb_row = await conn.fetchrow(
                            "SELECT id FROM recipe_feedback WHERE user_id=$1 AND recipe_id=$2",
                            user_id, recipe_id,
                        )
                        if fb_row:
                            await conn.executemany(
                                """
                                INSERT INTO feedback_missing_ingredients
                                    (feedback_id, ingredient_name, ner_token)
                                VALUES ($1, $2, $3)
                                """,
                                [(fb_row["id"], ing, ing.lower()) for ing in payload.missing_ingredients],
                            )
        except Exception as e:
            logger.warning(f"Feedback DB persist échoué (non critique) : {e}")

    # 3. Substituts pour les ingrédients manquants
    substitutes = []
    if payload.missing_ingredients:
        substitutes = get_substitutes_for_missing(payload.missing_ingredients)

    msg_parts = []
    if payload.liked is True:
        msg_parts.append("Like enregistré ✅")
    elif payload.liked is False:
        msg_parts.append("Dislike enregistré — cette recette ne sera plus proposée.")
    if payload.cooked:
        msg_parts.append("Super, tu l'as cuisiné ! 🍳")
    if payload.missing_ingredients:
        msg_parts.append(f"{len(payload.missing_ingredients)} ingrédient(s) manquant(s) notés.")

    return FeedbackResponse(
        message     = " ".join(msg_parts) or "Feedback enregistré.",
        substitutes = substitutes,
    )


@router.get("/users/me/feedback")
async def get_feedback_history(
    request: Request,
    user_id: int = FDepends(get_current_user_id),
):
    """Historique des feedbacks de l'utilisateur connecté."""
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT rf.id, r.title, rf.liked, rf.cooked,
                   rf.created_at::text, rf.updated_at::text,
                   ARRAY_AGG(fmi.ingredient_name) FILTER (WHERE fmi.ingredient_name IS NOT NULL)
                       AS missing_ingredients
            FROM recipe_feedback rf
            JOIN recipes r ON r.id = rf.recipe_id
            LEFT JOIN feedback_missing_ingredients fmi ON fmi.feedback_id = rf.id
            WHERE rf.user_id = $1
            GROUP BY rf.id, r.title, rf.liked, rf.cooked, rf.created_at, rf.updated_at
            ORDER BY rf.updated_at DESC
            """,
            user_id,
        )
    return [dict(r) for r in rows]


# ── Substituts ─────────────────────────────────────────────────────────────

@router.post("/substitute", response_model=SubstituteResult)
def get_substitute(payload: SubstituteRequest):
    """Retourne des substituts pour un ingrédient donné."""
    return find_substitutes(payload.ingredient, payload.dietary_restrictions)


# ── Admin ──────────────────────────────────────────────────────────────────

@router.post("/admin/refit", dependencies=[FDepends(verify_api_key)])
async def trigger_refit(request: Request):
    """Re-fit complet du Two-Tower manuellement."""
    updater = getattr(request.app.state, "model_updater", None)
    if not updater:
        raise HTTPException(status_code=503, detail="ModelUpdater non initialisé.")
    await updater._refit()
    return {
        "status":      "ok",
        "refit_count": updater._refit_count,
        "recipes":     updater.recommender.n_recipes,
    }


@router.post("/admin/recipes/add", dependencies=[FDepends(verify_api_key)])
async def add_recipes(recipes: List[dict], request: Request):
    """Ajoute des recettes à chaud sans redémarrage."""
    updater = getattr(request.app.state, "model_updater", None)
    if not updater:
        raise HTTPException(status_code=503, detail="ModelUpdater non initialisé.")
    added = await updater.add_recipes(recipes)
    return {"added": added, "total": updater.recommender.n_recipes}