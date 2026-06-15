"""
Recipe Recommendation Microservice — Food.com + Two-Tower
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import logging

from src.utils.schemas import (
    IngredientsInput,
    RecipeRecommendationResponse,
    FeedbackInput,
    FeedbackResponse,
    SubstituteRequest,
    SubstituteResult,
    MealType,
)
from src.models.recommender import RecipeRecommender
from src.utils.feedback_store import get_feedback_store
from src.utils.substitutor import get_substitutes_for_missing, find_substitutes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

recommender: RecipeRecommender = None
feedback_store = get_feedback_store()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    try:
        logger.info("Chargement du modèle Two-Tower…")
        recommender = RecipeRecommender(
            dataset_path="data/recipes.json",
            translate=False,
        )
        recommender.fit()
        logger.info(f"✅ Modèle prêt — {recommender.n_recipes} recettes indexées.")
    except Exception as e:
        logger.error(f"❌ Erreur au chargement : {e}", exc_info=True)
    yield


app = FastAPI(
    title="WhatIEat — Recipe Service",
    description="Recommandation de recettes avec Two-Tower scoring, filtres nutritionnels et feedback.",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_feedback_context(session_id: Optional[str]):
    if not session_id:
        return None, None
    return (
        feedback_store.get_missing_counts(session_id),
        feedback_store.get_disliked_titles(session_id),
    )


def _recommend(payload: IngredientsInput, meal_type: Optional[str] = None):
    if not recommender:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    feedback_history, disliked_titles = _get_feedback_context(
        getattr(payload, "session_id", None)
    )

    results = recommender.recommend(
        ingredients      = payload.ingredients,
        top_n            = payload.top_n,
        min_score        = payload.min_score,
        meal_type        = meal_type,
        profile          = payload.user_profile,
        feedback_history = feedback_history,
        disliked_titles  = disliked_titles,
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Aucune recette trouvée{' pour ' + meal_type if meal_type else ''}.",
        )

    return RecipeRecommendationResponse(
        query_ingredients = payload.ingredients,
        fridge            = payload.fridge_dict,
        meal_type         = meal_type,
        recipes           = results,
    )


# ── Routes recommandation ──────────────────────────────────────────────────

@app.post("/recommend", response_model=RecipeRecommendationResponse)
def recommend_recipes(payload: IngredientsInput):
    """
    Recommandation générale — toutes catégories de repas.

    Exemple :
    {
      "fridge_dict": {"egg": 4, "tomato": 3, "garlic": 2},
      "user_profile": {
        "calorie_target": 2100,
        "sports_objective": "muscle_gain",
        "dietary_restrictions": ["gluten"]
      },
      "top_n": 5
    }
    """
    return _recommend(payload, meal_type=None)


@app.post("/recommend/breakfast", response_model=RecipeRecommendationResponse)
def recommend_breakfast(payload: IngredientsInput):
    """Recettes petit-déjeuner adaptées au profil calorique (25% objectif journalier)."""
    return _recommend(payload, meal_type="breakfast")


@app.post("/recommend/lunch", response_model=RecipeRecommendationResponse)
def recommend_lunch(payload: IngredientsInput):
    """Recettes déjeuner adaptées au profil calorique (35% objectif journalier)."""
    return _recommend(payload, meal_type="lunch")


@app.post("/recommend/dinner", response_model=RecipeRecommendationResponse)
def recommend_dinner(payload: IngredientsInput):
    """Recettes dîner adaptées au profil calorique (35% objectif journalier)."""
    return _recommend(payload, meal_type="dinner")


@app.post("/recommend/snack", response_model=RecipeRecommendationResponse)
def recommend_snack(payload: IngredientsInput):
    """Recettes snack adaptées au profil calorique (10% objectif journalier)."""
    return _recommend(payload, meal_type="snack")


# ── Route feedback ─────────────────────────────────────────────────────────

@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackInput):
    """
    Enregistre le feedback utilisateur et retourne des substituts
    pour les ingrédients manquants.

    Exemple :
    {
      "session_id": "user_123",
      "recipe_title": "Scrambled Eggs",
      "liked": null,
      "missing_ingredients": ["cream", "chives"],
      "cooked": false
    }
    """
    feedback_store.record(
        session_id          = payload.session_id,
        recipe_title        = payload.recipe_title,
        liked               = payload.liked,
        missing_ingredients = payload.missing_ingredients,
        cooked              = payload.cooked,
    )

    # Propose des substituts pour les ingrédients manquants
    substitutes = []
    if payload.missing_ingredients:
        restrictions = []   # à récupérer depuis le profil si disponible
        substitutes = get_substitutes_for_missing(
            payload.missing_ingredients,
            dietary_restrictions=restrictions,
        )

    msg_parts = []
    if payload.liked is True:
        msg_parts.append("Like enregistré ✅")
    elif payload.liked is False:
        msg_parts.append("Dislike enregistré — cette recette ne sera plus proposée.")
    if payload.cooked:
        msg_parts.append("Super, tu l'as cuisiné ! 🍳")
    if payload.missing_ingredients:
        msg_parts.append(
            f"{len(payload.missing_ingredients)} ingrédient(s) manquant(s) notés "
            "— les prochaines suggestions en tiendront compte."
        )

    return FeedbackResponse(
        message     = " ".join(msg_parts) or "Feedback enregistré.",
        substitutes = substitutes,
    )


# ── Route substituts ───────────────────────────────────────────────────────

@app.post("/substitute", response_model=SubstituteResult)
def get_substitute(payload: SubstituteRequest):
    """
    Retourne des substituts pour un ingrédient donné.

    Exemple :
    {
      "ingredient": "butter",
      "dietary_restrictions": ["lactose"]
    }
    """
    return find_substitutes(payload.ingredient, payload.dietary_restrictions)


# ── Routes utilitaires ─────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "recipes_loaded": recommender.n_recipes if recommender else 0,
        "version": "4.0.0",
    }


@app.get("/vocabulary")
def vocabulary():
    if not recommender:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    vocab = recommender.vocabulary
    return {"size": len(vocab), "tokens": vocab}