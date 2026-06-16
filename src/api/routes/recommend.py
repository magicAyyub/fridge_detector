import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends

from src.utils.schemas import (
    IngredientsInput,
    RecipeRecommendationResponse,
)
from src.utils.feedback_store import get_feedback_store
from src.api.dependencies import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recommend", tags=["recommend"], dependencies=[Depends(verify_api_key)])
feedback_store = get_feedback_store()


def _get_feedback_context(session_id: Optional[str]):
    if not session_id:
        return None, None
    return (
        feedback_store.get_missing_counts(session_id),
        feedback_store.get_disliked_titles(session_id),
    )


def _recommend(request: Request, payload: IngredientsInput, meal_type: Optional[str] = None):
    recommender = request.app.state.recommender
    if not recommender:
        raise HTTPException(status_code=503, detail="Model not loaded.")

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
            detail=f"No recipe found{' for ' + meal_type if meal_type else ''}.",
        )

    return RecipeRecommendationResponse(
        query_ingredients = payload.ingredients,
        fridge            = payload.fridge_dict,
        meal_type         = meal_type,
        recipes           = results,
    )


@router.post("", response_model=RecipeRecommendationResponse)
def recommend_recipes(request: Request, payload: IngredientsInput):
    """General recommendations across all meal types."""
    return _recommend(request, payload, meal_type=None)


@router.post("/breakfast", response_model=RecipeRecommendationResponse)
def recommend_breakfast(request: Request, payload: IngredientsInput):
    """Breakfast recommendations adapted to caloric goal (25% of daily target)."""
    return _recommend(request, payload, meal_type="breakfast")


@router.post("/lunch", response_model=RecipeRecommendationResponse)
def recommend_lunch(request: Request, payload: IngredientsInput):
    """Lunch recommendations adapted to caloric goal (35% of daily target)."""
    return _recommend(request, payload, meal_type="lunch")


@router.post("/dinner", response_model=RecipeRecommendationResponse)
def recommend_dinner(request: Request, payload: IngredientsInput):
    """Dinner recommendations adapted to caloric goal (35% of daily target)."""
    return _recommend(request, payload, meal_type="dinner")


@router.post("/snack", response_model=RecipeRecommendationResponse)
def recommend_snack(request: Request, payload: IngredientsInput):
    """Snack recommendations adapted to caloric goal (10% of daily target)."""
    return _recommend(request, payload, meal_type="snack")


@router.get("/vocabulary")
def vocabulary(request: Request):
    """Get the full recommender vocab size and token list."""
    recommender = request.app.state.recommender
    if not recommender:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    vocab = recommender.vocabulary
    return {"size": len(vocab), "tokens": vocab}
