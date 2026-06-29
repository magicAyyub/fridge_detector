"""
src/api/routes/recommend.py
────────────────────────────
Routes de recommandation avec feedback implicite des likes.
"""

import logging
from typing import Optional, List, Set
from fastapi import APIRouter, HTTPException, Request, Depends

from src.utils.schemas import IngredientsInput, RecipeRecommendationResponse
from src.utils.feedback_store import get_feedback_store
from src.utils.auth_utils import get_optional_user_id
from src.api.dependencies import verify_api_key
from src.models.vectorizer import tokenize

logger         = logging.getLogger(__name__)
router         = APIRouter(prefix="/recommend", tags=["recommend"], dependencies=[Depends(verify_api_key)])
feedback_store = get_feedback_store()


async def _load_liked_data(pool, user_id: Optional[int]) -> tuple[Set[str], List[Set[str]], Set[str]]:
    """
    Charge depuis NeonDB :
    - liked_titles       : titres des recettes likées (pour exclusion / profil)
    - liked_recipes_tokens : tokens NER des recettes likées (pour frigo virtuel)
    - cooked_recent      : cuisinées dans les 7 derniers jours (pour pénalité)

    Retourne (liked_titles, liked_recipes_tokens, cooked_recent)
    """
    if not pool or not user_id:
        return set(), [], set()

    try:
        async with pool.acquire() as conn:
            # Recettes likées avec leurs ingrédients (pour le frigo virtuel)
            liked_rows = await conn.fetch(
                """
                SELECT r.title, r.ingredients
                FROM user_saved_recipes usr
                JOIN recipes r ON r.id = usr.recipe_id
                WHERE usr.user_id = $1
                """,
                user_id,
            )

            # Cuisinées récemment
            cooked_rows = await conn.fetch(
                """
                SELECT DISTINCT r.title
                FROM recipe_preparations rp
                JOIN recipes r ON r.id = rp.recipe_id
                WHERE rp.user_id = $1
                  AND rp.prepared_at > NOW() - INTERVAL '7 days'
                """,
                user_id,
            )

        liked_titles   = set()
        liked_tokens   = []
        cooked_recent  = {r["title"] for r in cooked_rows}

        for row in liked_rows:
            liked_titles.add(row["title"])
            # Parse les ingrédients JSONB → tokens NER
            raw = row["ingredients"]
            if raw:
                try:
                    ings = list(raw) if not isinstance(raw, str) else __import__('json').loads(raw)
                    token_set = set()
                    for ing in ings:
                        token_set.update(tokenize(ing))
                    if token_set:
                        liked_tokens.append(token_set)
                except Exception:
                    pass

        logger.debug(
            f"User {user_id} — {len(liked_titles)} recettes likées, "
            f"{len(liked_tokens)} avec tokens, "
            f"{len(cooked_recent)} cuisinées récemment"
        )
        return liked_titles, liked_tokens, cooked_recent

    except Exception as e:
        logger.warning(f"Chargement données liked échoué : {e}")
        return set(), [], set()


def _get_feedback_context(session_id: Optional[str]):
    if not session_id:
        return None, None
    return (
        feedback_store.get_missing_counts(session_id),
        feedback_store.get_disliked_titles(session_id),
    )


async def _recommend(
    request:   Request,
    payload:   IngredientsInput,
    meal_type: Optional[str] = None,
    user_id:   Optional[int] = None,
):
    recommender = request.app.state.recommender
    if not recommender:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    pool = getattr(request.app.state, "db_pool", None)

    # Feedback mémoire (session courante)
    feedback_history, disliked_titles = _get_feedback_context(
        getattr(payload, "session_id", None)
    )

    # Données liked depuis la DB
    liked_titles, liked_recipes_tokens, cooked_recent = await _load_liked_data(pool, user_id)

    results = recommender.recommend(
        ingredients          = payload.ingredients,
        top_n                = payload.top_n,
        min_score            = payload.min_score,
        meal_type            = meal_type,
        profile              = payload.user_profile,
        feedback_history     = feedback_history,
        disliked_titles      = disliked_titles,
        liked_titles         = liked_titles,
        cooked_recently      = cooked_recent,
        liked_recipes_tokens = liked_recipes_tokens,
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
async def recommend_recipes(request: Request, payload: IngredientsInput, user_id: Optional[int] = Depends(get_optional_user_id)):
    return await _recommend(request, payload, meal_type=None, user_id=user_id)

@router.post("/breakfast", response_model=RecipeRecommendationResponse)
async def recommend_breakfast(request: Request, payload: IngredientsInput, user_id: Optional[int] = Depends(get_optional_user_id)):
    return await _recommend(request, payload, meal_type="breakfast", user_id=user_id)

@router.post("/lunch", response_model=RecipeRecommendationResponse)
async def recommend_lunch(request: Request, payload: IngredientsInput, user_id: Optional[int] = Depends(get_optional_user_id)):
    return await _recommend(request, payload, meal_type="lunch", user_id=user_id)

@router.post("/dinner", response_model=RecipeRecommendationResponse)
async def recommend_dinner(request: Request, payload: IngredientsInput, user_id: Optional[int] = Depends(get_optional_user_id)):
    return await _recommend(request, payload, meal_type="dinner", user_id=user_id)

@router.post("/snack", response_model=RecipeRecommendationResponse)
async def recommend_snack(request: Request, payload: IngredientsInput, user_id: Optional[int] = Depends(get_optional_user_id)):
    return await _recommend(request, payload, meal_type="snack", user_id=user_id)

@router.get("/vocabulary")
def vocabulary(request: Request):
    recommender = request.app.state.recommender
    if not recommender:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    vocab = recommender.vocabulary
    return {"size": len(vocab), "tokens": vocab}