import logging
from fastapi import APIRouter, HTTPException, Depends

from src.utils.schemas import (
    FeedbackInput,
    FeedbackResponse,
    SubstituteRequest,
    SubstituteResult,
)
from src.utils.feedback_store import get_feedback_store
from src.utils.substitutor import get_substitutes_for_missing, find_substitutes
from src.api.dependencies import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["feedback"], dependencies=[Depends(verify_api_key)])
feedback_store = get_feedback_store()


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackInput):
    """
    Enregistre le feedback utilisateur et retourne des substituts
    pour les ingrédients manquants.
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
        restrictions = []   # can retrieve from user profile if stored in future
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


@router.post("/substitute", response_model=SubstituteResult)
def get_substitute(payload: SubstituteRequest):
    """
    Retourne des substituts pour un ingrédient donné.
    """
    return find_substitutes(payload.ingredient, payload.dietary_restrictions)
