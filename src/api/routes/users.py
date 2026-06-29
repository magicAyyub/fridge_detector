"""
src/api/routes/users.py
────────────────────────
Routes profil utilisateur : GET/PUT /users/me
"""

import json
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src.utils.auth_utils import get_current_user_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["users"])


# ── Schemas ────────────────────────────────────────────────────────────────

class ProfileUpdateInput(BaseModel):
    first_name:               Optional[str]       = None
    age:                      Optional[int]        = None
    weight_kg:                Optional[float]      = None
    height_cm:                Optional[float]      = None
    sports_objective:         Optional[str]        = None
    activity_level:           Optional[str]        = None
    calorie_target:           Optional[int]        = None
    has_completed_onboarding: Optional[bool]       = None
    dietary_restrictions:     Optional[List[str]]  = None


# ── Routes ─────────────────────────────────────────────────────────────────

@router.get("/me")
async def get_profile(
    request: Request,
    user_id: int = Depends(get_current_user_id),
):
    """Récupère le profil complet de l'utilisateur connecté."""
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            """
            SELECT id, email, first_name, age, weight_kg, height_cm,
                   sports_objective, activity_level, calorie_target,
                   has_completed_onboarding, dietary_restrictions, created_at
            FROM users WHERE id = $1
            """,
            user_id,
        )
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")

    row = dict(user)
    # JSONB → list Python
    dr = row.get("dietary_restrictions")
    if isinstance(dr, str):
        row["dietary_restrictions"] = json.loads(dr)
    elif dr is None:
        row["dietary_restrictions"] = []
    return row


@router.put("/me")
async def update_profile(
    payload: ProfileUpdateInput,
    request: Request,
    user_id: int = Depends(get_current_user_id),
):
    """Met à jour un ou plusieurs champs du profil."""
    pool    = request.app.state.db_pool
    updates = payload.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="Aucune donnée à mettre à jour.")

    if "dietary_restrictions" in updates:
        updates["dietary_restrictions"] = json.dumps(updates["dietary_restrictions"])

    fields = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(updates.keys()))
    values = list(updates.values())

    async with pool.acquire() as conn:
        result = await conn.execute(
            f"UPDATE users SET {fields} WHERE id = $1",
            user_id, *values,
        )
    if result == "UPDATE 0":
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")

    return {"message": "Profil mis à jour."}


@router.delete("/me")
async def delete_account(
    request: Request,
    user_id: int = Depends(get_current_user_id),
):
    """Supprime le compte et toutes les données associées (CASCADE)."""
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM users WHERE id = $1", user_id)
    return {"message": "Compte supprimé."}