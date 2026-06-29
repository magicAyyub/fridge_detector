"""
src/api/routes/fridge.py
─────────────────────────
CRUD frigo + catégories + déduction ingrédients.
"""

import logging
from datetime import date
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src.utils.auth_utils import get_current_user_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users/me/fridge", tags=["fridge"])


def parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


# ── Schemas ────────────────────────────────────────────────────────────────

class FridgeItemInput(BaseModel):
    ingredient_name: str
    quantity:        float
    unit:            Optional[str] = None
    expires_at:      Optional[str] = None
    category:        Optional[str] = None


class FridgeItemUpdate(BaseModel):
    quantity:   Optional[float] = None
    unit:       Optional[str]   = None
    expires_at: Optional[str]   = None
    category:   Optional[str]   = None


# ── Routes ─────────────────────────────────────────────────────────────────

@router.get("/categories")
async def get_categories(
    request: Request,
    user_id: int = Depends(get_current_user_id),
):
    """Catégories distinctes du frigo de l'utilisateur."""
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT category
            FROM fridge_items
            WHERE user_id = $1 AND category IS NOT NULL
            ORDER BY category
            """,
            user_id,
        )
    return [r["category"] for r in rows]


@router.get("")
async def get_fridge(
    request:  Request,
    user_id:  int           = Depends(get_current_user_id),
    category: Optional[str] = None,
    search:   Optional[str] = None,
):
    """Liste les ingrédients avec filtres optionnels ?category=Protein&search=chick"""
    pool   = request.app.state.db_pool
    query  = """
        SELECT id, ingredient_name, quantity, unit, category,
               expires_at::text AS expires_at,
               updated_at::text AS updated_at
        FROM fridge_items
        WHERE user_id = $1
    """
    params = [user_id]

    if category:
        params.append(category)
        query += f" AND category = ${len(params)}"
    if search:
        params.append(f"%{search.lower()}%")
        query += f" AND LOWER(ingredient_name) LIKE ${len(params)}"

    query += " ORDER BY expires_at ASC NULLS LAST, ingredient_name"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    return [dict(r) for r in rows]


@router.post("", status_code=201)
async def add_fridge_item(
    payload: FridgeItemInput,
    request: Request,
    user_id: int = Depends(get_current_user_id),
):
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO fridge_items
                (user_id, ingredient_name, quantity, unit, expires_at, category)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (user_id, ingredient_name) DO UPDATE SET
                quantity   = EXCLUDED.quantity,
                unit       = EXCLUDED.unit,
                expires_at = EXCLUDED.expires_at,
                category   = COALESCE(EXCLUDED.category, fridge_items.category),
                updated_at = NOW()
            RETURNING id
            """,
            user_id, payload.ingredient_name, payload.quantity,
            payload.unit, parse_date(payload.expires_at), payload.category,
        )
    return {"id": row["id"], "message": "Ingrédient ajouté."}


@router.post("/bulk", status_code=201)
async def add_fridge_items_bulk(
    items:   List[FridgeItemInput],
    request: Request,
    user_id: int = Depends(get_current_user_id),
):
    pool     = request.app.state.db_pool
    inserted = 0
    async with pool.acquire() as conn:
        for item in items:
            await conn.execute(
                """
                INSERT INTO fridge_items
                    (user_id, ingredient_name, quantity, unit, expires_at, category)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id, ingredient_name) DO UPDATE SET
                    quantity   = EXCLUDED.quantity,
                    unit       = EXCLUDED.unit,
                    expires_at = EXCLUDED.expires_at,
                    category   = COALESCE(EXCLUDED.category, fridge_items.category),
                    updated_at = NOW()
                """,
                user_id, item.ingredient_name, item.quantity,
                item.unit, parse_date(item.expires_at), item.category,
            )
            inserted += 1
    return {"inserted": inserted, "message": f"{inserted} ingrédient(s) ajouté(s)."}


@router.put("/{item_id}")
async def update_fridge_item(
    item_id: int,
    payload: FridgeItemUpdate,
    request: Request,
    user_id: int = Depends(get_current_user_id),
):
    pool    = request.app.state.db_pool
    updates = payload.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="Aucune donnée à mettre à jour.")
    if "expires_at" in updates:
        updates["expires_at"] = parse_date(updates["expires_at"])

    fields = ", ".join(f"{k} = ${i+3}" for i, k in enumerate(updates.keys()))
    values = list(updates.values())

    async with pool.acquire() as conn:
        result = await conn.execute(
            f"UPDATE fridge_items SET {fields}, updated_at=NOW() WHERE id=$1 AND user_id=$2",
            item_id, user_id, *values,
        )
    if result == "UPDATE 0":
        raise HTTPException(status_code=404, detail="Ingrédient introuvable.")
    return {"message": "Ingrédient mis à jour."}


@router.delete("/{item_id}")
async def delete_fridge_item(
    item_id: int,
    request: Request,
    user_id: int = Depends(get_current_user_id),
):
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM fridge_items WHERE id=$1 AND user_id=$2",
            item_id, user_id,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Ingrédient introuvable.")
    return {"message": "Ingrédient supprimé."}


@router.delete("")
async def clear_fridge(
    request: Request,
    user_id: int = Depends(get_current_user_id),
):
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM fridge_items WHERE user_id=$1", user_id
        )
    deleted = int(result.split()[-1])
    return {"message": f"{deleted} ingrédient(s) supprimé(s).", "deleted": deleted}