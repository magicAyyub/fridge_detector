"""
src/api/routes/recipes.py
──────────────────────────
Recettes likées + préparations multi-jours via recipe_preparations.

Logique préparation :
  - user_saved_recipes.is_prepared = booléen "déjà cuisinée au moins une fois"
  - recipe_preparations = log de chaque préparation avec date
  - Le lendemain, is_prepared reste TRUE mais l'utilisateur peut re-préparer
  - GET /users/me/recipes/preparations?date=YYYY-MM-DD → repas d'un jour donné
"""

import json
import logging
from datetime import date as date_type

def parse_date(value: str) -> date_type:
    """Convertit YYYY-MM-DD en objet date Python pour asyncpg."""
    try:
        return date_type.fromisoformat(str(value))
    except (ValueError, TypeError):
        return date_type.today()

def parse_date(value: str) -> date_type:
    """Convertit YYYY-MM-DD en objet date Python pour asyncpg."""
    try:
        return date_type.fromisoformat(value)
    except (ValueError, TypeError):
        return date_type.today()
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src.utils.auth_utils import get_current_user_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users/me/recipes", tags=["recipes"])


# ── Schemas ────────────────────────────────────────────────────────────────

class RecipeStepInput(BaseModel):
    step:        int
    instruction: str


class SaveRecipeInput(BaseModel):
    title:            str
    meal_type:        Optional[str]                   = None
    minutes:          Optional[int]                   = None
    calories:         Optional[float]                 = None
    protein_g:        Optional[float]                 = None
    carbs_g:          Optional[float]                 = None
    total_fat_g:      Optional[float]                 = None
    saturated_fat_g:  Optional[float]                 = None
    sugar_g:          Optional[float]                 = None
    sodium_mg:        Optional[float]                 = None
    steps:            Optional[List[RecipeStepInput]] = None
    all_ingredients:  Optional[List[str]]             = None   # tous les ingrédients
    matched_ingredients: Optional[List[str]]          = None   # possédés par l'user
    missing_ingredients: Optional[List[str]]          = None   # manquants


class PrepareRecipeInput(BaseModel):
    matched_ingredients: Optional[List[str]] = None  # pour déduire du frigo


class SavedRecipeResponse(BaseModel):
    id:          int
    title:       str
    meal_type:   Optional[str]   = None
    minutes:     Optional[int]   = None
    calories:    Optional[float] = None
    protein_g:   Optional[float] = None
    carbs_g:     Optional[float] = None
    total_fat_g: Optional[float] = None
    is_prepared: bool
    saved_at:    str
    prepared_at: Optional[str]   = None
    prep_count:  int             = 0


# ── Routes ─────────────────────────────────────────────────────────────────

@router.get("", response_model=List[SavedRecipeResponse])
async def get_saved_recipes(
    request:  Request,
    user_id:  int            = Depends(get_current_user_id),
    prepared: Optional[bool] = None,
):
    """
    Recettes likées.
    ?prepared=true → seulement celles cuisinées au moins une fois.
    Inclut prep_count pour savoir combien de fois cuisinée.
    """
    pool  = request.app.state.db_pool
    query = """
        SELECT r.id, r.title, r.meal_type, r.minutes, r.calories,
               r.protein_g, r.carbs_g, r.total_fat_g,
               r.ingredients,
               usr.is_prepared,
               usr.saved_at::text    AS saved_at,
               usr.prepared_at::text AS prepared_at,
               COALESCE(p.prep_count, 0) AS prep_count
        FROM user_saved_recipes usr
        JOIN recipes r ON r.id = usr.recipe_id
        LEFT JOIN (
            SELECT recipe_id, COUNT(*) AS prep_count
            FROM recipe_preparations
            WHERE user_id = $1
            GROUP BY recipe_id
        ) p ON p.recipe_id = r.id
        WHERE usr.user_id = $1
    """
    params = [user_id]

    if prepared is not None:
        params.append(prepared)
        query += f" AND usr.is_prepared = ${len(params)}"

    query += " ORDER BY usr.saved_at DESC"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    result = []
    for r in rows:
        row = dict(r)
        raw = row.get("ingredients")
        if raw:
            try:
                if isinstance(raw, str):
                    row["all_ingredients"] = json.loads(raw)
                elif isinstance(raw, (list, tuple)):
                    row["all_ingredients"] = list(raw)
                else:
                    row["all_ingredients"] = []
            except Exception:
                row["all_ingredients"] = []
        else:
            row["all_ingredients"] = []
        result.append(row)
    return result


@router.get("/preparations")
async def get_preparations_by_date(
    request: Request,
    user_id: int           = Depends(get_current_user_id),
    date:    Optional[str] = None,   # ?date=2025-06-15, défaut = today
):
    """
    Préparations d'un jour donné.
    Utilisé par nutrition-store pour charger les calories du jour.
    Renvoie toutes les préparations même si la recette a été cuisinée
    plusieurs fois le même jour.
    """
    pool   = request.app.state.db_pool
    target = parse_date(date) if date else date_type.today()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT rp.id          AS preparation_id,
                   rp.prepared_at::text,
                   r.id           AS recipe_id,
                   r.title, r.meal_type, r.minutes,
                   r.calories, r.protein_g, r.carbs_g, r.total_fat_g
            FROM recipe_preparations rp
            JOIN recipes r ON r.id = rp.recipe_id
            WHERE rp.user_id = $1
              AND rp.prepared_at::date = $2
            ORDER BY rp.prepared_at
            """,
            user_id, target,
        )
    return [dict(r) for r in rows]


@router.get("/{recipe_id}")
async def get_saved_recipe(
    recipe_id: int,
    request:   Request,
    user_id:   int = Depends(get_current_user_id),
):
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        recipe = await conn.fetchrow(
            """
            SELECT r.id, r.title, r.meal_type, r.minutes, r.calories,
                   r.protein_g, r.carbs_g, r.total_fat_g,
                   r.saturated_fat_g, r.sugar_g, r.sodium_mg,
                   r.ingredients,
                   usr.is_prepared,
                   usr.saved_at::text    AS saved_at,
                   usr.prepared_at::text AS prepared_at
            FROM user_saved_recipes usr
            JOIN recipes r ON r.id = usr.recipe_id
            WHERE usr.user_id = $1 AND r.id = $2
            """,
            user_id, recipe_id,
        )
        if not recipe:
            raise HTTPException(status_code=404, detail="Recette non trouvée.")
        steps = await conn.fetch(
            "SELECT step_number AS step, instruction FROM recipe_steps WHERE recipe_id=$1 ORDER BY step_number",
            recipe_id,
        )

    result = dict(recipe)
    result["steps"] = [dict(s) for s in steps]

    # Parse JSONB ingredients → all_ingredients
    raw_ing = result.get("ingredients")
    if raw_ing:
        try:
            if isinstance(raw_ing, str):
                all_ings = json.loads(raw_ing)
            elif isinstance(raw_ing, (list, tuple)):
                all_ings = list(raw_ing)
            else:
                all_ings = []
        except Exception:
            all_ings = []
    else:
        all_ings = []
    result["all_ingredients"] = all_ings

    # Calcule matched/missing en croisant avec le frigo actuel
    # C'est plus fiable que les données stockées au moment du like
    if all_ings:
        async with pool.acquire() as conn:
            fridge_rows = await conn.fetch(
                "SELECT ingredient_name FROM fridge_items WHERE user_id=$1",
                user_id,
            )
        fridge_names = {r["ingredient_name"].lower() for r in fridge_rows}

        def _matches(ing: str) -> bool:
            ing_lower = ing.lower()
            ing_words = {w for w in ing_lower.split() if len(w) > 3}
            for fname in fridge_names:
                fname_words = {w for w in fname.split() if len(w) > 3}
                common = ing_words & fname_words
                if common or ing_lower in fname or fname in ing_lower:
                    return True
            return False

        matched = [ing for ing in all_ings if _matches(ing)]
        missing = [ing for ing in all_ings if not _matches(ing)]
        result["matched_ingredients"] = matched
        result["missing_ingredients"] = missing
    else:
        result["matched_ingredients"] = []
        result["missing_ingredients"] = []

    return result


@router.post("", status_code=201)
async def save_recipe(
    payload: SaveRecipeInput,
    request: Request,
    user_id: int = Depends(get_current_user_id),
):
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        async with conn.transaction():
            recipe = await conn.fetchrow(
                """
                INSERT INTO recipes
                    (title, meal_type, minutes, calories, protein_g, carbs_g,
                     total_fat_g, saturated_fat_g, sugar_g, sodium_mg, ingredients)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                ON CONFLICT (title) DO UPDATE SET
                    meal_type   = EXCLUDED.meal_type,
                    calories    = COALESCE(EXCLUDED.calories,    recipes.calories),
                    protein_g   = COALESCE(EXCLUDED.protein_g,  recipes.protein_g),
                    carbs_g     = COALESCE(EXCLUDED.carbs_g,    recipes.carbs_g),
                    total_fat_g = COALESCE(EXCLUDED.total_fat_g,recipes.total_fat_g),
                    ingredients = COALESCE(EXCLUDED.ingredients, recipes.ingredients)
                RETURNING id
                """,
                payload.title, payload.meal_type, payload.minutes,
                payload.calories, payload.protein_g, payload.carbs_g,
                payload.total_fat_g, payload.saturated_fat_g,
                payload.sugar_g, payload.sodium_mg,
                json.dumps(payload.all_ingredients or []),
            )
            recipe_id = recipe["id"]

            if payload.steps:
                await conn.execute("DELETE FROM recipe_steps WHERE recipe_id=$1", recipe_id)
                await conn.executemany(
                    "INSERT INTO recipe_steps (recipe_id, step_number, instruction) VALUES ($1,$2,$3)",
                    [(recipe_id, s.step, s.instruction) for s in payload.steps],
                )

            await conn.execute(
                """
                INSERT INTO user_saved_recipes (user_id, recipe_id)
                VALUES ($1, $2)
                ON CONFLICT (user_id, recipe_id) DO NOTHING
                """,
                user_id, recipe_id,
            )

    return {"recipe_id": recipe_id, "message": "Recette sauvegardée."}


@router.post("/{recipe_id}/prepared")
async def mark_prepared(
    recipe_id: int,
    payload:   PrepareRecipeInput,
    request:   Request,
    user_id:   int = Depends(get_current_user_id),
):
    """
    Cuisine une recette :
    1. Crée un log dans recipe_preparations (multi-jours possible)
    2. Met à jour user_saved_recipes.is_prepared = TRUE
    3. Déduit les ingrédients disponibles du frigo
    """
    pool = request.app.state.db_pool

    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT 1 FROM user_saved_recipes WHERE user_id=$1 AND recipe_id=$2",
            user_id, recipe_id,
        )
        if not exists:
            raise HTTPException(status_code=404, detail="Recette non likée.")

        async with conn.transaction():
            # 1. Log préparation
            await conn.execute(
                "INSERT INTO recipe_preparations (user_id, recipe_id) VALUES ($1,$2)",
                user_id, recipe_id,
            )

            # 2. Marque cuisinée
            await conn.execute(
                """
                UPDATE user_saved_recipes
                SET is_prepared=TRUE, prepared_at=NOW()
                WHERE user_id=$1 AND recipe_id=$2
                """,
                user_id, recipe_id,
            )

            # 3. Déduction frigo
            # Stratégie : pour chaque ingrédient de la recette, cherche une correspondance
            # dans le frigo par matching bidirectionnel (ing dans fridge OU fridge dans ing)
            deducted = []
            all_ings_to_check = payload.matched_ingredients or []
            if all_ings_to_check:
                # Charge tout le frigo une fois (évite N requêtes)
                fridge_rows = await conn.fetch(
                    "SELECT id, ingredient_name, quantity FROM fridge_items WHERE user_id=$1",
                    user_id,
                )
                fridge = [(r["id"], r["ingredient_name"].lower(), float(r["quantity"])) for r in fridge_rows]

                for ing in all_ings_to_check:
                    ing_lower  = ing.lower()
                    ing_words  = set(ing_lower.split())
                    matched_id = None
                    matched_qty= None

                    for fid, fname, fqty in fridge:
                        fname_words = set(fname.split())
                        # Match si partage au moins un mot significatif (>3 chars)
                        common = {w for w in ing_words & fname_words if len(w) > 3}
                        if common or ing_lower in fname or fname in ing_lower:
                            matched_id  = fid
                            matched_qty = fqty
                            break

                    if matched_id is None:
                        continue

                    new_qty = matched_qty - 1
                    if new_qty <= 0:
                        await conn.execute(
                            "DELETE FROM fridge_items WHERE id=$1 AND user_id=$2",
                            matched_id, user_id,
                        )
                        # Retire du cache local
                        fridge = [(fid, fn, fq) for fid, fn, fq in fridge if fid != matched_id]
                    else:
                        await conn.execute(
                            "UPDATE fridge_items SET quantity=$1, updated_at=NOW() WHERE id=$2",
                            new_qty, matched_id,
                        )
                        fridge = [(fid, fn, new_qty if fid == matched_id else fq) for fid, fn, fq in fridge]
                    deducted.append(ing)

    return {"message": "Recette cuisinée.", "deducted": deducted}


@router.delete("/{recipe_id}/prepared")
async def unmark_prepared(
    recipe_id: int,
    request:   Request,
    user_id:   int = Depends(get_current_user_id),
):
    """Annule la dernière préparation du jour."""
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        await conn.execute(
            """
            DELETE FROM recipe_preparations WHERE id = (
                SELECT id FROM recipe_preparations
                WHERE user_id=$1 AND recipe_id=$2
                  AND prepared_at::date = CURRENT_DATE
                ORDER BY prepared_at DESC LIMIT 1
            )
            """,
            user_id, recipe_id,
        )
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM recipe_preparations WHERE user_id=$1 AND recipe_id=$2",
            user_id, recipe_id,
        )
        if count == 0:
            await conn.execute(
                "UPDATE user_saved_recipes SET is_prepared=FALSE, prepared_at=NULL WHERE user_id=$1 AND recipe_id=$2",
                user_id, recipe_id,
            )
    return {"message": "Préparation annulée."}


@router.delete("/{recipe_id}")
async def unsave_recipe(
    recipe_id: int,
    request:   Request,
    user_id:   int = Depends(get_current_user_id),
):
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM user_saved_recipes WHERE user_id=$1 AND recipe_id=$2",
            user_id, recipe_id,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Recette non trouvée.")
    return {"message": "Recette retirée des favoris."}