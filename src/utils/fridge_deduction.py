"""
fridge_deduction.py
───────────────────
Log and restore fridge deductions when marking / unmarking recipe preparations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.utils.quantity_utils import compute_deduction

logger = logging.getLogger(__name__)

DeductionRow = Dict[str, Any]


async def ensure_deductions_table(conn) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS preparation_fridge_deductions (
            id               SERIAL PRIMARY KEY,
            preparation_id   INTEGER NOT NULL
                             REFERENCES recipe_preparations(id) ON DELETE CASCADE,
            ingredient_name  VARCHAR(255) NOT NULL,
            quantity_deducted DOUBLE PRECISION NOT NULL,
            unit             VARCHAR(50),
            fridge_item_id   INTEGER
        )
        """
    )


def _match_fridge_item(
    ing: str,
    fridge: List[Tuple[int, str, float, Optional[str]]],
) -> Optional[Tuple[int, str, float, Optional[str]]]:
    ing_lower = ing.lower()
    ing_words = set(ing_lower.split())

    for fid, fname, fqty, funit in fridge:
        fname_words = set(fname.split())
        common = {w for w in ing_words & fname_words if len(w) > 3}
        if common or ing_lower in fname or fname in ing_lower:
            return fid, fname, fqty, funit
    return None


async def apply_fridge_deductions(
    conn,
    user_id: int,
    preparation_id: int,
    matched_ingredients: List[str],
) -> List[str]:
    """Deduct matched ingredients from the fridge and log each deduction."""
    deducted_names: List[str] = []
    if not matched_ingredients:
        return deducted_names

    await ensure_deductions_table(conn)

    fridge_rows = await conn.fetch(
        "SELECT id, ingredient_name, quantity, unit FROM fridge_items WHERE user_id=$1",
        user_id,
    )
    fridge = [
        (r["id"], r["ingredient_name"].lower(), float(r["quantity"]), r["unit"])
        for r in fridge_rows
    ]
    # Keep display names for restore
    display_names = {r["id"]: r["ingredient_name"] for r in fridge_rows}

    for ing in matched_ingredients:
        match = _match_fridge_item(ing, fridge)
        if match is None:
            continue

        matched_id, fname_lower, matched_qty, matched_unit = match
        display_name = display_names.get(matched_id, fname_lower)
        deduction = compute_deduction(ing, matched_qty, matched_unit)
        new_qty = matched_qty - deduction

        await conn.execute(
            """
            INSERT INTO preparation_fridge_deductions
                (preparation_id, ingredient_name, quantity_deducted, unit, fridge_item_id)
            VALUES ($1, $2, $3, $4, $5)
            """,
            preparation_id,
            display_name,
            deduction,
            matched_unit,
            matched_id,
        )

        if new_qty <= 0:
            await conn.execute(
                "DELETE FROM fridge_items WHERE id=$1 AND user_id=$2",
                matched_id, user_id,
            )
            fridge = [(fid, fn, fq, fu) for fid, fn, fq, fu in fridge if fid != matched_id]
        else:
            await conn.execute(
                "UPDATE fridge_items SET quantity=$1, updated_at=NOW() WHERE id=$2",
                new_qty, matched_id,
            )
            fridge = [
                (fid, fn, new_qty if fid == matched_id else fq, fu)
                for fid, fn, fq, fu in fridge
            ]

        deducted_names.append(ing)

    return deducted_names


async def restore_fridge_deductions(
    conn,
    user_id: int,
    preparation_id: int,
) -> List[str]:
    """Restore fridge quantities logged for a given preparation."""
    await ensure_deductions_table(conn)

    rows = await conn.fetch(
        """
        SELECT ingredient_name, quantity_deducted, unit, fridge_item_id
        FROM preparation_fridge_deductions
        WHERE preparation_id = $1
        ORDER BY id
        """,
        preparation_id,
    )

    restored: List[str] = []
    for row in rows:
        name = row["ingredient_name"]
        qty = float(row["quantity_deducted"])
        unit = row["unit"]
        item_id = row["fridge_item_id"]

        target = None
        if item_id is not None:
            target = await conn.fetchrow(
                "SELECT id, quantity FROM fridge_items WHERE id=$1 AND user_id=$2",
                item_id, user_id,
            )

        if target is None:
            target = await conn.fetchrow(
                """
                SELECT id, quantity FROM fridge_items
                WHERE user_id=$1 AND LOWER(ingredient_name)=LOWER($2)
                """,
                user_id, name,
            )

        if target:
            new_qty = float(target["quantity"]) + qty
            await conn.execute(
                "UPDATE fridge_items SET quantity=$1, updated_at=NOW() WHERE id=$2",
                new_qty, target["id"],
            )
        else:
            await conn.execute(
                """
                INSERT INTO fridge_items (user_id, ingredient_name, quantity, unit)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, ingredient_name) DO UPDATE SET
                    quantity   = fridge_items.quantity + EXCLUDED.quantity,
                    unit       = COALESCE(EXCLUDED.unit, fridge_items.unit),
                    updated_at = NOW()
                """,
                user_id, name, qty, unit,
            )

        restored.append(name)

    return restored
