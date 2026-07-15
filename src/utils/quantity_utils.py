"""
quantity_utils.py
Helpers for parsing recipe ingredient amounts and computing fridge deductions.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

COUNT_UNITS = {"pieces", "piece", "count", "unit", "units", "tbsp", "tsp", "cup", "cups"}
WEIGHT_G_UNITS = {"g", "gram", "grams"}
WEIGHT_KG_UNITS = {"kg", "kilogram", "kilograms"}
VOLUME_ML_UNITS = {"ml", "milliliter", "milliliters"}
VOLUME_L_UNITS = {"l", "liter", "liters"}

# Portion par défaut quand la recette ne précise pas la quantité.
PORTION_G = 150.0
PORTION_ML = 150.0

DEFAULT_SERVING = {
    "g":  PORTION_G,
    "kg": PORTION_G / 1000.0,
    "ml": PORTION_ML,
    "l":  PORTION_ML / 1000.0,
}


def _normalize_unit(unit: Optional[str]) -> str:
    if not unit:
        return "pieces"
    u = unit.lower().strip()
    if u in WEIGHT_G_UNITS:
        return "g"
    if u in WEIGHT_KG_UNITS:
        return "kg"
    if u in VOLUME_ML_UNITS:
        return "ml"
    if u in VOLUME_L_UNITS:
        return "l"
    if u in COUNT_UNITS:
        return "pieces"
    return u


def parse_ingredient_amount(ingredient: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract a numeric amount from strings like '200g pasta', '2 cups milk', '0.5 kg chicken'.
    Returns (amount, normalized_unit) or (None, None).
    """
    text = ingredient.lower().strip()

    patterns = [
        (r"(\d+(?:\.\d+)?)\s*(kg)\b", "kg"),
        (r"(\d+(?:\.\d+)?)\s*(g|grams?)\b", "g"),
        (r"(\d+(?:\.\d+)?)\s*(ml|milliliters?)\b", "ml"),
        (r"(\d+(?:\.\d+)?)\s*(l|liters?)\b", "l"),
        (r"(\d+(?:\.\d+)?)\s*(cups?|tbsp|tsp|pieces?|units?)\b", "pieces"),
    ]

    for pattern, unit in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)), unit

    return None, None


def _convert_amount(amount: float, from_unit: str, to_unit: str) -> float:
    if from_unit == to_unit:
        return amount
    if from_unit == "g" and to_unit == "kg":
        return amount / 1000.0
    if from_unit == "kg" and to_unit == "g":
        return amount * 1000.0
    if from_unit == "ml" and to_unit == "l":
        return amount / 1000.0
    if from_unit == "l" and to_unit == "ml":
        return amount * 1000.0
    return amount


def compute_deduction(
    ingredient: str,
    fridge_qty: float,
    fridge_unit: Optional[str],
) -> float:
    """
    How much to subtract from the fridge for one recipe preparation.

    - Explicit amount in ingredient text ('200g pasta') when present
    - Weight/volume units: 150 g (or 150 ml) per matched ingredient
    - Count units: 1 piece
    """
    unit = _normalize_unit(fridge_unit)
    parsed_qty, parsed_unit = parse_ingredient_amount(ingredient)

    if parsed_qty is not None and parsed_unit is not None:
        if parsed_unit == unit or (
            parsed_unit in {"g", "kg"} and unit in {"g", "kg"}
        ) or (
            parsed_unit in {"ml", "l"} and unit in {"ml", "l"}
        ):
            amount = _convert_amount(parsed_qty, parsed_unit, unit)
            return min(fridge_qty, max(amount, 0.0))

        if parsed_unit == "pieces" and unit == "pieces":
            return min(fridge_qty, max(parsed_qty, 1.0))

    if unit in {"g", "kg", "ml", "l"}:
        default = DEFAULT_SERVING.get(unit, PORTION_G)
        return min(fridge_qty, default)

    return min(fridge_qty, 1.0)
