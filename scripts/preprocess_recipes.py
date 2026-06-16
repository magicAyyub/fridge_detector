"""
scripts/preprocess_recipes.py
──────────────────────────────
Parse le dataset Food.com (RAW_recipes.csv) et produit data/recipes.json (sans traduction).

Dataset source :
  https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

Colonnes utilisées :
  name, id, minutes, tags, nutrition, steps, ingredients

Usage :
  python scripts/preprocess_recipes.py --max 10000
"""

import argparse
import ast
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas requis : pip install pandas")

# ── Configuration ──────────────────────────────────────────────────────────
RAW_CSV         = Path("data/RAW_recipes.csv")
OUTPUT          = Path("data/recipes.json")
MIN_INGREDIENTS = 3
MIN_STEPS       = 1

# Valeurs de référence FDA pour convertir PDV → grammes
# [calories(kcal), fat(g), sugar(g), sodium(mg), protein(g), sat_fat(g), carbs(g)]
FDA_REF = {
    "total_fat_g":      78.0,
    "sugar_g":          50.0,
    "sodium_mg":      2300.0,
    "protein_g":        50.0,
    "saturated_fat_g":  20.0,
    "carbs_g":         275.0,
}

# Tags Food.com → type de repas
MEAL_TYPE_TAGS = {
    "breakfast": [
        "breakfast", "brunch", "morning", "pancakes",
        "waffles", "eggs", "oatmeal", "smoothie",
    ],
    "lunch": [
        "lunch", "salad", "sandwich", "soup", "wrap",
        "light-meals-snacks", "salads",
    ],
    "dinner": [
        "dinner", "main-dish", "main-course", "supper",
        "weeknight", "roast", "stew", "pasta",
        "beef", "chicken", "pork", "seafood", "fish",
    ],
    "snack": [
        "snack", "appetizer", "finger-food", "dessert",
        "cookies", "cake", "bread",
    ],
}


# ── Parsing ────────────────────────────────────────────────────────────────

def parse_list_field(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    items = re.split(r"[\n\r]+|,(?![^[]*\])", raw)
    return [x.strip().strip("'\"") for x in items if x.strip()]


def parse_nutrition(raw) -> dict:
    empty = {
        "calories": None,
        "total_fat_g": None,
        "sugar_g": None,
        "sodium_mg": None,
        "protein_g": None,
        "saturated_fat_g": None,
        "carbs_g": None,
    }
    try:
        values = ast.literal_eval(str(raw)) if not isinstance(raw, list) else raw
        if not isinstance(values, list) or len(values) < 7:
            return empty
        calories     = round(float(values[0]), 1)
        total_fat_g  = round(float(values[1]) / 100 * FDA_REF["total_fat_g"], 1)
        sugar_g      = round(float(values[2]) / 100 * FDA_REF["sugar_g"], 1)
        sodium_mg    = round(float(values[3]) / 100 * FDA_REF["sodium_mg"], 1)
        protein_g    = round(float(values[4]) / 100 * FDA_REF["protein_g"], 1)
        sat_fat_g    = round(float(values[5]) / 100 * FDA_REF["saturated_fat_g"], 1)
        carbs_g      = round(float(values[6]) / 100 * FDA_REF["carbs_g"], 1)
        return {
            "calories":        calories,
            "total_fat_g":     total_fat_g,
            "sugar_g":         sugar_g,
            "sodium_mg":       sodium_mg,
            "protein_g":       protein_g,
            "saturated_fat_g": sat_fat_g,
            "carbs_g":         carbs_g,
        }
    except Exception:
        return empty


def infer_meal_type(tags: list[str]) -> str | None:
    tags_lower = [t.lower() for t in tags]
    for meal_type, keywords in MEAL_TYPE_TAGS.items():
        if any(kw in tags_lower for kw in keywords):
            return meal_type
    return None


def _save(recipes: list, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)
    logger.info(f"\n✅  {len(recipes)} recettes sauvegardées → {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    type=str, default=str(RAW_CSV))
    parser.add_argument("--output", type=str, default=str(OUTPUT))
    parser.add_argument("--max",    type=int, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"Fichier introuvable : {csv_path}")

    logger.info(f"Lecture de {csv_path}…")
    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    logger.info(f"  {len(df)} lignes — colonnes : {list(df.columns)}")

    if args.max:
        df = df.head(args.max)

    logger.info(f"Traitement de {len(df)} recettes…")
    recipes, skipped = [], 0

    for _, row in df.iterrows():
        ingredients = parse_list_field(row.get("ingredients", ""))
        steps       = parse_list_field(row.get("steps", ""))
        tags        = parse_list_field(row.get("tags", ""))

        if len(ingredients) < MIN_INGREDIENTS or len(steps) < MIN_STEPS:
            skipped += 1
            continue

        nutrition  = parse_nutrition(row.get("nutrition"))
        meal_type  = infer_meal_type(tags)

        # Filtre recettes sans calories
        if nutrition["calories"] is None or nutrition["calories"] <= 0:
            skipped += 1
            continue

        recipe: dict = {
            "title":       str(row.get("name", "")).strip(),
            "ingredients": ingredients,
            "steps":       steps,
            "ner":         ingredients,
            "tags":        tags,
            "meal_type":   meal_type,
            "minutes":     int(row["minutes"]) if pd.notna(row.get("minutes")) else None,
            **nutrition,
        }
        recipes.append(recipe)

    _save(recipes, Path(args.output))
    logger.info(f"   {skipped} recettes ignorées")


if __name__ == "__main__":
    main()
