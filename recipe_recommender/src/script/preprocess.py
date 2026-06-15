"""
scripts/preprocess.py
──────────────────────
Parse le dataset Food.com (RAW_recipes.csv) et produit data/recipes.json

Dataset source :
  https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

Colonnes utilisées :
  name, id, minutes, tags, nutrition, steps, ingredients

Colonne `nutrition` format :
  [calories, total_fat_PDV, sugar_PDV, sodium_PDV, protein_PDV, sat_fat_PDV, carbs_PDV]
  PDV = % Daily Value — on convertit en grammes absolus avec les valeurs de référence FDA.

Usage :
  python scripts/preprocess.py --max 10000 --no-translate
  python scripts/preprocess.py --max 5000  # avec traduction FR
"""

import argparse
import ast
import json
import logging
import re
import sys
import time
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

# Objectif sportif → cibles macro pour le scoring calorique
# Format : (kcal_ratio_per_meal, min_protein_g, max_carbs_g)
OBJECTIVE_NUTRITION = {
    "weight_loss":   {"meal_ratio": 0.30, "min_protein_g": 20, "max_carbs_g": 40},
    "muscle_gain":   {"meal_ratio": 0.35, "min_protein_g": 30, "max_carbs_g": 80},
    "maintenance":   {"meal_ratio": 0.33, "min_protein_g": 15, "max_carbs_g": 70},
    "endurance":     {"meal_ratio": 0.35, "min_protein_g": 15, "max_carbs_g": 100},
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
    """
    Parse la colonne nutrition Food.com et retourne les valeurs absolues.
    Format source : [calories, fat_PDV, sugar_PDV, sodium_PDV, protein_PDV, sat_fat_PDV, carbs_PDV]
    """
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
    """
    Déduit le type de repas depuis les tags Food.com.
    Retourne 'breakfast' | 'lunch' | 'dinner' | 'snack' | None
    """
    tags_lower = [t.lower() for t in tags]
    for meal_type, keywords in MEAL_TYPE_TAGS.items():
        if any(kw in tags_lower for kw in keywords):
            return meal_type
    return None


# ── Traduction ─────────────────────────────────────────────────────────────

def translate_batch(translator, texts: list[str], desc: str) -> list[str]:
    total, results, bs = len(texts), [], translator.batch_size
    for start in range(0, total, bs):
        chunk      = texts[start : start + bs]
        translated = translator.translate(chunk)
        results.extend(translated if isinstance(translated, list) else [translated])
        logger.info(f"  {desc}: {min(start + bs, total)}/{total}")
    return results


# ── Build sans traduction ──────────────────────────────────────────────────

def build_no_translate(df: pd.DataFrame, output: Path) -> None:
    logger.info(f"Traitement de {len(df)} recettes (sans traduction)…")
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

        # Filtre recettes sans calories (données trop incomplètes)
        if nutrition["calories"] is None or nutrition["calories"] <= 0:
            skipped += 1
            continue

        recipe: dict = {
            "title":       str(row.get("name", "")).strip(),
            "ingredients": ingredients,
            "steps":       steps,
            "ner":         ingredients,          # Food.com n'a pas de champ NER séparé
            "tags":        tags,
            "meal_type":   meal_type,
            "minutes":     int(row["minutes"]) if pd.notna(row.get("minutes")) else None,
            **nutrition,
        }
        recipes.append(recipe)

    _save(recipes, output)
    logger.info(f"   {skipped} recettes ignorées")


# ── Build avec traduction ──────────────────────────────────────────────────

def build_with_translate(df: pd.DataFrame, translator, output: Path) -> None:
    logger.info(f"Traitement de {len(df)} recettes (avec traduction EN→FR)…")

    titles_en, ingredients_en, steps_en = [], [], []
    ing_offsets, step_offsets           = [], []
    rows_valid                          = []
    skipped = 0

    for _, row in df.iterrows():
        ingredients = parse_list_field(row.get("ingredients", ""))
        steps       = parse_list_field(row.get("steps", ""))
        tags        = parse_list_field(row.get("tags", ""))
        nutrition   = parse_nutrition(row.get("nutrition"))

        if len(ingredients) < MIN_INGREDIENTS or len(steps) < MIN_STEPS:
            skipped += 1
            continue
        if nutrition["calories"] is None or nutrition["calories"] <= 0:
            skipped += 1
            continue

        rows_valid.append({
            "title":       str(row.get("name", "")).strip(),
            "ingredients": ingredients,
            "steps":       steps,
            "tags":        tags,
            "meal_type":   infer_meal_type(tags),
            "minutes":     int(row["minutes"]) if pd.notna(row.get("minutes")) else None,
            **nutrition,
        })

        titles_en.append(rows_valid[-1]["title"])
        ing_offsets.append(len(ingredients_en))
        ingredients_en.extend(ingredients)
        step_offsets.append(len(steps_en))
        steps_en.extend(steps)

    logger.info(f"  {len(rows_valid)} recettes valides ({skipped} ignorées)")

    logger.info("Étape 1/3 — Titres…")
    t0        = time.time()
    titles_fr = translate_batch(translator, titles_en, "titres")
    logger.info(f"  ✅ {time.time()-t0:.0f}s")

    logger.info("Étape 2/3 — Ingrédients…")
    t0             = time.time()
    ingredients_fr = translate_batch(translator, ingredients_en, "ingrédients")
    logger.info(f"  ✅ {time.time()-t0:.0f}s")

    logger.info("Étape 3/3 — Étapes…")
    t0       = time.time()
    steps_fr = translate_batch(translator, steps_en, "étapes")
    logger.info(f"  ✅ {time.time()-t0:.0f}s")

    recipes = []
    for i, row in enumerate(rows_valid):
        start_ing = ing_offsets[i]
        end_ing   = ing_offsets[i+1] if i+1 < len(ing_offsets) else len(ingredients_fr)
        ings_fr   = [t for t in ingredients_fr[start_ing:end_ing] if t.strip()]

        start_stp = step_offsets[i]
        end_stp   = step_offsets[i+1] if i+1 < len(step_offsets) else len(steps_fr)
        stps_fr   = [t for t in steps_fr[start_stp:end_stp] if t.strip()]

        if not ings_fr or not stps_fr:
            continue

        recipe = {
            "title":       titles_fr[i],
            "ingredients": ings_fr,
            "steps":       stps_fr,
            "ner":         row["ingredients"],   # EN — matching
            "tags":        row["tags"],
            "meal_type":   row["meal_type"],
            "minutes":     row["minutes"],
            "calories":    row["calories"],
            "protein_g":   row["protein_g"],
            "carbs_g":     row["carbs_g"],
            "total_fat_g": row["total_fat_g"],
            "sugar_g":     row["sugar_g"],
            "sodium_mg":   row["sodium_mg"],
        }
        recipes.append(recipe)

    _save(recipes, output)


def _save(recipes: list, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)
    logger.info(f"\n✅  {len(recipes)} recettes sauvegardées → {output}")


# ── Point d'entrée ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",          type=str, default=str(RAW_CSV))
    parser.add_argument("--output",       type=str, default=str(OUTPUT))
    parser.add_argument("--max",          type=int, default=None)
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--no-translate", action="store_true")
    parser.add_argument("--device",       type=str, default="cpu")
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

    if args.no_translate:
        build_no_translate(df, Path(args.output))
    else:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.models.translator import Translator
        translator = Translator(batch_size=args.batch_size, device=args.device)
        build_with_translate(df, translator, Path(args.output))


if __name__ == "__main__":
    main()