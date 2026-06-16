"""
utils/substitutor.py
─────────────────────
Propose des substituts pour les ingrédients manquants.

Table hand-crafted enrichie par catégorie.
Peut être remplacée par un appel LLM ou une API nutrition.
"""

from typing import List
from .schemas import SubstituteResult

# ── Table de substitution ──────────────────────────────────────────────────
# Format : ingrédient → [(substitut, note)]

SUBSTITUTES: dict[str, list[tuple[str, str]]] = {
    # Laitier
    "butter":        [("margarine", "même quantité"), ("coconut oil", "¾ de la quantité"), ("olive oil", "¾ de la quantité")],
    "milk":          [("oat milk", "même quantité"), ("almond milk", "même quantité"), ("soy milk", "même quantité")],
    "cream":         [("coconut cream", "même quantité"), ("greek yogurt", "même quantité")],
    "egg":           [("flax egg (1 tbsp flaxseed + 3 tbsp water)", "par œuf"), ("chia egg", "par œuf"), ("applesauce (¼ cup)", "par œuf dans les gâteaux")],
    "cheese":        [("nutritional yeast", "pour la saveur"), ("tofu", "pour la texture")],

    # Protéines
    "chicken":       [("turkey", "même quantité"), ("tofu", "même quantité"), ("chickpeas", "même quantité")],
    "beef":          [("lentils", "même quantité"), ("mushrooms", "pour la texture"), ("turkey", "même quantité")],
    "pork":          [("chicken", "même quantité"), ("turkey", "même quantité")],
    "salmon":        [("tuna", "même quantité"), ("cod", "même quantité")],
    "shrimp":        [("scallops", "même quantité"), ("firm tofu", "même quantité")],

    # Féculent / farine
    "flour":         [("almond flour", "même quantité (sans gluten)"), ("oat flour", "même quantité"), ("rice flour", "même quantité")],
    "bread":         [("tortilla", "à adapter"), ("lettuce wrap", "sans gluten")],
    "pasta":         [("zucchini noodles", "même quantité"), ("rice noodles", "même quantité")],
    "rice":          [("quinoa", "même quantité"), ("cauliflower rice", "même quantité"), ("barley", "même quantité")],
    "potato":        [("sweet potato", "même quantité"), ("cauliflower", "même quantité")],

    # Aromates / saveurs
    "garlic":        [("garlic powder (¼ tsp)", "par gousse"), ("shallot", "1 échalote par gousse")],
    "onion":         [("shallot", "même quantité"), ("leek", "même quantité"), ("onion powder", "1 tsp par oignon")],
    "tomato":        [("canned tomatoes", "même quantité"), ("red pepper", "pour la douceur"), ("sun-dried tomatoes", "moins de quantité")],
    "lemon":         [("lime", "même quantité"), ("white wine vinegar (½ tsp)", "par c. à soupe de jus")],
    "olive oil":     [("avocado oil", "même quantité"), ("coconut oil", "même quantité"), ("butter", "même quantité")],
    "sugar":         [("honey (¾ qty)", "légèrement différent"), ("maple syrup (¾ qty)", ""), ("coconut sugar", "même quantité")],
    "salt":          [("soy sauce (½ qty)", "ajoute de l'umami"), ("miso paste", "pour les sauces")],

    # Légumes
    "spinach":       [("kale", "même quantité"), ("arugula", "même quantité"), ("swiss chard", "même quantité")],
    "broccoli":      [("cauliflower", "même quantité"), ("broccolini", "même quantité")],
    "carrot":        [("parsnip", "même quantité"), ("sweet potato", "même quantité")],
    "zucchini":      [("cucumber (cru)", ""), ("yellow squash", "même quantité")],
    "mushroom":      [("eggplant", "pour la texture"), ("sun-dried tomatoes", "pour l'umami")],

    # Légumineuses
    "chickpeas":     [("white beans", "même quantité"), ("lentils", "même quantité")],
    "lentils":       [("split peas", "même quantité"), ("chickpeas", "même quantité")],

    # Bouillon / liquide
    "chicken broth": [("vegetable broth", "même quantité"), ("water + herbs", "même quantité")],
    "beef broth":    [("vegetable broth", "même quantité"), ("mushroom broth", "pour l'umami")],
    "wine":          [("grape juice + splash of vinegar", "même quantité"), ("broth", "même quantité")],
}

# Normalisation des clés pour le matching partiel
_KEYS_LOWER = {k.lower(): v for k, v in SUBSTITUTES.items()}


def find_substitutes(
    ingredient: str,
    dietary_restrictions: list[str] | None = None,
) -> SubstituteResult:
    """
    Cherche des substituts pour un ingrédient.
    Filtre selon les restrictions alimentaires si fournies.
    """
    ing_lower = ingredient.lower().strip()
    restrictions = [r.lower() for r in (dietary_restrictions or [])]

    # Matching exact d'abord, puis partiel
    candidates = _KEYS_LOWER.get(ing_lower)
    if candidates is None:
        for key, val in _KEYS_LOWER.items():
            if key in ing_lower or ing_lower in key:
                candidates = val
                break

    if not candidates:
        return SubstituteResult(
            original=ingredient,
            substitutes=[],
            notes=f"Aucun substitut connu pour '{ingredient}'. Essaie une recherche en ligne.",
        )

    # Filtre restrictions alimentaires
    filtered = [
        (sub, note) for sub, note in candidates
        if not any(r in sub.lower() for r in restrictions)
    ]

    substitutes = [
        f"{sub} ({note})" if note else sub
        for sub, note in filtered[:3]
    ]

    return SubstituteResult(
        original=ingredient,
        substitutes=substitutes,
        notes=f"Substituts pour '{ingredient}' compatibles avec tes restrictions." if substitutes
              else f"Aucun substitut compatible avec tes restrictions pour '{ingredient}'.",
    )


def get_substitutes_for_missing(
    missing_ingredients: list[str],
    dietary_restrictions: list[str] | None = None,
) -> list[SubstituteResult]:
    return [
        find_substitutes(ing, dietary_restrictions)
        for ing in missing_ingredients
    ]
