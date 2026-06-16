"""
models/two_tower.py
────────────────────
Two-Tower scoring pour la recommandation de recettes.

Tower 1 — User :
  - ingrédients du frigo (TF-IDF sparse)
  - profil utilisateur (objectif, calories cibles)
  - historique feedback (ingrédients manquants fréquents, recettes dislikées)

Tower 2 — Recipe :
  - ingrédients NER (TF-IDF sparse)
  - nutrition (calories, protein, carbs, fat normalisés)
  - meal_type (one-hot)

Score final = cosine(user_vec, recipe_vec)
            + bonus_nutrition(profil, recette)
            - pénalité_feedback(session, recette)

Note : il s'agit d'un Two-Tower "léger" basé sur des features hand-crafted,
sans entraînement neural. Il peut être remplacé par un vrai réseau de neurones
une fois qu'on dispose de suffisamment de données d'interaction.
"""

import math
import logging
from typing import Any, Dict, List, Optional

from .vectorizer import tokenize

logger = logging.getLogger(__name__)

MEAL_TYPES = ["breakfast", "lunch", "dinner", "snack"]

# Poids des composantes du score final
W_COSINE    = 0.50   # similarité ingrédients
W_NUTRITION = 0.30   # adéquation nutritionnelle
W_JACCARD   = 0.20   # couverture ingrédients

# Tolérance calorique ±% autour de la cible par repas
CALORIE_TOLERANCE = 0.35


# ── Normalisation ──────────────────────────────────────────────────────────

def _safe(val, default=0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot    = sum(a.get(k, 0.0) * v for k, v in b.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def jaccard(sa: set, sb: set) -> float:
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ── Tower 1 : encodeur utilisateur ────────────────────────────────────────

class UserEncoder:
    """
    Produit un vecteur utilisateur à partir de :
      - tokens des ingrédients du frigo
      - profil (objectif, calorie_target)
      - feedback history (ingrédients manquants pénalisés)
    """

    def __init__(self, idf: Dict[str, float]):
        self.idf = idf

    def encode(
        self,
        fridge_tokens: List[str],
        profile: Optional[Any] = None,          # UserProfileInput
        feedback_history: Optional[Dict] = None, # {ingredient: miss_count}
    ) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        for t in fridge_tokens:
            counts[t] = counts.get(t, 0) + 1
        n = max(len(fridge_tokens), 1)

        vector: Dict[str, float] = {}
        for tok, cnt in counts.items():
            if tok in self.idf:
                # Downweight les tokens souvent manquants dans le feedback
                miss_penalty = 1.0
                if feedback_history and tok in feedback_history:
                    miss_penalty = max(0.3, 1.0 - feedback_history[tok] * 0.1)
                vector[tok] = (cnt / n) * self.idf[tok] * miss_penalty

        return vector


# ── Tower 2 : encodeur recette ─────────────────────────────────────────────

class RecipeEncoder:
    """
    Produit un vecteur recette à partir de :
      - tokens NER (ingrédients normalisés EN)
      - features nutritionnelles normalisées
      - meal_type one-hot
    """

    def __init__(self, idf: Dict[str, float], nutrition_stats: Dict[str, float]):
        self.idf = idf
        self.nutrition_stats = nutrition_stats   # max de chaque feature pour normaliser

    def encode(self, recipe: Dict[str, Any]) -> Dict[str, float]:
        # ── Partie ingrédients (TF-IDF) ────────────────────────────────
        ner_list = recipe.get("ner") or recipe.get("ingredients", [])
        tokens: List[str] = []
        for item in ner_list:
            tokens.extend(tokenize(item))

        counts: Dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        n = max(len(tokens), 1)

        vector: Dict[str, float] = {}
        for tok, cnt in counts.items():
            if tok in self.idf:
                vector[tok] = (cnt / n) * self.idf[tok]

        # ── Partie nutrition (features numériques normalisées) ──────────
        # On les ajoute dans un espace de features séparé avec préfixe "nutr_"
        for feat in ["calories", "protein_g", "carbs_g", "total_fat_g"]:
            val = _safe(recipe.get(feat))
            max_val = self.nutrition_stats.get(feat, 1.0)
            if max_val > 0:
                vector[f"nutr_{feat}"] = val / max_val

        # ── Meal type one-hot ───────────────────────────────────────────
        meal = recipe.get("meal_type") or ""
        for mt in MEAL_TYPES:
            vector[f"meal_{mt}"] = 1.0 if meal == mt else 0.0

        return vector


# ── Scoring nutrition ──────────────────────────────────────────────────────

def nutrition_score(
    recipe: Dict[str, Any],
    profile: Any,            # UserProfileInput
    meal_type: Optional[str] = None,
) -> tuple[float, str]:
    """
    Retourne (bonus_score, calorie_fit_label).
    calorie_fit : 'perfect' | 'low' | 'high' | 'unknown'
    """
    from src.utils.schemas import MEAL_CALORIE_RATIO, OBJECTIVE_TARGETS

    if profile is None:
        return 0.0, "unknown"

    recipe_kcal = _safe(recipe.get("calories"))
    if recipe_kcal <= 0:
        return 0.0, "unknown"

    # Cible calorique pour ce repas
    mt = meal_type or "lunch"
    meal_ratio   = MEAL_CALORIE_RATIO.get(mt, 0.33)
    target_kcal  = profile.calorie_target * meal_ratio

    # Adéquation calorique
    ratio = recipe_kcal / target_kcal if target_kcal > 0 else 1.0
    if abs(1.0 - ratio) <= CALORIE_TOLERANCE:
        cal_fit   = "perfect"
        cal_bonus = 0.20
    elif ratio < 1.0 - CALORIE_TOLERANCE:
        cal_fit   = "low"
        cal_bonus = 0.05
    else:
        cal_fit   = "high"
        cal_bonus = -0.10   # pénalité si trop calorique

    # Bonus macro selon objectif
    obj     = profile.sports_objective
    targets = OBJECTIVE_TARGETS.get(obj, OBJECTIVE_TARGETS["maintenance"])
    macro_bonus = 0.0

    protein = _safe(recipe.get("protein_g"))
    carbs   = _safe(recipe.get("carbs_g"))
    fat     = _safe(recipe.get("total_fat_g"))

    if protein >= targets["min_protein_g"]:
        macro_bonus += 0.05
    if carbs <= targets["max_carbs_g"]:
        macro_bonus += 0.03
    if fat <= targets.get("max_fat_g", 999):
        macro_bonus += 0.02

    total = cal_bonus + macro_bonus
    return round(min(max(total, -0.15), 0.30), 4), cal_fit


# ── Filtre allergènes ──────────────────────────────────────────────────────

def passes_allergy_filter(recipe: Dict[str, Any], restrictions: List[str]) -> bool:
    if not restrictions:
        return True
    ner_text = " ".join(recipe.get("ner") or recipe.get("ingredients", [])).lower()
    tags_text = " ".join(recipe.get("tags", [])).lower()
    full_text = ner_text + " " + tags_text
    for allergen in restrictions:
        if allergen.lower() in full_text:
            return False
    return True


# ── Two-Tower principal ────────────────────────────────────────────────────

class TwoTowerScorer:
    """
    Orchestre les deux encodeurs et calcule le score final.
    """

    def __init__(self):
        self.user_encoder:   Optional[UserEncoder]   = None
        self.recipe_encoder: Optional[RecipeEncoder] = None
        self._recipe_vectors: List[Dict[str, float]] = []
        self._recipe_token_sets: List[set] = []
        self._fitted = False

    def fit(self, recipes: List[Dict[str, Any]]) -> "TwoTowerScorer":
        """Construit IDF + vecteurs recettes."""
        logger.info("Two-Tower : calcul IDF…")

        # ── IDF sur le corpus NER ──────────────────────────────────────
        from collections import Counter
        doc_freq: Counter = Counter()
        corpus_tokens = []

        for recipe in recipes:
            ner_list = recipe.get("ner") or recipe.get("ingredients", [])
            tokens = []
            for item in ner_list:
                tokens.extend(tokenize(item))
            corpus_tokens.append(tokens)
            for tok in set(tokens):
                doc_freq[tok] += 1

        n_docs = len(recipes)
        idf: Dict[str, float] = {
            tok: math.log((n_docs + 1) / (df + 1)) + 1.0
            for tok, df in doc_freq.items()
        }

        # ── Stats nutrition pour normalisation ─────────────────────────
        nutrition_stats: Dict[str, float] = {}
        for feat in ["calories", "protein_g", "carbs_g", "total_fat_g"]:
            vals = [_safe(r.get(feat)) for r in recipes if r.get(feat)]
            nutrition_stats[feat] = max(vals) if vals else 1.0

        # ── Instanciation des encodeurs ─────────────────────────────────
        self.user_encoder   = UserEncoder(idf)
        self.recipe_encoder = RecipeEncoder(idf, nutrition_stats)

        # ── Pré-calcul des vecteurs recettes ────────────────────────────
        logger.info("Two-Tower : encodage des recettes…")
        self._recipe_vectors    = [self.recipe_encoder.encode(r) for r in recipes]
        self._recipe_token_sets = [set(t) for t in corpus_tokens]

        self._fitted = True
        logger.info(f"Two-Tower prêt — {len(recipes)} recettes encodées")
        return self

    def score_all(
        self,
        fridge_tokens: List[str],
        recipes: List[Dict[str, Any]],
        profile: Optional[Any] = None,
        meal_type: Optional[str] = None,
        feedback_history: Optional[Dict[str, int]] = None,
        disliked_titles: Optional[set] = None,
    ) -> List[tuple[float, int, str]]:
        """
        Retourne [(score, idx, calorie_fit)] triés par score décroissant.
        """
        if not self._fitted:
            raise RuntimeError("Appeler fit() avant score_all().")

        fridge_set  = set(fridge_tokens)
        user_vector = self.user_encoder.encode(
            fridge_tokens, profile, feedback_history
        )

        results = []
        for i, recipe in enumerate(recipes):
            # Filtre allergènes
            if profile and not passes_allergy_filter(
                recipe, profile.dietary_restrictions
            ):
                continue

            # Filtre meal_type si spécifié
            if meal_type and recipe.get("meal_type") != meal_type:
                continue

            # Pénalise les recettes dislikées
            if disliked_titles and recipe.get("title") in disliked_titles:
                continue

            cos  = cosine_sparse(user_vector, self._recipe_vectors[i])
            jac  = jaccard(fridge_set, self._recipe_token_sets[i])
            nutr_bonus, cal_fit = nutrition_score(recipe, profile, meal_type)

            score = W_COSINE * cos + W_JACCARD * jac + W_NUTRITION * (nutr_bonus + 0.5)
            score = round(min(max(score, 0.0), 1.0), 4)

            results.append((score, i, cal_fit))

        results.sort(key=lambda x: x[0], reverse=True)
        return results
