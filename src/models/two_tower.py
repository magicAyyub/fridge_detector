"""
models/two_tower.py
────────────────────
Two-Tower scoring avec apprentissage implicite depuis les likes.

Nouveautés vs version précédente :
──────────────────────────────────
1. FRIGO VIRTUEL (implicit_fridge)
   Les ingrédients des recettes likées enrichissent le vecteur utilisateur
   avec un poids réduit (α = 0.3). Si l'user like 3 recettes avec "parmesan",
   "parmesan" reçoit un poids = 0.3 × fréquence dans les likes.
   → Le système propose des recettes avec ces ingrédients même si absents du frigo.

2. BOOST DE SIMILARITÉ (taste_profile_boost)
   Score cosinus entre la recette candidate et les recettes likées.
   Si une recette partage ≥40% des tokens NER avec une recette likée → boost.
   → Le système détecte "l'utilisateur aime ce type de cuisine"
     sans avoir besoin des ingrédients exacts.

3. PÉNALITÉ CUISINÉE RÉCEMMENT (cooked_recently)
   Pénalité douce -0.08 si la recette a été cuisinée dans les 7 derniers jours.
   → Variété sans bannir définitivement.
"""

import math
import logging
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Set

from .vectorizer import tokenize

logger = logging.getLogger(__name__)

MEAL_TYPES = ["breakfast", "lunch", "dinner", "snack"]

# Poids du score final
W_COSINE    = 0.45
W_NUTRITION = 0.25
W_JACCARD   = 0.20
W_TASTE     = 0.10   # boost similarité avec recettes likées (NEW)

CALORIE_TOLERANCE = 0.35

# Poids du frigo virtuel (ingrédients des recettes likées)
# 0.3 = présent dans les likes vaut 30% d'un ingrédient réellement dans le frigo
IMPLICIT_FRIDGE_WEIGHT = 0.30


# ── Helpers ────────────────────────────────────────────────────────────────

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


# ── Construction du frigo virtuel ─────────────────────────────────────────

def build_implicit_fridge(
    liked_recipes_tokens: List[Set[str]],
    weight: float = IMPLICIT_FRIDGE_WEIGHT,
) -> Dict[str, float]:
    """
    Construit un "frigo virtuel" à partir des tokens des recettes likées.

    Logique :
    - Un token présent dans 3 recettes likées a plus de poids qu'un token
      présent dans 1 seule recette likée.
    - Le poids max est plafonné à `weight` (30% d'un ingrédient réel).

    Retourne : {token: poids_implicite}
    """
    if not liked_recipes_tokens:
        return {}

    freq: Counter = Counter()
    for token_set in liked_recipes_tokens:
        for tok in token_set:
            freq[tok] += 1

    n_liked = len(liked_recipes_tokens)
    implicit: Dict[str, float] = {}
    for tok, count in freq.items():
        # Normalise par le nombre de recettes likées, plafonne à `weight`
        implicit[tok] = min(weight, weight * (count / n_liked))

    return implicit


# ── UserEncoder ────────────────────────────────────────────────────────────

class UserEncoder:
    def __init__(self, idf: Dict[str, float]):
        self.idf = idf

    def encode(
        self,
        fridge_tokens:        List[str],
        profile:              Optional[Any]           = None,
        feedback_history:     Optional[Dict[str, int]] = None,
        implicit_fridge:      Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Vecteur utilisateur = frigo réel (TF-IDF) + frigo virtuel (likes).

        implicit_fridge : {token: poids} construit depuis les recettes likées.
        Un token du frigo virtuel vaut IMPLICIT_FRIDGE_WEIGHT × son poids IDF,
        contre 1.0 × IDF pour un ingrédient réel du frigo.
        """
        # ── Frigo réel ─────────────────────────────────────────────────
        counts: Dict[str, int] = {}
        for t in fridge_tokens:
            counts[t] = counts.get(t, 0) + 1
        n = max(len(fridge_tokens), 1)

        vector: Dict[str, float] = {}
        for tok, cnt in counts.items():
            if tok in self.idf:
                miss_penalty = 1.0
                if feedback_history and tok in feedback_history:
                    miss_penalty = max(0.3, 1.0 - feedback_history[tok] * 0.1)
                vector[tok] = (cnt / n) * self.idf[tok] * miss_penalty

        # ── Frigo virtuel (ingrédients des recettes likées) ────────────
        if implicit_fridge:
            for tok, impl_weight in implicit_fridge.items():
                if tok in self.idf:
                    # N'écrase pas le frigo réel — additionne
                    existing = vector.get(tok, 0.0)
                    virtual  = impl_weight * self.idf[tok]
                    vector[tok] = existing + virtual

        return vector


# ── RecipeEncoder ──────────────────────────────────────────────────────────

class RecipeEncoder:
    def __init__(self, idf: Dict[str, float], nutrition_stats: Dict[str, float]):
        self.idf             = idf
        self.nutrition_stats = nutrition_stats

    def encode(self, recipe: Dict[str, Any]) -> Dict[str, float]:
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

        for feat in ["calories", "protein_g", "carbs_g", "total_fat_g"]:
            val     = _safe(recipe.get(feat))
            max_val = self.nutrition_stats.get(feat, 1.0)
            if max_val > 0:
                vector[f"nutr_{feat}"] = val / max_val

        meal = recipe.get("meal_type") or ""
        for mt in MEAL_TYPES:
            vector[f"meal_{mt}"] = 1.0 if meal == mt else 0.0

        return vector


# ── Nutrition score ────────────────────────────────────────────────────────

def nutrition_score(
    recipe:    Dict[str, Any],
    profile:   Any,
    meal_type: Optional[str] = None,
) -> tuple[float, str]:
    from src.utils.schemas import MEAL_CALORIE_RATIO, OBJECTIVE_TARGETS

    if profile is None:
        return 0.0, "unknown"

    recipe_kcal = _safe(recipe.get("calories"))
    if recipe_kcal <= 0:
        return 0.0, "unknown"

    mt          = meal_type or "lunch"
    meal_ratio  = MEAL_CALORIE_RATIO.get(mt, 0.33)
    target_kcal = profile.calorie_target * meal_ratio

    ratio = recipe_kcal / target_kcal if target_kcal > 0 else 1.0
    if abs(1.0 - ratio) <= CALORIE_TOLERANCE:
        cal_fit, cal_bonus = "perfect", 0.20
    elif ratio < 1.0 - CALORIE_TOLERANCE:
        cal_fit, cal_bonus = "low", 0.05
    else:
        cal_fit, cal_bonus = "high", -0.10

    obj     = profile.sports_objective
    targets = OBJECTIVE_TARGETS.get(obj, OBJECTIVE_TARGETS["maintenance"])
    macro_bonus = 0.0

    if _safe(recipe.get("protein_g")) >= targets["min_protein_g"]:
        macro_bonus += 0.05
    if _safe(recipe.get("carbs_g"))   <= targets["max_carbs_g"]:
        macro_bonus += 0.03
    if _safe(recipe.get("total_fat_g")) <= targets.get("max_fat_g", 999):
        macro_bonus += 0.02

    return round(min(max(cal_bonus + macro_bonus, -0.15), 0.30), 4), cal_fit


# ── Filtre allergènes ──────────────────────────────────────────────────────

def passes_allergy_filter(recipe: Dict[str, Any], restrictions: List[str]) -> bool:
    if not restrictions:
        return True
    full_text = " ".join(recipe.get("ner") or recipe.get("ingredients", [])).lower()
    full_text += " " + " ".join(recipe.get("tags", [])).lower()
    return not any(a.lower() in full_text for a in restrictions)


# ── Two-Tower principal ────────────────────────────────────────────────────

class TwoTowerScorer:
    def __init__(self):
        self.user_encoder:        Optional[UserEncoder]        = None
        self.recipe_encoder:      Optional[RecipeEncoder]      = None
        self._recipe_vectors:     List[Dict[str, float]]       = []
        self._recipe_token_sets:  List[set]                    = []
        self._fitted = False

    def fit(self, recipes: List[Dict[str, Any]]) -> "TwoTowerScorer":
        logger.info("Two-Tower : calcul IDF…")
        from collections import Counter

        doc_freq: Counter = Counter()
        corpus_tokens     = []

        for recipe in recipes:
            ner_list = recipe.get("ner") or recipe.get("ingredients", [])
            tokens   = []
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

        nutrition_stats: Dict[str, float] = {}
        for feat in ["calories", "protein_g", "carbs_g", "total_fat_g"]:
            vals = [_safe(r.get(feat)) for r in recipes if r.get(feat)]
            nutrition_stats[feat] = max(vals) if vals else 1.0

        self.user_encoder   = UserEncoder(idf)
        self.recipe_encoder = RecipeEncoder(idf, nutrition_stats)

        logger.info("Two-Tower : encodage des recettes…")
        self._recipe_vectors    = [self.recipe_encoder.encode(r) for r in recipes]
        self._recipe_token_sets = [set(t) for t in corpus_tokens]

        self._fitted = True
        logger.info(f"Two-Tower prêt — {len(recipes)} recettes encodées")
        return self

    def score_all(
        self,
        fridge_tokens:        List[str],
        recipes:              List[Dict[str, Any]],
        profile:              Optional[Any]            = None,
        meal_type:            Optional[str]            = None,
        feedback_history:     Optional[Dict[str, int]] = None,
        disliked_titles:      Optional[Set[str]]       = None,
        liked_titles:         Optional[Set[str]]       = None,
        cooked_recently:      Optional[Set[str]]       = None,
        liked_recipes_tokens: Optional[List[Set[str]]] = None,
    ) -> List[tuple[float, int, str]]:
        """
        liked_recipes_tokens : liste de sets de tokens des recettes likées.
            Passé depuis le recommender qui charge les tokens depuis la DB.
            Utilisé pour construire le frigo virtuel et le profil de goût.
        """
        if not self._fitted:
            raise RuntimeError("Appeler fit() avant score_all().")

        fridge_set = set(fridge_tokens)

        # ── Frigo virtuel depuis les recettes likées ───────────────────
        implicit_fridge = build_implicit_fridge(liked_recipes_tokens or [])

        # ── Profil de goût : vecteur moyen des recettes likées ─────────
        liked_vectors: List[Dict[str, float]] = []
        for i, recipe in enumerate(recipes):
            if liked_titles and recipe.get("title") in liked_titles:
                liked_vectors.append(self._recipe_vectors[i])

        avg_liked_vector: Dict[str, float] = {}
        if liked_vectors:
            all_keys = set(k for v in liked_vectors for k in v)
            for k in all_keys:
                avg_liked_vector[k] = sum(v.get(k, 0.0) for v in liked_vectors) / len(liked_vectors)

        # ── Vecteur utilisateur enrichi ────────────────────────────────
        user_vector = self.user_encoder.encode(
            fridge_tokens    = fridge_tokens,
            profile          = profile,
            feedback_history = feedback_history,
            implicit_fridge  = implicit_fridge,
        )

        results = []
        for i, recipe in enumerate(recipes):
            if profile and not passes_allergy_filter(recipe, profile.dietary_restrictions):
                continue
            if meal_type and recipe.get("meal_type") != meal_type:
                continue
            if disliked_titles and recipe.get("title") in disliked_titles:
                continue

            cos         = cosine_sparse(user_vector, self._recipe_vectors[i])
            jac         = jaccard(fridge_set, self._recipe_token_sets[i])
            nutr_bonus, cal_fit = nutrition_score(recipe, profile, meal_type)

            # ── Boost profil de goût (similarité avec les recettes likées) ──
            taste_boost = 0.0
            if avg_liked_vector:
                taste_sim   = cosine_sparse(avg_liked_vector, self._recipe_vectors[i])
                taste_boost = taste_sim * 0.5   # plafonné à 0.5 pour ne pas dominer

            # ── Pénalité cuisinée récemment ────────────────────────────
            cooked_penalty = -0.08 if (cooked_recently and recipe.get("title") in cooked_recently) else 0.0

            score = (
                W_COSINE    * cos
              + W_JACCARD   * jac
              + W_NUTRITION * (nutr_bonus + 0.5)
              + W_TASTE     * taste_boost
              + cooked_penalty
            )
            score = round(min(max(score, 0.0), 1.0), 4)
            results.append((score, i, cal_fit))

        results.sort(key=lambda x: x[0], reverse=True)
        return results