"""
models/recommender.py
──────────────────────
Orchestrateur principal du Two-Tower.

Évolutivité :
  - Charge les tokens des recettes likées depuis la DB
  - Construit le frigo virtuel (ingrédients implicites des likes)
  - Passe le profil de goût au scorer
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .two_tower import TwoTowerScorer, build_implicit_fridge
from .vectorizer import tokenize
from src.utils.schemas import RecipeResult, RecipeStep, NutritionInfo

logger = logging.getLogger(__name__)


class RecipeRecommender:
    def __init__(self, dataset_path: str = "data/recipes.json"):
        self.dataset_path = Path(dataset_path)
        self._recipes:  List[Dict[str, Any]] = []
        self._scorer    = TwoTowerScorer()
        self._fitted    = False

    # ──────────────────────────────────────────────────────────────────
    # Fit
    # ──────────────────────────────────────────────────────────────────

    def fit(self) -> "RecipeRecommender":
        logger.info(f"Chargement dataset : {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self._recipes = json.load(f)
        self._scorer.fit(self._recipes)
        self._fitted = True
        logger.info(f"Recommender prêt — {len(self._recipes)} recettes")
        return self

    # ──────────────────────────────────────────────────────────────────
    # Recommend
    # ──────────────────────────────────────────────────────────────────

    def recommend(
        self,
        ingredients:          List[str],
        top_n:                int                      = 15,
        min_score:            float                    = 0.05,
        meal_type:            Optional[str]            = None,
        profile=None,
        feedback_history:     Optional[Dict[str, int]] = None,
        disliked_titles:      Optional[Set[str]]       = None,
        liked_titles:         Optional[Set[str]]       = None,
        cooked_recently:      Optional[Set[str]]       = None,
        liked_recipes_tokens: Optional[List[Set[str]]] = None,
    ) -> List[RecipeResult]:
        """
        liked_recipes_tokens : tokens des recettes likées par l'utilisateur.
            Permet de construire :
            1. Le frigo virtuel (ingrédients implicites)
            2. Le profil de goût (similarité avec les likes)
        """
        if not self._fitted:
            raise RuntimeError("Appeler fit() avant recommend().")

        fridge_tokens = []
        for ing in ingredients:
            fridge_tokens.extend(tokenize(ing))
        fridge_set = set(fridge_tokens)

        # ── Frigo enrichi (réel + virtuel depuis les likes) ────────────
        implicit_fridge = build_implicit_fridge(liked_recipes_tokens or [])
        if implicit_fridge:
            logger.debug(
                f"Frigo virtuel : {len(implicit_fridge)} tokens depuis "
                f"{len(liked_recipes_tokens or [])} recettes likées"
            )

        scored = self._scorer.score_all(
            fridge_tokens        = fridge_tokens,
            recipes              = self._recipes,
            profile              = profile,
            meal_type            = meal_type,
            feedback_history     = feedback_history,
            disliked_titles      = disliked_titles,
            liked_titles         = liked_titles,
            cooked_recently      = cooked_recently,
            liked_recipes_tokens = liked_recipes_tokens,
        )

        scored = [(s, i, cf) for s, i, cf in scored if s >= min_score][:top_n]

        results = []
        for score, idx, cal_fit in scored:
            recipe = self._recipes[idx]

            all_ings  = recipe.get("ingredients", [])
            steps     = recipe.get("steps", [])

            # Matched/missing basés sur le frigo RÉEL (pas le virtuel)
            # pour ne pas induire l'user en erreur sur ce qu'il possède
            matched = [ing for ing in all_ings if any(t in fridge_set for t in tokenize(ing))]
            missing = [ing for ing in all_ings if ing not in matched]

            steps_out = [
                RecipeStep(step=j + 1, instruction=s)
                for j, s in enumerate(steps)
            ]

            nutrition = NutritionInfo(
                calories        = recipe.get("calories"),
                protein_g       = recipe.get("protein_g"),
                carbs_g         = recipe.get("carbs_g"),
                total_fat_g     = recipe.get("total_fat_g"),
                sugar_g         = recipe.get("sugar_g"),
                sodium_mg       = recipe.get("sodium_mg"),
                saturated_fat_g = recipe.get("saturated_fat_g"),
            )

            results.append(RecipeResult(
                title               = recipe.get("title", "Untitled"),
                score               = score,
                matched_ingredients = matched,
                missing_ingredients = missing,
                all_ingredients     = all_ings,
                steps               = steps_out,
                nutrition           = nutrition,
                meal_type           = recipe.get("meal_type"),
                minutes             = recipe.get("minutes"),
                calorie_fit         = cal_fit,
            ))

        return results

    @property
    def n_recipes(self) -> int:
        return len(self._recipes)

    @property
    def vocabulary(self) -> List[str]:
        if self._scorer._fitted:
            vocab = set()
            for tokens in self._scorer._recipe_token_sets:
                vocab.update(tokens)
            return sorted(vocab)
        return []