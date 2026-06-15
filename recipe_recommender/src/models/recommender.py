"""
models/recommender.py
──────────────────────
Orchestrateur principal — utilise TwoTowerScorer pour le scoring.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.two_tower import TwoTowerScorer
from src.models.vectorizer import tokenize
from src.utils.schemas import RecipeResult, RecipeStep, NutritionInfo

logger = logging.getLogger(__name__)


class RecipeRecommender:
    def __init__(
        self,
        dataset_path: str = "data/recipes.json",
        translate: bool = False,
    ):
        self.dataset_path = Path(dataset_path)
        self.translate    = translate

        self._recipes: List[Dict[str, Any]] = []
        self._scorer  = TwoTowerScorer()
        self._fitted  = False
        self._translator = None

    def _get_translator(self):
        if self._translator is None:
            from src.models.translator import Translator
            self._translator = Translator()
        return self._translator

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self) -> "RecipeRecommender":
        logger.info(f"Chargement du dataset : {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self._recipes = json.load(f)

        self._scorer.fit(self._recipes)
        self._fitted = True
        logger.info(f"Recommender prêt — {len(self._recipes)} recettes")
        return self

    # ------------------------------------------------------------------
    # Recommend
    # ------------------------------------------------------------------

    def recommend(
        self,
        ingredients: List[str],
        top_n: int = 5,
        min_score: float = 0.05,
        meal_type: Optional[str] = None,
        profile=None,                      # UserProfileInput | None
        feedback_history: Optional[Dict[str, int]] = None,
        disliked_titles: Optional[set] = None,
    ) -> List[RecipeResult]:
        if not self._fitted:
            raise RuntimeError("Appeler fit() avant recommend().")

        fridge_tokens = []
        for ing in ingredients:
            fridge_tokens.extend(tokenize(ing))
        fridge_set = set(fridge_tokens)

        scored = self._scorer.score_all(
            fridge_tokens   = fridge_tokens,
            recipes         = self._recipes,
            profile         = profile,
            meal_type       = meal_type,
            feedback_history= feedback_history,
            disliked_titles = disliked_titles,
        )

        # Filtre min_score + top_n
        scored = [(s, i, cf) for s, i, cf in scored if s >= min_score][:top_n]

        results = []
        for score, idx, cal_fit in scored:
            recipe = self._recipes[idx]

            all_ingredients_en = recipe.get("ingredients", [])
            steps_en           = recipe.get("steps", [])
            title_en           = recipe.get("title", "Untitled")

            matched_en = [
                ing for ing in all_ingredients_en
                if any(t in fridge_set for t in tokenize(ing))
            ]
            missing_en = [ing for ing in all_ingredients_en if ing not in matched_en]

            if self.translate:
                tr = self._get_translator()
                title_fr           = tr.translate(title_en)
                all_ingredients_fr = tr.translate_batch(all_ingredients_en)
                matched_fr         = tr.translate_batch(matched_en)
                missing_fr         = tr.translate_batch(missing_en)
                steps_fr           = tr.translate_steps(steps_en)
            else:
                title_fr           = title_en
                all_ingredients_fr = all_ingredients_en
                matched_fr         = matched_en
                missing_fr         = missing_en
                steps_fr           = steps_en

            steps_out = [
                RecipeStep(step=j + 1, instruction=s)
                for j, s in enumerate(steps_fr)
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
                title               = title_fr,
                score               = score,
                matched_ingredients = matched_fr,
                missing_ingredients = missing_fr,
                all_ingredients     = all_ingredients_fr,
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