"""
utils/model_updater.py
───────────────────────
Niveau 2 — MAJ automatique du Two-Tower depuis le feedback.

Deux modes :
  A) Re-fit complet  : recharge recipes.json + re-index TF-IDF/BM25
                       déclenché quand recipes.json est modifié
  B) Ajout incrémental : ajoute de nouvelles recettes sans re-fit complet

Niveau 3 — Collecte annotations pour FRCNN :
  Stocke les corrections de détection (faux positifs/négatifs)
  pour un futur réentraînement.

Usage dans main.py :
  updater = ModelUpdater(recommender, pool)
  # Lance le watcher en arrière-plan
  asyncio.create_task(updater.watch_recipes_file())
  asyncio.create_task(updater.periodic_refit(interval_hours=24))
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.models.recommender import RecipeRecommender

logger = logging.getLogger(__name__)


def _file_hash(path: Path) -> str:
    """Hash MD5 du fichier pour détecter les modifications."""
    return hashlib.md5(path.read_bytes()).hexdigest()


class ModelUpdater:
    def __init__(
        self,
        recommender: "RecipeRecommender",
        pool=None,
        recipes_path: str = "data/recipes.json",
    ):
        self.recommender   = recommender
        self.pool          = pool
        self.recipes_path  = Path(recipes_path)
        self._last_hash    = None
        self._refit_count  = 0

    # ------------------------------------------------------------------
    # Niveau 2A — Watcher fichier recipes.json
    # ------------------------------------------------------------------

    async def watch_recipes_file(self, check_interval_seconds: int = 60) -> None:
        """
        Surveille recipes.json et re-fit le Two-Tower si le fichier change.
        Tourne en arrière-plan via asyncio.create_task().
        """
        logger.info(f"📡 Watcher démarré sur {self.recipes_path} (vérif toutes les {check_interval_seconds}s)")
        self._last_hash = _file_hash(self.recipes_path) if self.recipes_path.exists() else None

        while True:
            await asyncio.sleep(check_interval_seconds)
            try:
                if not self.recipes_path.exists():
                    continue
                current_hash = _file_hash(self.recipes_path)
                if current_hash != self._last_hash:
                    logger.info("🔄 recipes.json modifié — re-fit du Two-Tower en cours...")
                    await self._refit()
                    self._last_hash = current_hash
            except Exception as e:
                logger.error(f"Watcher erreur : {e}")

    # ------------------------------------------------------------------
    # Niveau 2B — Re-fit périodique basé sur le feedback
    # ------------------------------------------------------------------

    async def periodic_refit(self, interval_hours: int = 24) -> None:
        """
        Re-fit automatique toutes les N heures.
        Utile pour intégrer le feedback accumulé dans le scoring.
        """
        logger.info(f"⏰ Re-fit périodique programmé toutes les {interval_hours}h")
        while True:
            await asyncio.sleep(interval_hours * 3600)
            logger.info(f"⏰ Re-fit périodique #{self._refit_count + 1}...")
            await self._refit()

    # ------------------------------------------------------------------
    # Niveau 2C — Ajout de nouvelles recettes à chaud
    # ------------------------------------------------------------------

    async def add_recipes(self, new_recipes: list[dict]) -> int:
        """
        Ajoute de nouvelles recettes sans re-fit complet.
        Retourne le nombre de recettes ajoutées.

        new_recipes : liste de dicts au format du dataset
          [{title, ingredients, steps, ner, calories, protein_g, ...}]
        """
        if not new_recipes:
            return 0

        # 1. Ajoute en mémoire
        self.recommender._recipes.extend(new_recipes)

        # 2. Re-encode uniquement les nouvelles recettes
        for recipe in new_recipes:
            vec   = self.recommender._scorer.recipe_encoder.encode(recipe)
            toks  = []
            for item in (recipe.get("ner") or recipe.get("ingredients", [])):
                from src.models.vectorizer import tokenize
                toks.extend(tokenize(item))
            self.recommender._scorer._recipe_vectors.append(vec)
            self.recommender._scorer._recipe_token_sets.append(set(toks))

        # 3. Sauvegarde dans recipes.json
        with open(self.recipes_path, "w", encoding="utf-8") as f:
            json.dump(self.recommender._recipes, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ {len(new_recipes)} recettes ajoutées à chaud.")
        return len(new_recipes)

   
    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    async def _refit(self) -> None:
        """Re-fit complet du Two-Tower dans un thread pour ne pas bloquer."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_refit)
        self._refit_count += 1
        logger.info(f"✅ Two-Tower re-fit #{self._refit_count} terminé — {self.recommender.n_recipes} recettes")

    def _sync_refit(self) -> None:
        """Re-fit synchrone (appelé dans un executor thread)."""
        self.recommender.fit()