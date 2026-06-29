"""
models/translator.py
─────────────────────
Traducteur EN → FR basé sur MarianMT (Helsinki-NLP/opus-mt-en-fr).

- Modèle Transformer léger (~300MB), tourne en local, pas d'API externe.
- Utilisé au preprocessing pour traduire ingrédients + étapes du dataset.
- Batching automatique pour éviter les OOM sur les gros corpus.

Installation :
    pip install transformers sentencepiece torch
"""

import logging
import re
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"
CACHE_DIR = Path("models/cache/marian")


class Translator:
    """
    Wrapper MarianMT EN → FR.
    Chargement lazy : le modèle n'est instancié qu'au premier appel.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = 32,
        max_length: int = 256,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # Chargement lazy
    # ------------------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError:
            raise ImportError(
                "Installe les dépendances :\n"
                "  pip install transformers sentencepiece torch"
            )

        logger.info(f"Chargement MarianMT ({self.model_name})…")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self._tokenizer = MarianTokenizer.from_pretrained(
            self.model_name, cache_dir=str(CACHE_DIR)
        )
        self._model = MarianMTModel.from_pretrained(
            self.model_name, cache_dir=str(CACHE_DIR)
        )
        self._model.eval()
        logger.info("Modèle MarianMT prêt.")

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def translate(self, texts: Union[str, list]) -> Union[str, list]:
        """
        Traduit une string ou une liste de strings EN → FR.
        Retourne le même type que l'entrée.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        # Séparer les vides pour ne pas les envoyer au modèle
        non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        results = [""] * len(texts)

        if not non_empty:
            return results[0] if single else results

        self._load()

        import torch

        indices, batch_texts = zip(*non_empty)
        translated = []

        for start in range(0, len(batch_texts), self.batch_size):
            chunk = list(batch_texts[start : start + self.batch_size])
            tokens = self._tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            with torch.no_grad():
                output_ids = self._model.generate(
                    **tokens,
                    max_length=self.max_length,
                    num_beams=4,
                    early_stopping=True,
                )
            decoded = self._tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            translated.extend(decoded)

        for idx, translation in zip(indices, translated):
            results[idx] = translation

        return results[0] if single else results

    def translate_batch(self, texts: list[str]) -> list[str]:
        """
        Alias explicite pour traduire une liste de strings EN → FR.
        Utilisé par le recommender pour les ingrédients et les matched/missing.
        """
        if not texts:
            return []
        result = self.translate(texts)
        # translate() retourne déjà une liste quand l'entrée est une liste
        return result if isinstance(result, list) else [result]

    def translate_steps(self, steps: list[str]) -> list[str]:
        """
        Traduit une liste d'étapes de recette EN → FR.
        Les étapes peuvent être longues : on les envoie une par une
        si elles dépassent max_length tokens (géré automatiquement par truncation).
        """
        if not steps:
            return []
        str_steps = [str(s).strip() for s in steps]
        result = self.translate(str_steps)
        return result if isinstance(result, list) else [result]

    def translate_ingredient(self, ingredient: str) -> str:
        """
        Traduit un ingrédient en supprimant d'abord les quantités.
        Ex: '2 cups all-purpose flour' → 'farine tout usage'
        """
        cleaned = re.sub(
            r"^\s*[\d½¼¾⅓⅔/\-]+\s*"
            r"(cups?|tbsp?|tsp?|oz|lb|g|kg|ml|cl|l|pkg?|cans?|jar|"
            r"bunch|clove|slice|piece|pound|quart|pint|dash|pinch)?\s*",
            "",
            ingredient,
            flags=re.IGNORECASE,
        ).strip()

        return self.translate(cleaned or ingredient)


# ------------------------------------------------------------------
# Singleton partagé
# ------------------------------------------------------------------

_translator: Translator | None = None


def get_translator() -> Translator:
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator