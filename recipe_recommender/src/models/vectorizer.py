"""
Vectoriseur TF-IDF sur les ingrédients — from scratch, sans sklearn.
"""

import math
import re
from collections import Counter
from typing import List, Dict


# ---------------------------------------------------------------------------
# Normalisation des ingrédients
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "de", "du", "des", "la", "le", "les", "un", "une", "et", "en",
    "à", "au", "avec", "pour", "par", "sur", "dans", "d", "l",
    "g", "kg", "ml", "cl", "l", "cs", "cc",  # unités
}

_ACCENT_MAP = str.maketrans(
    "àâäéèêëîïôöùûüçæœ",
    "aaaeeeeiioouuucao",
)


def normalize(text: str) -> str:
    """Normalise un ingrédient : minuscules, sans accents, sans stopwords."""
    text = text.lower().translate(_ACCENT_MAP)
    tokens = re.findall(r"[a-z]+", text)
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def tokenize(text: str) -> List[str]:
    return normalize(text).split()


# ---------------------------------------------------------------------------
# TF-IDF from scratch
# ---------------------------------------------------------------------------

class TFIDFVectorizer:
    """
    TF-IDF binaire sur les ingrédients d'une liste de documents (recettes).
    Chaque document = liste de tokens d'ingrédients.
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}          # token → index
        self.idf: Dict[str, float] = {}          # token → score IDF
        self._n_docs: int = 0

    def fit(self, corpus: List[List[str]]) -> "TFIDFVectorizer":
        """
        corpus : liste de documents, chaque document étant une liste de tokens.
        """
        self._n_docs = len(corpus)
        doc_freq: Counter = Counter()

        for doc in corpus:
            for token in set(doc):  # unique par doc pour DF
                doc_freq[token] += 1

        # Vocabulaire trié pour reproductibilité
        self.vocab = {tok: i for i, tok in enumerate(sorted(doc_freq))}

        # IDF lissé : log((N + 1) / (df + 1)) + 1
        for tok, df in doc_freq.items():
            self.idf[tok] = math.log((self._n_docs + 1) / (df + 1)) + 1.0

        return self

    def transform(self, tokens: List[str]) -> Dict[str, float]:
        """
        Retourne un vecteur TF-IDF sparse (dict token → score).
        TF binaire (présent / absent) × IDF.
        """
        counts = Counter(tokens)
        n = max(len(tokens), 1)
        vector: Dict[str, float] = {}
        for tok, cnt in counts.items():
            if tok in self.idf:
                tf = cnt / n
                vector[tok] = tf * self.idf[tok]
        return vector

    @property
    def vocabulary(self) -> List[str]:
        return list(self.vocab.keys())
