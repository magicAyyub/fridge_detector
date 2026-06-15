"""
utils/feedback_store.py
────────────────────────
Stockage in-memory du feedback utilisateur par session.

Structure par session :
  {
    "missing_counts":  {"ingredient": n},   # nb de fois manquant
    "disliked_titles": {"Recipe Title"},     # recettes dislikées
    "cooked_titles":   {"Recipe Title"},     # recettes cuisinées (feedback positif fort)
    "liked_titles":    {"Recipe Title"},
  }

Le feedback ajuste le scoring Two-Tower :
  - missing_counts  → downweight les ingrédients souvent absents du frigo
  - disliked_titles → exclut les recettes dislikées
  - cooked_titles   → boost les recettes similaires
"""

from collections import defaultdict
from typing import Dict, Set


class FeedbackStore:
    def __init__(self):
        # session_id → données feedback
        self._sessions: Dict[str, Dict] = defaultdict(lambda: {
            "missing_counts":  defaultdict(int),
            "disliked_titles": set(),
            "liked_titles":    set(),
            "cooked_titles":   set(),
        })

    def record(
        self,
        session_id: str,
        recipe_title: str,
        liked: bool | None,
        missing_ingredients: list[str],
        cooked: bool,
    ) -> None:
        session = self._sessions[session_id]

        # Ingrédients manquants — incrémente le compteur
        for ing in missing_ingredients:
            session["missing_counts"][ing.lower()] += 1

        if liked is True:
            session["liked_titles"].add(recipe_title)
            session["disliked_titles"].discard(recipe_title)
        elif liked is False:
            session["disliked_titles"].add(recipe_title)
            session["liked_titles"].discard(recipe_title)

        if cooked:
            session["cooked_titles"].add(recipe_title)

    def get_missing_counts(self, session_id: str) -> Dict[str, int]:
        return dict(self._sessions[session_id]["missing_counts"])

    def get_disliked_titles(self, session_id: str) -> Set[str]:
        return self._sessions[session_id]["disliked_titles"]

    def get_liked_titles(self, session_id: str) -> Set[str]:
        return self._sessions[session_id]["liked_titles"]

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    def clear_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]


# Singleton partagé
_store = FeedbackStore()

def get_feedback_store() -> FeedbackStore:
    return _store