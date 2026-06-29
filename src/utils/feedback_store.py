"""
utils/feedback_store.py
────────────────────────
Stockage feedback avec persistance NeonDB.

Niveau 1 — Feedback → scoring :
  - missing_counts  → downweight ingrédients souvent absents
  - disliked_titles → exclut recettes dislikées
  - liked_titles    → boost recettes similaires

Niveau 2 — Collecte pour réentraînement :
  - Toutes les interactions sont loguées en DB
  - Un job périodique peut relire ces données pour re-fit le Two-Tower
"""

from collections import defaultdict
from typing import Dict, Optional, Set
import logging

logger = logging.getLogger(__name__)


class FeedbackStore:
    def __init__(self):
        self._sessions: Dict[str, Dict] = defaultdict(lambda: {
            "missing_counts":  defaultdict(int),
            "disliked_titles": set(),
            "liked_titles":    set(),
            "cooked_titles":   set(),
        })
        # Pool DB optionnel — injecté depuis main.py
        self._pool = None

    def set_pool(self, pool) -> None:
        """Injecte le pool NeonDB pour la persistance."""
        self._pool = pool

    # ------------------------------------------------------------------
    # Écriture
    # ------------------------------------------------------------------

    def record(
        self,
        session_id: str,
        recipe_title: str,
        liked: Optional[bool],
        missing_ingredients: list[str],
        cooked: bool,
    ) -> None:
        """Enregistre en mémoire + async vers DB si pool disponible."""
        session = self._sessions[session_id]

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

    async def record_async(
        self,
        session_id: str,
        recipe_title: str,
        liked: Optional[bool],
        missing_ingredients: list[str],
        cooked: bool,
        user_id: Optional[int] = None,
    ) -> None:
        """
        Version async : enregistre en mémoire ET persiste en DB.
        Utilisé dans les routes FastAPI async.
        """
        # 1. Mémoire (immédiat)
        self.record(session_id, recipe_title, liked, missing_ingredients, cooked)

        # 2. DB (persistance)
        if self._pool:
            try:
                async with self._pool.acquire() as conn:
                    async with conn.transaction():
                        # Récupère ou crée la recette en DB
                        recipe_row = await conn.fetchrow(
                            "SELECT id FROM recipes WHERE title = $1", recipe_title
                        )
                        recipe_id = recipe_row["id"] if recipe_row else None

                        if recipe_id and user_id:
                            # Upsert feedback
                            await conn.execute(
                                """
                                INSERT INTO recipe_feedback
                                    (user_id, recipe_id, liked, cooked)
                                VALUES ($1, $2, $3, $4)
                                ON CONFLICT (user_id, recipe_id)
                                DO UPDATE SET
                                    liked      = EXCLUDED.liked,
                                    cooked     = EXCLUDED.cooked,
                                    updated_at = NOW()
                                """,
                                user_id, recipe_id, liked, cooked,
                            )

                        # Log ingrédients manquants pour le réentraînement
                        if missing_ingredients and user_id and recipe_id:
                            feedback_row = await conn.fetchrow(
                                "SELECT id FROM recipe_feedback WHERE user_id=$1 AND recipe_id=$2",
                                user_id, recipe_id,
                            )
                            if feedback_row:
                                await conn.executemany(
                                    """
                                    INSERT INTO feedback_missing_ingredients
                                        (feedback_id, ingredient_name, ner_token)
                                    VALUES ($1, $2, $3)
                                    """,
                                    [
                                        (feedback_row["id"], ing, ing.lower())
                                        for ing in missing_ingredients
                                    ],
                                )

            except Exception as e:
                logger.warning(f"Feedback DB persist échoué (non critique) : {e}")

    # ------------------------------------------------------------------
    # Lecture (pour le Two-Tower)
    # ------------------------------------------------------------------

    def get_missing_counts(self, session_id: str) -> Dict[str, int]:
        return dict(self._sessions[session_id]["missing_counts"])

    def get_disliked_titles(self, session_id: str) -> Set[str]:
        return self._sessions[session_id]["disliked_titles"]

    def get_liked_titles(self, session_id: str) -> Set[str]:
        return self._sessions[session_id]["liked_titles"]

    async def load_from_db(self, session_id: str, user_id: int) -> None:
        """
        Recharge le feedback depuis la DB au démarrage d'une session.
        Permet de restaurer l'historique après redémarrage du serveur.
        """
        if not self._pool:
            return
        try:
            async with self._pool.acquire() as conn:
                # Ingrédients manquants
                rows = await conn.fetch(
                    """
                    SELECT fmi.ner_token, COUNT(*) as cnt
                    FROM feedback_missing_ingredients fmi
                    JOIN recipe_feedback rf ON rf.id = fmi.feedback_id
                    WHERE rf.user_id = $1 AND fmi.ner_token IS NOT NULL
                    GROUP BY fmi.ner_token
                    """,
                    user_id,
                )
                for row in rows:
                    self._sessions[session_id]["missing_counts"][row["ner_token"]] = row["cnt"]

                # Recettes dislikées
                disliked = await conn.fetch(
                    """
                    SELECT r.title FROM recipe_feedback rf
                    JOIN recipes r ON r.id = rf.recipe_id
                    WHERE rf.user_id = $1 AND rf.liked = FALSE
                    """,
                    user_id,
                )
                for row in disliked:
                    self._sessions[session_id]["disliked_titles"].add(row["title"])

                # Recettes aimées
                liked = await conn.fetch(
                    """
                    SELECT r.title FROM recipe_feedback rf
                    JOIN recipes r ON r.id = rf.recipe_id
                    WHERE rf.user_id = $1 AND rf.liked = TRUE
                    """,
                    user_id,
                )
                for row in liked:
                    self._sessions[session_id]["liked_titles"].add(row["title"])

                logger.info(f"Feedback chargé depuis DB pour user {user_id} → session {session_id}")
        except Exception as e:
            logger.warning(f"Impossible de charger le feedback depuis DB : {e}")

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    def clear_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]


_store = FeedbackStore()

def get_feedback_store() -> FeedbackStore:
    return _store