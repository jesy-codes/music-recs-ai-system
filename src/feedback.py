"""Turn 👍/👎 votes into an adjusted search: relevance feedback, not fine-tuning or reinforcement learning.

Nothing about the models changes. Only this session's search moves:

1. Query vector (Rocchio, a classic information-retrieval method):
       new = 1.0 x original + 0.7 x mean(liked songs) - 0.3 x mean(rejected songs), then normalized
   so the embedding search drifts toward what the listener liked and away from what they didn't.
2. Audio targets: the energy and valence ranges shift halfway toward the liked songs' average. With no
   range in the plan and 2+ liked songs, a range is created around their average (±0.2).
3. Genres: a requested genre is dropped if the listener rejected songs from it and liked none.

The state is discarded when the session ends, so nothing about the listener is stored.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from src.query import Range, SearchQuery
from src.ranker import Recommendation

logger = logging.getLogger(__name__)

ALPHA, BETA, GAMMA = 1.0, 0.7, 0.3   # weights for original query, liked, rejected
RANGE_SHIFT = 0.5                    # how far a range moves toward the liked average (0 = not at all, 1 = all the way)
NEW_RANGE_HALF_WIDTH = 0.2
MIN_LIKED_FOR_NEW_RANGE = 2
MIN_LIKED_FOR_NEW_GENRE = 2


def rocchio(original: np.ndarray, liked: Sequence[np.ndarray], rejected: Sequence[np.ndarray]) -> np.ndarray:
    """Move the query vector toward liked vectors and away from rejected ones; returns a unit vector."""
    vector = ALPHA * original.astype(np.float32)
    if len(liked):
        vector = vector + BETA * np.mean(liked, axis=0)
    if len(rejected):
        vector = vector - GAMMA * np.mean(rejected, axis=0)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else original


def shift_range(target: Optional[Range], liked_values: Sequence[float]) -> Optional[Range]:
    """Move a 0-1 target range toward the liked songs' average value."""
    if not liked_values:
        return target
    mean = float(np.mean(liked_values))
    if target is None:
        if len(liked_values) < MIN_LIKED_FOR_NEW_RANGE:
            return None
        return Range(mean - NEW_RANGE_HALF_WIDTH, mean + NEW_RANGE_HALF_WIDTH)
    center, half = (target.low + target.high) / 2, (target.high - target.low) / 2
    new_center = (1 - RANGE_SHIFT) * center + RANGE_SHIFT * mean
    return Range(new_center - half, new_center + half)


def adapt_genres(genres: Sequence[str], liked: Sequence[Recommendation],
                 rejected: Sequence[Recommendation]) -> tuple[list[str], list[str], list[str]]:
    """Return (genres to favor, dropped, learned).

    - Dropped: a requested genre whose shown songs were all rejected.
    - Learned: a genre shared by at least MIN_LIKED_FOR_NEW_GENRE liked songs and no rejected song.
    """
    liked_counts: dict[str, int] = {}
    for s in liked:
        for g in s.genres.split("; "):
            liked_counts[g] = liked_counts.get(g, 0) + 1
    rejected_genres = {g for s in rejected for g in s.genres.split("; ")}
    kept = [g for g in genres if g in liked_counts or g not in rejected_genres]
    dropped = [g for g in genres if g not in kept]
    learned = [g for g, n in sorted(liked_counts.items(), key=lambda kv: -kv[1])
               if n >= MIN_LIKED_FOR_NEW_GENRE and g not in rejected_genres and g not in kept]
    return kept + learned, dropped, learned


def apply_feedback(query: SearchQuery, liked: Sequence[Recommendation], rejected: Sequence[Recommendation]) -> tuple[SearchQuery, list[str]]:
    """Return the adjusted query for the next round and plain-language notes on what changed."""
    changes: list[str] = []
    energy = shift_range(query.energy, [s.features["energy"] for s in liked])
    if energy != query.energy:
        changes.append(f"energy {query.energy or 'any'} -> {energy}")
    valence = shift_range(query.valence, [s.features["valence"] for s in liked])
    if valence != query.valence:
        changes.append(f"mood/valence {query.valence or 'any'} -> {valence}")
    genres, dropped, learned = adapt_genres(query.genres, liked, rejected)
    if dropped:
        changes.append(f"stopped favoring {', '.join(dropped)} (all its songs were rejected)")
    if learned:
        changes.append(f"now favoring {', '.join(learned)} (shared by liked songs, none rejected)")
    adjusted = replace(query, energy=energy, valence=valence, genres=genres, notes=[])
    for change in changes:
        logger.info("Feedback adjustment: %s", change)
    return adjusted, changes
