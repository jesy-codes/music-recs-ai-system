"""Rank retrieved candidates and pick a varied final list.

Score per song = 0.6 x retrieval (fused semantic + keyword) + 0.3 x audio-feature fit + 0.1 x genre match.
Then MMR (maximal marginal relevance) picks the final k one at a time, trading score against similarity
to songs already picked, so the list isn't five near-identical tracks. Each song keeps its score parts
and plain-language reasons, which the explainer and the self-check use later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.query import SearchQuery

logger = logging.getLogger(__name__)

WEIGHTS = {"retrieval": 0.6, "features": 0.3, "genre": 0.1}
MMR_LAMBDA = 0.75          # 1.0 = pure score, 0.0 = pure variety
MMR_POOL = 60              # only the top-scored candidates are considered for the final list
PER_GENRE_POOL = 15        # ...plus the best few from each requested genre
GENRE_COVERAGE_BONUS = 0.1  # nudges MMR to cover every requested genre ("jazz AND indie")
MAX_PER_ARTIST = 1


@dataclass
class Recommendation:
    song_id: int
    track_id: str
    title: str
    artist: str
    genres: str
    score: float
    parts: dict[str, float]
    reasons: list[str] = field(default_factory=list)
    row: int = -1                  # index into the embeddings matrix
    features: dict[str, float] = field(default_factory=dict)

    @property
    def spotify_url(self) -> str:
        return f"https://open.spotify.com/track/{self.track_id}"


def feature_fit(song: pd.Series, query: SearchQuery) -> float:
    """How well a song's audio features match the plan, 0-1. Neutral (1.0) when the plan sets none."""
    fits = []
    if query.energy:
        fits.append(query.energy.closeness(song["energy"]))
    if query.valence:
        fits.append(query.valence.closeness(song["valence"]))
    if query.acoustic:
        fits.append(float(song["acousticness"]))
    if query.instrumental:
        fits.append(float(song["instrumentalness"]))
    return float(np.mean(fits)) if fits else 1.0


def genre_match(song_genres: set[str], query: SearchQuery) -> float:
    if not query.genres:
        return 0.0
    return 1.0 if song_genres & set(query.genres) else 0.0


def reasons_for(song: pd.Series, query: SearchQuery) -> list[str]:
    """Short, factual reasons built only from catalog data (the template fallback for the explainer)."""
    reasons = []
    matched = [g for g in query.genres if g in song["genres"].split("; ")]
    if matched:
        reasons.append("genre: " + ", ".join(matched))
    tags = [t for t in song["vibe_tags"].split(", ") if t]
    wanted = [t for t in tags if t in query.moods or (query.energy and t in ("calm", "mellow", "chill", "energetic", "intense"))]
    if wanted:
        reasons.append("feels " + ", ".join(dict.fromkeys(wanted)))
    if query.energy:
        status = "in" if query.energy.contains(song["energy"]) else "near"
        reasons.append(f"energy {song['energy']:.2f} ({status} target {query.energy})")
    if query.valence:
        status = "in" if query.valence.contains(song["valence"]) else "near"
        reasons.append(f"mood/valence {song['valence']:.2f} ({status} target {query.valence})")
    if query.acoustic and song["acousticness"] > 0.6:
        reasons.append(f"acoustic ({song['acousticness']:.2f})")
    if query.instrumental and song["instrumentalness"] > 0.5:
        reasons.append(f"instrumental ({song['instrumentalness']:.2f})")
    if not reasons:
        reasons.append("closest match to your description")
    return reasons


def score_candidates(candidates: pd.DataFrame, query: SearchQuery) -> pd.DataFrame:
    """Add feature, genre and final score columns, sorted best first."""
    if candidates.empty:
        return candidates.assign(features=[], genre=[], score=[])
    scored = candidates.copy()
    scored["features"] = [feature_fit(song, query) for _, song in scored.iterrows()]
    scored["genre"] = [genre_match(set(g.split("; ")), query) for g in scored["genres"]]
    scored["score"] = (
        WEIGHTS["retrieval"] * scored["fused"]
        + WEIGHTS["features"] * scored["features"]
        + WEIGHTS["genre"] * scored["genre"]
    )
    return scored.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)


def select_mmr(scored: pd.DataFrame, embeddings: np.ndarray, query: SearchQuery, k: int,
               lam: float = MMR_LAMBDA, max_per_artist: int = MAX_PER_ARTIST) -> list[int]:
    """Pick k positions from `scored` with MMR, an artist cap and a bonus for uncovered requested genres."""
    # Pool: the top-scored songs plus the best few from each requested genre, so a genre that scores a
    # little lower overall (e.g. indie next to jazz) still has songs available to balance the list.
    keep = set(range(min(MMR_POOL, len(scored))))
    for genre in query.genres:
        in_genre = [i for i, g in enumerate(scored["genres"]) if genre in g.split("; ")]
        keep.update(in_genre[:PER_GENRE_POOL])
    pool = scored.iloc[sorted(keep)].reset_index(drop=True)
    if pool.empty:
        return []
    vectors = embeddings[pool["row"].to_numpy()]
    scores = pool["score"].to_numpy()
    genres = [set(g.split("; ")) for g in pool["genres"]]
    artists = pool["artist"].str.lower().tolist()

    chosen: list[int] = []
    artist_count: dict[str, int] = {}
    wanted = set(query.genres)
    genre_count = {g: 0 for g in wanted}
    quota = -(-k // len(wanted)) if wanted else 0   # ceil(k / number of requested genres)
    while len(chosen) < k:
        best, best_value = None, -np.inf
        for i in range(len(pool)):
            if i in chosen or artist_count.get(artists[i], 0) >= max_per_artist:
                continue
            similarity = float(np.max(vectors[chosen] @ vectors[i])) if chosen else 0.0
            # Balance requested genres: favor songs from a genre below its fair share, and hold back
            # songs that don't help while some requested genre is still short.
            bonus = 0.0
            if wanted and any(count < quota for count in genre_count.values()):
                helps = any(genre_count[g] < quota for g in genres[i] & wanted)
                bonus = GENRE_COVERAGE_BONUS if helps else -GENRE_COVERAGE_BONUS
            value = lam * scores[i] - (1 - lam) * similarity + bonus
            if value > best_value:
                best, best_value = i, value
        if best is None:
            break
        chosen.append(best)
        artist_count[artists[best]] = artist_count.get(artists[best], 0) + 1
        for g in genres[best] & wanted:
            genre_count[g] += 1
    return chosen


def rank(candidates: pd.DataFrame, query: SearchQuery, embeddings: np.ndarray, k: int | None = None) -> list[Recommendation]:
    """Score candidates and return the final k recommendations with their score parts and reasons."""
    k = k or query.k
    scored = score_candidates(candidates, query)
    picks = select_mmr(scored, embeddings, query, k)
    if len(picks) < k:
        logger.warning("Only %d of %d songs could be picked for %r", len(picks), k, query.text)
    results = []
    for position in picks:
        song = scored.iloc[position]
        results.append(Recommendation(
            song_id=int(song["id"]), track_id=str(song["track_id"]), title=str(song["title"]),
            artist=str(song["artist"]), genres=str(song["genres"]), score=round(float(song["score"]), 3),
            parts={"retrieval": round(float(song["fused"]), 3), "features": round(float(song["features"]), 3),
                   "genre": float(song["genre"]), "semantic": round(float(song["semantic"]), 3)},
            reasons=reasons_for(song, query), row=int(song["row"]),
            features={f: float(song[f]) for f in ("energy", "valence", "acousticness", "instrumentalness", "danceability")},
        ))
    logger.info("Ranked: %s", "; ".join(f"{r.title} - {r.artist} ({r.score})" for r in results))
    return results
