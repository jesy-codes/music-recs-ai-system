"""Self-check: does a list of picks actually satisfy the plan?

The agent runs this after every ranking. Each failed rule becomes a Problem with a code the agent uses to
decide how to adjust the next attempt, and a message the user and logs can read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.query import Range, SearchQuery
from src.ranker import Recommendation

TOLERANCE = 0.05   # a song 0.05 outside an energy/valence range still counts as a fit


@dataclass
class Problem:
    code: str          # TOO_FEW, MISSING_GENRE, ENERGY_OUT, VALENCE_OUT, EXCLUDED_GENRE, DUPLICATE_ARTIST
    message: str
    song_ids: list[int] = field(default_factory=list)


@dataclass
class CheckResult:
    passed: bool
    problems: list[Problem] = field(default_factory=list)

    def codes(self) -> set[str]:
        return {p.code for p in self.problems}

    def summary(self) -> str:
        return "passed" if self.passed else "; ".join(p.message for p in self.problems)


def _fits(value: float, target: Range) -> bool:
    return target.low - TOLERANCE <= value <= target.high + TOLERANCE


def check(query: SearchQuery, songs: list[Recommendation]) -> CheckResult:
    problems: list[Problem] = []
    if len(songs) < query.k:
        problems.append(Problem("TOO_FEW", f"only {len(songs)} of {query.k} songs found"))

    song_genres = [set(s.genres.split("; ")) for s in songs]
    for genre in query.genres:
        if songs and not any(genre in g for g in song_genres):
            problems.append(Problem("MISSING_GENRE", f"no {genre} song in the list"))

    banned = set(query.exclude_genres)
    bad = [s.song_id for s, g in zip(songs, song_genres) if g & banned]
    if bad:
        problems.append(Problem("EXCLUDED_GENRE", f"{len(bad)} song(s) from an excluded genre", bad))

    for feature, target, code in (("energy", query.energy, "ENERGY_OUT"), ("valence", query.valence, "VALENCE_OUT")):
        if target is None:
            continue
        outside = [s for s in songs if not _fits(s.features[feature], target)]
        if outside:
            details = ", ".join(f"{s.title} ({s.features[feature]:.2f})" for s in outside)
            problems.append(Problem(code, f"{feature} outside {target}: {details}", [s.song_id for s in outside]))

    artists = [s.artist.lower() for s in songs]
    if len(artists) != len(set(artists)):
        problems.append(Problem("DUPLICATE_ARTIST", "the same artist appears more than once"))

    return CheckResult(passed=not problems, problems=problems)


def filter_to_targets(candidates: pd.DataFrame, query: SearchQuery, widen: float = 0.0) -> pd.DataFrame:
    """Keep only candidates inside the plan's energy/valence ranges (plus tolerance and `widen`).

    Used on retries when the self-check found songs outside the targets: the ranges become hard filters.
    """
    keep = pd.Series(True, index=candidates.index)
    for feature, target in (("energy", query.energy), ("valence", query.valence)):
        if target is not None:
            margin = TOLERANCE + widen
            keep &= candidates[feature].between(target.low - margin, target.high + margin)
    return candidates[keep]
