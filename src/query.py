"""The structured search plan the agent works from.

A SearchQuery is what the planner produces from the user's request (the rule-based parser now, Gemini in
Phase 3) and what the retriever and ranker consume. Keeping it a plain, validated data object means every
step can be logged, tested and explained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

MIN_K, MAX_K = 1, 20


@dataclass(frozen=True)
class Range:
    """An inclusive 0-1 range for an audio feature, e.g. energy 0.0-0.45 for "chill"."""

    low: float
    high: float

    def __post_init__(self) -> None:
        # Clamp into 0-1 so a bad plan (e.g. energy 1.5) can't push scores negative.
        low, high = sorted((min(max(self.low, 0.0), 1.0), min(max(self.high, 0.0), 1.0)))
        object.__setattr__(self, "low", round(low, 3))
        object.__setattr__(self, "high", round(high, 3))

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def closeness(self, value: float) -> float:
        """1.0 inside the range, falling linearly to 0.0 at 0.5 outside it."""
        if self.contains(value):
            return 1.0
        distance = self.low - value if value < self.low else value - self.high
        return max(0.0, 1.0 - 2.0 * distance)

    def __str__(self) -> str:
        return f"{self.low:.2f}-{self.high:.2f}"


@dataclass
class SearchQuery:
    """Everything the retriever and ranker need, derived from one user request."""

    text: str                                   # the original request
    search_text: str = ""                       # optional rewrite for semantic search (set by the Gemini planner)
    genres: list[str] = field(default_factory=list)          # catalog genres to favor
    exclude_genres: list[str] = field(default_factory=list)  # catalog genres to drop ("no metal")
    unknown_genres: list[str] = field(default_factory=list)  # genres asked for but not in the catalog
    moods: list[str] = field(default_factory=list)           # vibe words found ("chill", "sad", ...)
    energy: Optional[Range] = None
    valence: Optional[Range] = None
    acoustic: Optional[bool] = None             # True: prefer acoustic songs
    instrumental: Optional[bool] = None         # True: prefer songs without vocals
    k: int = 5
    notes: list[str] = field(default_factory=list)           # parser remarks shown to the user / logs

    def __post_init__(self) -> None:
        self.text = self.text.strip()
        self.k = min(max(int(self.k), MIN_K), MAX_K)

    def semantic_text(self) -> str:
        """Text for embedding search: the planner's rewrite if there is one, else the request."""
        return self.search_text.strip() or self.text

    def keyword_text(self) -> str:
        """Words for keyword (BM25) search: the request plus the normalized genres and moods."""
        return " ".join([self.semantic_text(), *self.genres, *self.moods])

    def summary(self) -> str:
        parts = []
        if self.genres:
            parts.append("genres " + ", ".join(self.genres))
        if self.exclude_genres:
            parts.append("excluding " + ", ".join(self.exclude_genres))
        if self.moods:
            parts.append("moods " + ", ".join(self.moods))
        if self.energy:
            parts.append(f"energy {self.energy}")
        if self.valence:
            parts.append(f"valence {self.valence}")
        if self.acoustic:
            parts.append("acoustic")
        if self.instrumental:
            parts.append("instrumental")
        return "; ".join(parts) or "no filters (semantic search only)"

    def to_dict(self) -> dict:
        return asdict(self)
