"""A listening session: round 1 from the request, then refine rounds driven by 👍/👎 votes.

    start("chill jazz and indie")   -> 5 songs                      (Gemini: plan = 1 call)
    refine({253: True, 92: False})  -> playlist so far + 5 new songs (Gemini: picks 5 of 30 candidates = 1 call)
    playlist()                      -> only the 👍 songs

Refine rounds reuse the round-1 plan; the votes adjust it in code (src/feedback.py), so they don't need
another planner call. MAX_REFINE_ROUNDS caps the rounds to keep Gemini usage within free-tier limits.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from src.agent import AgentResult, PlaylistAgent, Step
from src.feedback import apply_feedback, rocchio
from src.query import SearchQuery
from src.ranker import Recommendation

logger = logging.getLogger(__name__)

MAX_REFINE_ROUNDS = int(os.getenv("MAX_REFINE_ROUNDS", "1"))
REFINE_MESSAGE = "Here's your playlist so far, plus {n} new songs based on your feedback:"


class SessionError(ValueError):
    """The action isn't allowed right now (no round started, unknown song id, round limit reached...)."""


class PlaylistSession:
    def __init__(self, agent: PlaylistAgent, k: int = 5, max_refine_rounds: Optional[int] = None,
                 use_feedback: bool = True):
        self.agent = agent
        self.k = k
        # use_feedback=False keeps the round-1 search and only skips songs already shown: the baseline the
        # simulated-user evaluation compares against.
        self.use_feedback = use_feedback
        self.max_refine_rounds = MAX_REFINE_ROUNDS if max_refine_rounds is None else max_refine_rounds
        self.request = ""
        self.rounds: list[AgentResult] = []
        self.shown: dict[int, Recommendation] = {}
        self.liked: dict[int, Recommendation] = {}      # insertion order = playlist order
        self.rejected: dict[int, Recommendation] = {}
        self.carried: list[list[Recommendation]] = []   # per round: the 👍 songs carried into it ("playlist so far")
        self.reasons_by_id: dict[int, str] = {}
        self.base_query: Optional[SearchQuery] = None
        self.base_vector: Optional[np.ndarray] = None
        self.plan_source = ""

    # --- round 1 ---

    def start(self, request: str) -> AgentResult:
        result = self.agent.run(request, k=self.k)
        self.request = request
        self.rounds = [result]
        self.carried = [[]]
        self.shown, self.liked, self.rejected = {}, {}, {}
        self.reasons_by_id = {}
        if result.status in ("ok", "partial") and result.query is not None:
            self.base_query = result.query
            self.base_vector = self.agent.retriever.embed_query(result.query.semantic_text())
            self.plan_source = result.plan_source
            self._remember(result)
        return result

    # --- votes and refine rounds ---

    @property
    def refine_rounds_used(self) -> int:
        return max(0, len(self.rounds) - 1)

    @property
    def can_refine(self) -> bool:
        return self.base_query is not None and self.refine_rounds_used < self.max_refine_rounds

    def vote(self, votes: dict[int, bool]) -> None:
        """Record 👍 (True) / 👎 (False) for shown songs. A later vote replaces an earlier one."""
        unknown = [song_id for song_id in votes if song_id not in self.shown]
        if unknown:
            raise SessionError(f"Song id(s) {unknown} weren't shown in this session.")
        for song_id, liked in votes.items():
            song = self.shown[song_id]
            self.liked.pop(song_id, None)
            self.rejected.pop(song_id, None)
            (self.liked if liked else self.rejected)[song_id] = song

    def refine(self, votes: dict[int, bool]) -> AgentResult:
        if self.base_query is None:
            raise SessionError("Start a session with a request first.")
        if not self.can_refine:
            raise SessionError(f"The refine limit ({self.max_refine_rounds} round(s)) is reached. Finish to see your playlist.")
        self.vote(votes)
        if not self.liked and not self.rejected:
            raise SessionError("Vote on at least one song first.")

        trace: list[Step] = []
        if self.use_feedback:
            rows = self.agent.retriever.row_by_id
            embeddings = self.agent.retriever.embeddings
            vector = rocchio(self.base_vector,
                             [embeddings[rows[i]] for i in self.liked],
                             [embeddings[rows[i]] for i in self.rejected])
            query, changes = apply_feedback(self.base_query, list(self.liked.values()), list(self.rejected.values()))
            detail = f"{len(self.liked)} liked, {len(self.rejected)} rejected; search moved toward liked songs"
            if changes:
                detail += "; " + "; ".join(changes)
        else:
            vector, query = self.base_vector, self.base_query
            detail = f"{len(self.liked)} liked, {len(self.rejected)} rejected; feedback ignored (baseline)"
        self.agent._step(trace, "feedback", detail)

        result = self.agent.search_round(
            self.request, query, self.plan_source, exclude=list(self.shown), trace=trace,
            query_vector=vector, ok_message=REFINE_MESSAGE.format(n=query.k),
            feedback=(list(self.liked.values()), list(self.rejected.values())) if self.use_feedback else None,
        )
        self.rounds.append(result)
        self.carried.append(list(self.liked.values()))
        self._remember(result)
        return result

    # --- output ---

    def playlist(self) -> list[Recommendation]:
        """The final playlist: only songs with a 👍, in the order they were liked."""
        return list(self.liked.values())

    def rounds_liked_before(self, index: int) -> list[Recommendation]:
        """The 👍 songs that were carried into round `index` (shown as "your playlist so far")."""
        return self.carried[index] if 0 <= index < len(self.carried) else []

    def _remember(self, result: AgentResult) -> None:
        for song in result.songs:
            self.shown[song.song_id] = song
            self.reasons_by_id[song.song_id] = result.reasons.get(song.song_id, "; ".join(song.reasons))
