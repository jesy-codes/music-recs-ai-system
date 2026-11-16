"""One recommendation round: request -> SearchQuery -> candidates -> ranked songs.

Phase 3 wraps this in the agent loop (Gemini planner, self-check and retries); this module stays the
deterministic core that the agent, CLI, app and evaluation all share.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

from src.parser import parse_request
from src.query import SearchQuery
from src.ranker import Recommendation, rank
from src.retriever import DEFAULT_CANDIDATES, Retriever

logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    query: SearchQuery
    songs: list[Recommendation]
    candidates_considered: int


def recommend(request: str, retriever: Retriever, k: int = 5, exclude_ids: Optional[Iterable[int]] = None,
              query: Optional[SearchQuery] = None, n_candidates: int = DEFAULT_CANDIDATES) -> RoundResult:
    """Run one round. Pass `query` to skip parsing (e.g. when a planner already built it)."""
    query = query or parse_request(request, k=k)
    candidates = retriever.search(query, n_candidates=n_candidates, exclude_ids=exclude_ids)
    songs = rank(candidates, query, retriever.embeddings)
    return RoundResult(query=query, songs=songs, candidates_considered=len(candidates))
