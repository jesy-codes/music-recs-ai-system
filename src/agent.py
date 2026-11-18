"""The playlist agent: plan -> retrieve -> rank -> self-check -> (adjust and retry) -> present.

    guardrail ──► plan (Gemini, or rules) ──► retrieve ──► rank ──► self-check
                                                 ▲                      │ fails
                                                 └──── adjust (≤ 2) ◄───┘
                                                                        │ passes (or out of retries)
                             refine rounds only: Gemini reranker ──►  self-check ──► present

Gemini is called at most twice per session: once to plan round 1, and once in the refine round to choose
the next songs from 30 candidates using the listener's 👍/👎 (src/reranker.py). Reasons shown in round 1
come from templates built from catalog data.

Every step is appended to `trace` and logged, so a run can be inspected afterwards (CLI --trace and
logs/app.log).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

import numpy as np

from src.checker import CheckResult, check, filter_to_targets
from src.explainer import explain
from src.guardrails import apply_closest_genres, check_request, looks_off_topic
from src.llm import LLMClient
from src.planner import plan_request
from src.query import SearchQuery
from src.ranker import Recommendation, rank
from src.reranker import POOL_SIZE, rerank
from src.retriever import DEFAULT_CANDIDATES, Retriever

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


@dataclass
class Step:
    name: str
    detail: str


@dataclass
class Attempt:
    number: int
    songs: list[Recommendation]
    check: CheckResult
    settings: str
    pool: list[Recommendation] = field(default_factory=list)   # wider candidate list for the reranker


@dataclass
class AgentResult:
    request: str
    status: str                       # "ok", "partial" (constraints not fully met), "rejected", "empty"
    message: str
    query: Optional[SearchQuery] = None
    songs: list[Recommendation] = field(default_factory=list)
    reasons: dict[int, str] = field(default_factory=dict)
    plan_source: str = ""             # "gemini" or "rules"
    explain_source: str = ""          # "template", "gemini-rerank" or "gemini-rerank+agent" (or gemini explainer)
    attempts: list[Attempt] = field(default_factory=list)
    trace: list[Step] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["attempts"] = [{"number": a.number, "settings": a.settings, "check": a.check.summary(),
                             "song_ids": [s.song_id for s in a.songs]} for a in self.attempts]
        return data


class PlaylistAgent:
    def __init__(self, retriever: Retriever, llm: Optional[LLMClient] = None, max_retries: int = MAX_RETRIES):
        self.retriever = retriever
        self.llm = llm
        self.max_retries = max_retries
        self.vibe_tags = dict(zip(retriever.catalog["id"].astype(int), retriever.catalog["vibe_tags"]))

    def _step(self, trace: list[Step], name: str, detail: str) -> None:
        trace.append(Step(name, detail))
        logger.info("[agent] %s: %s", name, detail)

    def run(self, request: str, k: int = 5, exclude_ids: Optional[Iterable[int]] = None) -> AgentResult:
        trace: list[Step] = []
        exclude = list(exclude_ids or [])

        # 1. Input guardrail (no API call).
        guard = check_request(request)
        if not guard.ok:
            self._step(trace, "guardrail", f"rejected: {guard.message}")
            return AgentResult(request=request, status="rejected", message=guard.message, trace=trace)
        self._step(trace, "guardrail", "request accepted")

        # 2. Plan.
        plan = plan_request(request, self.retriever.known_genres, self.llm, k=k)
        detail = f"{plan.source}: {plan.query.summary()}"
        if plan.fallback_reason and plan.source == "rules":
            detail += f" (fallback: {plan.fallback_reason})"
        self._step(trace, "plan", detail)
        off_topic = not plan.on_topic or (plan.source == "rules" and looks_off_topic(request, plan.query))
        if off_topic:
            reason = plan.off_topic_reason or "it doesn't look like a music request"
            message = f"I can only help with music recommendations ({reason}). Try something like 'chill jazz for studying'."
            self._step(trace, "guardrail", f"rejected off-topic: {reason}")
            return AgentResult(request=request, status="rejected", message=message, query=plan.query,
                               plan_source=plan.source, trace=trace)
        query = apply_closest_genres(plan.query)
        for note in query.notes:
            self._step(trace, "note", note)
        return self.search_round(request, query, plan.source, exclude, trace)

    def search_round(self, request: str, query: SearchQuery, plan_source: str, exclude: list[int],
                     trace: list[Step], query_vector: Optional[np.ndarray] = None,
                     ok_message: str = "This is the recommended playlist, but let us know if it looks good:",
                     feedback: Optional[tuple[list[Recommendation], list[Recommendation]]] = None,
                     explain_with_llm: bool = False) -> AgentResult:
        """Steps 3-4 for an already-planned query: retrieve, rank, self-check (with retries), present.

        The feedback loop calls this directly for refine rounds, reusing the round-1 plan (no new planner
        call) with a feedback-adjusted query vector, and passes `feedback=(liked, rejected)` so Gemini
        can choose the final songs from a wider candidate pool (the reranker).
        `explain_with_llm` turns on the optional Gemini explainer; by default reasons come from templates.
        """
        # 3. Retrieve -> rank -> self-check, adjusting and retrying when the check fails.
        attempts: list[Attempt] = []
        strict, widen, n_candidates = False, 0.0, DEFAULT_CANDIDATES
        for number in range(1, self.max_retries + 2):
            settings = f"{n_candidates} candidates" + (f", hard energy/mood filter (+{widen:.1f})" if strict else "")
            candidates = self.retriever.search(query, n_candidates=n_candidates, exclude_ids=exclude,
                                               query_vector=query_vector)
            if strict:
                candidates = filter_to_targets(candidates, query, widen)
            self._step(trace, "retrieve", f"attempt {number}: searched {settings} -> {len(candidates)} kept")
            songs = rank(candidates, query, self.retriever.embeddings)
            self._step(trace, "rank", f"attempt {number}: " + ", ".join(f"{s.title} - {s.artist}" for s in songs))
            result = check(query, songs)
            self._step(trace, "check", f"attempt {number}: {result.summary()}")
            pool: list[Recommendation] = []
            if feedback is not None:
                # Only songs inside the energy/mood targets go to Gemini, so its picks can't fail on those.
                in_range = filter_to_targets(candidates, query, widen)
                pool = rank(in_range, query, self.retriever.embeddings, k=POOL_SIZE)
            attempts.append(Attempt(number, songs, result, settings, pool))
            if result.passed or number > self.max_retries:
                break

            codes, changes = result.codes(), []
            if codes & {"ENERGY_OUT", "VALENCE_OUT"} and not strict:
                strict = True
                changes.append("use the energy/mood targets as hard filters")
            elif "TOO_FEW" in codes and strict:
                widen += 0.1
                changes.append("widen the energy/mood ranges by 0.1")
            if codes & {"MISSING_GENRE", "TOO_FEW"} and n_candidates < len(self.retriever.catalog):
                n_candidates = min(n_candidates * 3, len(self.retriever.catalog))
                changes.append(f"search {n_candidates} candidates")
            if not changes:
                self._step(trace, "adjust", "no adjustment left to try")
                break
            self._step(trace, "adjust", "; ".join(changes))

        best = next((a for a in attempts if a.check.passed), None) or min(
            attempts, key=lambda a: (len(a.check.problems), -len(a.songs)))
        if best is not attempts[-1]:
            self._step(trace, "select", f"keeping attempt {best.number} (fewest problems)")

        if not best.songs:
            message = "No songs matched. Try a broader description or fewer exclusions."
            self._step(trace, "present", "no songs")
            return AgentResult(request=request, status="empty", message=message, query=query,
                               plan_source=plan_source, attempts=attempts, trace=trace)

        # 4. Refine rounds: let Gemini choose from a wider pool using the votes; the self-check still applies.
        songs, reasons, reasons_source = best.songs, None, ""
        if feedback is not None and self.llm is not None and len(best.pool) > query.k:
            liked, rejected = feedback
            picked = rerank(self.llm, query, liked, rejected, best.pool, query.k, self.vibe_tags)
            if picked.fallback_reason:
                self._step(trace, "rerank", f"Gemini unavailable ({picked.fallback_reason}); keeping the agent's picks")
            else:
                verdict = check(query, picked.songs)
                note = f" (dropped ids not in the candidates: {picked.dropped_ids})" if picked.dropped_ids else ""
                if verdict.passed or len(verdict.problems) <= len(best.check.problems):
                    self._step(trace, "rerank", f"Gemini chose {picked.from_gemini} of {len(best.pool)} candidates: "
                               + ", ".join(f"{s.title} - {s.artist}" for s in picked.songs) + note)
                    self._step(trace, "check", f"Gemini's picks: {verdict.summary()}")
                    best = Attempt(best.number, picked.songs, verdict, best.settings + ", Gemini picks", best.pool)
                    songs, reasons = picked.songs, picked.reasons
                    reasons_source = "gemini-rerank" if picked.from_gemini == query.k else "gemini-rerank+agent"
                else:
                    self._step(trace, "rerank", f"Gemini's picks failed the self-check ({verdict.summary()}); "
                               "keeping the agent's picks" + note)
        elif feedback is not None and self.llm is None:
            self._step(trace, "rerank", "offline mode; the agent's own picks are used")

        # 5. Reasons (grounded in catalog data) and present.
        if reasons is None:
            explained = explain(query, songs, self.vibe_tags, self.llm if explain_with_llm else None)
            reasons, reasons_source = explained.reasons, explained.source
            self._step(trace, "explain", f"{explained.source}" + (f" (dropped invented ids {explained.dropped_ids})" if explained.dropped_ids else ""))

        if best.check.passed:
            status, message = "ok", ok_message
        else:
            status = "partial"
            message = ("Here's the closest I could find, but not everything you asked for could be met: "
                       + best.check.summary() + ".")
        self._step(trace, "present", f"{status}: {len(songs)} songs")
        return AgentResult(
            request=request, status=status, message=message, query=query, songs=songs,
            reasons=reasons, plan_source=plan_source, explain_source=reasons_source,
            attempts=attempts, trace=trace,
        )
