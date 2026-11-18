"""Golden-prompt evaluation: score the agent on a fixed set of requests, for both planners.

For every prompt in eval/golden_prompts.json:
  - precision@5      share of returned songs that meet the prompt's song rules (genre, energy, mood...),
                     averaged over all runs
  - planner accuracy does the run-1 plan match the expected plan (genres, exclusions, energy/mood...)
  - consistency      the same request run N times: mean Jaccard overlap of the song ids
  - made-up songs    shown songs whose id isn't in the catalog, plus ids Gemini invented that were dropped
  - self-check       how often the first attempt failed the check, and whether a retry fixed it
Then one refine round per prompt (a simulated listener likes exactly the songs that meet the rules):
  - refine precision for Gemini's picks vs the agent's own picks from the same candidates
  - reasons with details not in the data (keyword check on Gemini's refine-round reasons)
And the six simulated users from src/simulator.py, one refine round each, Gemini's picks vs the agent's.

    python -m eval.run_eval                       # both planners (Gemini needs GEMINI_API_KEY)
    python -m eval.run_eval --planners rules      # offline only, no API calls, deterministic
    python -m eval.run_eval --runs 3 --delay 4    # 3 runs per prompt, at least 4 s between Gemini calls
    python -m eval.run_eval --rescore eval/results.json --runs 2   # re-score saved runs, no API calls

Results are printed as Markdown tables and saved to eval/results.json (--out to change).
Gemini calls: about (runs + 1) per prompt plus 2 per simulated user, i.e. ~76 with the defaults
(3 runs). --rescore recomputes precision and consistency from the song ids a previous run saved, using
only its first --runs runs, so changing the scoring or the number of runs costs no quota.
If Gemini fails or runs out of quota, the agent falls back to the rules; those runs are counted as
fallbacks and left out of the Gemini column, so the numbers never mix the two planners silently.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Type, TypeVar

from pydantic import BaseModel

from src.agent import AgentResult, PlaylistAgent
from src.catalog import CatalogError, load_catalog
from src.checker import TOLERANCE
from src.explainer import template_reason
from src.llm import LLMClient, LLMError, LLMQuotaError, get_client
from src.parser import UNKNOWN_GENRES
from src.query import SearchQuery
from src.ranker import Recommendation
from src.retriever import Retriever
from src.session import PlaylistSession
from src.simulator import PERSONAS, Persona

logger = logging.getLogger("eval")

GOLDEN_PATH = Path(__file__).with_name("golden_prompts.json")
RESULTS_PATH = Path(__file__).with_name("results.json")
PLAN_FIELDS = {"genres_all", "genres_any", "exclude_genres", "unknown_genres", "energy", "valence",
               "acoustic", "instrumental"}
SONG_FIELDS = {"genres", "exclude_genres", "energy", "valence", "acoustic", "instrumental"}
FEATURE_FLOOR = 0.5   # acousticness / instrumentalness needed for "acoustic" / "instrumental"

T = TypeVar("T", bound=BaseModel)


class GoldenSetError(ValueError):
    """golden_prompts.json is missing or malformed."""


# --- golden set ---------------------------------------------------------------------------------------

def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise GoldenSetError(f"Golden prompts not found at {path}.") from error
    except json.JSONDecodeError as error:
        raise GoldenSetError(f"{path} is not valid JSON: {error}") from error
    prompts = data.get("prompts") if isinstance(data, dict) else None
    if not prompts:
        raise GoldenSetError(f"{path} has no 'prompts' list.")
    seen = set()
    for item in prompts:
        pid = item.get("id")
        if not pid or not item.get("request"):
            raise GoldenSetError(f"Every golden prompt needs an 'id' and a 'request': {item}")
        if pid in seen:
            raise GoldenSetError(f"Duplicate golden prompt id {pid!r}.")
        seen.add(pid)
        bad = set(item.get("plan", {})) - PLAN_FIELDS or set(item.get("songs", {})) - SONG_FIELDS
        if bad:
            raise GoldenSetError(f"Prompt {pid!r} has unknown rule(s): {sorted(bad)}")
    return prompts


# --- scoring ------------------------------------------------------------------------------------------

def song_genres(song: Recommendation) -> set[str]:
    return set(song.genres.split("; "))


def song_matches(song: Recommendation, rules: dict) -> bool:
    """True if the song meets every rule in the prompt's `songs` block."""
    genres = song_genres(song)
    if rules.get("genres") and not genres & set(rules["genres"]):
        return False
    if genres & set(rules.get("exclude_genres", [])):
        return False
    for feature in ("energy", "valence"):
        if feature in rules:
            low, high = rules[feature]
            if not low - TOLERANCE <= song.features[feature] <= high + TOLERANCE:
                return False
    if rules.get("acoustic") and song.features["acousticness"] < FEATURE_FLOOR:
        return False
    if rules.get("instrumental") and song.features["instrumentalness"] < FEATURE_FLOOR:
        return False
    return True


def precision(songs: Sequence[Recommendation], rules: dict) -> Optional[float]:
    if not songs:
        return 0.0
    return sum(song_matches(s, rules) for s in songs) / len(songs)


def direction(target) -> str:
    if target is None:
        return "none"
    return "low" if (target.low + target.high) / 2 < 0.5 else "high"


def plan_mismatches(result: AgentResult, expected: dict) -> list[str]:
    """What the plan got wrong (empty list = correct). Only the fields in the golden prompt are checked."""
    should_be_on_topic = expected.get("on_topic", True)
    rejected = result.status == "rejected"
    if not should_be_on_topic:
        return [] if rejected else ["should have been rejected as off-topic"]
    if rejected or result.query is None:
        return [f"rejected an on-topic request ({result.message})"]

    query: SearchQuery = result.query
    rules = expected.get("plan", {})
    wrong = []
    missing = [g for g in rules.get("genres_all", []) if g not in query.genres]
    if missing:
        wrong.append(f"missing genres {missing}")
    if rules.get("genres_any") and not set(rules["genres_any"]) & set(query.genres):
        wrong.append(f"none of the genres {rules['genres_any']}")
    not_excluded = [g for g in rules.get("exclude_genres", []) if g not in query.exclude_genres]
    if not_excluded:
        wrong.append(f"didn't exclude {not_excluded}")
    not_flagged = [g for g in rules.get("unknown_genres", []) if g not in query.unknown_genres]
    if not_flagged:
        wrong.append(f"didn't flag {not_flagged} as not in the catalog")
    for feature in ("energy", "valence"):
        if feature in rules and direction(getattr(query, feature)) != rules[feature]:
            wrong.append(f"{feature} {direction(getattr(query, feature))}, expected {rules[feature]}")
    for flag in ("acoustic", "instrumental"):
        if rules.get(flag) and not getattr(query, flag):
            wrong.append(f"didn't ask for {flag}")
    return wrong


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def mean(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


# --- grounding check for Gemini's reasons --------------------------------------------------------------

# Details the catalog never contains. A word here is flagged unless it also appears in the facts Gemini
# was given (e.g. "piano" is fine for a song in the piano genre, "acoustic" for a song tagged acoustic).
UNGROUNDED_WORDS = [
    r"vocals?", r"voices?", r"vocalists?", r"sings?", r"singing", r"lyric(?:s|al)?", r"words",
    r"guitars?", r"piano", r"keys", r"drums?", r"percussion", r"bass(?:line)?s?", r"synths?", r"synthesizers?",
    r"horns?", r"sax(?:ophone)?", r"trumpets?", r"strings", r"violins?", r"orchestra(?:l)?", r"choir",
    r"harmon(?:y|ies)", r"falsetto", r"riffs?", r"solos?", r"production", r"produced", r"released",
    r"album", r"debut", r"award\w*", r"grammy", r"hits?", r"chart\w*", r"classic", r"iconic", r"legendary",
    r"famous", r"anthem\w*", r"cover", r"remix\w*", r"live", r"recorded", r"decade",
    r"(?:19|20)?\d0s", r"(?:19|20)\d\d",r"nostalgi\w*", r"storytelling", r"heartfelt",
]
UNGROUNDED_PATTERN = re.compile(r"\b(" + "|".join(UNGROUNDED_WORDS) + r")\b", re.I)
# Catalog genres that are really mood words; they're used as adjectives too ("a chill track"), so they're
# not checked as genre claims.
MOOD_GENRES = {"acoustic", "chill", "happy", "sad", "sleep", "study"}


def ungrounded_details(reason: str, song: Recommendation, context: str, catalog_genres: set[str]) -> list[str]:
    """Words in a reason that state something the facts given to Gemini don't support.

    `context` is everything Gemini saw that the reason may refer to: the request and the facts of this song
    and of the liked/rejected songs. Flags (1) catalog-free details like vocals, instruments, years and
    awards, (2) genre names that neither this song nor the context has, and (3) invented compound genres
    such as "indie-jazz". It's a keyword check: it can miss a paraphrase and over-flag a word used
    loosely, so every flag is listed in the output for a human to read.
    """
    text = reason.lower()
    allowed = context.lower()
    flags = [m.group(0) for m in UNGROUNDED_PATTERN.finditer(text) if m.group(0).lower() not in allowed]

    genre_words = (catalog_genres | set(UNKNOWN_GENRES)) - MOOD_GENRES
    this_song = song_genres(song)
    for word in re.findall(r"[a-z&]+(?:-[a-z&]+)*", text):
        parts = word.split("-")
        if word not in catalog_genres and len(parts) > 1 and all(p in genre_words for p in parts):
            flags.append(f"{word} (compound genre not in the catalog)")
        elif word in genre_words and word not in this_song and word not in allowed:
            flags.append(f"{word} (genre the song doesn't have)")
    return list(dict.fromkeys(flags))


def facts_text(songs: Sequence[Recommendation], vibe_tags: dict[int, str]) -> str:
    return " ".join(f"{s.title} {s.artist} {s.genres} {vibe_tags.get(s.song_id, '')}" for s in songs)


# --- Gemini wrapper: pacing and quota waits -------------------------------------------------------------

class PacedLLM:
    """Wraps the Gemini client: keeps `delay` seconds between calls and waits out a 429 instead of falling back.

    The free tier allows a limited number of requests per minute, so a full eval run would otherwise hit
    429s and silently turn into a rules run.
    """

    def __init__(self, inner: LLMClient, delay: float, quota_waits: int = 2, quota_wait_seconds: float = 65.0):
        self.inner = inner
        self.model = inner.model
        self.delay = delay
        self.quota_waits = quota_waits
        self.quota_wait_seconds = quota_wait_seconds
        self.calls = 0
        self.failures = 0
        self._last = 0.0

    def generate_json(self, system: str, prompt: str, schema: Type[T], temperature: float = 0.0) -> T:
        for attempt in range(self.quota_waits + 1):
            wait = self._last + self.delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.calls += 1
            try:
                return self.inner.generate_json(system, prompt, schema, temperature)
            except LLMQuotaError:
                if attempt == self.quota_waits:
                    self.failures += 1
                    raise
                logger.warning("Gemini quota hit; waiting %.0fs before retrying (%d/%d).",
                               self.quota_wait_seconds, attempt + 1, self.quota_waits)
                time.sleep(self.quota_wait_seconds)
            except LLMError:
                self.failures += 1
                raise
        raise AssertionError("unreachable")


# --- running the eval --------------------------------------------------------------------------------

@dataclass
class PromptResult:
    id: str
    request: str
    on_topic: bool
    plan_sources: list[str] = field(default_factory=list)       # one per run
    plan_summary: str = ""
    plan_summaries: list[str] = field(default_factory=list)      # one per run, to see how plans vary
    plan_problems: list[str] = field(default_factory=list)
    status: str = ""
    song_ids: list[list[int]] = field(default_factory=list)      # one list per run
    songs: list[str] = field(default_factory=list)               # run 1, "title - artist (genres)"
    precision: Optional[float] = None                            # mean over all runs
    precisions: list[float] = field(default_factory=list)        # one per run
    consistency: Optional[float] = None
    made_up_ids: list[int] = field(default_factory=list)
    first_check_failed: bool = False
    fixed_by_retry: bool = False
    refine_source: str = ""                                      # "gemini-rerank", "template" (agent's picks)...
    refine_precision: Optional[float] = None                     # the list the agent showed in round 2
    refine_agent_precision: Optional[float] = None               # the agent's own picks for the same round
    rerank_invented_ids: list[int] = field(default_factory=list)
    gemini_reasons: list[dict] = field(default_factory=list)     # {"song", "reason", "flags"}


def made_up(songs: Sequence[Recommendation], catalog_ids: set[int]) -> list[int]:
    return [s.song_id for s in songs if s.song_id not in catalog_ids]


def invented_ids(result: AgentResult) -> list[int]:
    ids = []
    for step in result.trace:
        if step.name == "rerank" and "dropped ids not in the candidates:" in step.detail:
            ids += [int(x) for x in re.findall(r"\d+", step.detail.split("candidates:")[-1])]
    return ids


def offline_copy(session: PlaylistSession, offline_agent: PlaylistAgent) -> PlaylistSession:
    """The same session state (round-1 plan, songs shown) but with an agent that never calls Gemini."""
    return copy.deepcopy(session, memo={id(session.agent): offline_agent})


def gemini_reasons(result: AgentResult, session: PlaylistSession, vibe_tags: dict[int, str],
                   catalog_genres: set[str]) -> list[dict]:
    """Reasons written by Gemini in a refine round (template reasons are skipped), each with its flags."""
    if not result.explain_source.startswith("gemini"):
        return []
    context = " ".join([result.request,
                        facts_text(list(session.liked.values()) + list(session.rejected.values()), vibe_tags)])
    out = []
    for song in result.songs:
        reason = result.reasons.get(song.song_id, "")
        if not reason or reason == template_reason(song):
            continue
        flags = ungrounded_details(reason, song, context + " " + facts_text([song], vibe_tags), catalog_genres)
        out.append({"song": f"{song.title} - {song.artist} ({song.genres})", "reason": reason, "flags": flags})
    return out


def evaluate_prompt(item: dict, agent: PlaylistAgent, offline_agent: PlaylistAgent, runs: int,
                    catalog_ids: set[int], catalog_genres: set[str]) -> PromptResult:
    on_topic = item.get("on_topic", True)
    out = PromptResult(id=item["id"], request=item["request"], on_topic=on_topic)
    rules = item.get("songs", {})

    # Run 1 goes through a session so the refine round can reuse its plan (no extra planner call).
    session = PlaylistSession(agent, max_refine_rounds=1)
    first = session.start(item["request"])
    results = [first] + [agent.run(item["request"]) for _ in range(runs - 1)]

    out.plan_sources = [r.plan_source or "guardrail" for r in results]
    out.status = first.status
    out.plan_summary = first.query.summary() if first.query else ""
    out.plan_summaries = [r.query.summary() if r.query else "" for r in results]
    out.plan_problems = plan_mismatches(first, item)
    out.song_ids = [[s.song_id for s in r.songs] for r in results]
    out.songs = [f"{s.title} - {s.artist} ({s.genres})" for s in first.songs]
    out.made_up_ids = sorted({i for r in results for i in made_up(r.songs, catalog_ids)})
    if not on_topic:
        return out

    out.precisions = [precision(r.songs, rules) for r in results]
    out.precision = mean(out.precisions)
    if runs > 1:
        out.consistency = mean([jaccard(a, b) for a, b in itertools.combinations(out.song_ids, 2)])
    if first.attempts:
        out.first_check_failed = not first.attempts[0].check.passed
        out.fixed_by_retry = out.first_check_failed and first.status == "ok"

    # Refine round: a listener who likes exactly the songs that meet the rules.
    if first.status in ("ok", "partial") and first.songs:
        votes = {s.song_id: song_matches(s, rules) for s in first.songs}
        agent_only = offline_copy(session, offline_agent)
        refined = session.refine(votes)
        agent_refined = agent_only.refine(votes)
        out.refine_source = refined.explain_source
        out.refine_precision = precision(refined.songs, rules)
        out.refine_agent_precision = precision(agent_refined.songs, rules)
        out.made_up_ids = sorted(set(out.made_up_ids) | set(made_up(refined.songs, catalog_ids)))
        out.rerank_invented_ids = invented_ids(refined)
        out.gemini_reasons = gemini_reasons(refined, session, agent.vibe_tags, catalog_genres)
    return out


@dataclass
class PersonaResult:
    name: str
    hidden_taste: str
    plan_source: str
    round1: float
    refine_source: str
    shown: float          # like rate of what the agent showed in the refine round
    agent_only: float     # like rate of the agent's own picks for the same round


def like_rate(songs: Sequence[Recommendation], persona: Persona) -> float:
    return sum(persona.likes(s) for s in songs) / len(songs) if songs else 0.0


def evaluate_persona(persona: Persona, agent: PlaylistAgent, offline_agent: PlaylistAgent) -> PersonaResult:
    session = PlaylistSession(agent, max_refine_rounds=1)
    first = session.start(persona.request)
    votes = {s.song_id: persona.likes(s) for s in first.songs}
    agent_only = offline_copy(session, offline_agent)
    refined = session.refine(votes)
    own = agent_only.refine(votes)
    return PersonaResult(persona.name, persona.hidden_taste, first.plan_source, like_rate(first.songs, persona),
                         refined.explain_source, like_rate(refined.songs, persona), like_rate(own.songs, persona))


def summarize(prompts: list[PromptResult], planner: str) -> dict:
    """Metrics for one planner. For Gemini, runs that fell back to the rules are left out and counted."""
    def used_planner(p: PromptResult) -> bool:
        return planner == "rules" or all(s in ("gemini", "guardrail") for s in p.plan_sources)

    valid = [p for p in prompts if used_planner(p)]
    on_topic = [p for p in valid if p.on_topic]
    reasons = [r for p in valid for r in p.gemini_reasons]
    refined = [p for p in on_topic if p.refine_precision is not None]
    reranked = [p for p in refined if p.refine_source.startswith("gemini")]
    retried = [p for p in on_topic if p.first_check_failed]
    return {
        "prompts": len(prompts),
        "scored": len(valid),
        "fallbacks": len(prompts) - len(valid),
        "precision_at_5": mean([p.precision for p in on_topic]),
        "planner_accuracy": mean([float(not p.plan_problems) for p in valid]),
        "plans_correct": sum(not p.plan_problems for p in valid),
        "consistency": mean([p.consistency for p in on_topic]),
        "made_up_songs": sum(len(p.made_up_ids) for p in valid),
        "rerank_invented_ids": sum(len(p.rerank_invented_ids) for p in valid),
        "first_check_failed": len(retried),
        "fixed_by_retry": sum(p.fixed_by_retry for p in retried),
        "refine_precision": mean([p.refine_precision for p in refined]),
        "refine_agent_precision": mean([p.refine_agent_precision for p in refined]),
        "refine_reranked": len(reranked),
        "gemini_reasons": len(reasons),
        "gemini_reasons_flagged": sum(bool(r["flags"]) for r in reasons),
    }


# --- output ------------------------------------------------------------------------------------------

def pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def num(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def summary_table(summaries: dict[str, dict]) -> str:
    names = list(summaries)
    rows = [
        ("Precision@5 (songs matching the expected genre/energy)", lambda s: pct(s["precision_at_5"])),
        ("Planner accuracy (plan matches the expected plan)",
         lambda s: f"{pct(s['planner_accuracy'])} ({s['plans_correct']}/{s['scored']})"),
        ("Consistency (same prompt N×, overlap of the top 5)", lambda s: num(s["consistency"])),
        ("Made-up songs shown", lambda s: str(s["made_up_songs"])),
        ("Invented ids caught by the reranker check", lambda s: str(s["rerank_invented_ids"])),
        ("Self-check failed on attempt 1 → fixed by a retry",
         lambda s: f"{s['first_check_failed']} → {s['fixed_by_retry']}"),
        ("Refine round precision: shown / agent's own picks",
         lambda s: f"{pct(s['refine_precision'])} / {pct(s['refine_agent_precision'])}"),
        ("Gemini reasons flagged for details not in the data",
         lambda s: f"{s['gemini_reasons_flagged']}/{s['gemini_reasons']}" if s["gemini_reasons"] else "n/a"),
        ("Runs that fell back to the rules", lambda s: str(s["fallbacks"])),
    ]
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += [f"| {label} | " + " | ".join(fmt(summaries[n]) for n in names) + " |" for label, fmt in rows]
    return "\n".join(lines)


def prompt_table(results: dict[str, list[PromptResult]]) -> str:
    names = list(results)
    lines = ["| Prompt | " + " | ".join(f"{n} P@5 | {n} plan" for n in names) + " |",
             "|---|" + "---|---|" * len(names)]
    for i, base in enumerate(results[names[0]]):
        cells = []
        for n in names:
            p = results[n][i]
            plan = "✓" if not p.plan_problems else "✗ " + "; ".join(p.plan_problems)
            if any(s == "rules" for s in p.plan_sources) and n != "rules":
                plan += " (fell back to rules)"
            cells += [pct(p.precision) if p.on_topic else "–", plan]
        lines.append(f"| {base.request} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def persona_table(personas: list[PersonaResult]) -> str:
    lines = ["| Persona | Round 1 | Refine: Gemini's list | Refine: agent's own picks | Gemini used |",
             "|---|---|---|---|---|"]
    for p in personas:
        used = "yes" if p.refine_source.startswith("gemini") else f"no ({p.refine_source or p.plan_source})"
        lines.append(f"| {p.name} | {pct(p.round1)} | {pct(p.shown)} | {pct(p.agent_only)} | {used} |")
    if personas:
        lines.append(f"| **Average** | {pct(mean([p.round1 for p in personas]))} | "
                     f"**{pct(mean([p.shown for p in personas]))}** | {pct(mean([p.agent_only for p in personas]))} | |")
    return "\n".join(lines)


def catalog_song(row) -> Recommendation:
    """A catalog row as a Recommendation, enough for song_matches (used when re-scoring saved runs)."""
    features = {name: float(row[name]) for name in ("energy", "valence", "acousticness", "instrumentalness")}
    return Recommendation(song_id=int(row["id"]), track_id=str(row["track_id"]), title=str(row["title"]),
                          artist=str(row["artist"]), genres=str(row["genres"]), score=0.0, parts={},
                          features=features)


def rescore(report: dict, golden: list[dict], catalog, runs: int) -> dict[str, list[PromptResult]]:
    """Recompute precision and consistency from a saved report, using only its first `runs` runs.

    Planner accuracy, the refine round and the reasons come from run 1 and are kept as saved.
    """
    rules_by_id = {item["id"]: item.get("songs", {}) for item in golden}
    rows = {int(r["id"]): r for r in catalog.to_dict("records")}
    known = set(PromptResult.__dataclass_fields__)
    results: dict[str, list[PromptResult]] = {}
    for planner, saved in report.get("prompts", {}).items():
        results[planner] = []
        for item in saved:
            p = PromptResult(**{k: v for k, v in item.items() if k in known})
            if p.id not in rules_by_id:
                raise GoldenSetError(f"Saved prompt {p.id!r} isn't in the golden set; re-run instead of re-scoring.")
            if len(p.song_ids) < runs:
                raise GoldenSetError(f"The saved results have {len(p.song_ids)} run(s) for {p.id!r}; "
                                     f"--runs {runs} needs a new run.")
            p.song_ids, p.plan_sources = p.song_ids[:runs], p.plan_sources[:runs]
            p.plan_summaries = p.plan_summaries[:runs]
            if p.on_topic:
                missing = [i for ids in p.song_ids for i in ids if i not in rows]
                if missing:
                    raise GoldenSetError(f"Song ids {missing} aren't in the catalog; was it rebuilt? Re-run instead.")
                p.precisions = [precision([catalog_song(rows[i]) for i in ids], rules_by_id[p.id]) for ids in p.song_ids]
                p.precision = mean(p.precisions)
                p.consistency = (mean([jaccard(a, b) for a, b in itertools.combinations(p.song_ids, 2)])
                                 if runs > 1 else None)
            results[planner].append(p)
    return results


def write_and_print(report: dict, results: dict[str, list[PromptResult]], out: Path) -> None:
    try:
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError as error:
        print(f"Warning: couldn't save results to {out}: {error}", file=sys.stderr)
    model = report.get("gemini_model")
    note = f", re-scored from a {report['rescored_from_runs']}-run report" if report.get("rescored_from_runs") else ""
    print(f"## Golden-prompt evaluation ({report['date']}, {report['runs_per_prompt']} run(s) per prompt"
          + (f", {model}, {report['gemini_calls']} Gemini calls" if model else "") + note + ")\n")
    print(summary_table(report["summary"]) + "\n")
    print(prompt_table(results) + "\n")
    for name, ps in report.get("personas", {}).items():
        print(f"### Simulated users, one refine round ({name})\n")
        print(persona_table([PersonaResult(**p) for p in ps]) + "\n")
    flagged = [(p.request, r) for rs in results.values() for p in rs for r in p.gemini_reasons if r["flags"]]
    if flagged:
        print("### Gemini reasons flagged by the grounding check (review by hand)\n")
        for request, r in flagged:
            print(f"- [{request}] {r['song']}: \"{r['reason']}\" → {', '.join(r['flags'])}")
    print(f"\nSaved full results to {out}", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from src.log_config import setup_logging

    parser = argparse.ArgumentParser(prog="python -m eval.run_eval", description=__doc__.split("\n\n")[0])
    parser.add_argument("--planners", default="rules,gemini", help="Comma-separated: rules, gemini (default %(default)s)")
    parser.add_argument("--runs", type=int, default=3, help="Runs per prompt for consistency (default %(default)s)")
    parser.add_argument("--delay", type=float, default=4.0, help="Seconds between Gemini calls (default %(default)s)")
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH, help="Golden prompts file")
    parser.add_argument("--out", type=Path, default=RESULTS_PATH, help="Where to save the JSON results")
    parser.add_argument("--no-personas", action="store_true", help="Skip the simulated-user comparison")
    parser.add_argument("--rescore", type=Path, metavar="RESULTS_JSON",
                        help="Re-score a saved results file with --runs runs instead of calling the planners")
    args = parser.parse_args(argv)
    setup_logging(logging.WARNING)

    planners = [p.strip() for p in args.planners.split(",") if p.strip()]
    if not planners or set(planners) - {"rules", "gemini"}:
        print("Error: --planners must list 'rules' and/or 'gemini'.", file=sys.stderr)
        return 2
    if args.runs < 1:
        print("Error: --runs must be at least 1.", file=sys.stderr)
        return 2
    if args.rescore:
        try:
            golden = load_golden(args.golden)
            report = json.loads(args.rescore.read_text(encoding="utf-8"))
            results = rescore(report, golden, load_catalog(), args.runs)
        except (OSError, json.JSONDecodeError) as error:
            print(f"Error: couldn't read {args.rescore}: {error}", file=sys.stderr)
            return 1
        except (GoldenSetError, CatalogError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        report["rescored_from_runs"] = report.get("rescored_from_runs") or report.get("runs_per_prompt")
        report["runs_per_prompt"] = args.runs
        report["summary"] = {name: summarize(rs, name) for name, rs in results.items()}
        report["prompts"] = {name: [p.__dict__ for p in rs] for name, rs in results.items()}
        write_and_print(report, results, args.out)
        return 0

    try:
        golden = load_golden(args.golden)
        retriever = Retriever.load()
    except (GoldenSetError, CatalogError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    offline_agent = PlaylistAgent(retriever, llm=None)
    agents: dict[str, PlaylistAgent] = {}
    paced: Optional[PacedLLM] = None
    for name in planners:
        if name == "rules":
            agents[name] = offline_agent
            continue
        client = get_client()
        if client is None:
            print("Error: Gemini isn't available (set GEMINI_API_KEY in .env), or run with --planners rules.",
                  file=sys.stderr)
            return 1
        paced = PacedLLM(client, args.delay)
        agents[name] = PlaylistAgent(retriever, llm=paced)

    catalog_ids = set(retriever.catalog["id"].astype(int))
    catalog_genres = set(retriever.known_genres)
    results: dict[str, list[PromptResult]] = {}
    personas: dict[str, list[PersonaResult]] = {}
    started = time.perf_counter()
    for name, agent in agents.items():
        label = name if name == "rules" else f"gemini ({agent.llm.model})"
        print(f"Evaluating {label}: {len(golden)} prompts × {args.runs} run(s)...", file=sys.stderr)
        results[name] = []
        for item in golden:
            print(f"  {item['id']}", file=sys.stderr)
            results[name].append(evaluate_prompt(item, agent, offline_agent, args.runs, catalog_ids, catalog_genres))
        if name == "gemini" and not args.no_personas:
            print("  simulated users", file=sys.stderr)
            personas[name] = [evaluate_persona(p, agent, offline_agent) for p in PERSONAS]

    summaries = {name: summarize(rs, name) for name, rs in results.items()}
    model = agents["gemini"].llm.model if "gemini" in agents else None
    report = {
        "date": time.strftime("%Y-%m-%d"),
        "gemini_model": model,
        "runs_per_prompt": args.runs,
        "gemini_calls": paced.calls if paced else 0,
        "gemini_failed_calls": paced.failures if paced else 0,
        "seconds": round(time.perf_counter() - started, 1),
        "summary": summaries,
        "prompts": {name: [p.__dict__ for p in rs] for name, rs in results.items()},
        "personas": {name: [p.__dict__ for p in ps] for name, ps in personas.items()},
    }
    write_and_print(report, results, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
