"""Simulated users: does the feedback loop actually learn?

Each persona makes a request but has a hidden taste the request doesn't fully say (e.g. asks for "chill
jazz and indie" but only likes acoustic songs). It votes 👍/👎 on every song it's shown, using that hidden
rule. We measure the like rate per round, with feedback and without it (baseline: same search, just
skipping songs already shown). If feedback works, later rounds should have a higher like rate than the
baseline.

Runs offline (rule-based planner, template reasons), so it costs no Gemini calls and gives the same
numbers every time.

    python -m src.simulator               # 3 refine rounds per persona
    python -m src.simulator --rounds 1    # match the app's default limit
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Callable

from src.agent import PlaylistAgent
from src.catalog import CatalogError
from src.ranker import Recommendation
from src.retriever import Retriever
from src.session import PlaylistSession


def has_genre(song: Recommendation, *genres: str) -> bool:
    return bool(set(song.genres.split("; ")) & set(genres))


@dataclass
class Persona:
    name: str
    request: str
    hidden_taste: str
    likes: Callable[[Recommendation], bool]


PERSONAS = [
    Persona("Acoustic purist", "something chill, with jazz and indie music", "only acoustic songs (acousticness ≥ 0.6)",
            lambda s: s.features["acousticness"] >= 0.6),
    Persona("Jazz head", "something chill, with jazz and indie music", "only jazz, no indie",
            lambda s: has_genre(s, "jazz")),
    Persona("Latin party", "upbeat music for a party", "only latin or reggae",
            lambda s: has_genre(s, "latin", "reggae")),
    Persona("Rainy-day crier", "sad songs for a rainy day", "very sad and acoustic (valence < 0.3, acousticness > 0.5)",
            lambda s: s.features["valence"] < 0.3 and s.features["acousticness"] > 0.5),
    Persona("Rock gym-goer", "workout music", "only rock, metal or alt-rock",
            lambda s: has_genre(s, "rock", "metal", "alt-rock")),
    Persona("No-vocals studier", "music for studying", "only instrumental (instrumentalness > 0.5)",
            lambda s: s.features["instrumentalness"] > 0.5),
]


@dataclass
class Run:
    like_rates: list[float] = field(default_factory=list)   # one per round
    liked_total: int = 0


def simulate(persona: Persona, agent: PlaylistAgent, refine_rounds: int, use_feedback: bool) -> Run:
    session = PlaylistSession(agent, max_refine_rounds=refine_rounds, use_feedback=use_feedback)
    result = session.start(persona.request)
    run = Run()
    for round_number in range(refine_rounds + 1):
        votes = {s.song_id: persona.likes(s) for s in result.songs}
        run.like_rates.append(sum(votes.values()) / len(votes) if votes else 0.0)
        if round_number == refine_rounds or not votes:
            session.vote(votes)
            break
        result = session.refine(votes)
    run.liked_total = len(session.playlist())
    return run


def format_rates(rates: list[float]) -> str:
    return " → ".join(f"{r:>4.0%}" for r in rates)


def main(argv=None) -> int:
    from src.log_config import setup_logging

    parser = argparse.ArgumentParser(prog="python -m src.simulator", description="Measure the feedback loop with simulated users.")
    parser.add_argument("--rounds", type=int, default=3, help="Refine rounds after round 1 (default %(default)s)")
    args = parser.parse_args(argv)
    setup_logging(logging.WARNING)
    try:
        agent = PlaylistAgent(Retriever.load(), llm=None)   # offline: no API calls, deterministic
    except CatalogError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    rounds = args.rounds
    print(f"Like rate per round (round 1 + {rounds} refine round(s)), 5 songs per round\n")
    print(f"{'Persona':<19} {'Hidden taste':<58} {'With feedback':<{7 * (rounds + 1)}}  Baseline (no feedback)")
    totals_fb, totals_base = [0.0] * (rounds + 1), [0.0] * (rounds + 1)
    for persona in PERSONAS:
        fb = simulate(persona, agent, rounds, use_feedback=True)
        base = simulate(persona, agent, rounds, use_feedback=False)
        print(f"{persona.name:<19} {persona.hidden_taste:<58} {format_rates(fb.like_rates):<{7 * (rounds + 1)}}  {format_rates(base.like_rates)}")
        for i in range(rounds + 1):
            totals_fb[i] += fb.like_rates[i] if i < len(fb.like_rates) else 0
            totals_base[i] += base.like_rates[i] if i < len(base.like_rates) else 0

    n = len(PERSONAS)
    avg_fb = [t / n for t in totals_fb]
    avg_base = [t / n for t in totals_base]
    print(f"\n{'Average':<19} {'':<58} {format_rates(avg_fb):<{7 * (rounds + 1)}}  {format_rates(avg_base)}")
    lift = avg_fb[1] - avg_base[1] if rounds >= 1 else 0.0
    print(f"\nRound 2 like rate: {avg_fb[1]:.0%} with feedback vs {avg_base[1]:.0%} without ({lift:+.0%}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
