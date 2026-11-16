"""Command-line front end for the playlist agent.

Usage:
    python -m src.cli "something chill, with jazz and indie music"
    python -m src.cli "chill jazz and indie" --interactive               # vote 👍/👎, get 5 more, see the playlist
    python -m src.cli "chill jazz and indie" --like 253 92 --dislike 278  # one refine round, no prompts
    python -m src.cli "sad acoustic songs for a rainy day" -k 8 --trace   # show every agent step
    python -m src.cli "upbeat latin, no reggae" --exclude 12 345          # skip songs already shown
    python -m src.cli "chill jazz" --offline                              # rule-based planner, no Gemini
    python -m src.cli "chill jazz" --json                                 # machine-readable output

Gemini is used when GEMINI_API_KEY is set (in .env or the environment); otherwise the agent runs offline.
Exit codes: 0 songs shown, 1 catalog problem, 2 request rejected by a guardrail, 3 no songs matched,
4 invalid votes.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from typing import Callable, Optional, Sequence

from src.agent import AgentResult, PlaylistAgent
from src.catalog import CatalogError
from src.llm import get_client
from src.query import MAX_K, MIN_K
from src.ranker import Recommendation
from src.retriever import Retriever
from src.session import PlaylistSession, SessionError

logger = logging.getLogger(__name__)

EXIT_CODES = {"ok": 0, "partial": 0, "rejected": 2, "empty": 3}
VOTE_PATTERN = re.compile(r"^([+-])(\d+)$")
VOTE_HELP = "Vote with +id to like and -id to reject, e.g. '+253 +92 -278'. Type 'done' to finish."


def format_song(i: int, song: Recommendation, reason: str) -> list[str]:
    return [f"\n{i}. {song.title} - {song.artist}   [{song.genres}]",
            f"   {reason}",
            f"   id {song.song_id} | {song.spotify_url}"]


def format_result(result: AgentResult, show_trace: bool = False,
                  playlist_so_far: Optional[list[Recommendation]] = None) -> str:
    lines = [f"Request: {result.request}"]
    if result.query is not None and result.status != "rejected":
        lines.append(f"Plan ({result.plan_source}): {result.query.summary()}")
        lines += [f"Note: {note}" for note in result.query.notes]
    lines.append("")
    lines.append(result.message)
    if playlist_so_far:
        lines.append(f"\nYour playlist so far ({len(playlist_so_far)}): 👍")
        lines += [f"  - {s.title} - {s.artist}   (id {s.song_id})" for s in playlist_so_far]
        lines.append("\nNew suggestions:")
    for i, song in enumerate(result.songs, start=1):
        lines += format_song(i, song, result.reasons.get(song.song_id, "; ".join(song.reasons)))
    if show_trace:
        lines.append("\nAgent steps:")
        lines += [f"  {i:>2}. {step.name:<9} {step.detail}" for i, step in enumerate(result.trace, start=1)]
    return "\n".join(lines)


def format_playlist(songs: list[Recommendation]) -> str:
    if not songs:
        return "\nYour playlist is empty: no songs got a 👍."
    lines = [f"\nYour playlist ({len(songs)} songs):"]
    lines += [f"  {i}. {s.title} - {s.artist}   {s.spotify_url}" for i, s in enumerate(songs, start=1)]
    return "\n".join(lines)


def parse_votes(text: str) -> dict[int, bool]:
    """'+253 +92 -278' -> {253: True, 92: True, 278: False}. Raises ValueError on anything else."""
    votes: dict[int, bool] = {}
    for token in text.replace(",", " ").split():
        match = VOTE_PATTERN.match(token)
        if not match:
            raise ValueError(f"Couldn't read {token!r}. {VOTE_HELP}")
        votes[int(match.group(2))] = match.group(1) == "+"
    return votes


def run_interactive(session: PlaylistSession, show_trace: bool, ask: Callable[[str], str] = input) -> None:
    """Prompt for votes after each round until the user is done or the refine limit is reached."""
    while True:
        prompt = VOTE_HELP if session.can_refine else "Last round: vote on the new songs, or type 'done'."
        print(f"\n{prompt}")
        try:
            answer = ask("> ").strip()
        except EOFError:
            answer = "done"
        if answer.lower() in ("done", "q", "quit", ""):
            break
        try:
            votes = parse_votes(answer)
            if not session.can_refine:
                session.vote(votes)
                break
            result = session.refine(votes)
        except (ValueError, SessionError) as error:
            print(f"Error: {error}")
            continue
        print(format_result(result, show_trace, playlist_so_far=session.playlist()))
    print(format_playlist(session.playlist()))


def main(argv: Sequence[str] | None = None, ask: Callable[[str], str] = input) -> int:
    from src.log_config import setup_logging

    parser = argparse.ArgumentParser(prog="python -m src.cli", description="Ask the playlist agent for songs.")
    parser.add_argument("request", nargs="+", help="What you want to listen to, in plain English")
    parser.add_argument("-k", type=int, default=5, help=f"Number of songs per round ({MIN_K}-{MAX_K}, default %(default)s)")
    parser.add_argument("--exclude", type=int, nargs="*", default=[], metavar="ID", help="Catalog ids to skip (songs already shown)")
    parser.add_argument("--interactive", action="store_true", help="Vote on songs and get a refined round")
    parser.add_argument("--like", type=int, nargs="*", default=[], metavar="ID", help="Songs from round 1 to 👍 (runs one refine round)")
    parser.add_argument("--dislike", type=int, nargs="*", default=[], metavar="ID", help="Songs from round 1 to 👎 (runs one refine round)")
    parser.add_argument("--offline", action="store_true", help="Don't call Gemini; use the rule-based planner and template reasons")
    parser.add_argument("--trace", action="store_true", help="Print every agent step after the results")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    parser.add_argument("--verbose", action="store_true", help="Show INFO logs on the console")
    args = parser.parse_args(argv)

    setup_logging(logging.INFO if args.verbose else logging.WARNING)
    request = " ".join(args.request)
    try:
        retriever = Retriever.load()
    except CatalogError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    llm = None if args.offline else get_client()
    if llm is None and not args.offline:
        print("Note: Gemini isn't configured (GEMINI_API_KEY not set), running offline.\n", file=sys.stderr)
    agent = PlaylistAgent(retriever, llm)

    voting = args.interactive or args.like or args.dislike
    if not voting:
        result = agent.run(request, k=args.k, exclude_ids=args.exclude)
        print(json.dumps(result.to_dict(), indent=2, default=str) if args.json else format_result(result, args.trace))
        return EXIT_CODES.get(result.status, 0)

    session = PlaylistSession(agent, k=args.k)
    first = session.start(request)
    print(format_result(first, args.trace))
    if first.status not in ("ok", "partial"):
        return EXIT_CODES.get(first.status, 0)
    if args.interactive:
        run_interactive(session, args.trace, ask)
        return 0

    votes = {**{i: True for i in args.like}, **{i: False for i in args.dislike}}
    try:
        second = session.refine(votes)
    except SessionError as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 4
    print("\n" + "=" * 60 + "\n")
    print(format_result(second, args.trace, playlist_so_far=session.playlist()))
    print(format_playlist(session.playlist()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
