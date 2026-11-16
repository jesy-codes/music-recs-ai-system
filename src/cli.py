"""Command-line front end for the playlist agent.

Usage:
    python -m src.cli "something chill, with jazz and indie music"
    python -m src.cli "sad acoustic songs for a rainy day" -k 8 --trace   # show every agent step
    python -m src.cli "upbeat latin, no reggae" --exclude 12 345          # skip songs already shown
    python -m src.cli "chill jazz" --offline                              # rule-based planner, no Gemini
    python -m src.cli "chill jazz" --json                                 # machine-readable output

Gemini is used when GEMINI_API_KEY is set (in .env or the environment); otherwise the agent runs offline.
Exit codes: 0 songs shown, 1 catalog problem, 2 request rejected by a guardrail, 3 no songs matched.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Sequence

from src.agent import AgentResult, PlaylistAgent
from src.catalog import CatalogError
from src.llm import get_client
from src.query import MAX_K, MIN_K
from src.retriever import Retriever

logger = logging.getLogger(__name__)

EXIT_CODES = {"ok": 0, "partial": 0, "rejected": 2, "empty": 3}


def format_result(result: AgentResult, show_trace: bool = False) -> str:
    lines = [f"Request: {result.request}"]
    if result.query is not None and result.status != "rejected":
        lines.append(f"Plan ({result.plan_source}): {result.query.summary()}")
        lines += [f"Note: {note}" for note in result.query.notes]
    lines.append("")
    lines.append(result.message)
    for i, song in enumerate(result.songs, start=1):
        lines.append(f"\n{i}. {song.title} - {song.artist}   [{song.genres}]")
        lines.append(f"   {result.reasons.get(song.song_id, '; '.join(song.reasons))}")
        lines.append(f"   id {song.song_id} | {song.spotify_url}")
    if show_trace:
        lines.append("\nAgent steps:")
        lines += [f"  {i:>2}. {step.name:<9} {step.detail}" for i, step in enumerate(result.trace, start=1)]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    from src.log_config import setup_logging

    parser = argparse.ArgumentParser(prog="python -m src.cli", description="Ask the playlist agent for songs.")
    parser.add_argument("request", nargs="+", help="What you want to listen to, in plain English")
    parser.add_argument("-k", type=int, default=5, help=f"Number of songs ({MIN_K}-{MAX_K}, default %(default)s)")
    parser.add_argument("--exclude", type=int, nargs="*", default=[], metavar="ID", help="Catalog ids to skip (songs already shown)")
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

    result = PlaylistAgent(retriever, llm).run(request, k=args.k, exclude_ids=args.exclude)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(format_result(result, show_trace=args.trace))
    return EXIT_CODES.get(result.status, 0)


if __name__ == "__main__":
    sys.exit(main())
