"""Command-line search: try requests against the catalog without the app or any API key.

Usage:
    python -m src.cli "something chill, with jazz and indie music"
    python -m src.cli "sad acoustic songs for a rainy day" -k 8
    python -m src.cli "upbeat latin, no reggae" --exclude 12 345   # skip songs by catalog id
    python -m src.cli "chill jazz" --json                           # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from typing import Sequence

from src.catalog import CatalogError
from src.query import MAX_K, MIN_K
from src.recommend import RoundResult, recommend
from src.retriever import Retriever

logger = logging.getLogger(__name__)


def format_round(result: RoundResult) -> str:
    query = result.query
    lines = [f"Request: {query.text}", f"Plan:    {query.summary()}"]
    lines += [f"Note:    {note}" for note in query.notes]
    lines.append(f"Picked {len(result.songs)} of {result.candidates_considered} candidates:\n")
    for i, song in enumerate(result.songs, start=1):
        lines.append(f"{i}. {song.title} - {song.artist}   [{song.genres}]   score {song.score:.2f}")
        lines.append(f"   id {song.song_id} | " + "; ".join(song.reasons))
        lines.append(f"   {song.spotify_url}")
    if not result.songs:
        lines.append("No songs matched. Try fewer exclusions or a broader description.")
    return "\n".join(lines)


def to_json(result: RoundResult) -> str:
    payload = {
        "query": result.query.to_dict(),
        "candidates_considered": result.candidates_considered,
        "songs": [{**asdict(s), "spotify_url": s.spotify_url} for s in result.songs],
    }
    return json.dumps(payload, indent=2, default=str)


def main(argv: Sequence[str] | None = None) -> int:
    from src.log_config import setup_logging

    parser = argparse.ArgumentParser(prog="python -m src.cli", description="Search the song catalog from the command line.")
    parser.add_argument("request", nargs="+", help="What you want to listen to, in plain English")
    parser.add_argument("-k", type=int, default=5, help=f"Number of songs ({MIN_K}-{MAX_K}, default %(default)s)")
    parser.add_argument("--exclude", type=int, nargs="*", default=[], metavar="ID", help="Catalog ids to skip (songs already shown)")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    parser.add_argument("--verbose", action="store_true", help="Show INFO logs on the console")
    args = parser.parse_args(argv)

    setup_logging(logging.INFO if args.verbose else logging.WARNING)
    request = " ".join(args.request)
    try:
        retriever = Retriever.load()
        result = recommend(request, retriever, k=args.k, exclude_ids=args.exclude)
    except ValueError as error:          # empty or unusable request
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except CatalogError as error:        # missing or out-of-date catalog files
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(to_json(result) if args.json else format_round(result))
    return 0 if result.songs else 3


if __name__ == "__main__":
    sys.exit(main())
