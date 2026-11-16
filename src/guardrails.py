"""Input and output guardrails.

Input: reject empty, overly long or clearly off-topic requests before any search or API call, and turn
requests for genres the catalog lacks into an explicit "closest match" instead of quietly weak results.
Output: only songs that exist in the catalog may be shown, whatever the LLM returns.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

from src.parser import CLOSEST_GENRE
from src.query import SearchQuery

logger = logging.getLogger(__name__)

MAX_REQUEST_CHARS = 500

MUSIC_WORDS = re.compile(
    r"\b(music|songs?|tracks?|tunes|playlists?|listen(?:ing)?|artists?|bands?|albums?|vibes?|beats|"
    r"genres?|singers?|mix|melod\w*|rhythm|jams?)\b", re.I)
TASK_WORDS = re.compile(
    r"\b(essay|homework|code|program(?:ming)?|debug|math|equation|solve|translate|email|recipe|weather|"
    r"news|stock|invest\w*|password|diagnos\w*|write (?:me )?an?)\b", re.I)


@dataclass
class GuardrailResult:
    ok: bool
    message: str = ""


def check_request(text: str) -> GuardrailResult:
    """Cheap checks that run before planning (no API call)."""
    if not text or not text.strip():
        return GuardrailResult(False, "The request is empty. Describe the music you want, e.g. 'chill jazz for studying'.")
    if len(text) > MAX_REQUEST_CHARS:
        return GuardrailResult(False, f"The request is too long ({len(text)} characters). Keep it under {MAX_REQUEST_CHARS}.")
    return GuardrailResult(True)


def looks_off_topic(text: str, query: SearchQuery) -> bool:
    """Rule-based off-topic check, used when Gemini isn't available to judge.

    Off-topic means: the parser found no music signal at all (no genre, mood or feature target) and the
    text contains task words like "essay" or "code" without any music words.
    """
    has_music_signal = bool(query.genres or query.unknown_genres or query.moods or query.energy
                            or query.valence or query.acoustic or query.instrumental)
    return not has_music_signal and bool(TASK_WORDS.search(text)) and not MUSIC_WORDS.search(text)


def apply_closest_genres(query: SearchQuery) -> SearchQuery:
    """If every requested genre is missing from the catalog, search the closest catalog genres instead.

    The substitution is written into query.notes so the user is told, rather than getting quietly
    weaker results.
    """
    if query.genres or not query.unknown_genres:
        return query
    substitutes = [CLOSEST_GENRE[g] for g in query.unknown_genres if g in CLOSEST_GENRE]
    substitutes = [g for g in dict.fromkeys(substitutes) if g not in query.exclude_genres]
    # Replace the planner's "isn't in the catalog" notes with one clear message.
    query.notes = [n for n in query.notes if "isn't in the catalog" not in n]
    if substitutes:
        query.genres = substitutes
        query.notes.append(
            f"Showing {', '.join(substitutes)} instead of {', '.join(query.unknown_genres)}, which isn't in the catalog."
        )
        logger.info("Substituted closest genres %s for unknown %s", substitutes, query.unknown_genres)
    else:
        query.notes.append(
            f"{', '.join(query.unknown_genres)} isn't in the catalog, so results are based on the description only."
        )
    return query


def filter_to_catalog(song_ids: Iterable[int], valid_ids: set[int]) -> tuple[list[int], list[int]]:
    """Split ids into (in catalog, not in catalog). Anything not in the catalog must never be shown."""
    kept, dropped = [], []
    for song_id in song_ids:
        (kept if song_id in valid_ids else dropped).append(song_id)
    if dropped:
        logger.warning("Dropped %d song ids that are not in the catalog: %s", len(dropped), dropped)
    return kept, dropped
