"""Rule-based request parser: plain English -> SearchQuery, with no API calls.

It's the offline fallback for the Gemini planner (Phase 3) and the baseline the evaluation compares Gemini
against. Every rule is a visible keyword list, so its behavior is easy to explain and test.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

from src.query import Range, SearchQuery

logger = logging.getLogger(__name__)

# Words the user might type -> catalog genre. Longer aliases are matched first, so "indie pop" wins over
# "indie" and "pop", and "hip hop" isn't read as "hop".
GENRE_ALIASES: dict[str, str] = {
    "jazz": "jazz", "jazzy": "jazz",
    "indie pop": "indie-pop", "indie-pop": "indie-pop", "indiepop": "indie-pop",
    "indie": "indie",
    "singer songwriter": "singer-songwriter", "singer-songwriter": "singer-songwriter",
    "folk": "folk",
    "soul": "soul",
    "rnb": "r-n-b", "r-n-b": "r-n-b", "r and b": "r-n-b", "rhythm and blues": "r-n-b",
    "blues": "blues", "bluesy": "blues",
    "piano": "piano",
    "classical": "classical", "orchestral": "classical",
    "ambient": "ambient",
    "pop": "pop",
    "alternative rock": "alt-rock", "alt rock": "alt-rock", "alt-rock": "alt-rock", "alternative": "alt-rock",
    "hard rock": "rock", "classic rock": "rock", "rock": "rock",
    "hip hop": "hip-hop", "hip-hop": "hip-hop", "hiphop": "hip-hop", "rap": "hip-hop",
    "electronica": "electronic", "electronic": "electronic",
    "dance": "dance",
    "edm": "edm",
    "latin": "latin",
    "heavy metal": "metal", "death metal": "metal", "black metal": "metal", "metal": "metal",
    "country": "country",
    "funk": "funk", "funky": "funk",
    "reggae": "reggae",
    "trip hop": "trip-hop", "trip-hop": "trip-hop", "triphop": "trip-hop",
}

# Genres people ask for that the catalog doesn't include -> closest catalog genre (used in the message).
UNKNOWN_GENRES: dict[str, str] = {
    "k-pop": "pop", "kpop": "pop", "j-pop": "pop", "jpop": "pop", "j-rock": "rock", "anime": "pop",
    "opera": "classical", "techno": "electronic", "house music": "edm", "deep house": "edm",
    "trance": "edm", "dubstep": "edm", "drum and bass": "edm", "dnb": "edm", "hardstyle": "edm",
    "punk": "rock", "grunge": "alt-rock", "emo": "alt-rock", "goth": "alt-rock", "gospel": "soul",
    "salsa": "latin", "samba": "latin", "tango": "latin", "reggaeton": "latin", "bossa nova": "latin",
    "disco": "funk", "ska": "reggae", "bluegrass": "country", "afrobeat": "dance",
    "cantopop": "pop", "mandopop": "pop", "bollywood": "pop", "show tunes": "pop",
}
CLOSEST_GENRE = UNKNOWN_GENRES

NEGATIONS = ("no", "not", "without", "except", "avoid", "skip", "minus", "hate")

# Mood words. Each list is checked with word boundaries; "low energy" phrases are removed before the
# high-energy words are checked, so "low energy" doesn't also count as "energy".
LOW_ENERGY_PHRASES = [r"low[- ]energy", r"not too (?:loud|intense|energetic)"]
LOW_ENERGY_WORDS = [
    r"chill(?:ed|y)?", r"calm(?:ing)?", r"relax(?:ing|ed)?", r"mellow", r"soft", r"sleep(?:y|ing)?",
    r"study(?:ing)?", r"focus(?:ed|ing)?", r"quiet", r"peaceful", r"lofi", r"slow", r"cozy", r"laid[- ]back",
    r"gentle", r"soothing", r"background",
]
HIGH_ENERGY_PHRASES = [r"high[- ]energy"]
HIGH_ENERGY_WORDS = [
    r"energetic", r"energy", r"workout", r"gym", r"running", r"party", r"hype",
    r"pump(?:ed)?(?: up)?", r"intense", r"fast", r"powerful", r"loud",
]
LOW_VALENCE_WORDS = [
    r"sad", r"melancholy", r"melancholic", r"heartbreak", r"heartbroken", r"moody", r"dark", r"rainy",
    r"cry(?:ing)?", r"lonely", r"gloomy", r"emotional",
]
HIGH_VALENCE_WORDS = [
    r"happy", r"cheerful", r"feel[- ]good", r"sunny", r"uplifting", r"joyful", r"upbeat", r"fun",
    r"bright", r"summery", r"summer",
]
ACOUSTIC_WORDS = [r"acoustic", r"unplugged"]
INSTRUMENTAL_PHRASES = [r"instrumental", r"no vocals", r"no lyrics", r"without vocals", r"without lyrics"]

LOW_ENERGY = Range(0.0, 0.45)
HIGH_ENERGY = Range(0.7, 1.0)
LOW_VALENCE = Range(0.0, 0.4)
HIGH_VALENCE = Range(0.6, 1.0)

COUNT_PATTERN = re.compile(r"\b(\d{1,2})\s+(?:songs|tracks|tunes)\b")


def normalize(text: str) -> str:
    """Lowercase and smooth over spelling variants before matching."""
    text = text.lower()
    text = text.replace("r&b", "rnb").replace("&", " and ").replace("lo-fi", "lofi")
    text = re.sub(r"[^\w\s'-]", " ", text)  # punctuation -> space, keep hyphens and apostrophes
    return re.sub(r"\s+", " ", text).strip()


def _find_all(patterns: Iterable[str], text: str) -> list[re.Match]:
    return [m for p in patterns for m in re.finditer(rf"\b{p}\b", text)]


def _blank(text: str, matches: Iterable[re.Match]) -> str:
    """Replace matched spans with spaces so later rules can't match them again."""
    chars = list(text)
    for m in matches:
        chars[m.start():m.end()] = " " * (m.end() - m.start())
    return "".join(chars)


def _negated(text: str, start: int) -> bool:
    before = text[:start].split()[-2:]
    return any(word in NEGATIONS for word in before)


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def parse_request(request: str, k: int = 5) -> SearchQuery:
    """Turn a free-text request into a SearchQuery using keyword rules."""
    if not request or not request.strip():
        raise ValueError("The request is empty. Describe the music you want, e.g. 'chill jazz for studying'.")
    text = normalize(request)
    notes: list[str] = []

    count = COUNT_PATTERN.search(text)
    if count:
        k = int(count.group(1))
        text = _blank(text, [count])

    # Instrumental phrases first: "no vocals" must not be read as a negated genre.
    instrumental_hits = _find_all(INSTRUMENTAL_PHRASES, text)
    text = _blank(text, instrumental_hits)

    # Genres (known and unknown), longest alias first so overlapping aliases don't double-count.
    genres: list[tuple[int, str]] = []
    excluded: list[tuple[int, str]] = []
    unknown: list[tuple[int, str]] = []
    for alias in sorted({**GENRE_ALIASES, **UNKNOWN_GENRES}, key=len, reverse=True):
        hits = list(re.finditer(rf"\b{re.escape(alias)}\b", text))
        for m in hits:
            if alias in UNKNOWN_GENRES:
                unknown.append((m.start(), alias))
            elif _negated(text, m.start()):
                excluded.append((m.start(), GENRE_ALIASES[alias]))
            else:
                genres.append((m.start(), GENRE_ALIASES[alias]))
        text = _blank(text, hits)

    genre_list = _unique(g for _, g in sorted(genres))
    excluded_list = _unique(g for _, g in sorted(excluded) if g not in genre_list)
    unknown_list = _unique(g for _, g in sorted(unknown))
    for name in unknown_list:
        notes.append(f"'{name}' isn't in the catalog; the closest genre is '{CLOSEST_GENRE[name]}'.")

    # Moods.
    low_phrase_hits = _find_all(LOW_ENERGY_PHRASES, text)
    text_no_low = _blank(text, low_phrase_hits)
    low_e = low_phrase_hits + _find_all(LOW_ENERGY_WORDS, text_no_low)
    high_phrase_hits = _find_all(HIGH_ENERGY_PHRASES, text_no_low)
    high_e = high_phrase_hits + _find_all(HIGH_ENERGY_WORDS, _blank(text_no_low, high_phrase_hits))
    low_v = _find_all(LOW_VALENCE_WORDS, text)
    high_v = _find_all(HIGH_VALENCE_WORDS, text)
    acoustic = bool(_find_all(ACOUSTIC_WORDS, text))

    energy = None
    if low_e and high_e:
        notes.append("The request asks for both calm and energetic music, so energy isn't filtered.")
    elif low_e:
        energy = LOW_ENERGY
    elif high_e:
        energy = HIGH_ENERGY

    valence = None
    if low_v and high_v:
        notes.append("The request asks for both sad and happy music, so mood isn't filtered.")
    elif low_v:
        valence = LOW_VALENCE
    elif high_v:
        valence = HIGH_VALENCE

    moods = _unique(m.group(0) for m in sorted(low_e + high_e + low_v + high_v, key=lambda m: m.start()))
    if acoustic:
        moods.append("acoustic")
    if instrumental_hits:
        moods.append("instrumental")

    query = SearchQuery(
        text=request,
        genres=genre_list,
        exclude_genres=excluded_list,
        unknown_genres=unknown_list,
        moods=moods,
        energy=energy,
        valence=valence,
        acoustic=True if acoustic else None,
        instrumental=True if instrumental_hits else None,
        k=k,
        notes=notes,
    )
    logger.info("Parsed request %r -> %s", request, query.summary())
    for note in notes:
        logger.info("Parser note: %s", note)
    return query
