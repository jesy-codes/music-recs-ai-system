"""Build and load the song catalog used for retrieval.

Pipeline (run once, outputs are committed):
    raw Kaggle/Hugging Face CSV (~114k rows, one row per track per genre)
      -> merge rows by track_id and by (title, artist), combining genres
      -> keep the most popular songs from a fixed list of genres
      -> add vibe tags derived from audio features ("calm", "acoustic", ...)
      -> write data/songs.csv
      -> embed one text description per song -> data/embeddings.npy

Usage:
    python -m src.catalog build                 # download if needed, build CSV + embeddings
    python -m src.catalog build --skip-embeddings
    python -m src.catalog build --per-genre 50 --force-download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import ssl
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Paths and settings -------------------------------------------------------

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
RAW_PATH = RAW_DIR / "spotify_tracks.csv"
CATALOG_PATH = DATA_DIR / "songs.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
META_PATH = DATA_DIR / "catalog_meta.json"

# "Spotify Tracks Dataset" (maharshipandya), BSD license, mirrored on Hugging Face so no Kaggle login is needed.
RAW_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"

# Static embedding model: numpy-only inference, so it runs on Intel Macs and small cloud hosts (no PyTorch).
DEFAULT_EMBEDDING_MODEL = "minishlab/potion-base-8M"

# Genres kept in the catalog. Chosen to cover the demo ("chill jazz and indie") plus enough contrast
# (rock, metal, edm, ...) for retrieval to be meaningful. Genres left out (e.g. k-pop) let us test the
# "not in catalog" guardrail.
SELECTED_GENRES: tuple[str, ...] = (
    "jazz", "indie", "indie-pop", "chill", "acoustic", "singer-songwriter", "folk", "soul",
    "r-n-b", "blues", "piano", "classical", "ambient", "study", "sleep", "sad", "happy",
    "pop", "rock", "alt-rock", "hip-hop", "electronic", "dance", "edm", "latin", "metal",
    "country", "funk", "reggae", "trip-hop",
)
DEFAULT_PER_GENRE = 50
DEFAULT_SEED = 42

RAW_REQUIRED_COLUMNS = (
    "track_id", "artists", "album_name", "track_name", "popularity", "duration_ms", "explicit",
    "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "time_signature", "track_genre",
)
FEATURES_0_TO_1 = ("danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence")

CATALOG_COLUMNS = (
    "id", "track_id", "title", "artist", "album", "genres", "popularity", "explicit", "duration_ms",
    "energy", "valence", "danceability", "acousticness", "instrumentalness", "speechiness", "liveness",
    "tempo", "loudness", "mode", "vibe_tags", "description",
)


class CatalogError(RuntimeError):
    """Raised when the catalog can't be built or loaded; the message says how to fix it."""


class Encoder(Protocol):
    def encode(self, sentences: Sequence[str]) -> np.ndarray: ...


# --- Download -----------------------------------------------------------------

def download_raw(url: str = RAW_URL, dest: Path = RAW_PATH, force: bool = False) -> Path:
    """Download the raw dataset CSV unless it's already on disk."""
    if dest.exists() and not force:
        logger.info("Raw dataset already present at %s (use --force-download to refresh)", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    logger.info("Downloading raw dataset from %s", url)
    try:
        # python.org builds on macOS ship without CA certificates; certifi's bundle avoids
        # "CERTIFICATE_VERIFY_FAILED" without asking users to run "Install Certificates.command".
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(url, timeout=60, context=context) as response, open(tmp, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        tmp.unlink(missing_ok=True)
        raise CatalogError(
            f"Couldn't download the dataset from {url}: {error}. Check your internet connection, "
            f"or download dataset.csv manually and save it as {dest}."
        ) from error
    tmp.replace(dest)
    logger.info("Saved raw dataset to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV and check it has the columns we rely on."""
    if not path.exists():
        raise CatalogError(f"Raw dataset not found at {path}. Run: python -m src.catalog build")
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise CatalogError(f"Raw dataset at {path} is not a readable CSV ({error}). Re-download with --force-download.") from error
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise CatalogError(f"Raw dataset at {path} is missing columns {missing}. Is it the Spotify Tracks Dataset?")
    logger.info("Loaded raw dataset: %d rows, %d genres", len(df), df["track_genre"].nunique())
    return df


# --- Cleaning -----------------------------------------------------------------

def _join_unique(values: Iterable[str]) -> str:
    return "; ".join(sorted({v for value in values for v in str(value).split("; ") if v}))


def merge_duplicates(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse the raw rows to one row per song, combining the genres of every duplicate.

    The raw data repeats a track once per genre it appears in, and often lists the same song under
    several track_ids (single vs. album release). We merge both kinds of duplicate and keep the
    most popular version's metadata.
    """
    df = raw.dropna(subset=["track_id", "track_name", "artists"]).copy()
    dropped = len(raw) - len(df)
    if dropped:
        logger.info("Dropped %d raw rows with no track id, title or artist", dropped)

    # 1) Same track_id in several genres -> one row with all genres.
    genres_by_id = df.groupby("track_id")["track_genre"].agg(_join_unique)
    df = df.sort_values(["popularity", "track_id"], ascending=[False, True]).drop_duplicates("track_id")
    df["genres"] = df["track_id"].map(genres_by_id)

    # 2) Same song (title + artists, case-insensitive) under several track_ids -> keep the most popular.
    df["_song_key"] = df["track_name"].str.strip().str.lower() + "|" + df["artists"].str.strip().str.lower()
    genres_by_song = df.groupby("_song_key")["genres"].agg(_join_unique)
    df = df.drop_duplicates("_song_key")  # already sorted by popularity, so the first is the most popular
    df["genres"] = df["_song_key"].map(genres_by_song)

    logger.info("Merged duplicates: %d raw rows -> %d unique songs", len(raw), len(df))
    return df.drop(columns=["_song_key", "track_genre"])


def sample_by_genre(
    songs: pd.DataFrame,
    genres: Sequence[str] = SELECTED_GENRES,
    per_genre: int = DEFAULT_PER_GENRE,
) -> pd.DataFrame:
    """Pick up to `per_genre` of the most popular songs for each selected genre.

    Songs are picked genre by genre without repeats, and only the selected genres are kept on each
    song, so a genre outside the list never shows up in the catalog. Deterministic: ties are broken
    by track_id.
    """
    if per_genre <= 0:
        raise CatalogError(f"--per-genre must be positive, got {per_genre}")
    selected = set(genres)
    songs = songs.copy()
    songs["genres"] = songs["genres"].map(lambda g: "; ".join(x for x in g.split("; ") if x in selected))
    songs = songs[(songs["genres"] != "") & (songs["popularity"] > 0)]
    songs = songs.sort_values(["popularity", "track_id"], ascending=[False, True])

    chosen: list[str] = []
    taken: set[str] = set()
    for genre in genres:
        in_genre = songs[songs["genres"].str.split("; ").map(lambda gs: genre in gs)]
        picks = [tid for tid in in_genre["track_id"] if tid not in taken][:per_genre]
        if len(picks) < per_genre:
            logger.warning("Genre %r has only %d songs available (wanted %d)", genre, len(picks), per_genre)
        chosen.extend(picks)
        taken.update(picks)

    sampled = songs.set_index("track_id").loc[chosen].reset_index()
    logger.info("Sampled %d songs across %d genres (%d per genre)", len(sampled), len(genres), per_genre)
    return sampled


# --- Vibe tags and descriptions ----------------------------------------------

def vibe_tags(song: pd.Series | dict) -> list[str]:
    """Translate audio features into plain words that a text embedding model understands.

    The thresholds are simple and documented so they can be explained (and criticized) in the model card.
    """
    tags: list[str] = []
    if song["energy"] < 0.4:
        tags += ["calm", "mellow", "chill"]
    elif song["energy"] > 0.7:
        tags += ["energetic", "intense"]
    if song["valence"] < 0.35:
        tags += ["sad", "moody"]
    elif song["valence"] > 0.65:
        tags += ["happy", "upbeat"]
    if song["acousticness"] > 0.6:
        tags += ["acoustic", "organic"]
    if song["instrumentalness"] > 0.5:
        tags += ["instrumental", "no vocals"]
    if song["danceability"] > 0.7:
        tags += ["danceable", "groovy"]
    if song["tempo"] < 90:
        tags.append("slow")
    elif song["tempo"] > 140:
        tags.append("fast")
    return tags


def describe(song: pd.Series | dict) -> str:
    """One sentence per song; this is the text that gets embedded for semantic search."""
    feels = song["vibe_tags"] or "neutral"
    return (
        f"{song['title']} by {song['artist']}. Genres: {song['genres'].replace('; ', ', ')}. "
        f"Album: {song['album']}. Feels: {feels}."
    )


def build_catalog(raw: pd.DataFrame, genres: Sequence[str] = SELECTED_GENRES, per_genre: int = DEFAULT_PER_GENRE) -> pd.DataFrame:
    """Turn the raw dataset into the cleaned, sampled catalog with vibe tags and descriptions."""
    sampled = sample_by_genre(merge_duplicates(raw), genres, per_genre)
    catalog = pd.DataFrame({
        "id": range(1, len(sampled) + 1),
        "track_id": sampled["track_id"],
        "title": sampled["track_name"].str.strip(),
        "artist": sampled["artists"].str.replace(";", ", ", regex=False).str.strip(),
        "album": sampled["album_name"].fillna("").str.strip(),
        "genres": sampled["genres"],
        "popularity": sampled["popularity"].astype(int),
        "explicit": sampled["explicit"].astype(bool),
        "duration_ms": sampled["duration_ms"].astype(int),
        "energy": sampled["energy"], "valence": sampled["valence"],
        "danceability": sampled["danceability"], "acousticness": sampled["acousticness"],
        "instrumentalness": sampled["instrumentalness"], "speechiness": sampled["speechiness"],
        "liveness": sampled["liveness"], "tempo": sampled["tempo"].round(1),
        "loudness": sampled["loudness"].round(2), "mode": sampled["mode"].astype(int),
    })
    catalog["vibe_tags"] = catalog.apply(lambda row: ", ".join(vibe_tags(row)), axis=1)
    catalog["description"] = catalog.apply(describe, axis=1)
    return catalog.reset_index(drop=True)


# --- Embeddings ---------------------------------------------------------------

def load_encoder(model_name: str | None = None) -> Encoder:
    """Load the static embedding model (downloaded once from Hugging Face, then cached)."""
    model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    try:
        from model2vec import StaticModel
    except ImportError as error:
        raise CatalogError("model2vec is not installed. Run: pip install -r requirements.txt") from error
    logger.info("Loading embedding model %s", model_name)
    try:
        return StaticModel.from_pretrained(model_name)
    except Exception as error:  # network errors, missing model, corrupt cache: all surface the same fix
        raise CatalogError(
            f"Couldn't load embedding model {model_name!r}: {error}. The first run needs internet access "
            "to download it from Hugging Face; later runs use the local cache."
        ) from error


def embed_texts(texts: Sequence[str], encoder: Encoder) -> np.ndarray:
    """Embed texts and L2-normalize the rows so cosine similarity is a plain dot product."""
    if not texts:
        raise CatalogError("Nothing to embed: the text list is empty.")
    vectors = np.asarray(encoder.encode(list(texts)), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise CatalogError(f"Encoder returned shape {vectors.shape} for {len(texts)} texts.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def catalog_fingerprint(path: Path = CATALOG_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- Save / load --------------------------------------------------------------

def save_catalog(catalog: pd.DataFrame, path: Path = CATALOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(path, index=False)
    logger.info("Wrote %d songs to %s", len(catalog), path)


def save_embeddings(vectors: np.ndarray, model_name: str, catalog_path: Path = CATALOG_PATH,
                    path: Path = EMBEDDINGS_PATH, meta_path: Path = META_PATH) -> None:
    np.save(path, vectors)
    meta = {
        "embedding_model": model_name,
        "catalog_sha256": catalog_fingerprint(catalog_path),
        "n_songs": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote embeddings %s to %s (model %s)", vectors.shape, path, model_name)


def load_catalog(path: Path = CATALOG_PATH) -> pd.DataFrame:
    """Load data/songs.csv and validate it, so bad data fails loudly instead of ranking oddly."""
    if not path.exists():
        raise CatalogError(f"Catalog not found at {path}. Build it with: python -m src.catalog build")
    catalog = pd.read_csv(path, keep_default_na=False)
    missing = [c for c in CATALOG_COLUMNS if c not in catalog.columns]
    if missing:
        raise CatalogError(f"Catalog {path} is missing columns {missing}. Rebuild with: python -m src.catalog build")
    if catalog.empty:
        raise CatalogError(f"Catalog {path} has no songs. Rebuild with: python -m src.catalog build")
    if catalog["track_id"].duplicated().any():
        raise CatalogError(f"Catalog {path} has duplicate track_ids. Rebuild with: python -m src.catalog build")
    for column in FEATURES_0_TO_1:
        values = pd.to_numeric(catalog[column], errors="coerce")
        if values.isna().any() or not values.between(0, 1).all():
            raise CatalogError(f"Catalog column {column!r} has values outside 0-1 or non-numbers. Rebuild the catalog.")
    logger.info("Loaded catalog: %d songs from %s", len(catalog), path)
    return catalog


def load_embeddings(catalog: pd.DataFrame, path: Path = EMBEDDINGS_PATH, meta_path: Path = META_PATH,
                    catalog_path: Path = CATALOG_PATH) -> tuple[np.ndarray, dict]:
    """Load the embedding matrix and check it matches the current catalog."""
    if not path.exists() or not meta_path.exists():
        raise CatalogError(f"Embeddings not found at {path}. Build them with: python -m src.catalog build")
    vectors = np.load(path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if vectors.shape[0] != len(catalog):
        raise CatalogError(
            f"Embeddings have {vectors.shape[0]} rows but the catalog has {len(catalog)} songs. "
            "Rebuild with: python -m src.catalog build"
        )
    if meta.get("catalog_sha256") != catalog_fingerprint(catalog_path):
        raise CatalogError("data/songs.csv changed since the embeddings were built. Rebuild with: python -m src.catalog build")
    return vectors, meta


# --- CLI ----------------------------------------------------------------------

def build(per_genre: int = DEFAULT_PER_GENRE, skip_embeddings: bool = False, force_download: bool = False) -> pd.DataFrame:
    download_raw(force=force_download)
    catalog = build_catalog(load_raw(), per_genre=per_genre)
    save_catalog(catalog)
    if skip_embeddings:
        logger.info("Skipping embeddings (--skip-embeddings)")
        return catalog
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    vectors = embed_texts(catalog["description"].tolist(), load_encoder(model_name))
    save_embeddings(vectors, model_name)
    return catalog


def main(argv: Sequence[str] | None = None) -> int:
    from src.log_config import setup_logging

    parser = argparse.ArgumentParser(prog="python -m src.catalog", description="Build the song catalog and embeddings.")
    sub = parser.add_subparsers(dest="command", required=True)
    build_cmd = sub.add_parser("build", help="Download the raw data if needed, then build data/songs.csv and embeddings")
    build_cmd.add_argument("--per-genre", type=int, default=DEFAULT_PER_GENRE, help="Songs to keep per genre (default %(default)s)")
    build_cmd.add_argument("--skip-embeddings", action="store_true", help="Only build the CSV")
    build_cmd.add_argument("--force-download", action="store_true", help="Download the raw dataset again")
    args = parser.parse_args(argv)

    setup_logging()
    try:
        catalog = build(args.per_genre, args.skip_embeddings, args.force_download)
    except CatalogError as error:
        logger.error("%s", error)
        return 1
    genre_counts = catalog["genres"].str.split("; ").explode().value_counts()
    logger.info("Done: %d songs. Songs per genre tag:\n%s", len(catalog), genre_counts.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
