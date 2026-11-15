"""Tests for src/catalog.py. They use small in-memory data and a fake encoder: no network, no model download."""

import numpy as np
import pandas as pd
import pytest

from src import catalog as cat
from src.catalog import CatalogError


def raw_row(track_id, title, artist, genre, popularity=50, **features):
    row = {
        "track_id": track_id, "artists": artist, "album_name": "Album", "track_name": title,
        "popularity": popularity, "duration_ms": 200_000, "explicit": False,
        "danceability": 0.5, "energy": 0.5, "key": 0, "loudness": -8.0, "mode": 1,
        "speechiness": 0.05, "acousticness": 0.3, "instrumentalness": 0.0, "liveness": 0.1,
        "valence": 0.5, "tempo": 110.0, "time_signature": 4, "track_genre": genre,
    }
    row.update(features)
    return row


@pytest.fixture
def raw():
    return pd.DataFrame([
        raw_row("a", "Blue Song", "Miles", "jazz", popularity=80, energy=0.2, acousticness=0.9),
        raw_row("a", "Blue Song", "Miles", "chill", popularity=80, energy=0.2, acousticness=0.9),  # same id, other genre
        raw_row("b", "Blue Song", "Miles", "sad", popularity=30),                                    # same song, other id
        raw_row("c", "Loud Song", "Metal Band", "metal", popularity=70, energy=0.95),
        raw_row("d", "K Song", "K Band", "k-pop", popularity=90),                                    # genre not selected
        raw_row("e", "Indie Song", "Indie Kid", "indie", popularity=60),
        raw_row("f", "Unknown Song", "Nobody", "indie", popularity=0),                               # popularity 0 dropped
    ])


class FakeEncoder:
    def encode(self, sentences):
        return np.array([[len(s), 1.0, 0.0] for s in sentences], dtype=np.float32)


# --- vibe tags and descriptions ---

def test_vibe_tags_low_energy_acoustic_slow():
    tags = cat.vibe_tags({"energy": 0.2, "valence": 0.2, "acousticness": 0.9, "instrumentalness": 0.8,
                          "danceability": 0.3, "tempo": 70})
    assert tags == ["calm", "mellow", "chill", "sad", "moody", "acoustic", "organic", "instrumental", "no vocals", "slow"]


def test_vibe_tags_high_energy_happy_fast():
    tags = cat.vibe_tags({"energy": 0.9, "valence": 0.9, "acousticness": 0.1, "instrumentalness": 0.0,
                          "danceability": 0.8, "tempo": 150})
    assert tags == ["energetic", "intense", "happy", "upbeat", "danceable", "groovy", "fast"]


def test_vibe_tags_middle_values_get_no_tags():
    assert cat.vibe_tags({"energy": 0.5, "valence": 0.5, "acousticness": 0.3, "instrumentalness": 0.1,
                          "danceability": 0.5, "tempo": 110}) == []


def test_describe_includes_title_artist_genres_and_feel():
    text = cat.describe({"title": "T", "artist": "A", "genres": "jazz; chill", "album": "Al", "vibe_tags": "calm"})
    assert text == "T by A. Genres: jazz, chill. Album: Al. Feels: calm."


def test_describe_without_tags_says_neutral():
    assert "Feels: neutral." in cat.describe({"title": "T", "artist": "A", "genres": "rock", "album": "", "vibe_tags": ""})


# --- cleaning ---

def test_merge_duplicates_combines_genres_and_keeps_most_popular(raw):
    merged = cat.merge_duplicates(raw)
    blue = merged[merged["track_name"] == "Blue Song"]
    assert len(blue) == 1
    assert blue.iloc[0]["track_id"] == "a"            # most popular version kept
    assert blue.iloc[0]["genres"] == "chill; jazz; sad"


def test_sample_by_genre_keeps_only_selected_genres_and_skips_zero_popularity(raw):
    sampled = cat.sample_by_genre(cat.merge_duplicates(raw), genres=("jazz", "chill", "indie", "metal"), per_genre=5)
    assert set(sampled["track_id"]) == {"a", "c", "e"}
    assert sampled.set_index("track_id").loc["a", "genres"] == "chill; jazz"   # "sad" not selected, so removed
    assert not sampled["genres"].str.contains("k-pop").any()


def test_sample_by_genre_caps_per_genre_without_repeats():
    raw = pd.DataFrame([raw_row(f"j{i}", f"Song {i}", "X", "jazz", popularity=100 - i) for i in range(10)])
    sampled = cat.sample_by_genre(cat.merge_duplicates(raw), genres=("jazz",), per_genre=3)
    assert list(sampled["track_id"]) == ["j0", "j1", "j2"]   # the 3 most popular, in order


def test_sample_by_genre_rejects_non_positive_size(raw):
    with pytest.raises(CatalogError):
        cat.sample_by_genre(cat.merge_duplicates(raw), per_genre=0)


def test_build_catalog_columns_and_ids(raw):
    catalog = cat.build_catalog(raw, genres=("jazz", "chill", "indie", "metal"), per_genre=5)
    assert tuple(catalog.columns) == cat.CATALOG_COLUMNS
    assert list(catalog["id"]) == [1, 2, 3]
    blue = catalog[catalog["title"] == "Blue Song"].iloc[0]
    assert "calm" in blue["vibe_tags"] and "acoustic" in blue["vibe_tags"]
    assert blue["description"].startswith("Blue Song by Miles.")


# --- embeddings ---

def test_embed_texts_returns_unit_vectors():
    vectors = cat.embed_texts(["a", "bbb"], FakeEncoder())
    assert vectors.shape == (2, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_embed_texts_rejects_empty_input():
    with pytest.raises(CatalogError):
        cat.embed_texts([], FakeEncoder())


def test_embed_texts_rejects_wrong_shape():
    class Broken:
        def encode(self, sentences):
            return np.zeros((1, 3))
    with pytest.raises(CatalogError):
        cat.embed_texts(["a", "b"], Broken())


# --- save / load ---

@pytest.fixture
def built(tmp_path, raw):
    catalog = cat.build_catalog(raw, genres=("jazz", "chill", "indie", "metal"), per_genre=5)
    paths = {"csv": tmp_path / "songs.csv", "npy": tmp_path / "emb.npy", "meta": tmp_path / "meta.json"}
    cat.save_catalog(catalog, paths["csv"])
    vectors = cat.embed_texts(catalog["description"].tolist(), FakeEncoder())
    cat.save_embeddings(vectors, "fake-model", catalog_path=paths["csv"], path=paths["npy"], meta_path=paths["meta"])
    return catalog, paths


def test_save_and_load_round_trip(built):
    catalog, paths = built
    loaded = cat.load_catalog(paths["csv"])
    assert list(loaded["track_id"]) == list(catalog["track_id"])
    vectors, meta = cat.load_embeddings(loaded, paths["npy"], paths["meta"], paths["csv"])
    assert vectors.shape[0] == len(loaded)
    assert meta["embedding_model"] == "fake-model"


def test_load_catalog_missing_file_says_how_to_build(tmp_path):
    with pytest.raises(CatalogError, match="python -m src.catalog build"):
        cat.load_catalog(tmp_path / "nope.csv")


def test_load_catalog_rejects_missing_columns(tmp_path):
    path = tmp_path / "songs.csv"
    pd.DataFrame({"id": [1], "title": ["x"]}).to_csv(path, index=False)
    with pytest.raises(CatalogError, match="missing columns"):
        cat.load_catalog(path)


def test_load_catalog_rejects_out_of_range_features(built):
    catalog, paths = built
    catalog.loc[0, "energy"] = 1.5
    catalog.to_csv(paths["csv"], index=False)
    with pytest.raises(CatalogError, match="energy"):
        cat.load_catalog(paths["csv"])


def test_load_embeddings_detects_changed_catalog(built):
    catalog, paths = built
    catalog.loc[0, "title"] = "Edited"
    catalog.to_csv(paths["csv"], index=False)
    with pytest.raises(CatalogError, match="changed since the embeddings were built"):
        cat.load_embeddings(catalog, paths["npy"], paths["meta"], paths["csv"])


def test_load_embeddings_detects_row_count_mismatch(built):
    catalog, paths = built
    with pytest.raises(CatalogError, match="rows"):
        cat.load_embeddings(catalog.iloc[:1], paths["npy"], paths["meta"], paths["csv"])


# --- raw data ---

def test_download_raw_skips_existing_file(tmp_path):
    dest = tmp_path / "raw.csv"
    dest.write_text("already here")
    assert cat.download_raw(url="http://invalid.invalid/x.csv", dest=dest) == dest
    assert dest.read_text() == "already here"


def test_download_raw_failure_raises_catalog_error(tmp_path):
    with pytest.raises(CatalogError, match="Couldn't download"):
        cat.download_raw(url="http://invalid.invalid/x.csv", dest=tmp_path / "raw.csv")
    assert not (tmp_path / "raw.part").exists()


def test_load_raw_rejects_wrong_dataset(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    with pytest.raises(CatalogError, match="missing columns"):
        cat.load_raw(path)
