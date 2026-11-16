"""Shared test data: a tiny catalog, a deterministic fake encoder and a fake Gemini client."""

import numpy as np
import pandas as pd

from src.llm import LLMError
from src.retriever import Retriever

VOCAB = ["jazz", "indie", "metal", "calm", "intense", "acoustic"]


class BagOfWordsEncoder:
    """Deterministic fake: one dimension per vocabulary word, plus a constant so no vector is zero."""

    def encode(self, sentences):
        return np.array([[s.lower().count(w) for w in VOCAB] + [0.1] for s in sentences], dtype=np.float32)


def song(i, title, artist, genres, energy, valence=0.5, acousticness=0.3, instrumentalness=0.0, tags=""):
    return {
        "id": i, "track_id": f"t{i}", "title": title, "artist": artist, "album": "A", "genres": genres,
        "popularity": 50, "explicit": False, "duration_ms": 1, "energy": energy, "valence": valence,
        "danceability": 0.5, "acousticness": acousticness, "instrumentalness": instrumentalness,
        "speechiness": 0.05, "liveness": 0.1, "tempo": 100.0, "loudness": -8.0, "mode": 1,
        "vibe_tags": tags, "description": f"{title} by {artist}. Genres: {genres.replace('; ', ', ')}. Feels: {tags}.",
    }


SONGS = [
    song(1, "Blue Calm", "Miles", "jazz", 0.2, tags="calm"),
    song(2, "Night Calm", "Miles", "jazz", 0.25, tags="calm"),          # same artist as 1
    song(3, "Soft Jazz", "Ella", "jazz", 0.3, tags="calm"),
    song(4, "Quiet Indie", "Phoebe", "indie", 0.3, acousticness=0.9, tags="calm, acoustic"),
    song(5, "Indie Calm", "Bon", "indie", 0.35, tags="calm"),
    song(6, "Loud Metal", "Iron", "metal", 0.95, tags="intense"),
    song(7, "More Metal", "Steel", "metal", 0.9, tags="intense"),
    song(8, "Jazz Indie", "Norah", "indie; jazz", 0.3, tags="calm"),
]


def make_retriever(songs=SONGS) -> Retriever:
    catalog = pd.DataFrame(songs)
    encoder = BagOfWordsEncoder()
    vectors = encoder.encode(catalog["description"].tolist())
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return Retriever(catalog, vectors, encoder)


class FakeLLM:
    """Stands in for GeminiClient. `responses` maps schema class name -> object, callable, or exception."""

    model = "fake-model"

    def __init__(self, **responses):
        self.responses = responses
        self.calls = []

    def generate_json(self, system, prompt, schema, temperature=0.0):
        self.calls.append((schema.__name__, prompt))
        response = self.responses.get(schema.__name__)
        if response is None:
            raise LLMError(f"no fake response for {schema.__name__}")
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(prompt)
        return response
