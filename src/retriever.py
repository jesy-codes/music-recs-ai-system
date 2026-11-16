"""Hybrid retrieval over the song catalog: embeddings + BM25 keywords, fused by rank.

- Semantic search: cosine similarity between the request and each song's description embedding. Good at
  vibes ("rainy day", "late night") that don't appear word for word.
- Keyword search: BM25 over the same descriptions. Good at exact words (genre names, artists) that
  embeddings can blur.
- Reciprocal rank fusion (RRF) combines the two rankings without having to calibrate their score scales.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.catalog import CatalogError, Encoder, embed_texts, load_catalog, load_embeddings, load_encoder
from src.query import SearchQuery

logger = logging.getLogger(__name__)

RRF_K = 60            # standard RRF constant: dampens the gap between rank 1 and rank 2
DEFAULT_CANDIDATES = 200

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and the of to for with by in on at is it me my i some something music songs song tracks track "
    "play give want like recommend please feels genres album".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping hyphenated genres ("indie-pop") whole and dropping filler words."""
    return [t for t in TOKEN_PATTERN.findall(text.lower()) if t not in STOPWORDS and not t.isdigit()]


class Retriever:
    """Holds the catalog, its embeddings, the encoder and a BM25 index; answers candidate searches."""

    def __init__(self, catalog: pd.DataFrame, embeddings: np.ndarray, encoder: Encoder):
        if len(catalog) != embeddings.shape[0]:
            raise CatalogError(f"Catalog has {len(catalog)} songs but embeddings have {embeddings.shape[0]} rows.")
        self.catalog = catalog.reset_index(drop=True)
        self.embeddings = embeddings
        self.encoder = encoder
        self.song_genres = [set(g.split("; ")) for g in self.catalog["genres"]]
        self.known_genres = sorted(set().union(*self.song_genres))
        self.bm25 = BM25Okapi([tokenize(d) for d in self.catalog["description"]])
        self.row_by_id = {int(song_id): row for row, song_id in enumerate(self.catalog["id"])}
        logger.info("Retriever ready: %d songs, %d genres", len(self.catalog), len(self.known_genres))

    @classmethod
    def load(cls) -> "Retriever":
        """Load the committed catalog, embeddings and the embedding model they were built with."""
        catalog = load_catalog()
        embeddings, meta = load_embeddings(catalog)
        return cls(catalog, embeddings, load_encoder(meta["embedding_model"]))

    def embed_query(self, text: str) -> np.ndarray:
        return embed_texts([text], self.encoder)[0]

    def semantic_scores(self, text: str, query_vector: Optional[np.ndarray] = None) -> np.ndarray:
        """Cosine similarity of every song to the request (or to a given vector, e.g. after feedback)."""
        if query_vector is None:
            query_vector = self.embed_query(text)
        return self.embeddings @ query_vector

    def keyword_scores(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        if not tokens:
            return np.zeros(len(self.catalog))
        return np.asarray(self.bm25.get_scores(tokens), dtype=float)

    def search(self, query: SearchQuery, n_candidates: int = DEFAULT_CANDIDATES,
               exclude_ids: Optional[Iterable[int]] = None,
               query_vector: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Return up to n_candidates songs ranked by fused score, skipping excluded songs and genres.

        `query_vector` replaces the embedding of the request text; the feedback loop passes a vector
        moved toward liked songs and away from rejected ones.

        The result keeps the catalog columns plus: row (index into the embeddings), semantic, keyword,
        fused (RRF, scaled so the best candidate is 1.0).
        """
        semantic = self.semantic_scores(query.semantic_text(), query_vector)
        keyword = self.keyword_scores(query.keyword_text())

        sem_rank = np.empty(len(semantic), dtype=int)
        sem_rank[np.argsort(-semantic, kind="stable")] = np.arange(1, len(semantic) + 1)
        kw_rank = np.empty(len(keyword), dtype=int)
        kw_rank[np.argsort(-keyword, kind="stable")] = np.arange(1, len(keyword) + 1)
        fused = 1.0 / (RRF_K + sem_rank) + np.where(keyword > 0, 1.0 / (RRF_K + kw_rank), 0.0)

        allowed = np.ones(len(self.catalog), dtype=bool)
        for song_id in exclude_ids or ():
            row = self.row_by_id.get(int(song_id))
            if row is not None:
                allowed[row] = False
        if query.exclude_genres:
            banned = set(query.exclude_genres)
            allowed &= np.array([not (genres & banned) for genres in self.song_genres])

        order = [row for row in np.argsort(-fused, kind="stable") if allowed[row]][:n_candidates]
        if not order:
            logger.warning("No candidates left after exclusions for %r", query.text)
            return self.catalog.iloc[0:0].assign(row=[], semantic=[], keyword=[], fused=[])

        candidates = self.catalog.iloc[order].copy()
        candidates["row"] = order
        candidates["semantic"] = semantic[order]
        candidates["keyword"] = keyword[order]
        candidates["fused"] = fused[order] / fused[order].max()
        logger.info(
            "Retrieved %d candidates for %r (excluded %d songs by id/genre)",
            len(candidates), query.text, int((~allowed).sum()),
        )
        return candidates.reset_index(drop=True)
