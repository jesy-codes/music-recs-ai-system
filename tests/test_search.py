"""Tests for the retriever, ranker, recommend() and the CLI, using a tiny catalog and a fake encoder."""

import pandas as pd
import pytest

from src import cli
from src.catalog import CatalogError
from src.parser import parse_request
from src.query import Range, SearchQuery
from src.ranker import feature_fit, rank, select_mmr, score_candidates
from src.recommend import recommend
from src.retriever import Retriever, tokenize


# The `retriever` fixture (8-song catalog, fake encoder) lives in tests/conftest.py.


def test_tokenize_keeps_hyphenated_genres_and_drops_filler():
    assert tokenize("Some indie-pop songs for ME") == ["indie-pop"]


def test_search_ranks_matching_genre_first(retriever):
    candidates = retriever.search(parse_request("calm jazz"))
    assert "jazz" in candidates.iloc[0]["genres"]
    assert candidates["fused"].max() == pytest.approx(1.0)


def test_search_respects_exclude_ids_and_genres(retriever):
    query = parse_request("jazz but no metal")
    candidates = retriever.search(query, exclude_ids=[1, 3])
    assert not set(candidates["id"]) & {1, 3}
    assert not candidates["genres"].str.contains("metal").any()


def test_search_with_everything_excluded_returns_empty(retriever):
    candidates = retriever.search(parse_request("jazz"), exclude_ids=range(1, 9))
    assert candidates.empty
    assert rank(candidates, parse_request("jazz"), retriever.embeddings) == []


def test_retriever_rejects_mismatched_embeddings(retriever):
    with pytest.raises(CatalogError):
        Retriever(retriever.catalog, retriever.embeddings[:3], retriever.encoder)


def test_feature_fit():
    query = SearchQuery(text="x", energy=Range(0.0, 0.4), acoustic=True)
    assert feature_fit(pd.Series({"energy": 0.3, "valence": 0.5, "acousticness": 0.8, "instrumentalness": 0}), query) == pytest.approx(0.9)
    assert feature_fit(pd.Series({"energy": 0.3}), SearchQuery(text="x")) == 1.0   # no targets -> neutral


def test_rank_balances_requested_genres_and_caps_artists(retriever):
    query = parse_request("calm jazz and indie")
    songs = rank(retriever.search(query), query, retriever.embeddings, k=4)
    assert len(songs) == 4
    genres = [set(s.genres.split("; ")) for s in songs]
    assert sum("jazz" in g for g in genres) >= 2 and sum("indie" in g for g in genres) >= 2
    artists = [s.artist for s in songs]
    assert len(artists) == len(set(artists))                   # max one song per artist
    assert all(s.reasons for s in songs)


def test_rank_respects_energy_target(retriever):
    query = parse_request("calm music")
    songs = rank(retriever.search(query), query, retriever.embeddings, k=3)
    assert all(s.features["energy"] <= 0.45 for s in songs)


def test_select_mmr_returns_fewer_when_pool_is_small(retriever):
    query = parse_request("metal")
    scored = score_candidates(retriever.search(query, n_candidates=2), query)
    assert len(select_mmr(scored, retriever.embeddings, query, k=5)) == 2


def test_recommend_round_excludes_seen_songs(retriever):
    first = recommend("calm jazz", retriever, k=2)
    seen = [s.song_id for s in first.songs]
    second = recommend("calm jazz", retriever, k=2, exclude_ids=seen)
    assert not set(seen) & {s.song_id for s in second.songs}


def test_cli_prints_songs(monkeypatch, capsys, retriever):
    monkeypatch.setattr(cli.Retriever, "load", classmethod(lambda cls: retriever))
    assert cli.main(["calm", "jazz", "-k", "2", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "Plan (rules):" in out and "open.spotify.com/track/" in out


def test_cli_json_output(monkeypatch, capsys, retriever):
    import json
    monkeypatch.setattr(cli.Retriever, "load", classmethod(lambda cls: retriever))
    assert cli.main(["calm jazz", "--json", "-k", "2", "--offline"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["songs"]) == 2 and payload["query"]["genres"] == ["jazz"]


def test_cli_missing_catalog_exits_1(monkeypatch, capsys):
    def fail(cls):
        raise CatalogError("Catalog not found. Build it with: python -m src.catalog build")
    monkeypatch.setattr(cli.Retriever, "load", classmethod(fail))
    assert cli.main(["jazz"]) == 1
    assert "python -m src.catalog build" in capsys.readouterr().err


def test_cli_blank_request_exits_2(monkeypatch, capsys, retriever):
    monkeypatch.setattr(cli.Retriever, "load", classmethod(lambda cls: retriever))
    assert cli.main(["  ", "--offline"]) == 2
    assert "empty" in capsys.readouterr().out
