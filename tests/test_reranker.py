"""Tests for the refine-round Gemini reranker (src/reranker.py): validation of Gemini's picks and fallbacks."""

import pytest

from src.llm import LLMQuotaError
from src.query import Range, SearchQuery
from src.ranker import Recommendation
from src.reranker import Pick, PickList, build_prompt, rerank
from tests.helpers import FakeLLM


def rec(song_id, artist=None, energy=0.3):
    return Recommendation(song_id=song_id, track_id=f"t{song_id}", title=f"Song {song_id}", artist=artist or f"Artist {song_id}",
                          genres="jazz", score=1.0, parts={}, reasons=[f"template {song_id}"],
                          features={"energy": energy, "valence": 0.5, "acousticness": 0.3, "instrumentalness": 0.0, "danceability": 0.5})


POOL = [rec(i) for i in range(1, 11)]
QUERY = SearchQuery(text="calm jazz", genres=["jazz"], k=3)


def picks(*items):
    return PickList(picks=[Pick(song_id=i, reason=r) for i, r in items])


def test_uses_gemini_choices_in_its_order():
    out = rerank(FakeLLM(PickList=picks((7, "like Moon River"), (2, "slow"), (9, "acoustic"))), QUERY, [], [], POOL, 3, {})
    assert [s.song_id for s in out.songs] == [7, 2, 9]
    assert out.reasons[7] == "like Moon River" and out.from_gemini == 3


def test_drops_invented_ids_and_fills_from_agent_ranking():
    out = rerank(FakeLLM(PickList=picks((999, "made up"), (5, "good"))), QUERY, [], [], POOL, 3, {})
    assert out.dropped_ids == [999]
    assert [s.song_id for s in out.songs] == [5, 1, 2]                   # gaps filled in pool order
    assert out.reasons[1] == "template 1" and out.from_gemini == 1


def test_skips_duplicates_and_second_song_by_same_artist():
    pool = [rec(1, "Same"), rec(2, "Same"), rec(3), rec(4)]
    out = rerank(FakeLLM(PickList=picks((1, "a"), (1, "again"), (2, "same artist"), (3, "b"))), QUERY, [], [], pool, 3, {})
    assert [s.song_id for s in out.songs] == [1, 3, 4]


def test_trims_long_reasons():
    out = rerank(FakeLLM(PickList=picks((1, "x" * 500))), QUERY, [], [], POOL, 1, {})
    assert len(out.reasons[1]) <= 200


def test_falls_back_to_agent_picks_on_error_or_offline():
    failed = rerank(FakeLLM(PickList=LLMQuotaError("quota")), QUERY, [], [], POOL, 3, {})
    assert [s.song_id for s in failed.songs] == [1, 2, 3] and "quota" in failed.fallback_reason
    offline = rerank(None, QUERY, [], [], POOL, 3, {})
    assert [s.song_id for s in offline.songs] == [1, 2, 3] and offline.fallback_reason


def test_prompt_lists_votes_and_candidates():
    prompt = build_prompt(QUERY, [rec(100)], [rec(200)], POOL[:2], {100: "calm"})
    assert "Song 100" in prompt and "Song 200" in prompt and '"song_id": 1' in prompt and "calm" in prompt


def test_agent_keeps_its_own_picks_when_gemini_picks_fail_the_self_check(retriever):
    """Gemini picks a loud metal song for a calm request: the self-check rejects it and the agent's picks stay."""
    from src.agent import PlaylistAgent

    loud = retriever.catalog[retriever.catalog["genres"] == "metal"]["id"].tolist()
    agent = PlaylistAgent(retriever, FakeLLM(PickList=picks(*[(i, "loud") for i in loud])))
    query = SearchQuery(text="calm music", energy=Range(0.0, 0.45), k=2)
    result = agent.search_round("calm music", query, "rules", [], [], feedback=([], []))
    assert not set(loud) & {s.song_id for s in result.songs}
    assert any(s.name == "rerank" and "failed the self-check" in s.detail for s in result.trace)
    assert result.explain_source == "template"
