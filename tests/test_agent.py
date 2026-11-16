"""Tests for Phase 3: planner, guardrails, self-check, explainer, the agent loop and Gemini error handling.

Gemini is always faked (tests/helpers.FakeLLM); conftest.py also blanks GEMINI_API_KEY.
"""

import pandas as pd
import pytest
from google.genai import errors

from src import llm as llm_module
from src.agent import PlaylistAgent
from src.checker import check, filter_to_targets
from src.explainer import ReasonList, SongReason, explain
from src.guardrails import apply_closest_genres, check_request, filter_to_catalog, looks_off_topic
from src.llm import GeminiClient, LLMError, LLMQuotaError, LLMUnavailable, get_client
from src.parser import parse_request
from src.planner import PlanModel, plan_request, plan_to_query
from src.query import Range, SearchQuery
from src.ranker import Recommendation
from tests.helpers import FakeLLM

KNOWN = ["indie", "jazz", "metal", "hip-hop", "pop"]


def rec(song_id, title="T", artist="A", genres="jazz", energy=0.3, valence=0.5):
    return Recommendation(song_id=song_id, track_id=f"t{song_id}", title=title, artist=artist, genres=genres,
                          score=1.0, parts={}, reasons=["genre: jazz"],
                          features={"energy": energy, "valence": valence, "acousticness": 0.3,
                                    "instrumentalness": 0.0, "danceability": 0.5})


# --- planner ---

def test_gemini_plan_is_validated_against_catalog():
    plan = PlanModel(on_topic=True, search_text="mellow late-night jazz", genres=["Jazz", "hip hop", "k-pop"],
                     exclude_genres=["metal"], moods=["Chill"], energy_min=0.0, energy_max=1.7, k=99)
    query = plan_to_query(plan, "req", set(KNOWN), k=5)
    assert query.genres == ["jazz", "hip-hop"]          # case and alias normalized
    assert query.unknown_genres == ["k-pop"]            # not in catalog -> moved to unknown
    assert "closest genre is 'pop'" in query.notes[0]
    assert query.energy == Range(0.0, 1.0)              # clamped
    assert query.k == 5                                 # out-of-range k ignored
    assert query.semantic_text() == "mellow late-night jazz"
    assert query.moods == ["chill"]


def test_plan_request_uses_gemini_when_available():
    fake = FakeLLM(PlanModel=PlanModel(on_topic=True, search_text="calm jazz", genres=["jazz"]))
    result = plan_request("calm jazz", KNOWN, fake)
    assert result.source == "gemini" and result.query.genres == ["jazz"]


def test_plan_request_falls_back_to_rules_on_llm_error():
    fake = FakeLLM(PlanModel=LLMQuotaError("quota reached"))
    result = plan_request("calm jazz", KNOWN, fake)
    assert result.source == "rules" and result.query.genres == ["jazz"]
    assert "quota" in result.fallback_reason


def test_plan_request_offline_uses_rules():
    result = plan_request("calm jazz", KNOWN, None)
    assert result.source == "rules" and "offline" in result.fallback_reason


def test_plan_request_off_topic():
    fake = FakeLLM(PlanModel=PlanModel(on_topic=False, off_topic_reason="homework request", search_text=""))
    result = plan_request("solve my math homework", KNOWN, fake)
    assert result.on_topic is False and result.off_topic_reason == "homework request"


# --- guardrails ---

def test_check_request_rejects_empty_and_long():
    assert not check_request("  ").ok
    assert not check_request("jazz " * 200).ok
    assert check_request("chill jazz").ok


def test_rule_based_off_topic_detection():
    assert looks_off_topic("write my essay about rome", parse_request("write my essay about rome"))
    assert not looks_off_topic("music to write my essay to", parse_request("music to write my essay to"))
    assert not looks_off_topic("chill jazz", parse_request("chill jazz"))


def test_apply_closest_genres_substitutes_and_tells_user():
    query = apply_closest_genres(parse_request("some k-pop"))
    assert query.genres == ["pop"]
    assert query.notes == ["Showing pop instead of k-pop, which isn't in the catalog."]   # one note, not two


def test_apply_closest_genres_keeps_known_genres():
    query = apply_closest_genres(parse_request("k-pop and jazz"))
    assert query.genres == ["jazz"]


def test_filter_to_catalog():
    assert filter_to_catalog([1, 99, 2], {1, 2}) == ([1, 2], [99])


# --- self-check ---

def test_check_passes_good_list():
    query = SearchQuery(text="x", genres=["jazz"], energy=Range(0.0, 0.45), k=2)
    assert check(query, [rec(1, artist="A"), rec(2, artist="B")]).passed


def test_check_reports_each_problem():
    query = SearchQuery(text="x", genres=["jazz", "indie"], exclude_genres=["metal"], energy=Range(0.0, 0.45), k=3)
    songs = [rec(1, artist="A", energy=0.9), rec(2, artist="A", genres="jazz; metal")]
    result = check(query, songs)
    assert result.codes() == {"TOO_FEW", "MISSING_GENRE", "EXCLUDED_GENRE", "ENERGY_OUT", "DUPLICATE_ARTIST"}
    energy_problem = next(p for p in result.problems if p.code == "ENERGY_OUT")
    assert energy_problem.song_ids == [1]


def test_check_tolerance():
    query = SearchQuery(text="x", energy=Range(0.0, 0.45), k=1)
    assert check(query, [rec(1, energy=0.49)]).passed        # within 0.05 tolerance
    assert not check(query, [rec(1, energy=0.51)]).passed


def test_filter_to_targets():
    df = pd.DataFrame({"energy": [0.1, 0.48, 0.6], "valence": [0.5, 0.5, 0.5]})
    query = SearchQuery(text="x", energy=Range(0.0, 0.45))
    assert len(filter_to_targets(df, query)) == 2
    assert len(filter_to_targets(df, query, widen=0.1)) == 3


# --- explainer ---

def test_explain_offline_uses_templates():
    result = explain(SearchQuery(text="x"), [rec(1)], {1: "calm"}, None)
    assert result.source == "template" and result.reasons[1] == "genre: jazz"


def test_explain_drops_invented_ids_and_fills_gaps():
    answer = ReasonList(reasons=[SongReason(song_id=1, reason="Mellow jazz for your night."),
                                 SongReason(song_id=999, reason="A song that doesn't exist.")])
    result = explain(SearchQuery(text="x"), [rec(1), rec(2)], {}, FakeLLM(ReasonList=answer))
    assert result.dropped_ids == [999]
    assert result.reasons == {1: "Mellow jazz for your night.", 2: "genre: jazz"}
    assert result.source == "gemini+template"


def test_explain_trims_long_reasons():
    answer = ReasonList(reasons=[SongReason(song_id=1, reason="x" * 500)])
    result = explain(SearchQuery(text="x"), [rec(1)], {}, FakeLLM(ReasonList=answer))
    assert len(result.reasons[1]) <= 200


def test_explain_falls_back_on_error():
    result = explain(SearchQuery(text="x"), [rec(1)], {}, FakeLLM(ReasonList=LLMError("boom")))
    assert result.source == "template" and "boom" in result.fallback_reason


# --- agent loop ---

def steps(result):
    return [s.name for s in result.trace]


def test_agent_happy_path_offline(retriever):
    result = PlaylistAgent(retriever).run("calm jazz and indie", k=4)
    assert result.status == "ok" and len(result.songs) == 4
    assert result.plan_source == "rules" and result.explain_source == "template"
    assert steps(result)[:4] == ["guardrail", "plan", "retrieve", "rank"]
    assert steps(result)[-2:] == ["explain", "present"]


def test_agent_retries_when_energy_is_off(monkeypatch, retriever):
    # The real ranker already prefers in-range songs, so force the first attempt to include a loud song
    # (as happens in the real catalog, e.g. "Light On" at energy 0.57 for a "chill" request).
    import src.agent as agent_module
    real_rank, calls = agent_module.rank, []

    def rank_with_loud_first(candidates, query, embeddings, k=None):
        calls.append(1)
        if len(calls) == 1:
            loud = candidates[candidates["genres"].str.contains("metal")].head(1)
            return real_rank(pd.concat([loud.assign(fused=5.0), candidates]), query, embeddings, k)
        return real_rank(candidates, query, embeddings, k)

    monkeypatch.setattr(agent_module, "rank", rank_with_loud_first)
    result = PlaylistAgent(retriever).run("calm music", k=2)
    assert len(result.attempts) == 2
    assert "ENERGY_OUT" in result.attempts[0].check.codes()
    assert result.attempts[1].check.passed and result.status == "ok"
    assert "adjust" in steps(result)
    assert all(s.features["energy"] <= 0.5 for s in result.songs)


def test_agent_reports_partial_when_constraints_cannot_be_met(retriever):
    result = PlaylistAgent(retriever).run("calm jazz", k=8)    # only 3 jazz songs by 2 artists
    assert result.status == "partial"
    assert "only" in result.message and result.songs
    assert len(result.attempts) <= 3


def test_agent_rejects_empty_request(retriever):
    result = PlaylistAgent(retriever).run("   ")
    assert result.status == "rejected" and steps(result) == ["guardrail"]


def test_agent_rejects_off_topic_offline(retriever):
    result = PlaylistAgent(retriever).run("write my essay on the french revolution")
    assert result.status == "rejected" and "music" in result.message


def test_agent_rejects_off_topic_with_gemini(retriever):
    fake = FakeLLM(PlanModel=PlanModel(on_topic=False, off_topic_reason="coding question", search_text=""))
    result = PlaylistAgent(retriever, fake).run("fix my python bug")
    assert result.status == "rejected" and "coding question" in result.message
    assert [c[0] for c in fake.calls] == ["PlanModel"]          # no explain call after rejection


def test_round_one_uses_gemini_plan_and_template_reasons(retriever):
    fake = FakeLLM(PlanModel=PlanModel(on_topic=True, search_text="calm jazz", genres=["jazz"],
                                       energy_min=0.0, energy_max=0.45))
    result = PlaylistAgent(retriever, fake).run("something for a quiet evening", k=2)
    assert result.plan_source == "gemini" and result.explain_source == "template"
    assert [c[0] for c in fake.calls] == ["PlanModel"]                # round 1 = one Gemini call


def test_optional_gemini_explainer_still_works(retriever):
    def reasons(prompt):
        import json
        ids = [s["song_id"] for s in json.loads(prompt.split("Songs:\n", 1)[1])]
        return ReasonList(reasons=[SongReason(song_id=i, reason=f"why {i}") for i in ids])

    agent = PlaylistAgent(retriever, FakeLLM(ReasonList=reasons))
    query = SearchQuery(text="calm jazz", genres=["jazz"], k=2)
    result = agent.search_round("calm jazz", query, "rules", [], [], explain_with_llm=True)
    assert result.explain_source == "gemini"
    assert all(result.reasons[s.song_id] == f"why {s.song_id}" for s in result.songs)


def test_agent_never_repeats_excluded_songs(retriever):
    first = PlaylistAgent(retriever).run("calm jazz and indie", k=2)
    seen = [s.song_id for s in first.songs]
    second = PlaylistAgent(retriever).run("calm jazz and indie", k=2, exclude_ids=seen)
    assert not set(seen) & {s.song_id for s in second.songs}


def test_agent_result_to_dict(retriever):
    data = PlaylistAgent(retriever).run("calm jazz", k=2).to_dict()
    assert data["status"] in ("ok", "partial") and data["attempts"][0]["song_ids"]


# --- Gemini client error handling (no network) ---

def test_get_client_without_key_is_offline():
    assert get_client() is None
    with pytest.raises(LLMUnavailable):
        GeminiClient()


class _RaisingModels:
    def __init__(self, error):
        self.error = error

    def generate_content(self, **kwargs):
        raise self.error


def _client_raising(error):
    client = object.__new__(GeminiClient)
    from google.genai import types
    client._types, client.model = types, "gemini-test"
    client._blocked, client._blocked_until = None, 0.0
    client._client = type("C", (), {"models": _RaisingModels(error)})()
    return client


@pytest.mark.parametrize("code, expected", [(429, LLMQuotaError), (403, LLMUnavailable), (404, LLMUnavailable), (400, LLMError)])
def test_client_errors_are_mapped(code, expected):
    error = errors.ClientError(code, {"error": {"code": code, "message": "nope", "status": "X"}})
    with pytest.raises(expected):
        _client_raising(error).generate_json("s", "p", PlanModel)


def test_unusable_model_is_not_called_again():
    calls = []

    class Models:
        def generate_content(self, **kwargs):
            calls.append(1)
            raise errors.ClientError(404, {"error": {"code": 404, "message": "no model", "status": "NOT_FOUND"}})

    client = _client_raising(None)
    client._client = type("C", (), {"models": Models()})()
    for _ in range(3):
        with pytest.raises(LLMUnavailable):
            client.generate_json("s", "p", PlanModel)
    assert len(calls) == 1          # planner failed once; explainer etc. fail fast without a request


def test_server_error_is_retried_once(monkeypatch):
    monkeypatch.setattr(llm_module, "RETRY_DELAY_SECONDS", 0)
    calls = []

    class Models:
        def generate_content(self, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise errors.ServerError(504, {"error": {"code": 504, "message": "deadline", "status": "X"}})
            return type("R", (), {"parsed": PlanModel(on_topic=True, search_text="ok"), "usage_metadata": None})()

    client = _client_raising(None)
    client._client = type("C", (), {"models": Models()})()
    assert client.generate_json("s", "p", PlanModel).search_text == "ok"
    assert len(calls) == 2


def test_server_and_network_errors_become_llm_error(monkeypatch):
    monkeypatch.setattr(llm_module, "RETRY_DELAY_SECONDS", 0)
    with pytest.raises(LLMError):
        _client_raising(errors.ServerError(503, {"error": {"code": 503, "message": "busy", "status": "X"}})).generate_json("s", "p", PlanModel)
    with pytest.raises(LLMError):
        _client_raising(TimeoutError("slow")).generate_json("s", "p", PlanModel)


def test_tests_never_see_a_real_key():
    assert llm_module.os.getenv("GEMINI_API_KEY") == ""
