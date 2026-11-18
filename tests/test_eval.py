"""Golden-prompt evaluation harness (eval/run_eval.py), on the tiny test catalog with a fake Gemini."""

import json

import pytest

from eval.run_eval import (
    GoldenSetError, PacedLLM, evaluate_prompt, jaccard, load_golden, plan_mismatches, song_matches,
    summarize, ungrounded_details,
)
from src.agent import PlaylistAgent
from src.llm import LLMQuotaError
from src.planner import PlanModel
from src.ranker import Recommendation
from src.reranker import Pick, PickList
from tests.helpers import FakeLLM

GENRES = {"jazz", "indie", "metal", "pop", "folk", "indie-pop"}


def rec(song_id=1, genres="jazz", energy=0.3, valence=0.5, acousticness=0.3, instrumentalness=0.0, title="T"):
    return Recommendation(song_id=song_id, track_id="t", title=title, artist=f"A{song_id}", genres=genres,
                          score=1.0, parts={}, features={"energy": energy, "valence": valence,
                                                         "acousticness": acousticness,
                                                         "instrumentalness": instrumentalness})


def test_golden_file_loads_and_covers_both_kinds_of_prompt():
    prompts = load_golden()
    assert len(prompts) >= 15
    assert any(p.get("on_topic") is False for p in prompts)
    assert all("songs" in p for p in prompts if p.get("on_topic", True))


def test_load_golden_rejects_unknown_rules(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"prompts": [{"id": "a", "request": "x", "songs": {"tempo": [1, 2]}}]}))
    with pytest.raises(GoldenSetError, match="unknown rule"):
        load_golden(path)


def test_load_golden_missing_file(tmp_path):
    with pytest.raises(GoldenSetError, match="not found"):
        load_golden(tmp_path / "nope.json")


def test_song_matches_every_rule_with_tolerance():
    rules = {"genres": ["jazz", "indie"], "energy": [0.0, 0.45], "exclude_genres": ["metal"]}
    assert song_matches(rec(genres="jazz", energy=0.48), rules)          # inside the 0.05 tolerance
    assert not song_matches(rec(genres="jazz", energy=0.6), rules)
    assert not song_matches(rec(genres="pop", energy=0.3), rules)
    assert not song_matches(rec(genres="jazz; metal", energy=0.3), rules)
    assert song_matches(rec(acousticness=0.7), {"acoustic": True})
    assert not song_matches(rec(acousticness=0.2), {"acoustic": True})


def test_jaccard():
    assert jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)


def test_plan_mismatches_lists_what_is_wrong(retriever):
    agent = PlaylistAgent(retriever)
    result = agent.run("intense metal")
    expected = {"plan": {"genres_all": ["metal"], "energy": "low"}}
    assert plan_mismatches(result, expected) == ["energy high, expected low"]
    assert plan_mismatches(result, {"plan": {"genres_all": ["metal"], "energy": "high"}}) == []
    assert plan_mismatches(result, {"on_topic": False}) == ["should have been rejected as off-topic"]


def test_off_topic_prompt_is_correct_when_rejected(retriever):
    agent = PlaylistAgent(retriever)
    result = evaluate_prompt({"id": "essay", "request": "write my history essay", "on_topic": False},
                             agent, agent, runs=2, catalog_ids={1}, catalog_genres=GENRES)
    assert result.status == "rejected" and result.plan_problems == []
    assert result.precision is None


def test_ungrounded_details_flags_details_not_in_the_facts():
    song = rec(genres="jazz", title="Moon River")
    context = "something chill with jazz and indie Moon River Andy jazz calm"
    assert ungrounded_details("Like Moon River, a calm jazz song.", song, context, GENRES) == []
    flags = ungrounded_details("Smooth vocals and a 1960s guitar riff.", song, context, GENRES)
    assert {"vocals", "1960s", "guitar", "riff"} <= set(flags)
    assert any("indie-jazz" in f for f in ungrounded_details("A mellow indie-jazz tune.", song, context, GENRES))
    assert any("metal" in f for f in ungrounded_details("A metal track.", song, context, GENRES))
    # A word that is in the facts (the song's genre here) is allowed.
    assert ungrounded_details("Soft piano.", rec(genres="piano"), "piano", GENRES | {"piano"}) == []


def test_offline_prompt_scores_and_refines(retriever):
    agent = PlaylistAgent(retriever)
    item = {"id": "jazz", "request": "calm jazz", "plan": {"genres_all": ["jazz"], "energy": "low"},
            "songs": {"genres": ["jazz", "indie"], "energy": [0.0, 0.45]}}
    ids = set(retriever.catalog["id"].astype(int))
    result = evaluate_prompt(item, agent, agent, runs=3, catalog_ids=ids, catalog_genres=GENRES)
    assert result.plan_sources == ["rules"] * 3
    assert result.plan_problems == []
    assert result.consistency == 1.0 and result.made_up_ids == []
    assert result.precision is not None and result.refine_precision is not None
    summary = summarize([result], "rules")
    assert summary["fallbacks"] == 0 and summary["planner_accuracy"] == 1.0


def test_gemini_prompt_compares_reranker_with_agent_and_checks_reasons(retriever):
    plan = PlanModel(on_topic=True, search_text="calm jazz", genres=["jazz", "indie"], energy_max=0.45)

    def pick_all(prompt):
        candidates = json.loads(prompt.split("choose from these only): ")[1])
        picks = [Pick(song_id=c["song_id"], reason=f"Like the liked songs, {c['title']} has smooth vocals.")
                 for c in candidates] + [Pick(song_id=999, reason="made up")]
        return PickList(picks=picks)

    llm = FakeLLM(PlanModel=plan, PickList=pick_all)
    agent, offline = PlaylistAgent(retriever, llm=llm), PlaylistAgent(retriever)
    item = {"id": "jazz", "request": "calm jazz and indie", "plan": {"genres_all": ["jazz"]},
            "songs": {"genres": ["jazz"], "energy": [0.0, 0.45]}}
    ids = set(retriever.catalog["id"].astype(int))
    result = evaluate_prompt(item, agent, offline, runs=2, catalog_ids=ids, catalog_genres=GENRES)
    assert result.plan_sources == ["gemini", "gemini"]
    assert result.made_up_ids == []
    # The tiny catalog has too few candidates for the reranker, so the agent's own picks are shown;
    # either way the comparison against the agent-only copy is recorded.
    assert result.refine_agent_precision is not None
    assert summarize([result], "gemini")["fallbacks"] == 0


def test_gemini_fallback_runs_are_left_out(retriever):
    agent = PlaylistAgent(retriever, llm=FakeLLM())       # every Gemini call fails -> rules planner
    item = {"id": "jazz", "request": "calm jazz", "songs": {"genres": ["jazz"]}}
    result = evaluate_prompt(item, agent, PlaylistAgent(retriever), runs=1, catalog_ids={1}, catalog_genres=GENRES)
    summary = summarize([result], "gemini")
    assert summary["fallbacks"] == 1 and summary["scored"] == 0 and summary["precision_at_5"] is None


def test_paced_llm_waits_out_a_quota_error(monkeypatch):
    sleeps = []
    monkeypatch.setattr("eval.run_eval.time.sleep", sleeps.append)
    calls = iter([LLMQuotaError("429"), "ok"])

    class Flaky:
        model = "m"

        def generate_json(self, *args, **kwargs):
            value = next(calls)
            if isinstance(value, Exception):
                raise value
            return value

    paced = PacedLLM(Flaky(), delay=0, quota_wait_seconds=5)
    assert paced.generate_json("s", "p", PlanModel) == "ok"
    assert paced.calls == 2 and paced.failures == 0 and 5 in sleeps


def test_paced_llm_gives_up_after_the_last_wait(monkeypatch):
    monkeypatch.setattr("eval.run_eval.time.sleep", lambda s: None)

    class AlwaysQuota:
        model = "m"

        def generate_json(self, *args, **kwargs):
            raise LLMQuotaError("429")

    paced = PacedLLM(AlwaysQuota(), delay=0, quota_waits=1)
    with pytest.raises(LLMQuotaError):
        paced.generate_json("s", "p", PlanModel)
    assert paced.calls == 2 and paced.failures == 1


def _saved_report(retriever, runs=3):
    agent = PlaylistAgent(retriever)
    item = {"id": "jazz", "request": "calm jazz", "plan": {"genres_all": ["jazz"]},
            "songs": {"genres": ["jazz"], "energy": [0.0, 0.45]}}
    ids = set(retriever.catalog["id"].astype(int))
    result = evaluate_prompt(item, agent, agent, runs=runs, catalog_ids=ids, catalog_genres=GENRES)
    return item, {"runs_per_prompt": runs, "prompts": {"rules": [result.__dict__]}}


def test_rescore_uses_only_the_first_runs_and_needs_no_planner(retriever):
    from eval.run_eval import rescore

    item, report = _saved_report(retriever)
    report["prompts"]["rules"][0]["song_ids"][1] = [6, 7]          # pretend run 2 returned two metal songs
    results = rescore(report, [item], retriever.catalog, runs=2)
    p = results["rules"][0]
    assert len(p.song_ids) == 2 and len(p.precisions) == 2
    assert p.precisions[1] == 0.0 and p.precision == pytest.approx(p.precisions[0] / 2)
    assert p.consistency == 0.0


def test_rescore_refuses_more_runs_than_were_saved(retriever):
    from eval.run_eval import rescore

    item, report = _saved_report(retriever, runs=2)
    with pytest.raises(GoldenSetError, match="needs a new run"):
        rescore(report, [item], retriever.catalog, runs=3)
