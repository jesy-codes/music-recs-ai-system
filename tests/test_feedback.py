"""Tests for Phase 4: feedback math, the listening session, CLI voting and the simulator (all offline)."""

import numpy as np
import pytest

from src import cli
from src.agent import PlaylistAgent
from src.feedback import adapt_genres, apply_feedback, rocchio, shift_range
from src.query import Range, SearchQuery
from src.ranker import Recommendation
from src.session import PlaylistSession, SessionError
from src.simulator import Persona, simulate
from tests.helpers import FakeLLM


def rec(song_id, genres="jazz", energy=0.3, valence=0.5):
    return Recommendation(song_id=song_id, track_id=f"t{song_id}", title=f"S{song_id}", artist=f"A{song_id}",
                          genres=genres, score=1.0, parts={}, reasons=["r"],
                          features={"energy": energy, "valence": valence, "acousticness": 0.3,
                                    "instrumentalness": 0.0, "danceability": 0.5})


# --- feedback math ---

def test_rocchio_moves_toward_liked_and_away_from_rejected():
    original, liked, rejected = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    vector = rocchio(original, [liked], [rejected])
    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert vector @ liked > original @ liked
    assert vector @ rejected < 0


def test_rocchio_without_votes_returns_original_direction():
    original = np.array([3.0, 4.0])
    assert np.allclose(rocchio(original, [], []), original / 5)


def test_shift_range_moves_halfway_toward_liked_mean():
    assert shift_range(Range(0.0, 0.4), [0.6, 0.6]) == Range(0.2, 0.6)   # center 0.2 -> 0.4, width kept


def test_shift_range_creates_range_only_with_two_likes():
    assert shift_range(None, [0.3]) is None
    assert shift_range(None, [0.3, 0.5]) == Range(0.2, 0.6)
    assert shift_range(Range(0.0, 0.4), []) == Range(0.0, 0.4)


def test_adapt_genres_drops_fully_rejected_and_learns_shared_likes():
    liked = [rec(1, "jazz; blues"), rec(2, "blues"), rec(3, "jazz")]
    rejected = [rec(4, "indie"), rec(5, "indie; pop")]
    genres, dropped, learned = adapt_genres(["jazz", "indie"], liked, rejected)
    assert dropped == ["indie"]
    assert learned == ["blues"]
    assert genres == ["jazz", "blues"]


def test_adapt_genres_does_not_learn_genre_that_was_also_rejected():
    _, _, learned = adapt_genres([], [rec(1, "pop"), rec(2, "pop")], [rec(3, "pop")])
    assert learned == []


def test_apply_feedback_reports_changes_and_keeps_original_query():
    query = SearchQuery(text="x", genres=["jazz", "indie"], energy=Range(0.0, 0.4))
    adjusted, changes = apply_feedback(query, [rec(1, energy=0.6), rec(2, energy=0.6)], [rec(3, "indie")])
    assert adjusted.energy == Range(0.2, 0.6) and adjusted.genres == ["jazz"]
    assert query.energy == Range(0.0, 0.4) and query.genres == ["jazz", "indie"]   # original untouched
    assert any("energy" in c for c in changes) and any("indie" in c for c in changes)


# --- session ---

@pytest.fixture
def session(retriever):
    return PlaylistSession(PlaylistAgent(retriever), k=2, max_refine_rounds=1)


def test_session_refine_shows_new_songs_and_keeps_liked(session):
    first = session.start("calm jazz and indie")
    a, b = [s.song_id for s in first.songs]
    second = session.refine({a: True, b: False})
    assert second.status in ("ok", "partial")
    assert not {a, b} & {s.song_id for s in second.songs}             # never shown twice
    assert [s.song_id for s in session.playlist()] == [a]             # only 👍 songs
    assert second.trace[0].name == "feedback"
    assert "playlist so far" in second.message


def test_session_enforces_refine_limit(session):
    first = session.start("calm jazz")
    session.refine({first.songs[0].song_id: True})
    assert not session.can_refine
    with pytest.raises(SessionError, match="limit"):
        session.refine({first.songs[0].song_id: True})


def test_session_rejects_unknown_ids_and_empty_votes(session):
    session.start("calm jazz")
    with pytest.raises(SessionError, match="weren't shown"):
        session.refine({999: True})
    with pytest.raises(SessionError, match="Vote on at least one"):
        session.refine({})


def test_session_refine_before_start_fails(session):
    with pytest.raises(SessionError, match="Start"):
        session.refine({1: True})


def test_vote_can_change_mind(session):
    first = session.start("calm jazz")
    song_id = first.songs[0].song_id
    session.vote({song_id: True})
    session.vote({song_id: False})
    assert session.playlist() == [] and song_id in session.rejected


def test_refine_round_uses_one_gemini_call_for_explanations(retriever):
    from src.explainer import ReasonList
    from src.planner import PlanModel
    fake = FakeLLM(PlanModel=PlanModel(on_topic=True, search_text="calm jazz", genres=["jazz"]),
                   ReasonList=ReasonList(reasons=[]))
    session = PlaylistSession(PlaylistAgent(retriever, fake), k=2)
    first = session.start("calm jazz")
    session.refine({first.songs[0].song_id: True})
    assert [c[0] for c in fake.calls] == ["PlanModel", "ReasonList", "ReasonList"]   # no second planner call


def test_baseline_session_ignores_feedback(retriever):
    session = PlaylistSession(PlaylistAgent(retriever), k=2, use_feedback=False)
    first = session.start("calm jazz")
    second = session.refine({first.songs[0].song_id: True})
    assert "baseline" in second.trace[0].detail


# --- CLI voting ---

def test_parse_votes():
    assert cli.parse_votes("+253, +92 -278") == {253: True, 92: True, 278: False}
    with pytest.raises(ValueError):
        cli.parse_votes("like 253")


def _use(monkeypatch, retriever):
    monkeypatch.setattr(cli.Retriever, "load", classmethod(lambda cls: retriever))


def test_cli_like_dislike_runs_one_refine_round(monkeypatch, capsys, retriever):
    _use(monkeypatch, retriever)
    # Round 1 for this request on the test catalog is: Indie Calm (id 5), Soft Jazz (id 3).
    assert cli.main(["calm jazz and indie", "-k", "2", "--offline", "--like", "5", "--dislike", "3"]) == 0
    out = capsys.readouterr().out
    assert "New suggestions:" in out
    assert "Your playlist (1 songs):" in out and "Indie Calm" in out.rsplit("Your playlist (1 songs):", 1)[1]


def test_cli_rejects_votes_for_unshown_songs(monkeypatch, capsys, retriever):
    _use(monkeypatch, retriever)
    assert cli.main(["calm jazz", "-k", "2", "--offline", "--like", "999"]) == 4
    assert "weren't shown" in capsys.readouterr().err


def test_cli_interactive_flow(monkeypatch, capsys, retriever):
    _use(monkeypatch, retriever)
    session_ids = {}

    def ask(_prompt, answers=iter(["oops", None, None])):
        answer = next(answers)
        if answer is None:   # vote +first song of the latest round
            import re
            out = capsys.readouterr().out
            session_ids["last"] = int(re.findall(r"id (\d+) \|", out)[-1])
            return f"+{session_ids['last']}"
        return answer

    assert cli.main(["calm jazz and indie", "-k", "2", "--offline", "--interactive"], ask=ask) == 0
    out = capsys.readouterr().out
    assert "Your playlist (" in out


# --- simulator ---

def test_simulate_runs_rounds_and_feedback_beats_baseline_on_easy_case(retriever):
    persona = Persona("Indie only", "calm music", "only indie", lambda s: "indie" in s.genres)
    agent = PlaylistAgent(retriever)
    with_fb = simulate(persona, agent, refine_rounds=1, use_feedback=True)
    base = simulate(persona, agent, refine_rounds=1, use_feedback=False)
    assert len(with_fb.like_rates) == 2 and len(base.like_rates) == 2
    assert with_fb.like_rates[1] >= base.like_rates[1]
