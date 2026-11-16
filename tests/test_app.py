"""Tests for Phase 5: CSV/Spotify export (Spotify mocked) and the Streamlit app flow (AppTest, offline)."""

import csv
import io
import json
import urllib.error
from datetime import datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import export
from src.export import ExportError
from src.ranker import Recommendation


def rec(song_id, title="Song", artist="Artist"):
    return Recommendation(song_id=song_id, track_id=f"track{song_id}", title=title, artist=artist,
                          genres="jazz; blues", score=1.0, parts={}, reasons=["r"], features={})


# --- CSV ---

def test_csv_has_header_and_rows_in_order():
    rows = list(csv.DictReader(io.StringIO(export.to_csv_bytes([rec(1, "A"), rec(2, "B")]).decode())))
    assert [r["title"] for r in rows] == ["A", "B"]
    assert rows[0]["position"] == "1" and rows[0]["genres"] == "jazz, blues"
    assert rows[0]["spotify_url"] == "https://open.spotify.com/track/track1"


def test_csv_handles_commas_and_quotes():
    data = export.to_csv_bytes([rec(1, 'Hello, "World"')]).decode()
    assert list(csv.DictReader(io.StringIO(data)))[0]["title"] == 'Hello, "World"'


def test_csv_filename_and_save(tmp_path):
    assert export.csv_filename(datetime(2026, 11, 15, 18, 5, 9)) == "playlist_20261115_180509.csv"
    path = export.save_csv([rec(1)], folder=tmp_path)
    assert path.exists() and path.read_text().startswith("position,title")


def test_save_empty_playlist_fails(tmp_path):
    with pytest.raises(ExportError, match="empty"):
        export.save_csv([], folder=tmp_path)


# --- Spotify (all HTTP mocked) ---

def test_playlist_name_never_uses_user_text():
    assert export.playlist_name(datetime(2026, 11, 15, 18, 5)) == "Vibe Mix 2026-11-15 18:05"


def test_spotify_not_configured_without_env(monkeypatch):
    monkeypatch.setattr(export, "_env_path", lambda: None)
    for key in ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET", "SPOTIFY_REFRESH_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    assert not export.spotify_configured()


def test_create_spotify_playlist_calls_new_endpoints(monkeypatch):
    calls = []

    def fake_request(url, *, method="GET", data=None, headers=None, timeout=15.0):
        calls.append((method, url, json.loads(data) if data else None))
        if url.endswith("/me/playlists"):
            return {"id": "pl1", "external_urls": {"spotify": "https://open.spotify.com/playlist/pl1"}}
        return {"snapshot_id": "x"}

    monkeypatch.setattr(export, "owner_access_token", lambda: "token")
    monkeypatch.setattr(export, "_request", fake_request)
    link = export.create_spotify_playlist([rec(1), rec(2)])
    assert link == "https://open.spotify.com/playlist/pl1"
    (m1, u1, body1), (m2, u2, body2) = calls
    assert (m1, u1) == ("POST", "https://api.spotify.com/v1/me/playlists")
    assert body1["public"] is True and body1["name"].startswith("Vibe Mix ")
    assert (m2, u2) == ("POST", "https://api.spotify.com/v1/playlists/pl1/items")
    assert body2 == {"uris": ["spotify:track:track1", "spotify:track:track2"]}


def test_create_spotify_playlist_rejects_empty():
    with pytest.raises(ExportError, match="empty"):
        export.create_spotify_playlist([])


def test_spotify_403_gets_development_mode_hint(monkeypatch):
    def raise_403(request, timeout, context):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, io.BytesIO(b'{"error": "forbidden"}'))

    monkeypatch.setattr(export.urllib.request, "urlopen", raise_403)
    with pytest.raises(ExportError, match="Development Mode"):
        export._request("https://api.spotify.com/v1/me/playlists", method="POST", data=b"{}")


def test_missing_refresh_token_explains_setup(monkeypatch):
    monkeypatch.delenv("SPOTIFY_REFRESH_TOKEN", raising=False)
    with pytest.raises(ExportError, match="authorize"):
        export.owner_access_token()


# --- Streamlit app (offline: conftest blanks the Gemini key) ---

@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(export, "spotify_status", lambda: (False, "not set up"))
    return AppTest.from_file(str(Path(__file__).resolve().parent.parent / "src" / "app.py"), default_timeout=120).run()


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()


def test_app_start_page(app):
    assert not app.exception
    assert any("Give me your" in m.value for m in app.markdown)       # hero title
    assert len(app.sidebar) == 0                                      # demo: no settings sidebar
    assert "something chill, with jazz and indie music" in [b.label for b in app.button]


def test_app_full_flow(app):
    click(app, "something chill, with jazz and indie music")
    assert not app.exception
    assert len(app.feedback) == 5                                    # round 1: 5 songs with 👍/👎
    refine = next(b for b in app.button if b.label == "Refine my playlist")
    assert refine.disabled                                           # no votes yet

    app.feedback[0].set_value(1).run()                               # 👍 first song
    app.feedback[1].set_value(0).run()                               # 👎 second song
    click(app, "Refine my playlist")
    assert not app.exception
    # round 1 (5, locked) + playlist so far (1) + 5 new songs
    assert len(app.feedback) == 11
    assert any("playlist so far" in m.value for m in app.markdown)
    assert all(f.disabled for f in app.feedback[:5])

    app.feedback[-1].set_value(1).run()                              # 👍 one new song
    click(app, "Looks good, I'm done")
    assert not app.exception
    assert any("2 songs you gave a 👍" in m.value for m in app.markdown)
    assert [b.label for b in app.get("download_button")] == ["Download CSV"]


def test_app_rejects_off_topic(app):
    app.chat_input[0].set_value("write my history essay about the french revolution").run()
    assert not app.exception
    assert any("only help with music" in w.value for w in app.warning)


def test_styles_escape_data_in_markup():
    from src.styles import playlist_html, song_info_html, user_bubble_html
    assert "<script>" not in user_bubble_html("<script>alert(1)</script>")
    assert "&lt;b&gt;" in song_info_html("<b>T</b>", "A", "jazz", "r")
    assert 'href="https://open.spotify.com/track/x&quot;' in playlist_html([("T", "A", 'https://open.spotify.com/track/x"')])


def test_visualizer_bars_are_deterministic_and_decorative():
    from src.styles import CSS, visualizer_html
    html = visualizer_html(bars=10)
    assert html.count("<span") == 10 and 'aria-hidden="true"' in html
    assert html == visualizer_html(bars=10)                        # same output on every render
    assert "prefers-reduced-motion" in CSS and "@keyframes eq-pulse" in CSS
