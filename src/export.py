"""Export the final playlist (👍 songs only): CSV for everyone, a Spotify share link when configured.

The app never writes to a visitor's Spotify account. "Get Spotify playlist link" creates a public playlist
in the *app owner's* account (with a refresh token stored once in .env / Streamlit secrets) and returns
its link; visitors can open it and save it if they want.

One-time setup (owner, needs Spotify Premium for Development Mode):
    python -m src.export authorize     # opens Spotify login, saves SPOTIFY_REFRESH_TOKEN to .env
    python -m src.export check         # confirms the token works
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import os
import secrets
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from src.ranker import Recommendation

logger = logging.getLogger(__name__)

CSV_COLUMNS = ["position", "title", "artist", "genres", "spotify_url", "track_id", "song_id"]
EXPORT_DIR = Path("exports")

TOKEN_URL = "https://accounts.spotify.com/api/token"
AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
API = "https://api.spotify.com/v1"
SCOPES = "playlist-modify-public playlist-modify-private"
PLAYLIST_DESCRIPTION = "Playlist proposal made with the Music Recs AI demo."


class ExportError(RuntimeError):
    """Export failed; the message says what to do."""


# --- CSV ----------------------------------------------------------------------

def playlist_rows(songs: Sequence[Recommendation]) -> list[dict]:
    return [{"position": i, "title": s.title, "artist": s.artist, "genres": s.genres.replace("; ", ", "),
             "spotify_url": s.spotify_url, "track_id": s.track_id, "song_id": s.song_id}
            for i, s in enumerate(songs, start=1)]


def to_csv_bytes(songs: Sequence[Recommendation]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(playlist_rows(songs))
    return buffer.getvalue().encode("utf-8")


def csv_filename(now: Optional[datetime] = None) -> str:
    return f"playlist_{(now or datetime.now()).strftime('%Y%m%d_%H%M%S')}.csv"


def save_csv(songs: Sequence[Recommendation], folder: Path = EXPORT_DIR) -> Path:
    if not songs:
        raise ExportError("The playlist is empty: give at least one song a 👍 first.")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / csv_filename()
    path.write_bytes(to_csv_bytes(songs))
    logger.info("Saved playlist CSV with %d songs to %s", len(songs), path)
    return path


# --- Spotify ------------------------------------------------------------------

def _env_path() -> Optional[Path]:
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return None
    path = find_dotenv(usecwd=True) or find_dotenv()
    if path:
        load_dotenv(path, override=False)
    return Path(path) if path else None


def _credentials() -> tuple[str, str]:
    _env_path()
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret or client_id.startswith("your-"):
        raise ExportError("SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET aren't set. See the Spotify section of the README.")
    return client_id, client_secret


def spotify_configured() -> bool:
    """True when the owner token is set up, so the share-link button can be offered."""
    try:
        _credentials()
    except ExportError:
        return False
    return bool(os.getenv("SPOTIFY_REFRESH_TOKEN", "").strip())


def _request(url: str, *, method: str = "GET", data: Optional[bytes] = None, headers: Optional[dict] = None,
             timeout: float = 15.0) -> dict:
    import certifi

    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:300]
        hint = ""
        if error.code == 401:
            hint = " The token was rejected; run `python -m src.export authorize` again."
        elif error.code == 403:
            hint = (" Spotify refused the request. In Development Mode the app owner needs Premium and must be "
                    "listed under User Management in the dashboard.")
        elif error.code == 429:
            hint = " Spotify rate limit reached; try again in a minute."
        raise ExportError(f"Spotify returned HTTP {error.code} for {url.split('?')[0]}: {detail}.{hint}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ExportError(f"Couldn't reach Spotify ({error}). Check your internet connection.") from error
    return json.loads(body) if body else {}


def _token_request(form: dict) -> dict:
    client_id, client_secret = _credentials()
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    return _request(TOKEN_URL, method="POST", data=urllib.parse.urlencode(form).encode(),
                    headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"})


def owner_access_token() -> str:
    refresh_token = os.getenv("SPOTIFY_REFRESH_TOKEN", "").strip()
    if not refresh_token:
        raise ExportError("SPOTIFY_REFRESH_TOKEN isn't set. Run `python -m src.export authorize` once.")
    payload = _token_request({"grant_type": "refresh_token", "refresh_token": refresh_token})
    token = payload.get("access_token")
    if not token:
        raise ExportError("Spotify didn't return an access token. Run `python -m src.export authorize` again.")
    return token


def playlist_name(now: Optional[datetime] = None) -> str:
    """Fixed format, never the user's text: the playlist is public under the owner's name."""
    return f"Vibe Mix {(now or datetime.now()).strftime('%Y-%m-%d %H:%M')}"


def create_spotify_playlist(songs: Sequence[Recommendation]) -> str:
    """Create a public playlist with the songs in the owner's account; return its open.spotify.com link."""
    if not songs:
        raise ExportError("The playlist is empty: give at least one song a 👍 first.")
    token = owner_access_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    playlist = _request(f"{API}/me/playlists", method="POST", headers=headers, data=json.dumps(
        {"name": playlist_name(), "description": PLAYLIST_DESCRIPTION, "public": True}).encode())
    playlist_id = playlist.get("id")
    if not playlist_id:
        raise ExportError("Spotify didn't return a playlist id.")
    uris = [f"spotify:track:{s.track_id}" for s in songs][:100]
    _request(f"{API}/playlists/{playlist_id}/items", method="POST", headers=headers,
             data=json.dumps({"uris": uris}).encode())
    link = (playlist.get("external_urls") or {}).get("spotify") or f"https://open.spotify.com/playlist/{playlist_id}"
    logger.info("Created Spotify playlist %s with %d songs: %s", playlist_id, len(uris), link)
    return link


# --- one-time owner authorization (CLI) ----------------------------------------

def authorize() -> int:
    """Log in to Spotify in the browser and save a refresh token to .env (owner, run once)."""
    import http.server
    import webbrowser

    client_id, _ = _credentials()
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "").strip() or "http://127.0.0.1:8888/callback"
    parsed = urllib.parse.urlparse(redirect_uri)
    state = secrets.token_urlsafe(16)
    url = AUTHORIZE_URL + "?" + urllib.parse.urlencode({
        "client_id": client_id, "response_type": "code", "redirect_uri": redirect_uri, "scope": SCOPES, "state": state})
    result: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server API)
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            result.update({k: v[0] for k, v in query.items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write("<h3>Done. You can close this tab and go back to the terminal.</h3>".encode())

        def log_message(self, *args):
            pass

    print(f"Opening Spotify login in your browser. If it doesn't open, visit:\n{url}\n")
    webbrowser.open(url)
    with http.server.HTTPServer((parsed.hostname or "127.0.0.1", parsed.port or 8888), Handler) as server:
        server.timeout = 300
        server.handle_request()

    if result.get("state") != state or "code" not in result:
        print(f"Error: authorization failed ({result.get('error', 'no code returned')}).", file=sys.stderr)
        return 1
    payload = _token_request({"grant_type": "authorization_code", "code": result["code"], "redirect_uri": redirect_uri})
    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        print("Error: Spotify didn't return a refresh token.", file=sys.stderr)
        return 1
    env_path = _env_path()
    if env_path is None:
        print("Error: no .env file found. Create one from .env.example first.", file=sys.stderr)
        return 1
    from dotenv import set_key
    set_key(str(env_path), "SPOTIFY_REFRESH_TOKEN", refresh_token)
    print(f"Saved SPOTIFY_REFRESH_TOKEN to {env_path}. The share-link button is now available.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    from src.log_config import setup_logging

    setup_logging(logging.WARNING)
    command = (list(argv) if argv is not None else sys.argv[1:] or ["help"])[0]
    try:
        if command == "authorize":
            return authorize()
        if command == "check":
            owner_access_token()
            print("Spotify owner token works. The share-link button is available.")
            return 0
    except ExportError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
