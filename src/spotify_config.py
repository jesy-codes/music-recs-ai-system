"""
Load and verify Spotify Developer app credentials from the environment.

Credentials are read from a `.env` file in the project root (see `.env.example`)
or from real environment variables, which take precedence.

Check your setup with:
    python -m src.spotify_config
"""

import base64
import json
import logging
import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import certifi
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
TOKEN_URL = "https://accounts.spotify.com/api/token"
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8888/callback"
PLACEHOLDER_PREFIX = "your-"


class SpotifyConfigError(Exception):
    """Raised when Spotify credentials are missing, placeholders, or rejected by Spotify."""


@dataclass(frozen=True)
class SpotifyConfig:
    client_id: str
    client_secret: str
    redirect_uri: str


def _require(name: str) -> str:
    """Return a required env var, or raise a clear error if it's missing or still a placeholder."""
    value = os.getenv(name, "").strip()
    if not value:
        raise SpotifyConfigError(
            f"{name} is not set. Copy .env.example to .env and fill in {name} "
            f"from your app's Settings page at https://developer.spotify.com/dashboard."
        )
    if value.startswith(PLACEHOLDER_PREFIX):
        raise SpotifyConfigError(f"{name} in .env still has the placeholder value '{value}'.")
    return value


def load_spotify_config(env_path: Path = ENV_PATH) -> SpotifyConfig:
    """Read Spotify credentials from `.env` / the environment and validate they are present."""
    if env_path.exists():
        load_dotenv(env_path, override=False)
        logger.debug("Loaded environment from %s", env_path)
    else:
        logger.warning("No .env file at %s; falling back to process environment variables.", env_path)

    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "").strip() or DEFAULT_REDIRECT_URI
    if "localhost" in redirect_uri:
        logger.warning(
            "SPOTIFY_REDIRECT_URI uses 'localhost', which Spotify rejects. Use http://127.0.0.1:<port>/... instead."
        )

    config = SpotifyConfig(
        client_id=_require("SPOTIFY_CLIENT_ID"),
        client_secret=_require("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=redirect_uri,
    )
    logger.info("Spotify config loaded (client id ending in ...%s)", config.client_id[-4:])
    return config


def fetch_app_token(config: SpotifyConfig, timeout: float = 10.0) -> dict:
    """
    Request an app-only access token (Client Credentials flow).

    A successful response proves the Client ID and Secret are valid.
    Raises SpotifyConfigError with Spotify's error message on failure.
    """
    basic = base64.b64encode(f"{config.client_id}:{config.client_secret}".encode()).decode()
    request = urllib.request.Request(
        TOKEN_URL,
        data=urllib.parse.urlencode({"grant_type": "client_credentials"}).encode(),
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    # Use certifi's CA bundle: python.org builds on macOS ship without root certificates.
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise SpotifyConfigError(
            f"Spotify rejected the credentials (HTTP {error.code}): {body}. "
            f"Double-check SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env."
        ) from error
    except urllib.error.URLError as error:
        raise SpotifyConfigError(f"Could not reach {TOKEN_URL}: {error.reason}") from error

    logger.info("Got Spotify app token (expires in %ss)", payload.get("expires_in"))
    return payload


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = load_spotify_config()
        fetch_app_token(config)
    except SpotifyConfigError as error:
        logger.error("%s", error)
        return 1
    print("Spotify credentials are valid.")
    print(f"Redirect URI to register in the dashboard: {config.redirect_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
