"""Thin wrapper around the Gemini API (google-genai SDK).

The rest of the code only sees `LLMClient.generate_json(...)` and three error types, so:
- tests swap in a fake client and never call the network,
- every caller can fall back to rule-based behavior when Gemini is missing, out of quota or failing.

Configuration (.env or environment):
    GEMINI_API_KEY   required to use Gemini; without it the app runs in offline (rules-only) mode
    GEMINI_MODEL     optional, default "gemini-3.5-flash" (tested Nov 2026; models change, so check the list below)

List the models your key can use:
    python -m src.llm --list-models
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional, Protocol, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash"
REQUEST_TIMEOUT_MS = 30_000
QUOTA_COOLDOWN_SECONDS = 60   # after a 429, skip Gemini for this long instead of retrying every step
TRANSIENT_RETRIES = 1         # one retry for 5xx / timeouts / network errors (Gemini 504s happen)
RETRY_DELAY_SECONDS = 1.0

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Gemini call failed (bad response, server error, network); callers should fall back."""


class LLMUnavailable(LLMError):
    """No API key or SDK, so Gemini can't be used at all (offline mode)."""


class LLMQuotaError(LLMError):
    """Rate limit or daily quota reached (HTTP 429)."""


class LLMTransientError(LLMError):
    """Server error (5xx), timeout or network failure; worth one retry."""


class LLMClient(Protocol):
    model: str

    def generate_json(self, system: str, prompt: str, schema: Type[T], temperature: float = 0.0) -> T: ...


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


class GeminiClient:
    """Calls Gemini with a JSON response schema and returns a validated pydantic object."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        _load_dotenv()
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key or api_key.startswith("your-"):
            raise LLMUnavailable("GEMINI_API_KEY is not set, so the agent runs in offline mode (rule-based planner).")
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise LLMUnavailable("google-genai is not installed. Run: pip install -r requirements.txt") from error
        self._types = types
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
        # Circuit breaker: once the key or model is unusable (or quota is hit), later calls fail fast
        # instead of repeating a request that can't succeed.
        self._blocked: Optional[LLMError] = None
        self._blocked_until = 0.0

    def generate_json(self, system: str, prompt: str, schema: Type[T], temperature: float = 0.0) -> T:
        if self._blocked is not None and time.monotonic() < self._blocked_until:
            raise self._blocked
        try:
            for attempt in range(TRANSIENT_RETRIES + 1):
                try:
                    return self._generate_json(system, prompt, schema, temperature)
                except LLMTransientError as error:
                    if attempt == TRANSIENT_RETRIES:
                        raise
                    logger.warning("%s Retrying once in %.0fs.", error, RETRY_DELAY_SECONDS)
                    time.sleep(RETRY_DELAY_SECONDS)
        except LLMQuotaError as error:
            self._blocked, self._blocked_until = error, time.monotonic() + QUOTA_COOLDOWN_SECONDS
            raise
        except LLMUnavailable as error:
            self._blocked, self._blocked_until = error, float("inf")
            raise

    def _generate_json(self, system: str, prompt: str, schema: Type[T], temperature: float) -> T:
        from google.genai import errors

        config = self._types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
        )
        started = time.perf_counter()
        try:
            response = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
        except errors.ClientError as error:
            if error.code == 429:
                raise LLMQuotaError(
                    f"Gemini quota or rate limit reached for {self.model}. Try again later or set GEMINI_MODEL to another model."
                ) from error
            if error.code in (401, 403):
                raise LLMUnavailable(f"Gemini rejected the API key ({error.code}). Check GEMINI_API_KEY in .env.") from error
            if error.code == 404:
                raise LLMUnavailable(
                    f"Gemini model {self.model!r} not found. Run `python -m src.llm --list-models` and set GEMINI_MODEL."
                ) from error
            raise LLMError(f"Gemini request failed ({error.code}): {error.message}") from error
        except errors.ServerError as error:
            raise LLMTransientError(f"Gemini server error ({error.code}).") from error
        except Exception as error:  # network errors and timeouts surface as different httpx types
            raise LLMTransientError(f"Gemini call failed: {type(error).__name__}: {error}") from error

        elapsed = time.perf_counter() - started
        usage = getattr(response, "usage_metadata", None)
        logger.info(
            "Gemini %s answered in %.1fs (prompt %s tokens, output %s tokens)", self.model, elapsed,
            getattr(usage, "prompt_token_count", "?"), getattr(usage, "candidates_token_count", "?"),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        try:
            return schema.model_validate_json(response.text or "")
        except (ValidationError, ValueError) as error:
            raise LLMError(f"Gemini returned JSON that doesn't match {schema.__name__}: {error}") from error

    def list_models(self) -> list[str]:
        return sorted(m.name for m in self._client.models.list() if "generateContent" in (m.supported_actions or []))


def get_client() -> Optional[LLMClient]:
    """Return a Gemini client, or None (offline mode) when no key or SDK is available."""
    try:
        return GeminiClient()
    except LLMUnavailable as error:
        logger.warning("%s", error)
        return None


def main() -> int:
    from src.log_config import setup_logging

    setup_logging(logging.WARNING)
    if "--list-models" not in sys.argv:
        print(__doc__)
        return 0
    try:
        client = GeminiClient()
        for name in client.list_models():
            print(name)
    except LLMError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Error: couldn't list models: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
