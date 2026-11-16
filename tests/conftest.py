import pytest

from tests.helpers import make_retriever


@pytest.fixture
def retriever():
    return make_retriever()


@pytest.fixture(autouse=True)
def no_real_gemini(monkeypatch):
    """Tests must never call the real API, even if a developer's .env has a key."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setattr("src.llm._load_dotenv", lambda: None)
