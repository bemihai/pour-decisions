"""Focused contract tests for the reusable web-search service."""

import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from src.services.web_search import WineWebSearchEngine


@pytest.fixture
def service_config(tmp_path):
    """Return a minimal service config with an isolated cache."""
    config = MagicMock()
    config.provider = "tavily"
    config.max_results = 5
    config.cache.enabled = True
    config.cache.max_entries = 100
    config.cache.db_path = str(tmp_path / "web_cache.db")
    config.tavily.api_key_env = "TAVILY_API_KEY"
    return config


def _engine(service_config, monkeypatch) -> WineWebSearchEngine:
    """Construct a service with provider initialization replaced."""
    monkeypatch.setattr(WineWebSearchEngine, "_init_client", lambda self, cfg: MagicMock())
    return WineWebSearchEngine(service_config)


@pytest.mark.parametrize(
    ("search_type", "expected_ttl"),
    [("general", 12), ("price", 24), ("review", 168), ("producer", 720)],
)
def test_search_uses_type_specific_ttl(
    service_config,
    monkeypatch,
    search_type: str,
    expected_ttl: int,
) -> None:
    """Every supported search type must preserve its established cache TTL."""
    engine = _engine(service_config, monkeypatch)
    engine._search_provider = MagicMock(
        return_value=[{"title": "Title", "snippet": "Snippet", "url": "https://example.test/result"}]
    )
    engine._cache.set = MagicMock()

    engine.search("query", search_type=search_type, max_results=3)

    assert engine._cache.set.call_args.args[-1] == expected_ttl


def test_provider_result_has_stable_structured_shape(service_config, monkeypatch) -> None:
    """Provider output must expose title, snippet, and URL to every consumer."""
    engine = _engine(service_config, monkeypatch)
    engine._client.search.return_value = {
        "results": [{"title": "Title", "content": "Snippet", "url": "https://example.test/result"}]
    }

    assert engine._search_provider("query", 1) == [
        {"title": "Title", "snippet": "Snippet", "url": "https://example.test/result"}
    ]


def test_retrieval_package_import_does_not_initialize_agent_tools() -> None:
    """The retrieval package must remain independent from agent tool initialization."""
    command = (
        "import sys; import src.retrieval; "
        "raise SystemExit(1 if 'src.agents.tools' in sys.modules else 0)"
    )

    result = subprocess.run([sys.executable, "-c", command], check=False)

    assert result.returncode == 0
