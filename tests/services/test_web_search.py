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
    command = "import sys; import src.retrieval; " "raise SystemExit(1 if 'src.agents.tools' in sys.modules else 0)"

    result = subprocess.run([sys.executable, "-c", command], check=False)

    assert result.returncode == 0


@pytest.mark.parametrize("domain", ["vivino.com", " VIVINO.COM. ", "Vivino.com"])
def test_restricted_provider_forwarding_and_filtering(
    service_config: MagicMock, monkeypatch: pytest.MonkeyPatch, domain: str
) -> None:
    """Restrictions include exact hosts and subdomains but exclude lookalikes and malformed URLs."""
    engine = _engine(service_config, monkeypatch)
    urls = [
        "https://vivino.com/wine",
        "https://WWW.VIVINO.COM./wine",
        "https://shop.vivino.com/wine",
        "https://evilvivino.com/wine",
        "https://vivino.com.evil.com/wine",
        "https://elsewhere.com/vivino.com",
        "https://vivino.com@evil.com/wine",
        "https://[broken",
        "/wine",
        "javascript://vivino.com/wine",
    ]
    engine._client.search.return_value = {
        "results": [{"title": "Wine", "content": "Notes", "url": url} for url in urls]
    }
    results = engine.search("wine", domain=domain)
    assert [result["url"] for result in results] == urls[:3]
    engine._client.search.assert_called_once_with("wine", max_results=5, include_domains=["vivino.com"])


@pytest.mark.parametrize(
    "domain",
    [
        "",
        "localhost",
        "wine.local",
        "wine.localhost",
        "wine.internal",
        "wine.test",
        "wine.invalid",
        "wine.example",
        "wine.onion",
        "home.arpa",
        "127.0.0.1",
        "8.8.8.8",
        "::1",
        "[::1]",
        "https://vivino.com",
        "vivino.com/path",
        "vivino.com:443",
        "user@vivino.com",
        "*.vivino.com",
        "wine..com",
        "-wine.com",
        "wine-.com",
        "wine_name.com",
        "wine.com..",
        "a" * 64 + ".com",
        ".".join(["a" * 63] * 4),
    ],
)
def test_invalid_domain_rejected_before_provider(
    service_config: MagicMock, monkeypatch: pytest.MonkeyPatch, domain: str
) -> None:
    """Invalid and non-public hostname syntax must never reach the provider."""
    engine = _engine(service_config, monkeypatch)
    with pytest.raises(ValueError, match="public hostname"):
        engine.search("wine", domain=domain)
    engine._client.search.assert_not_called()


def test_domain_cache_identity_and_unrestricted_compatibility(
    service_config: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equivalent hosts share results; other hosts and unrestricted requests remain isolated."""
    engine = _engine(service_config, monkeypatch)
    engine._client.search.return_value = {
        "results": [
            {"title": "A", "content": "A", "url": "https://vivino.com/wine"},
            {"title": "B", "content": "B", "url": "https://decanter.com/wine"},
        ]
    }
    assert len(engine.search("wine")) == 2
    engine._client.search.assert_called_once_with("wine", max_results=5)
    assert len(engine.search("wine", domain="vivino.com")) == 1
    assert len(engine.search("wine", domain=" VIVINO.COM. ")) == 1
    assert len(engine.search("wine", domain="decanter.com")) == 1
    assert len(engine.search("wine")) == 2
    assert engine._client.search.call_count == 3


def test_restricted_empty_results_do_not_retry(service_config: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Filtering everything out must not trigger an unrestricted provider request."""
    engine = _engine(service_config, monkeypatch)
    engine._client.search.return_value = {
        "results": [
            {"title": "Other", "content": "Notes", "url": "https://decanter.com/wine"},
        ]
    }
    assert engine.search("wine", domain="vivino.com") == []
    engine._client.search.assert_called_once_with("wine", max_results=5, include_domains=["vivino.com"])


def test_international_hostname_normalization() -> None:
    """Unicode and ASCII representations use the same cache identity."""
    from src.services.web_search import _query_hash, normalize_domain

    assert normalize_domain("BÜCHER.de.") == "xn--bcher-kva.de"
    assert _query_hash("wine", domain="BÜCHER.de.") == _query_hash("wine", domain="xn--bcher-kva.de")
