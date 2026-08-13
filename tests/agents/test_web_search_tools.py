"""
Unit tests for the shared web-search service and its LangChain wrappers.

All tests use an in-memory SQLite cache and mocked provider clients so no
network calls or file I/O are made during the test run.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.agents.tools.web_search_tools import (
    _format_search_results,
    search_web_for_wine,
    search_wine_price,
    search_wine_reviews,
)
from src.services.web_search import (
    WebSearchCache,
    WineWebSearchEngine,
    _normalize_query,
    _query_hash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path) -> WebSearchCache:
    """File-based cache instance, isolated per test via tmp_path."""
    return WebSearchCache(tmp_path / "test_cache.db")


@pytest.fixture
def fake_results() -> list[dict]:
    return [
        {"title": "Sassicaia 2019 price", "snippet": "Available from $200", "url": "https://wine-searcher.com/1"},
        {"title": "Buy Sassicaia 2019", "snippet": "Retail $220 per bottle", "url": "https://vivino.com/1"},
    ]


@pytest.fixture
def engine_cfg():
    """Minimal config object that mimics the omegaconf DictConfig structure."""
    cfg = MagicMock()
    cfg.provider = "tavily"
    cfg.max_results = 5
    cfg.cache.enabled = True
    cfg.cache.ttl_hours = 24
    cfg.cache.max_entries = 1000
    cfg.cache.db_path = ":memory:"
    cfg.tavily.api_key_env = "TAVILY_API_KEY"
    return cfg


# ---------------------------------------------------------------------------
# _normalize_query
# ---------------------------------------------------------------------------

class TestNormalizeQuery:
    def test_lowercases_input(self):
        assert _normalize_query("BAROLO") == _normalize_query("barolo")

    def test_removes_stop_words(self):
        result = _normalize_query("What is the price of Barolo")
        assert "what" not in result
        assert "the" not in result
        assert "is" not in result
        assert "of" not in result

    def test_sorts_tokens_alphabetically(self):
        assert _normalize_query("price barolo") == _normalize_query("barolo price")

    def test_equivalent_queries_produce_same_hash(self):
        h1 = _query_hash("What is the price of Barolo")
        h2 = _query_hash("Barolo price")
        assert h1 == h2

    def test_different_queries_produce_different_hashes(self):
        assert _query_hash("Barolo price") != _query_hash("Amarone review")


# ---------------------------------------------------------------------------
# WebSearchCache
# ---------------------------------------------------------------------------

class TestWebSearchCache:
    def test_miss_returns_none(self, cache):
        assert cache.get("nonexistent_hash") is None

    def test_set_and_get_roundtrip(self, cache, fake_results):
        h = _query_hash("test")
        cache.set(h, "test", "price", fake_results, ttl_hours=1)
        retrieved = cache.get(h)
        assert retrieved is not None
        assert retrieved[0]["title"] == fake_results[0]["title"]

    def test_expired_entry_returns_none(self, cache, fake_results):
        h = _query_hash("expired")
        cache.set(h, "expired", "price", fake_results, ttl_hours=1)
        # Manually backdate the expires_at via the cache's own connection
        past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        with cache._connect() as conn:
            conn.execute(
                "UPDATE web_search_cache SET expires_at = ? WHERE query_hash = ?",
                (past, h),
            )
        assert cache.get(h) is None

    def test_hit_increments_hit_count(self, cache, fake_results):
        h = _query_hash("hit_count_test")
        cache.set(h, "hit_count_test", "general", fake_results, ttl_hours=1)
        cache.get(h)
        cache.get(h)
        with cache._connect() as conn:
            row = conn.execute(
                "SELECT hit_count FROM web_search_cache WHERE query_hash = ?", (h,)
            ).fetchone()
        assert row[0] == 2

    def test_evict_expired_removes_stale_rows(self, cache, fake_results):
        h = _query_hash("stale")
        cache.set(h, "stale", "price", fake_results, ttl_hours=1)
        past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        with cache._connect() as conn:
            conn.execute(
                "UPDATE web_search_cache SET expires_at = ? WHERE query_hash = ?",
                (past, h),
            )
        cache.evict_expired()
        with cache._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM web_search_cache").fetchone()[0]
        assert count == 0

    def test_evict_lru_keeps_most_recent(self, cache, fake_results):
        for i in range(5):
            cache.set(_query_hash(f"query_{i}"), f"query_{i}", "general", fake_results, ttl_hours=1)
        cache.evict_lru(max_entries=3)
        with cache._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM web_search_cache").fetchone()[0]
        assert count == 3

    def test_evict_lru_keeps_recently_accessed(self, cache, fake_results):
        """Entries accessed via get() survive eviction over unaccessed entries."""
        for i in range(4):
            cache.set(_query_hash(f"query_{i}"), f"query_{i}", "general", fake_results, ttl_hours=1)

        # Access query_0 and query_1 — they should be kept over query_2 and query_3
        cache.get(_query_hash("query_0"))
        cache.get(_query_hash("query_1"))

        cache.evict_lru(max_entries=2)

        with cache._connect() as conn:
            remaining = {
                row["query_text"] for row in conn.execute(
                    "SELECT query_text FROM web_search_cache"
                ).fetchall()
            }
        assert remaining == {"query_0", "query_1"}

    def test_clear_removes_all_rows(self, cache, fake_results):
        for i in range(3):
            cache.set(_query_hash(f"q{i}"), f"q{i}", "general", fake_results, ttl_hours=1)
        cache.clear()
        with cache._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM web_search_cache").fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# WineWebSearchEngine
# ---------------------------------------------------------------------------

class TestWineWebSearchEngine:
    def _make_engine(self, engine_cfg, mock_client, monkeypatch, tmp_path) -> WineWebSearchEngine:
        """Build an engine with a mocked Tavily client and a per-test file cache."""
        monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
        engine_cfg.cache.db_path = str(tmp_path / "test_cache.db")
        with patch("src.services.web_search.find_project_root", return_value=str(tmp_path)):
            with patch("tavily.TavilyClient", return_value=mock_client):
                engine = WineWebSearchEngine(cfg=engine_cfg)
        engine._client = mock_client
        return engine

    def test_cache_miss_calls_provider(self, engine_cfg, fake_results, monkeypatch, tmp_path):
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [{"title": r["title"], "content": r["snippet"], "url": r["url"]} for r in fake_results]
        }
        engine = self._make_engine(engine_cfg, mock_client, monkeypatch, tmp_path)
        results = engine.search("Sassicaia 2019 price", search_type="price")
        mock_client.search.assert_called_once()
        assert len(results) == 2

    def test_cache_hit_skips_provider(self, engine_cfg, fake_results, monkeypatch, tmp_path):
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [{"title": r["title"], "content": r["snippet"], "url": r["url"]} for r in fake_results]
        }
        engine = self._make_engine(engine_cfg, mock_client, monkeypatch, tmp_path)
        engine.search("Sassicaia 2019 price", search_type="price")   # first call — populates cache
        engine.search("Sassicaia 2019 price", search_type="price")   # second call — should hit cache
        assert mock_client.search.call_count == 1

    def test_provider_error_returns_empty_list(self, engine_cfg, monkeypatch, tmp_path):
        mock_client = MagicMock()
        mock_client.search.side_effect = RuntimeError("API unavailable")
        engine = self._make_engine(engine_cfg, mock_client, monkeypatch, tmp_path)
        results = engine.search("anything", search_type="general")
        assert results == []

    def test_cache_disabled_skips_file_io(self, engine_cfg, fake_results, monkeypatch, tmp_path):
        engine_cfg.cache.enabled = False
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [{"title": r["title"], "content": r["snippet"], "url": r["url"]} for r in fake_results]
        }
        engine = self._make_engine(engine_cfg, mock_client, monkeypatch, tmp_path)
        assert engine._cache is None
        assert not (tmp_path / engine_cfg.cache.db_path).exists()
        results = engine.search("Sassicaia 2019 price", search_type="price")
        assert len(results) == 2

    def test_url_deduplication(self, engine_cfg, monkeypatch, tmp_path):
        dup_url = "https://wine-searcher.com/1"
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {"title": "Result A", "content": "snippet A", "url": dup_url},
                {"title": "Result B", "content": "snippet B", "url": dup_url},
            ]
        }
        engine = self._make_engine(engine_cfg, mock_client, monkeypatch, tmp_path)
        results = engine.search("duplicate test", search_type="general")
        assert len(results) == 1
        assert results[0]["title"] == "Result A"


# ---------------------------------------------------------------------------
# _format_search_results
# ---------------------------------------------------------------------------

class TestFormatSearchResults:
    def test_empty_list_returns_fallback(self):
        assert "No web search" in _format_search_results([])

    def test_numbered_output(self, fake_results):
        out = _format_search_results(fake_results)
        assert "[1]" in out
        assert "[2]" in out

    def test_includes_source_url(self, fake_results):
        out = _format_search_results(fake_results)
        assert "Source:" in out
        assert fake_results[0]["url"] in out

    def test_includes_snippet(self, fake_results):
        out = _format_search_results(fake_results)
        assert fake_results[0]["snippet"] in out


# ---------------------------------------------------------------------------
# LangChain tool functions
# ---------------------------------------------------------------------------

class TestTools:
    @pytest.fixture(autouse=True)
    def _patch_engine(self, fake_results, monkeypatch):
        """Replace the module-level singleton engine with a mock for all tool tests."""
        mock_engine = MagicMock()
        mock_engine.search.return_value = fake_results
        monkeypatch.setattr(
            "src.agents.tools.web_search_tools._engine", mock_engine
        )
        self.mock_engine = mock_engine

    def test_search_web_for_wine_returns_formatted_string(self):
        out = search_web_for_wine.invoke({"query": "Barolo news", "search_type": "general"})
        assert "[1]" in out
        assert "Source:" in out

    def test_search_web_for_wine_appends_producer_suffix(self):
        search_web_for_wine.invoke({"query": "Domaine Leflaive", "search_type": "producer"})
        call_args = self.mock_engine.search.call_args
        assert "winery winemaker producer" in call_args[0][0]

    def test_search_wine_price_appends_price_keywords(self):
        search_wine_price.invoke({"wine_name": "Sassicaia", "vintage": 2019})
        call_args = self.mock_engine.search.call_args
        assert "wine price buy retail" in call_args[0][0]

    def test_search_wine_price_includes_vintage(self):
        search_wine_price.invoke({"wine_name": "Barolo", "vintage": 2018})
        call_args = self.mock_engine.search.call_args
        assert "2018" in call_args[0][0]

    def test_search_wine_reviews_appends_review_keywords(self):
        search_wine_reviews.invoke({"wine_name": "Penfolds Grange", "vintage": 2018})
        call_args = self.mock_engine.search.call_args
        assert "wine review score rating" in call_args[0][0]

    def test_search_wine_reviews_includes_reviewer(self):
        search_wine_reviews.invoke({
            "wine_name": "Barolo Monfortino",
            "vintage": 2016,
            "reviewer": "Wine Advocate",
        })
        call_args = self.mock_engine.search.call_args
        assert "Wine Advocate" in call_args[0][0]

    def test_search_web_for_wine_clamps_max_results(self):
        search_web_for_wine.invoke({"query": "test", "max_results": 99})
        call_args = self.mock_engine.search.call_args
        assert call_args[1]["max_results"] == 10






