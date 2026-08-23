"""Provider-neutral web search with a persistent, bounded cache."""

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.utils import find_project_root, get_config, logger


_CACHE_TTL_HOURS: dict[str, int] = {
    "price": 24,
    "review": 168,
    "producer": 720,
    "general": 12,
}

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "of",
        "for",
        "in",
        "and",
        "or",
        "to",
        "me",
        "what",
        "where",
        "when",
        "how",
        "who",
        "which",
        "why",
        "does",
        "do",
        "did",
        "are",
        "was",
        "were",
        "been",
        "can",
        "could",
        "would",
        "should",
        "tell",
        "about",
        "much",
        "cost",
        "i",
        "my",
    }
)


def _normalize_query(query: str) -> str:
    """Return a normalized, hash-ready representation of a search query."""
    tokens = [token for token in query.lower().split() if token not in _STOP_WORDS]
    return " ".join(sorted(tokens))


def _query_hash(query: str, search_type: str = "general", max_results: int | None = None) -> str:
    """Hash a normalized query together with cache-affecting parameters."""
    key = f"{_normalize_query(query)}|{search_type}|{max_results}"
    return hashlib.sha256(key.encode()).hexdigest()


class WebSearchCache:
    """SQLite-backed cache for web search result dictionaries."""

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS web_search_cache (
            query_hash    TEXT PRIMARY KEY,
            query_text    TEXT NOT NULL,
            search_type   TEXT NOT NULL,
            results_json  TEXT NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at    TIMESTAMP NOT NULL,
            hit_count     INTEGER DEFAULT 0,
            last_access_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    _MIGRATIONS: list[str] = [
        "ALTER TABLE web_search_cache ADD COLUMN last_access_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    ]

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._in_memory = self._db_path == ":memory:"
        if self._in_memory:
            self._db_path = "file::memory:?cache=shared"
        with self._connect() as conn:
            conn.execute(self._CREATE_TABLE)
            self._apply_migrations(conn)
        self.evict_expired()

    def _apply_migrations(self, conn: sqlite3.Connection) -> None:
        """Apply cache-table additions missing from an existing database."""
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(web_search_cache)")}
        for ddl in self._MIGRATIONS:
            column = ddl.split("ADD COLUMN")[1].strip().split()[0]
            if column not in existing_columns:
                conn.execute(ddl)
                logger.debug("Applied cache migration: %s", ddl)

    def _connect(self) -> sqlite3.Connection:
        """Open one cache connection."""
        conn = sqlite3.connect(self._db_path, uri=self._in_memory)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _now_iso() -> str:
        """Return the current UTC time as an ISO 8601 string."""
        return datetime.now(timezone.utc).isoformat()

    def get(self, query_hash: str) -> list[dict[str, Any]] | None:
        """Return unexpired cached results, otherwise ``None``."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT results_json, expires_at FROM web_search_cache WHERE query_hash = ?",
                (query_hash,),
            ).fetchone()
        if row is None or datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
            return None

        with self._connect() as conn:
            conn.execute(
                "UPDATE web_search_cache SET hit_count = hit_count + 1, last_access_at = ? WHERE query_hash = ?",
                (self._now_iso(), query_hash),
            )
        return json.loads(row["results_json"])

    def set(
        self,
        query_hash: str,
        query_text: str,
        search_type: str,
        results: list[dict[str, Any]],
        ttl_hours: int,
    ) -> None:
        """Insert or replace one cache entry."""
        expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO web_search_cache
                    (query_hash, query_text, search_type, results_json, expires_at, hit_count, last_access_at)
                VALUES (?, ?, ?, ?, ?, 0, ?)
                """,
                (query_hash, query_text, search_type, json.dumps(results), expires_at.isoformat(), self._now_iso()),
            )

    def evict_expired(self) -> None:
        """Delete expired entries."""
        with self._connect() as conn:
            conn.execute("DELETE FROM web_search_cache WHERE expires_at <= ?", (self._now_iso(),))

    def evict_lru(self, max_entries: int) -> None:
        """Bound the cache by deleting least-recently-accessed entries."""
        with self._connect() as conn:
            excess = conn.execute("SELECT COUNT(*) FROM web_search_cache").fetchone()[0] - max_entries
            if excess > 0:
                conn.execute(
                    """
                    DELETE FROM web_search_cache WHERE query_hash IN (
                        SELECT query_hash FROM web_search_cache
                        ORDER BY last_access_at ASC LIMIT ?
                    )
                    """,
                    (excess,),
                )

    def clear(self) -> None:
        """Delete all cached entries."""
        with self._connect() as conn:
            conn.execute("DELETE FROM web_search_cache")
        logger.info("Web search cache cleared")


class WineWebSearchEngine:
    """Provider-neutral web search engine with integrated caching."""

    def __init__(self, cfg: Any | None = None) -> None:
        if cfg is None:
            cfg = get_config().web_search
        self._provider: str = cfg.provider
        self._max_results: int = cfg.max_results
        self._cache_enabled: bool = cfg.cache.enabled
        self._cache_max_entries: int = cfg.cache.max_entries

        self._cache: WebSearchCache | None = None
        if self._cache_enabled:
            db_path = Path(find_project_root()) / cfg.cache.db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache = WebSearchCache(db_path)

        self._client = self._init_client(cfg)
        logger.info("WineWebSearchEngine initialized with provider: %s", self._provider)

    def _init_client(self, cfg: Any) -> Any:
        """Instantiate the configured provider client."""
        if self._provider == "tavily":
            from tavily import TavilyClient

            key_env = cfg.tavily.api_key_env
            api_key = os.environ.get(key_env, "")
            if not api_key:
                raise ValueError(f"Tavily API key not found. Set the {key_env} environment variable.")
            return TavilyClient(api_key=api_key)
        raise ValueError(
            f"Unsupported web search provider: '{self._provider}'. "
            "Add a new branch in WineWebSearchEngine._init_client to support it."
        )

    def _search_provider(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Execute one provider request and normalize its result dictionaries."""
        if self._provider == "tavily":
            response = self._client.search(query, max_results=max_results)
            return [
                {"title": result["title"], "snippet": result["content"], "url": result["url"]}
                for result in response.get("results", [])
            ]
        return []

    def search(
        self,
        query: str,
        search_type: str = "general",
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search the web, returning normalized and URL-deduplicated dictionaries."""
        result_limit = max_results or self._max_results
        ttl = _CACHE_TTL_HOURS.get(search_type, _CACHE_TTL_HOURS["general"])
        query_hash = _query_hash(query, search_type, result_limit)

        if self._cache_enabled and self._cache is not None:
            cached = self._cache.get(query_hash)
            if cached is not None:
                logger.debug("Web search cache hit: %s", query[:60])
                return cached

        try:
            results = self._search_provider(query, result_limit)
        except Exception:
            logger.exception("Unexpected web search provider failure (%s)", self._provider)
            raise

        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for result in results:
            url = str(result.get("url", ""))
            if url not in seen:
                seen.add(url)
                unique.append(result)

        if self._cache_enabled and self._cache is not None and unique:
            self._cache.set(query_hash, query, search_type, unique, ttl)
            self._cache.evict_lru(self._cache_max_entries)

        logger.info(
            "Web search (%s, type=%s): %d results for '%s'",
            self._provider,
            search_type,
            len(unique),
            query[:60],
        )
        return unique
