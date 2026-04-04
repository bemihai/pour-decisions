"""
Web search tools for real-time wine information retrieval.

Provides LangChain-compatible tools that query the web for current wine prices,
reviews, producer details, and availability. Results are cached in a dedicated
SQLite database to minimise external API calls.

Active provider: Tavily (configurable via app_config.yml web_search.provider).
The WineWebSearchEngine abstraction allows adding a new provider by implementing
a single branch without touching the tools or agents.
"""

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from src.utils import get_config, logger, find_project_root


# Cache TTL in hours per search type
_CACHE_TTL_HOURS: dict[str, int] = {
    "price": 24,
    "review": 168,    # 7 days
    "producer": 720,  # 30 days
    "general": 12,
}

# Stop words stripped before query hashing (question words + common fillers)
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "of", "for", "in", "and", "or", "to", "me",
    "what", "where", "when", "how", "who", "which", "why",
    "does", "do", "did", "are", "was", "were", "been",
    "can", "could", "would", "should", "tell", "about",
    "much", "does", "cost", "i", "my",
})


# ---------------------------------------------------------------------------
# Query normalisation
# ---------------------------------------------------------------------------

def _normalize_query(query: str) -> str:
    """Return a normalized, hash-ready representation of a search query.

    Lowercases, strips stop words, and sorts tokens so minor variations of the
    same query share the same cache key.

    Args:
        query: Raw search query string.

    Returns:
        Normalized string suitable for SHA-256 hashing.

    Example:
        >>> _normalize_query("What is the price of Barolo")
        'barolo price'
        >>> _normalize_query("barolo price")
        'barolo price'
    """
    tokens = query.lower().split()
    tokens = [t for t in tokens if t not in _STOP_WORDS]
    return " ".join(sorted(tokens))


def _query_hash(query: str, search_type: str = "general", max_results: int | None = None) -> str:
    """Return the SHA-256 hash of a normalized query string combined with search parameters.

    Including search_type and max_results ensures that calls with different parameters
    do not collide in the cache (e.g. different result counts or search categories).
    Pass max_results=None (default) when the count is not relevant to the cache key.
    """
    key = f"{_normalize_query(query)}|{search_type}|{max_results}"
    return hashlib.sha256(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------

class WebSearchCache:
    """SQLite-backed cache for web search results.

    Uses a dedicated database separate from wine_cellar.db so the cache can
    be cleared or deleted without affecting cellar data.

    Args:
        db_path: Path to the SQLite database file. Pass ':memory:' for tests.
    """

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

    # Incremental DDL statements applied when the table already exists but is
    # missing columns added in later versions of the schema.
    _MIGRATIONS: list[str] = [
        "ALTER TABLE web_search_cache ADD COLUMN last_access_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    ]

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        # For in-memory databases use a shared-cache URI so all connections see
        # the same database (required for tests and for evict_expired on init).
        self._in_memory = self._db_path == ":memory:"
        if self._in_memory:
            self._db_path = "file::memory:?cache=shared"
        with self._connect() as conn:
            conn.execute(self._CREATE_TABLE)
            self._apply_migrations(conn)
        self.evict_expired()

    def _apply_migrations(self, conn: sqlite3.Connection) -> None:
        """Add any columns that are present in _MIGRATIONS but absent from the table.

        SQLite does not support IF NOT EXISTS on ALTER TABLE, so existing columns
        are detected via PRAGMA table_info before attempting each statement.

        Args:
            conn: An open SQLite connection to use for the migration.
        """
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(web_search_cache)")}
        for ddl in self._MIGRATIONS:
            col = ddl.split("ADD COLUMN")[1].strip().split()[0]
            if col not in existing_columns:
                conn.execute(ddl)
                logger.debug(f"Applied cache migration: {ddl}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, uri=self._in_memory)
        conn.row_factory = sqlite3.Row
        return conn

    def _now_iso(self) -> str:
        """Return the current UTC time as an ISO 8601 string."""
        return datetime.now(timezone.utc).isoformat()

    def get(self, query_hash: str) -> list[dict] | None:
        """Return cached results if present and not expired, else None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT results_json, expires_at FROM web_search_cache WHERE query_hash = ?",
                (query_hash,),
            ).fetchone()

        if row is None:
            return None

        if datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
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
        results: list[dict],
        ttl_hours: int,
    ) -> None:
        """Insert or replace a cache entry."""
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
        """Delete all rows whose TTL has elapsed."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM web_search_cache WHERE expires_at <= ?",
                (self._now_iso(),),
            )

    def evict_lru(self, max_entries: int) -> None:
        """Delete the least recently accessed entries when the cache exceeds max_entries."""
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM web_search_cache").fetchone()[0]
            excess = count - max_entries
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


# ---------------------------------------------------------------------------
# Search engine wrapper
# ---------------------------------------------------------------------------

class WineWebSearchEngine:
    """Provider-agnostic web search engine with integrated caching.

    Reads provider and cache config from app_config.yml on init. All provider
    interaction is isolated to this class; adding a new provider requires only a
    new branch in _search_provider — no changes to the tools or agents.

    Args:
        cfg: Omegaconf DictConfig node for web_search. Loaded from app_config.yml
             if not provided.

    Example:
        >>> engine = WineWebSearchEngine()
        >>> results = engine.search("Sassicaia 2019 price", search_type="price")
    """

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

        # Resolve provider client
        self._client = self._init_client(cfg)
        logger.info(f"WineWebSearchEngine initialized with provider: {self._provider}")

    def _init_client(self, cfg: Any) -> Any:
        """Instantiate the provider client."""
        if self._provider == "tavily":
            from tavily import TavilyClient
            key_env = cfg.tavily.api_key_env
            api_key = os.environ.get(key_env, "")
            if not api_key:
                raise ValueError(
                    f"Tavily API key not found. Set the {key_env} environment variable."
                )
            return TavilyClient(api_key=api_key)
        raise ValueError(
            f"Unsupported web search provider: '{self._provider}'. "
            "Add a new branch in WineWebSearchEngine._init_client to support it."
        )

    def _search_provider(self, query: str, max_results: int) -> list[dict]:
        """Execute the search against the active provider and return normalised results."""
        if self._provider == "tavily":
            response = self._client.search(query, max_results=max_results)
            return [
                {"title": r["title"], "snippet": r["content"], "url": r["url"]}
                for r in response.get("results", [])
            ]
        return []

    def search(self, query: str, search_type: str = "general", max_results: int | None = None) -> list[dict]:
        """Search the web for wine information, using cache when available.

        Args:
            query: The search query string.
            search_type: One of 'general', 'price', 'review', 'producer'.
                         Determines cache TTL.
            max_results: Number of results to return. Falls back to config default.

        Returns:
            List of result dicts with keys 'title', 'snippet', 'url'.
            Returns empty list on provider error.
        """
        n = max_results or self._max_results
        ttl = _CACHE_TTL_HOURS.get(search_type, _CACHE_TTL_HOURS["general"])
        qhash = _query_hash(query, search_type, n)

        if self._cache_enabled:
            cached = self._cache.get(qhash)
            if cached is not None:
                logger.debug(f"Web search cache hit: {query[:60]}")
                return cached

        try:
            results = self._search_provider(query, n)
        except Exception as e:
            logger.warning(f"Web search provider error ({self._provider}): {e}")
            return []

        # URL-based deduplication
        seen: set[str] = set()
        unique: list[dict] = []
        for r in results:
            url = r.get("url", "")
            if url not in seen:
                seen.add(url)
                unique.append(r)

        if self._cache_enabled and unique:
            self._cache.set(qhash, query, search_type, unique, ttl)
            self._cache.evict_lru(self._cache_max_entries)

        logger.info(f"Web search ({self._provider}, type={search_type}): {len(unique)} results for '{query[:60]}'")
        return unique


# ---------------------------------------------------------------------------
# Result formatter
# ---------------------------------------------------------------------------

def _format_search_results(results: list[dict]) -> str:
    """Format a list of search result dicts into an LLM-readable string.

    Args:
        results: List of dicts with 'title', 'snippet', and 'url' keys.

    Returns:
        Numbered string suitable for inclusion in an LLM prompt, or a fallback
        message if results is empty.
    """
    if not results:
        return "No web search results found for this query."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title', 'No title')}")
        lines.append(f"    {r.get('snippet', '')}")
        lines.append(f"    Source: {r.get('url', '')}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Module-level engine singleton (lazy init)
# ---------------------------------------------------------------------------

_engine: WineWebSearchEngine | None = None


def _get_engine() -> WineWebSearchEngine:
    """Return the shared engine instance, initialising on first call."""
    global _engine
    if _engine is None:
        _engine = WineWebSearchEngine()
    return _engine


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------

@tool
def search_web_for_wine(
    query: str,
    search_type: str = "general",
    max_results: int = 5,
) -> str:
    """Search the web for real-time wine information.

    Use this tool when the user asks about current wine prices, availability,
    recent reviews, producer news, or any information that is not covered by
    the local wine knowledge base or the cellar database.

    Do NOT use for general wine education questions (what is tannin, how is
    Champagne made, etc.) — the RAG knowledge base answers those better.

    Args:
        query: The search query. Be specific: include wine name, vintage, and
               region when known. Examples:
               - "Chateau Margaux 2015 current price"
               - "Domaine Leflaive winemaker 2025"
               - "Bordeaux 2023 en primeur results"
        search_type: Query category — affects cache TTL. One of:
                     'general' (12 h), 'price' (24 h), 'review' (7 days),
                     'producer' (30 days). Default 'general'.
        max_results: Number of results to return (1-10). Default 5.

    Returns:
        Numbered list of search results with title, snippet, and source URL.
        Returns a fallback message if no results are found.

    Example:
        >>> search_web_for_wine.invoke({"query": "Sassicaia 2019 release news", "search_type": "general"})
        >>> search_web_for_wine.invoke({"query": "Ridge Monte Bello winemaker", "search_type": "producer"})
    """
    try:
        max_results = min(max(max_results, 1), 10)
        suffix = {
            "price": " wine price buy retail",
            "review": " wine review score rating",
            "producer": " winery winemaker producer",
        }.get(search_type, "")
        full_query = f"{query}{suffix}".strip()
        results = _get_engine().search(full_query, search_type=search_type, max_results=max_results)
        return _format_search_results(results)
    except Exception as e:
        logger.error(f"search_web_for_wine failed: {e}")
        return f"Web search unavailable: {e}"


@tool
def search_wine_price(
    wine_name: str,
    vintage: int | None = None,
) -> str:
    """Search the web for current retail prices of a specific wine.

    Use this tool when the user asks how much a wine costs, whether a price is
    fair, or where to buy a wine. Prefer this over search_web_for_wine when the
    question is specifically about pricing.

    Args:
        wine_name: Full wine name including producer if known, e.g. "Opus One",
                   "Giacomo Conterno Barolo Riserva", "Chateau Petrus".
        vintage: Optional vintage year, e.g. 2018. Omit for non-vintage wines.

    Returns:
        Numbered list of search results focused on price and availability.

    Example:
        >>> search_wine_price.invoke({"wine_name": "Sassicaia", "vintage": 2019})
        >>> search_wine_price.invoke({"wine_name": "Domaine Leflaive Batard-Montrachet", "vintage": 2020})
    """
    try:
        vintage_str = str(vintage) if vintage else ""
        query = f"{wine_name} {vintage_str} wine price buy retail".strip()
        results = _get_engine().search(query, search_type="price")
        return _format_search_results(results)
    except Exception as e:
        logger.error(f"search_wine_price failed: {e}")
        return f"Web search unavailable: {e}"


@tool
def search_wine_reviews(
    wine_name: str,
    vintage: int | None = None,
    reviewer: str | None = None,
) -> str:
    """Search the web for critic reviews and scores for a specific wine.

    Use this tool when the user asks about ratings, scores, tasting notes from
    major publications, or critic opinions for a wine.

    Args:
        wine_name: Full wine name including producer if known.
        vintage: Optional vintage year.
        reviewer: Optional publication or critic name, e.g. "Wine Advocate",
                  "Vinous", "James Suckling", "Jancis Robinson", "Decanter".
                  Omit to search all sources.

    Returns:
        Numbered list of search results focused on reviews and scores.

    Example:
        >>> search_wine_reviews.invoke({"wine_name": "Barolo Riserva Monfortino", "vintage": 2016})
        >>> search_wine_reviews.invoke({"wine_name": "Penfolds Grange", "vintage": 2018, "reviewer": "Wine Advocate"})
    """
    try:
        vintage_str = str(vintage) if vintage else ""
        reviewer_str = reviewer or ""
        query = f"{wine_name} {vintage_str} {reviewer_str} wine review score rating".strip()
        results = _get_engine().search(query, search_type="review")
        return _format_search_results(results)
    except Exception as e:
        logger.error(f"search_wine_reviews failed: {e}")
        return f"Web search unavailable: {e}"










