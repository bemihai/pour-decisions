"""LangChain wrappers for the shared wine web-search service."""

from langchain_core.tools import tool

from src.services.web_search import WebSearchCache, WineWebSearchEngine, _normalize_query, _query_hash
from src.utils import logger


def _format_search_results(results: list[dict]) -> str:
    """Format structured search results for agent tool output."""
    if not results:
        return "No web search results found for this query."

    lines: list[str] = []
    for index, result in enumerate(results, 1):
        lines.append(f"[{index}] {result.get('title', 'No title')}")
        lines.append(f"    {result.get('snippet', '')}")
        lines.append(f"    Source: {result.get('url', '')}")
        lines.append("")
    return "\n".join(lines).rstrip()


_engine: WineWebSearchEngine | None = None


def _get_engine() -> WineWebSearchEngine:
    """Return the lazily initialized shared service instance."""
    global _engine
    if _engine is None:
        _engine = WineWebSearchEngine()
    return _engine


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
    except Exception as exc:
        logger.error("search_web_for_wine failed: %s", exc)
        return f"Web search unavailable: {exc}"


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
    except Exception as exc:
        logger.error("search_wine_price failed: %s", exc)
        return f"Web search unavailable: {exc}"


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
    except Exception as exc:
        logger.error("search_wine_reviews failed: %s", exc)
        return f"Web search unavailable: {exc}"
