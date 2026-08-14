"""Automatic web-search fallback for low-confidence book retrieval."""

import hashlib
from typing import Any, Protocol

from src.services.web_search import WineWebSearchEngine
from src.utils import logger

from .confidence import RetrievalResult


class WebSearchEngine(Protocol):
    """Small dependency boundary required by the fallback adapter."""

    def search(
        self,
        query: str,
        search_type: str = "general",
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return normalized web result dictionaries."""


class WebSearchFallback:
    """Append cached web evidence when book retrieval confidence is low."""

    def __init__(self, enabled: bool, engine: WebSearchEngine | None = None) -> None:
        self.enabled = enabled
        self._engine = engine

    def should_trigger(self, result: RetrievalResult) -> bool:
        """Return whether this result is eligible for automatic fallback."""
        return self.enabled and result.low_confidence

    def fetch_and_merge(self, query: str, book_results: RetrievalResult) -> RetrievalResult:
        """Append normalized web documents after book documents when eligible."""
        if not self.should_trigger(book_results):
            return book_results

        try:
            engine = self._engine or WineWebSearchEngine()
            web_results = engine.search(query, search_type="general")
        except Exception as exc:
            logger.warning("Automatic web fallback unavailable: %s", exc)
            return book_results

        web_documents = [_to_document(result) for result in web_results]
        if not web_documents:
            logger.warning("Automatic web fallback returned no results")
            return book_results

        return RetrievalResult(
            documents=[*book_results.documents, *web_documents],
            confidence=book_results.confidence,
            low_confidence=book_results.low_confidence,
            web_fallback_used=True,
        )


def _to_document(result: dict[str, Any]) -> dict[str, Any]:
    """Convert one normalized service result to a retrieval document."""
    url = str(result.get("url", ""))
    title = str(result.get("title", ""))
    snippet = str(result.get("snippet", result.get("content", "")))
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    return {
        "id": f"web_{url_hash}",
        "document": f"{title}\n\n{snippet}",
        "metadata": {
            "source": "web",
            "url": url,
            "title": title,
            "quality_score": 0.6,
            "filename": "web_search",
        },
        "rerank_score": 0.0,
    }
