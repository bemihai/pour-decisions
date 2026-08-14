"""Tests for the low-confidence web fallback adapter."""

import hashlib
from unittest.mock import MagicMock, patch

from src.retrieval.confidence import RetrievalResult
from src.retrieval.web_fallback import WebSearchFallback


def _result(*, low_confidence: bool) -> RetrievalResult:
    """Build a minimal book-retrieval result."""
    return RetrievalResult(
        documents=[{"id": "book", "document": "Book evidence", "metadata": {"source": "book.pdf"}}],
        confidence=0.0 if low_confidence else 0.9,
        low_confidence=low_confidence,
    )


def test_disabled_fallback_never_triggers() -> None:
    """Configuration must override low confidence."""
    assert WebSearchFallback(enabled=False).should_trigger(_result(low_confidence=True)) is False


def test_high_confidence_never_triggers() -> None:
    """High-confidence book evidence must avoid an external call."""
    assert WebSearchFallback(enabled=True).should_trigger(_result(low_confidence=False)) is False


def test_enabled_low_confidence_triggers() -> None:
    """Enabled fallback must identify low-confidence results."""
    assert WebSearchFallback(enabled=True).should_trigger(_result(low_confidence=True)) is True


def test_successful_fetch_appends_normalized_web_documents() -> None:
    """Book evidence retains precedence and web evidence gets stable metadata."""
    url = "https://example.test/current-wine-report"
    engine = MagicMock()
    engine.search.return_value = [{"title": "Current report", "snippet": "Fresh facts", "url": url}]

    result = WebSearchFallback(enabled=True, engine=engine).fetch_and_merge(
        "latest wine report",
        _result(low_confidence=True),
    )

    engine.search.assert_called_once_with("latest wine report", search_type="general")
    assert [document["id"] for document in result.documents] == [
        "book",
        f"web_{hashlib.sha256(url.encode()).hexdigest()}",
    ]
    assert result.documents[1] == {
        "id": f"web_{hashlib.sha256(url.encode()).hexdigest()}",
        "document": "Current report\n\nFresh facts",
        "metadata": {
            "source": "web",
            "url": url,
            "title": "Current report",
            "quality_score": 0.6,
            "filename": "web_search",
        },
        "rerank_score": 0.0,
    }
    assert result.web_fallback_used is True


def test_provider_failure_preserves_original_result() -> None:
    """External errors must leave usable book results unchanged."""
    engine = MagicMock()
    engine.search.side_effect = RuntimeError("provider unavailable")
    original = _result(low_confidence=True)

    result = WebSearchFallback(enabled=True, engine=engine).fetch_and_merge("query", original)

    assert result is original
    assert result.web_fallback_used is False


def test_missing_api_key_preserves_original_result() -> None:
    """Lazy service construction failures must fail safe."""
    original = _result(low_confidence=True)
    with patch(
        "src.retrieval.web_fallback.WineWebSearchEngine",
        side_effect=ValueError("Tavily API key not found"),
    ):
        result = WebSearchFallback(enabled=True).fetch_and_merge("query", original)

    assert result is original


def test_empty_provider_result_preserves_original_result() -> None:
    """An empty search result is not reported as a successful fallback."""
    engine = MagicMock()
    engine.search.return_value = []
    original = _result(low_confidence=True)

    result = WebSearchFallback(enabled=True, engine=engine).fetch_and_merge("query", original)

    assert result is original
