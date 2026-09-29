"""Runtime tests for the approved bounded corrective retrieval attempt."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from src.retrieval.correction import CorrectionAttemptBudget
from src.retrieval.rag_service import execute_production_rag, execute_production_rag_async
from src.retrieval.web_fallback import WebSearchFallback


_ELIGIBLE_QUERY = "What are the ageing classifications for Rioja wines?"


def _config(*, enabled: bool = True, timeout_seconds: float = 3.0) -> SimpleNamespace:
    """Build the complete minimal configuration consumed by production RAG."""
    return SimpleNamespace(
        web_search=SimpleNamespace(auto_fallback=False),
        chroma=SimpleNamespace(
            retrieval=SimpleNamespace(
                n_results=2,
                enable_metadata_boost=False,
                rerank_top_k=2,
                rerank_threshold=0.0,
                min_retrieval_confidence=0.3,
                use_deduplication=False,
                enable_compression=False,
                correction=SimpleNamespace(
                    enabled=enabled,
                    timeout_seconds=timeout_seconds,
                ),
            ),
            chunking=SimpleNamespace(enable_small_to_big=False),
            settings=SimpleNamespace(embedder="test-embedder"),
        ),
    )


class _LowConfidenceReranker:
    """Retain every document with one stable low-confidence score."""

    def rerank(self, query: str, documents: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Return stable low-scored documents."""
        _ = query
        return [dict(document, rerank_score=-1.0) for document in documents[:top_k]]


def _document(document_id: str) -> dict[str, Any]:
    """Build one isolated retriever document."""
    return {
        "id": document_id,
        "document": f"Evidence from {document_id}.",
        "metadata": {"filename": "rioja.pdf", "page_number": 4},
        "similarity": 0.8,
    }


class _SequenceRetriever:
    """Return one first pass followed by one configurable correction result."""

    def __init__(
        self,
        corrected: list[dict[str, Any]],
        *,
        correction_error: Exception | None = None,
        correction_delay: float = 0.0,
    ) -> None:
        self.corrected = corrected
        self.correction_error = correction_error
        self.correction_delay = correction_delay
        self.sync_calls: list[str] = []
        self.async_calls: list[str] = []

    def retrieve(self, query: str, n_results: int) -> list[dict[str, Any]]:
        """Return the first or corrected synchronous result."""
        _ = n_results
        self.sync_calls.append(query)
        if len(self.sync_calls) == 1:
            return [_document("first")]
        if self.correction_delay:
            import time

            time.sleep(self.correction_delay)
        if self.correction_error is not None:
            raise self.correction_error
        return [dict(document) for document in self.corrected]

    async def aretrieve(self, query: str, n_results: int) -> list[dict[str, Any]]:
        """Return the first or corrected asynchronous result."""
        _ = n_results
        self.async_calls.append(query)
        if len(self.async_calls) == 1:
            return [_document("first")]
        if self.correction_delay:
            await asyncio.sleep(self.correction_delay)
        if self.correction_error is not None:
            raise self.correction_error
        return [dict(document) for document in self.corrected]


def test_sync_correction_selects_nonempty_novel_result() -> None:
    """An eligible call selects the successful correction only when it adds evidence."""
    retriever = _SequenceRetriever([_document("corrected"), _document("additional")])

    result = execute_production_rag(
        prompt=_ELIGIBLE_QUERY,
        config=_config(),
        model=None,
        retriever=retriever,
        reranker=None,
        message_history=[],
        generation_enabled=False,
    )

    assert len(retriever.sync_calls) == 2
    assert [chunk.id for chunk in result.context_chunks] == ["corrected", "additional"]
    assert [chunk.id for chunk in result.raw_retrieved_chunks] == ["corrected", "additional"]
    assert result.correction.status == "completed"
    assert result.correction.selected_result == "corrected"
    assert result.correction.selection_reason == "corrected_novel_chunks"
    assert result.correction.attempt_count == 1
    assert result.correction.model_attempts == 0
    assert result.correction.first_pass_chunk_count == 1
    assert result.correction.corrected_chunk_count == 2
    assert result.correction.novel_corrected_chunk_count == 2


@pytest.mark.parametrize(
    ("corrected", "expected_reason"),
    [([], "corrected_empty"), ([_document("first")], "no_novel_chunks")],
)
def test_sync_correction_ties_and_empty_results_preserve_first_pass(
    corrected: list[dict[str, Any]],
    expected_reason: str,
) -> None:
    """Successful correction without novel context cannot replace the first pass."""
    result = execute_production_rag(
        prompt=_ELIGIBLE_QUERY,
        config=_config(),
        model=None,
        retriever=_SequenceRetriever(corrected),
        reranker=None,
        message_history=[],
        generation_enabled=False,
    )

    assert [chunk.id for chunk in result.context_chunks] == ["first"]
    assert result.correction.selected_result == "first_pass"
    assert result.correction.selection_reason == expected_reason


def test_sync_correction_failure_and_timeout_preserve_first_pass() -> None:
    """Ordinary correction failures stay bounded and do not erase valid evidence."""
    failed = execute_production_rag(
        prompt=_ELIGIBLE_QUERY,
        config=_config(),
        model=None,
        retriever=_SequenceRetriever([], correction_error=RuntimeError("unavailable")),
        reranker=None,
        message_history=[],
        generation_enabled=False,
    )
    timed_out = execute_production_rag(
        prompt=_ELIGIBLE_QUERY,
        config=_config(timeout_seconds=0.005),
        model=None,
        retriever=_SequenceRetriever([_document("late")], correction_delay=0.05),
        reranker=None,
        message_history=[],
        generation_enabled=False,
    )

    assert [chunk.id for chunk in failed.context_chunks] == ["first"]
    assert failed.retrieval_error is None
    assert failed.correction.status == "failed"
    assert failed.correction.failure_reason == "retrieval_error"
    assert [chunk.id for chunk in timed_out.context_chunks] == ["first"]
    assert timed_out.correction.status == "timed_out"
    assert timed_out.correction.failure_reason == "timeout"


def test_web_fallback_and_generation_run_once_after_correction_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downstream external fallback and generation consume only the selected result."""
    config = _config()
    config.web_search.auto_fallback = True
    config.chroma.retrieval.rerank_threshold = None
    config.chroma.retrieval.min_retrieval_confidence = 0.8
    web_calls: list[str] = []
    generation_contexts: list[str] = []

    def _search(query: str, **_kwargs: Any) -> list[dict[str, str]]:
        web_calls.append(query)
        return [{"title": "Report", "snippet": "Web evidence", "url": "https://example.test"}]

    def _generate(_model: Any, _prompt: str, context: str, *_args: Any) -> str:
        generation_contexts.append(context)
        return "Answer [1]."

    monkeypatch.setattr(
        "src.retrieval.rag_service.build_web_fallback_from_config",
        lambda _config: WebSearchFallback(enabled=True, engine=SimpleNamespace(search=_search)),
    )
    monkeypatch.setattr("src.retrieval.rag_service.process_user_prompt", _generate)

    result = execute_production_rag(
        prompt=_ELIGIBLE_QUERY,
        config=config,
        model=object(),
        retriever=_SequenceRetriever([_document("corrected")]),
        reranker=_LowConfidenceReranker(),
        message_history=[],
    )

    assert web_calls == [_ELIGIBLE_QUERY]
    assert len(generation_contexts) == 1
    assert "Evidence from corrected." in generation_contexts[0]
    assert result.feature_usage.web_fallback is True


@pytest.mark.asyncio
async def test_async_correction_matches_selection_and_propagates_cancellation() -> None:
    """Async correction uses the same selection rule and never swallows caller cancellation."""
    retriever = _SequenceRetriever([_document("corrected")])
    result = await execute_production_rag_async(
        prompt=_ELIGIBLE_QUERY,
        config=_config(),
        model=None,
        retriever=retriever,
        reranker=None,
        message_history=[],
        generation_enabled=False,
        correction_budget=CorrectionAttemptBudget(),
    )

    assert len(retriever.async_calls) == 2
    assert [chunk.id for chunk in result.context_chunks] == ["corrected"]
    assert result.correction.selected_result == "corrected"

    cancelled = _SequenceRetriever([], correction_error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await execute_production_rag_async(
            prompt=_ELIGIBLE_QUERY,
            config=_config(),
            model=None,
            retriever=cancelled,
            reranker=None,
            message_history=[],
            generation_enabled=False,
        )


@pytest.mark.asyncio
async def test_async_correction_timeout_preserves_first_pass() -> None:
    """The async deadline cancels correction work and returns the valid first pass."""
    result = await execute_production_rag_async(
        prompt=_ELIGIBLE_QUERY,
        config=_config(timeout_seconds=0.005),
        model=None,
        retriever=_SequenceRetriever([_document("late")], correction_delay=0.05),
        reranker=None,
        message_history=[],
        generation_enabled=False,
    )

    assert [chunk.id for chunk in result.context_chunks] == ["first"]
    assert result.correction.status == "timed_out"
    assert result.correction.failure_reason == "timeout"
