"""Deterministic parity matrix for synchronous and asynchronous production RAG."""

import asyncio
import threading
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models import BaseChatModel

from src.agents.llm import ModelInternalError
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.rag_service import execute_production_rag, execute_production_rag_async
from src.retrieval.web_fallback import WebSearchFallback


class _DualRetriever:
    """Return isolated vector results through both execution modes."""

    def __init__(self, documents: list[dict[str, Any]], *, error: str | None = None) -> None:
        self.documents = documents
        self.error = error
        self.sync_calls: list[tuple[str, int]] = []
        self.async_calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, n_results: int) -> list[dict[str, Any]]:
        """Record one synchronous retrieval."""
        self.sync_calls.append((query, n_results))
        if self.error is not None:
            raise RuntimeError(self.error)
        return [dict(document) for document in self.documents[:n_results]]

    async def aretrieve(self, query: str, n_results: int) -> list[dict[str, Any]]:
        """Record one native async retrieval."""
        self.async_calls.append((query, n_results))
        if self.error is not None:
            raise RuntimeError(self.error)
        return [dict(document) for document in self.documents[:n_results]]


class _DualReranker:
    """Apply deterministic logits through sync and async reranker siblings."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, str, float | None, int | None]] = []

    def rerank(self, query: str, documents: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Apply rank-only scoring synchronously."""
        self.calls.append(("sync", query, None, top_k))
        return self._score(documents)[:top_k]

    async def arerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Apply rank-only scoring asynchronously."""
        self.calls.append(("async", query, None, top_k))
        return self._score(documents)[:top_k]

    def rerank_with_threshold(
        self,
        query: str,
        documents: list[dict[str, Any]],
        threshold: float,
        top_k: int | None,
    ) -> list[dict[str, Any]]:
        """Apply threshold scoring synchronously."""
        self.calls.append(("sync", query, threshold, top_k))
        return self._threshold(documents, threshold, top_k)

    async def arerank_with_threshold(
        self,
        query: str,
        documents: list[dict[str, Any]],
        threshold: float,
        top_k: int | None,
    ) -> list[dict[str, Any]]:
        """Apply threshold scoring asynchronously."""
        self.calls.append(("async", query, threshold, top_k))
        return self._threshold(documents, threshold, top_k)

    def _score(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach stable scores and sort descending."""
        scored = [dict(document, rerank_score=self.scores[str(document["id"])]) for document in documents]
        return sorted(scored, key=lambda document: document["rerank_score"], reverse=True)

    def _threshold(
        self,
        documents: list[dict[str, Any]],
        threshold: float,
        top_k: int | None,
    ) -> list[dict[str, Any]]:
        """Filter the stable scored result."""
        scored = [document for document in self._score(documents) if document["rerank_score"] >= threshold]
        return scored if top_k is None else scored[:top_k]


class _DenseChannel(_DualRetriever):
    """Adapt the dual vector fake to hybrid keyword arguments."""

    def retrieve(self, query: str, n_results: int, **_kwargs: Any) -> list[dict[str, Any]]:
        """Return the synchronous dense pool."""
        return super().retrieve(query, n_results)

    async def aretrieve(self, query: str, n_results: int, **_kwargs: Any) -> list[dict[str, Any]]:
        """Return the asynchronous dense pool."""
        return await super().aretrieve(query, n_results)


class _SparseChannel:
    """Return isolated sparse candidates and count searches."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return the configured sparse pool."""
        self.calls.append((query, top_k))
        return [dict(document) for document in self.documents[:top_k]]


class _RecordingSpan:
    """Minimal span context manager used to compare tracing calls."""

    def __init__(self, name: str, events: list[tuple[str, object]]) -> None:
        self.name = name
        self.events = events

    def __enter__(self) -> "_RecordingSpan":
        self.events.append(("enter", self.name))
        return self

    def __exit__(self, *_args: object) -> None:
        self.events.append(("exit", self.name))


class _RecordingTracer:
    """Create named spans without depending on an exporter."""

    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events

    def start_as_current_span(self, name: str) -> _RecordingSpan:
        """Return one recording span."""
        return _RecordingSpan(name, self.events)


def _config(
    *,
    threshold: float | None = None,
    auto_fallback: bool = False,
    metadata_boost: bool = False,
    deduplication: bool = False,
    small_to_big: bool = False,
    compression: bool = False,
) -> SimpleNamespace:
    """Build every configuration field consumed by production RAG."""
    return SimpleNamespace(
        web_search=SimpleNamespace(auto_fallback=auto_fallback),
        chroma=SimpleNamespace(
            retrieval=SimpleNamespace(
                n_results=2,
                enable_metadata_boost=metadata_boost,
                metadata_boost_factor=0.1,
                rerank_top_k=2,
                rerank_threshold=threshold,
                min_retrieval_confidence=0.8,
                use_deduplication=deduplication,
                deduplication_threshold=0.9,
                enable_compression=compression,
                compression_max_chars=80,
            ),
            chunking=SimpleNamespace(enable_small_to_big=small_to_big),
            settings=SimpleNamespace(embedder="test-embedder"),
        ),
    )


def _document(
    document_id: str = "book-1",
    *,
    score: float | None = None,
    similarity: float = 0.91,
) -> dict[str, Any]:
    """Build one source-bearing deterministic document."""
    metadata: dict[str, Any] = {
        "filename": "/books/wine-atlas.pdf",
        "page_number": 42,
        "region": "barolo",
    }
    if score is not None:
        metadata["score"] = score
    return {
        "id": document_id,
        "document": f"Evidence for {document_id} about Barolo and Nebbiolo.",
        "metadata": metadata,
        "similarity": similarity,
    }


def _semantic_result(result: object) -> dict[str, Any]:
    """Exclude only nondeterministic retrieval timing values from equality."""
    value = asdict(result)
    for collection in (value["raw_retrieved_chunks"], value["context_chunks"]):
        for chunk in collection:
            chunk["retrieval_diagnostics"].pop("dense_latency_ms", None)
            chunk["retrieval_diagnostics"].pop("sparse_latency_ms", None)
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("documents", [[_document()], []], ids=["success", "empty-context"])
async def test_vector_fallback_generation_and_tracing_match(
    monkeypatch: pytest.MonkeyPatch,
    documents: list[dict[str, Any]],
) -> None:
    """Vector-only success and empty context should preserve generation and spans."""
    retriever = _DualRetriever(documents)
    generation_calls: list[tuple[str, object, str, list[dict[str, Any]], dict[str, str] | None]] = []

    def _generate(
        _model: BaseChatModel,
        prompt: str,
        context: str,
        history: list[dict[str, Any]],
        trace_context: dict[str, str] | None,
    ) -> str:
        generation_calls.append(("sync", prompt, context, history, trace_context))
        return "Barolo uses Nebbiolo [1]."

    async def _generate_async(
        _model: BaseChatModel,
        prompt: str,
        context: str,
        history: list[dict[str, Any]],
        trace_context: dict[str, str] | None,
    ) -> str:
        generation_calls.append(("async", prompt, context, history, trace_context))
        return "Barolo uses Nebbiolo [1]."

    trace_events: list[tuple[str, object]] = []
    monkeypatch.setattr("src.retrieval.rag_service.process_user_prompt", _generate)
    monkeypatch.setattr("src.retrieval.rag_service.process_user_prompt_async", _generate_async)
    monkeypatch.setattr(
        "src.retrieval.rag_service.otel_trace.get_tracer",
        lambda _name: _RecordingTracer(trace_events),
    )
    monkeypatch.setattr(
        "src.retrieval.rag_service.set_span_attributes",
        lambda span, attributes: trace_events.append((span.name, attributes)),
    )
    history = [{"role": "human", "content": "Tell me about Piedmont."}]
    trace_context = {"request_id": "phase3", "agent_mode": "rag_only"}
    kwargs = {
        "prompt": "What is Barolo?",
        "config": _config(metadata_boost=True),
        "model": MagicMock(spec=BaseChatModel),
        "retriever": retriever,
        "reranker": None,
        "message_history": history,
        "trace_context": trace_context,
    }

    sync_result = execute_production_rag(**kwargs)
    sync_trace_events = list(trace_events)
    trace_events.clear()
    async_result = await execute_production_rag_async(**kwargs)

    assert _semantic_result(async_result) == _semantic_result(sync_result)
    assert retriever.async_calls == retriever.sync_calls
    assert generation_calls[1][1:] == generation_calls[0][1:]
    assert trace_events == sync_trace_events


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_fallback", [False, True], ids=["disabled", "enabled"])
@pytest.mark.parametrize("threshold", [None, -2.0], ids=["rank-only", "thresholded"])
async def test_threshold_deduplication_and_web_fallback_call_counts_match(
    monkeypatch: pytest.MonkeyPatch,
    auto_fallback: bool,
    threshold: float | None,
) -> None:
    """Low-confidence fallback should spend the same external calls in both modes."""
    event_loop_thread = threading.get_ident()
    web_calls: list[tuple[str, str, int]] = []

    def _search(
        query: str,
        search_type: str = "general",
        max_results: int | None = None,
    ) -> list[dict[str, str]]:
        _ = max_results
        web_calls.append((query, search_type, threading.get_ident()))
        return [
            {
                "title": "Current Barolo report",
                "snippet": "Current external evidence.",
                "url": "https://example.test/barolo",
            }
        ]

    monkeypatch.setattr(
        "src.retrieval.rag_service.build_web_fallback_from_config",
        lambda config: WebSearchFallback(
            enabled=config.web_search.auto_fallback,
            engine=SimpleNamespace(search=_search),
        ),
    )
    monkeypatch.setattr(
        "src.retrieval.rag_service.deduplicate_chunks",
        lambda documents, **_kwargs: [dict(document) for document in documents],
    )

    async def _deduplicate_async(
        documents: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        return [dict(document) for document in documents]

    monkeypatch.setattr("src.retrieval.rag_service.deduplicate_chunks_async", _deduplicate_async)
    retriever = _DualRetriever([_document(score=-1.0)])
    reranker = _DualReranker({"book-1": -1.0})
    kwargs = {
        "prompt": "What is Barolo?",
        "config": _config(threshold=threshold, auto_fallback=auto_fallback, deduplication=True),
        "model": None,
        "retriever": retriever,
        "reranker": reranker,
        "message_history": [],
        "generation_enabled": False,
    }

    sync_result = execute_production_rag(**kwargs)
    async_result = await execute_production_rag_async(**kwargs)

    assert _semantic_result(async_result) == _semantic_result(sync_result)
    assert reranker.calls[1][1:] == reranker.calls[0][1:]
    assert len(web_calls) == (2 if auto_fallback else 0)
    if auto_fallback:
        assert web_calls[0][:2] == web_calls[1][:2] == ("What is Barolo?", "general")
        assert web_calls[0][2] == event_loop_thread
        assert web_calls[1][2] != event_loop_thread


@pytest.mark.asyncio
async def test_hybrid_generation_disabled_matches_except_literal_latencies() -> None:
    """Hybrid union, diagnostics, and generation-disabled state should match."""
    dense = _DenseChannel([_document("dense")])
    sparse = _SparseChannel(
        [
            {
                "id": "sparse",
                "document": "Exact sparse evidence.",
                "metadata": {"filename": "terms.pdf", "page_number": 3},
                "bm25_score": 4.0,
            }
        ]
    )
    retriever = HybridRetriever(dense, sparse, semantic_candidate_pool=2, bm25_candidate_pool=2)
    kwargs = {
        "prompt": "What is Barolo?",
        "config": _config(),
        "model": None,
        "retriever": retriever,
        "reranker": None,
        "message_history": [],
        "generation_enabled": False,
    }

    sync_result = execute_production_rag(**kwargs)
    async_result = await execute_production_rag_async(**kwargs)

    assert _semantic_result(async_result) == _semantic_result(sync_result)
    assert dense.async_calls == dense.sync_calls
    assert sparse.calls[0] == sparse.calls[1]
    assert async_result.feature_usage.generation is False


@pytest.mark.asyncio
async def test_optional_parent_expansion_and_compression_match_and_use_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled blocking optional stages should preserve output off the loop."""
    event_loop_thread = threading.get_ident()
    expansion_threads: list[int] = []
    compression_threads: list[int] = []

    def _expand(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expansion_threads.append(threading.get_ident())
        return [dict(document, document=f"{document['document']} Expanded parent context.") for document in documents]

    def _compress(context: str, *, max_chars: int) -> str:
        compression_threads.append(threading.get_ident())
        return context[:max_chars]

    monkeypatch.setattr("src.retrieval.rag_service._expand_to_parent_context", _expand)
    monkeypatch.setattr("src.retrieval.rag_service.compress_context", _compress)
    retriever = _DualRetriever([_document()])
    kwargs = {
        "prompt": "What is Barolo?",
        "config": _config(small_to_big=True, compression=True),
        "model": None,
        "retriever": retriever,
        "reranker": None,
        "message_history": [],
        "generation_enabled": False,
    }

    sync_result = execute_production_rag(**kwargs)
    async_result = await execute_production_rag_async(**kwargs)

    assert _semantic_result(async_result) == _semantic_result(sync_result)
    assert expansion_threads[0] == compression_threads[0] == event_loop_thread
    assert expansion_threads[1] != event_loop_thread
    assert compression_threads[1] != event_loop_thread


@pytest.mark.asyncio
async def test_retrieval_failure_generation_disabled_matches_and_skips_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both modes should preserve the established fail-soft retrieval boundary."""
    sync_generation = MagicMock(side_effect=AssertionError("generation must remain disabled"))
    async_generation = AsyncMock(side_effect=AssertionError("generation must remain disabled"))
    monkeypatch.setattr("src.retrieval.rag_service.process_user_prompt", sync_generation)
    monkeypatch.setattr("src.retrieval.rag_service.process_user_prompt_async", async_generation)
    retriever = _DualRetriever([], error="phase3 retrieval unavailable")
    kwargs = {
        "prompt": "What is Barolo?",
        "config": _config(),
        "model": None,
        "retriever": retriever,
        "reranker": None,
        "message_history": [],
        "generation_enabled": False,
    }

    sync_result = execute_production_rag(**kwargs)
    async_result = await execute_production_rag_async(**kwargs)

    assert _semantic_result(async_result) == _semantic_result(sync_result)
    assert async_result.retrieval_error == "phase3 retrieval unavailable"
    sync_generation.assert_not_called()
    async_generation.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_failure_answer_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native async generation should retain the synchronous fail-soft answer."""
    failure = ModelInternalError("phase3 model unavailable")
    monkeypatch.setattr("src.agents.llm.invoke_llm", MagicMock(side_effect=failure))
    monkeypatch.setattr("src.agents.llm.ainvoke_llm", AsyncMock(side_effect=failure))
    kwargs = {
        "prompt": "What is Barolo?",
        "config": _config(),
        "model": MagicMock(spec=BaseChatModel),
        "retriever": None,
        "reranker": None,
        "message_history": [],
        "enable_retrieval": False,
    }

    sync_result = execute_production_rag(**kwargs)
    async_result = await execute_production_rag_async(**kwargs)

    assert _semantic_result(async_result) == _semantic_result(sync_result)
    assert async_result.answer == failure.default_message


@pytest.mark.asyncio
async def test_async_retrieval_cancellation_propagates() -> None:
    """The async service must not convert cancellation into retrieval failure."""

    class _CancelledRetriever:
        async def aretrieve(self, query: str, n_results: int) -> list[dict[str, Any]]:
            _ = query
            _ = n_results
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await execute_production_rag_async(
            prompt="What is Barolo?",
            config=_config(),
            model=None,
            retriever=_CancelledRetriever(),
            reranker=None,
            message_history=[],
            generation_enabled=False,
        )


@pytest.mark.asyncio
async def test_generation_requires_model_in_both_modes() -> None:
    """Both entry points should reject missing generation models identically."""
    kwargs = {
        "prompt": "What is Barolo?",
        "config": _config(),
        "model": None,
        "retriever": None,
        "reranker": None,
        "message_history": [],
    }

    with pytest.raises(ValueError, match="RAG generation requires a model"):
        execute_production_rag(**kwargs)
    with pytest.raises(ValueError, match="RAG generation requires a model"):
        await execute_production_rag_async(**kwargs)
