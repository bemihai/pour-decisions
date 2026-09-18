"""Semantic parity and thread-placement tests for async hybrid retrieval."""

import threading

import pytest

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_analyzer import build_retrieval_query_plan


class _DualModeDenseRetriever:
    """Return the same dense pool from sync and async entry points."""

    def __init__(self, documents: list[dict[str, object]]) -> None:
        self.documents = documents
        self.sync_calls: list[tuple[str, int, dict[str, object]]] = []
        self.async_calls: list[tuple[str, int, dict[str, object], int]] = []

    def retrieve(self, query: str, n_results: int, **kwargs: object) -> list[dict[str, object]]:
        """Record and return the configured dense pool."""
        self.sync_calls.append((query, n_results, kwargs))
        return [dict(document) for document in self.documents[:n_results]]

    async def aretrieve(self, query: str, n_results: int, **kwargs: object) -> list[dict[str, object]]:
        """Record event-loop execution and return the configured dense pool."""
        self.async_calls.append((query, n_results, kwargs, threading.get_ident()))
        return [dict(document) for document in self.documents[:n_results]]


class _RecordingSparseIndex:
    """Return a fixed sparse pool while recording execution threads."""

    def __init__(self, documents: list[dict[str, object]]) -> None:
        self.documents = documents
        self.calls: list[tuple[str, int, int]] = []

    def search(self, query: str, top_k: int) -> list[dict[str, object]]:
        """Record and return the configured sparse pool."""
        self.calls.append((query, top_k, threading.get_ident()))
        return [dict(document) for document in self.documents[:top_k]]


def _doc(document_id: str, **scores: float) -> dict[str, object]:
    """Build one deterministic channel result."""
    return {
        "id": document_id,
        "document": f"Evidence for {document_id}.",
        "metadata": {},
        **scores,
    }


def _without_latencies(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Remove only nondeterministic timing values from retrieval diagnostics."""
    normalized: list[dict[str, object]] = []
    for result in results:
        candidate = dict(result)
        diagnostics = dict(candidate["retrieval_diagnostics"])
        diagnostics.pop("dense_latency_ms")
        diagnostics.pop("sparse_latency_ms")
        candidate["retrieval_diagnostics"] = diagnostics
        normalized.append(candidate)
    return normalized


@pytest.mark.asyncio
@pytest.mark.parametrize("use_rrf_fallback", [True, False])
async def test_aretrieve_matches_sync_union_and_places_blocking_work_off_loop(
    use_rrf_fallback: bool,
) -> None:
    """Async hybrid retrieval should preserve inputs, dictionaries, and ordering."""
    event_loop_thread = threading.get_ident()
    dense = _DualModeDenseRetriever(
        [_doc("dense", similarity=0.9), _doc("shared", similarity=0.8)]
    )
    sparse = _RecordingSparseIndex(
        [_doc("sparse", bm25_score=4.0), _doc("shared", bm25_score=3.0)]
    )
    retriever = HybridRetriever(
        dense,
        sparse,
        semantic_candidate_pool=2,
        bm25_candidate_pool=2,
        reranker_input_limit=3,
    )
    plan = build_retrieval_query_plan("What are the primary flavour characteristics of Nebbiolo?")

    sync_results = retriever.retrieve(
        "ignored",
        n_results=3,
        query_plan=plan,
        use_rrf_fallback=use_rrf_fallback,
        where={"source": "book.pdf"},
    )
    async_results = await retriever.aretrieve(
        "ignored",
        n_results=3,
        query_plan=plan,
        use_rrf_fallback=use_rrf_fallback,
        where={"source": "book.pdf"},
    )

    assert _without_latencies(async_results) == _without_latencies(sync_results)
    assert dense.async_calls == [
        (
            dense.sync_calls[0][0],
            dense.sync_calls[0][1],
            dense.sync_calls[0][2],
            event_loop_thread,
        )
    ]
    assert sparse.calls[0][:2] == sparse.calls[1][:2]
    assert sparse.calls[0][2] == event_loop_thread
    assert sparse.calls[1][2] != event_loop_thread

