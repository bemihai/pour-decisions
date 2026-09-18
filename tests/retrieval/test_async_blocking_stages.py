"""Parity and thread-placement tests for optional blocking retrieval stages."""

import threading
from typing import Any

import pytest

from src.retrieval.context_builder import deduplicate_chunks, deduplicate_chunks_async
from src.retrieval.reranker import DocumentReranker


class _RecordingCrossEncoder:
    """Return fixed logits while recording prediction threads and inputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[tuple[str, str]], int]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Record one prediction and return deterministic scores."""
        self.calls.append((pairs, threading.get_ident()))
        return [-0.25, 0.75]


def _documents() -> list[dict[str, Any]]:
    """Build fresh documents because rank-only reranking mutates its inputs."""
    return [
        {"id": "first", "document": "Light-bodied wine.", "metadata": {}},
        {"id": "second", "document": "Structured Nebbiolo.", "metadata": {}},
    ]


@pytest.mark.asyncio
async def test_arerank_matches_sync_output_and_runs_prediction_off_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank-only async output should retain scoring, mutation, and ordering."""
    event_loop_thread = threading.get_ident()
    model = _RecordingCrossEncoder()
    monkeypatch.setattr("src.retrieval.reranker._get_reranker", lambda _name: model)
    reranker = DocumentReranker("test-model")

    sync_documents = _documents()
    async_documents = _documents()
    sync_results = reranker.rerank("Nebbiolo", sync_documents, top_k=2)
    async_results = await reranker.arerank("Nebbiolo", async_documents, top_k=2)

    assert async_results == sync_results
    assert async_documents == sync_documents
    assert model.calls[0][0] == model.calls[1][0]
    assert model.calls[0][1] == event_loop_thread
    assert model.calls[1][1] != event_loop_thread


@pytest.mark.asyncio
async def test_arerank_with_threshold_matches_sync_output_and_runs_off_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thresholded async output should retain copying, filtering, and ordering."""
    event_loop_thread = threading.get_ident()
    model = _RecordingCrossEncoder()
    monkeypatch.setattr("src.retrieval.reranker._get_reranker", lambda _name: model)
    reranker = DocumentReranker("test-model")

    sync_documents = _documents()
    async_documents = _documents()
    sync_results = reranker.rerank_with_threshold(
        "Nebbiolo",
        sync_documents,
        threshold=0.0,
        top_k=1,
    )
    async_results = await reranker.arerank_with_threshold(
        "Nebbiolo",
        async_documents,
        threshold=0.0,
        top_k=1,
    )

    assert async_results == sync_results
    assert async_documents == sync_documents
    assert model.calls[0][0] == model.calls[1][0]
    assert model.calls[0][1] == event_loop_thread
    assert model.calls[1][1] != event_loop_thread


@pytest.mark.asyncio
async def test_async_semantic_deduplication_matches_sync_and_runs_off_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dedup bridge should preserve arguments and retained dictionaries."""
    event_loop_thread = threading.get_ident()
    calls: list[tuple[list[dict[str, Any]], float, str, bool, int]] = []

    def _deduplicate_context(
        documents: list[dict[str, Any]],
        *,
        similarity_threshold: float,
        embedding_model: str,
        use_hash_first: bool,
    ) -> list[dict[str, Any]]:
        calls.append(
            (
                documents,
                similarity_threshold,
                embedding_model,
                use_hash_first,
                threading.get_ident(),
            )
        )
        return [dict(documents[1])]

    monkeypatch.setattr("src.chroma.deduplication.deduplicate_context", _deduplicate_context)
    documents = _documents()

    sync_results = deduplicate_chunks(documents, similarity_threshold=0.85, embedding_model="test-embedder")
    async_results = await deduplicate_chunks_async(
        documents,
        similarity_threshold=0.85,
        embedding_model="test-embedder",
    )

    assert async_results == sync_results
    assert calls[0][:-1] == calls[1][:-1]
    assert calls[0][-1] == event_loop_thread
    assert calls[1][-1] != event_loop_thread

