"""Parity and thread-placement tests for async vector retrieval."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.retrieval.vector_retriever import ChromaRetriever


_RESULTS = {
    "ids": [["strong", "weak"]],
    "documents": [["Strong evidence.", "Weak evidence."]],
    "metadatas": [[{"source": "book.pdf"}, {"source": "book.pdf"}]],
    "distances": [[0.1, 0.8]],
}


class _RecordingEmbedder:
    """Return a fixed embedding while recording execution threads."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def embed_query(self, query: str) -> list[float]:
        """Record the worker and return one deterministic embedding."""
        self.calls.append((query, threading.get_ident()))
        return [0.25, 0.75]


@pytest.mark.asyncio
async def test_async_construction_and_retrieval_match_sync_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both paths should share query arguments, filtering, and result dictionaries."""
    event_loop_thread = threading.get_ident()
    load_threads: list[int] = []
    sync_embedder = _RecordingEmbedder()
    async_embedder = _RecordingEmbedder()

    def _load_embedder(*, model_name: str) -> _RecordingEmbedder:
        load_threads.append(threading.get_ident())
        return async_embedder if len(load_threads) == 2 else sync_embedder

    monkeypatch.setattr("src.retrieval.vector_retriever.get_embedder", _load_embedder)
    sync_collection = MagicMock()
    sync_collection.query.return_value = _RESULTS
    sync_client = MagicMock()
    sync_client.get_collection.return_value = sync_collection

    async_query_threads: list[int] = []

    async def _query(**kwargs: object) -> dict[str, list[list[object]]]:
        _ = kwargs
        async_query_threads.append(threading.get_ident())
        return _RESULTS

    async_collection = MagicMock()
    async_collection.query = AsyncMock(side_effect=_query)
    async_client = MagicMock()
    async_client.get_collection = AsyncMock(return_value=async_collection)

    sync_retriever = ChromaRetriever(
        sync_client,
        "wine_books",
        "test-embedder",
        similarity_threshold=0.3,
        enable_cache=False,
        enable_query_expansion=False,
    )
    async_retriever = await ChromaRetriever.create_async(
        async_client,
        "wine_books",
        "test-embedder",
        similarity_threshold=0.3,
        enable_cache=False,
        enable_query_expansion=False,
    )

    sync_result = sync_retriever.retrieve(
        "Barolo",
        n_results=2,
        where={"region": "Piedmont"},
        where_document={"$contains": "Nebbiolo"},
    )
    async_result = await async_retriever.aretrieve(
        "Barolo",
        n_results=2,
        where={"region": "Piedmont"},
        where_document={"$contains": "Nebbiolo"},
    )

    assert async_result == sync_result == [
        {
            "id": "strong",
            "document": "Strong evidence.",
            "metadata": {"source": "book.pdf"},
            "distance": 0.1,
            "similarity": 0.9,
        }
    ]
    assert async_collection.query.call_args.kwargs == sync_collection.query.call_args.kwargs
    assert load_threads[0] == event_loop_thread
    assert load_threads[1] != event_loop_thread
    assert async_embedder.calls[0][1] != event_loop_thread
    assert async_query_threads == [event_loop_thread]
    async_client.get_collection.assert_awaited_once_with("wine_books")


@pytest.mark.asyncio
async def test_aretrieve_preserves_cache_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    """The async path should reuse the existing cache key and hit accounting."""
    embedder = _RecordingEmbedder()
    monkeypatch.setattr("src.retrieval.vector_retriever.get_embedder", lambda **_kwargs: embedder)
    collection = MagicMock()
    collection.query = AsyncMock(return_value=_RESULTS)
    client = MagicMock()
    client.get_collection = AsyncMock(return_value=collection)
    retriever = await ChromaRetriever.create_async(
        client,
        "wine_books",
        "test-embedder",
        similarity_threshold=0.3,
        enable_query_expansion=False,
    )

    first = await retriever.aretrieve("Barolo", n_results=2)
    second = await retriever.aretrieve("Barolo", n_results=2)

    assert second is first
    collection.query.assert_awaited_once()
    assert len(embedder.calls) == 1
    assert retriever.get_cache_stats() == {
        "size": 1,
        "max_size": 100,
        "hits": 1,
        "misses": 1,
        "hit_rate": 0.5,
    }


@pytest.mark.asyncio
async def test_aretrieve_preserves_fail_soft_query_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async Chroma failures should retain the established empty-result contract."""
    monkeypatch.setattr(
        "src.retrieval.vector_retriever.get_embedder",
        lambda **_kwargs: _RecordingEmbedder(),
    )
    collection = MagicMock()
    collection.query = AsyncMock(side_effect=RuntimeError("Chroma unavailable"))
    client = MagicMock()
    client.get_collection = AsyncMock(return_value=collection)
    retriever = await ChromaRetriever.create_async(client, "wine_books", "test-embedder")

    assert await retriever.aretrieve("Barolo") == []
