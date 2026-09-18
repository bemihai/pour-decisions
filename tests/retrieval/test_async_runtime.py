"""Tests for lifespan-owned asynchronous RAG resources."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.retrieval import AsyncRAGRuntimeResources
from src.retrieval import async_runtime


def _config(*, enable_hybrid: bool = False) -> SimpleNamespace:
    """Return the minimal configuration consumed by the async factory."""
    retrieval = SimpleNamespace(
        n_results=5,
        similarity_threshold=0.0,
        enable_hybrid=enable_hybrid,
        enable_reranking=True,
    )
    chroma = SimpleNamespace(
        client=SimpleNamespace(host="chroma.test", port=8100),
        collections=[SimpleNamespace(name="wine_books")],
        settings=SimpleNamespace(embedder="test-embedder", batch_size=10),
        retrieval=retrieval,
    )
    return SimpleNamespace(chroma=chroma)


@pytest.mark.asyncio
async def test_build_async_rag_runtime_returns_owned_resources_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful factory call should expose and close one shared client."""
    client = object()
    collection = object()
    vector_retriever = MagicMock(collection=collection)
    owner = MagicMock(client=client)
    owner.close = AsyncMock()
    reranker = object()

    monkeypatch.setattr(async_runtime._AsyncChromaClientOwner, "open", AsyncMock(return_value=owner))
    create_async = AsyncMock(return_value=vector_retriever)
    monkeypatch.setattr(async_runtime.ChromaRetriever, "create_async", create_async)
    monkeypatch.setattr(async_runtime, "build_reranker_from_config", lambda _cfg: reranker)

    config = _config()
    resources = await async_runtime.build_async_rag_runtime(config)

    assert isinstance(resources, AsyncRAGRuntimeResources)
    assert resources.config is config
    assert resources.chroma_client is client
    assert resources.collection is collection
    assert resources.retriever is vector_retriever
    assert resources.bm25_index is None
    assert resources.reranker is reranker
    create_async.assert_awaited_once_with(
        client=client,
        collection_name="wine_books",
        embedding_model="test-embedder",
        n_results=5,
        similarity_threshold=0.0,
        enable_cache=True,
        enable_query_expansion=False,
    )

    await resources.close()
    await resources.close()

    owner.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_build_async_rag_runtime_closes_partial_retriever_and_returns_degraded_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retriever initialization failure should not leak its async client."""
    owner = MagicMock(client=object())
    owner.close = AsyncMock()
    reranker = object()
    monkeypatch.setattr(async_runtime._AsyncChromaClientOwner, "open", AsyncMock(return_value=owner))
    monkeypatch.setattr(
        async_runtime.ChromaRetriever,
        "create_async",
        AsyncMock(side_effect=RuntimeError("missing collection")),
    )
    monkeypatch.setattr(async_runtime, "build_reranker_from_config", lambda _cfg: reranker)

    resources = await async_runtime.build_async_rag_runtime(_config())

    owner.close.assert_awaited_once_with()
    assert resources.chroma_client is None
    assert resources.collection is None
    assert resources.retriever is None
    assert resources.bm25_index is None
    assert resources.reranker is reranker


@pytest.mark.asyncio
async def test_build_async_rag_runtime_propagates_cancellation_after_partial_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation during retriever construction should close the client and propagate."""
    owner = MagicMock(client=object())
    owner.close = AsyncMock()
    monkeypatch.setattr(async_runtime._AsyncChromaClientOwner, "open", AsyncMock(return_value=owner))
    monkeypatch.setattr(
        async_runtime.ChromaRetriever,
        "create_async",
        AsyncMock(side_effect=async_runtime.asyncio.CancelledError),
    )

    with pytest.raises(async_runtime.asyncio.CancelledError):
        await async_runtime.build_async_rag_runtime(_config())

    owner.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_build_async_rag_runtime_closes_client_when_reranker_wait_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation after retriever construction should close the owned client."""
    owner = MagicMock(client=object())
    owner.close = AsyncMock()
    vector_retriever = MagicMock(collection=object())
    monkeypatch.setattr(async_runtime._AsyncChromaClientOwner, "open", AsyncMock(return_value=owner))
    monkeypatch.setattr(
        async_runtime.ChromaRetriever,
        "create_async",
        AsyncMock(return_value=vector_retriever),
    )
    monkeypatch.setattr(
        async_runtime.asyncio,
        "to_thread",
        AsyncMock(side_effect=async_runtime.asyncio.CancelledError),
    )

    with pytest.raises(async_runtime.asyncio.CancelledError):
        await async_runtime.build_async_rag_runtime(_config())

    owner.close.assert_awaited_once_with()
