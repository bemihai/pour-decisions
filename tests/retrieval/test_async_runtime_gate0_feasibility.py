"""Live Gate 0 probes for installed async resources and shared retrieval state."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from typing import Any

import chromadb
from chromadb.api.async_fastapi import AsyncFastAPI
from chromadb.api.shared_system_client import SharedSystemClient
import pytest

from src.retrieval import HybridRetriever, build_reranker_from_config, build_retriever_from_config
from src.retrieval.rag_service import RAGExecutionResult, execute_production_rag
from src.utils import get_config


_HOST = os.getenv("CHROMA_HOST", "localhost")
_PORT = int(os.getenv("CHROMA_PORT", "8100"))
_COLLECTION = "wine_books"
_QUERY = "What grape is used in Barolo?"


@dataclass
class _InstalledAsyncChromaOwner:
    """Prototype explicit ownership around Chroma 1.3.7's public client factory."""

    client: Any
    system_identifier: str

    @classmethod
    async def start(
        cls,
        *,
        host: str,
        port: int,
        tenant: str = "default_tenant",
    ) -> "_InstalledAsyncChromaOwner":
        """Create a client and clean newly allocated systems if validation fails."""
        previous_identifiers = set(SharedSystemClient._identifier_to_system)
        try:
            client = await chromadb.AsyncHttpClient(host=host, port=port, tenant=tenant)
        except Exception:
            await _cleanup_new_chroma_systems(previous_identifiers)
            raise
        return cls(client=client, system_identifier=client._identifier)

    async def close(self) -> None:
        """Close installed HTTP transports and remove the owned shared system."""
        system = SharedSystemClient._identifier_to_system.pop(self.system_identifier)
        await self.client._server._cleanup()
        await asyncio.to_thread(system.stop)


async def _cleanup_new_chroma_systems(previous_identifiers: set[str]) -> None:
    """Clean systems allocated by an AsyncHttpClient call that did not return."""
    new_identifiers = set(SharedSystemClient._identifier_to_system) - previous_identifiers
    for identifier in new_identifiers:
        system = SharedSystemClient._identifier_to_system.pop(identifier)
        server = system.instance(AsyncFastAPI)
        await server._cleanup()
        await asyncio.to_thread(system.stop)


def _semantic_signature(result: RAGExecutionResult) -> tuple[Any, ...]:
    """Return stable retrieval fields while excluding measured stage latencies."""
    return (
        result.answer,
        result.context,
        result.normalized_query,
        result.retrieval_query_plan,
        [chunk.id for chunk in result.raw_retrieved_chunks],
        [chunk.id for chunk in result.context_chunks],
        [source.to_dict() for source in result.sources],
        result.feature_usage.to_dict(),
        result.retrieval_error,
        result.low_confidence,
        result.rerank_threshold,
    )


@pytest.mark.integration
async def test_installed_async_chroma_supports_concurrent_reads_and_normal_shutdown() -> None:
    """Use one installed async client concurrently, then close its transport."""
    owner = await _InstalledAsyncChromaOwner.start(host=_HOST, port=_PORT)
    try:
        collection = await owner.client.get_collection(_COLLECTION)
        counts = await asyncio.gather(collection.count(), collection.count(), collection.count())
        heartbeats = await asyncio.gather(owner.client.heartbeat(), owner.client.heartbeat())

        assert counts == [counts[0]] * 3
        assert counts[0] > 0
        assert all(heartbeat > 0 for heartbeat in heartbeats)
        assert AsyncFastAPI._clients
    finally:
        await owner.close()

    assert not AsyncFastAPI._clients
    assert owner.system_identifier not in SharedSystemClient._identifier_to_system


@pytest.mark.integration
async def test_installed_async_chroma_cleans_partial_startup_failure() -> None:
    """Remove the transport and system allocated before tenant validation fails."""
    previous_identifiers = set(SharedSystemClient._identifier_to_system)

    with pytest.raises(Exception):
        await _InstalledAsyncChromaOwner.start(
            host=_HOST,
            port=_PORT,
            tenant="m06b-gate0-missing-tenant",
        )

    assert set(SharedSystemClient._identifier_to_system) == previous_identifiers
    assert not AsyncFastAPI._clients


@pytest.mark.integration
async def test_shared_production_retrieval_resources_are_concurrency_safe() -> None:
    """Exercise real cache, BM25, reranker, and semantic deduplication concurrently."""
    config = get_config()
    retriever = build_retriever_from_config(config)
    reranker = build_reranker_from_config(config)

    assert isinstance(retriever, HybridRetriever)
    assert reranker is not None
    assert config.chroma.retrieval.use_deduplication is True

    def _execute() -> RAGExecutionResult:
        return execute_production_rag(
            prompt=_QUERY,
            config=config,
            model=None,
            retriever=retriever,
            reranker=reranker,
            message_history=[],
            generation_enabled=False,
        )

    results = await asyncio.gather(*[asyncio.to_thread(_execute) for _ in range(3)])
    signatures = [_semantic_signature(result) for result in results]

    assert all(result.retrieval_error is None for result in results)
    assert all(result.context_chunks for result in results)
    assert signatures == [signatures[0]] * 3
    assert retriever.vector_retriever.get_cache_stats()["size"] == 1
