"""Lifespan-owned resources for the asynchronous production RAG path."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.async_fastapi import AsyncFastAPI
from chromadb.api.shared_system_client import SharedSystemClient

from src.chroma.bm25_builder import (
    BM25SyncError,
    compute_chunk_ids_sha256,
    load_bm25_sync_manifest,
)
from src.utils import logger

from .factory import _resolve_bm25_manifest_path, build_reranker_from_config
from .hybrid_retriever import HybridRetriever
from .keyword_search import BM25Index
from .reranker import DocumentReranker
from .vector_retriever import ChromaRetriever


@dataclass
class _AsyncChromaClientOwner:
    """Own the transport and shared system allocated by Chroma 1.3.7."""

    client: Any
    system_identifier: str
    _closed: bool = field(default=False, init=False)

    @classmethod
    async def open(cls, *, host: str, port: int) -> "_AsyncChromaClientOwner":
        """Create an async client and clean allocations if validation fails."""
        previous_identifiers = set(SharedSystemClient._identifier_to_system)
        try:
            client = await chromadb.AsyncHttpClient(host=host, port=port)
        except BaseException:
            await _cleanup_new_chroma_systems(previous_identifiers)
            raise
        return cls(client=client, system_identifier=client._identifier)

    async def close(self) -> None:
        """Close the installed async transport and stop its shared system once."""
        if self._closed:
            return
        self._closed = True
        system = SharedSystemClient._identifier_to_system.pop(self.system_identifier, None)
        if system is None:
            return
        try:
            await self.client._server._cleanup()
        finally:
            await asyncio.to_thread(system.stop)


@dataclass
class AsyncRAGRuntimeResources:
    """Typed resources shared by async RAG callers for one API lifespan."""

    config: Any
    chroma_client: Any | None = None
    collection: Any | None = None
    retriever: HybridRetriever | ChromaRetriever | None = None
    bm25_index: BM25Index | None = None
    reranker: DocumentReranker | None = None
    _chroma_owner: _AsyncChromaClientOwner | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    async def close(self) -> None:
        """Close owned async resources in reverse construction order once."""
        if self._closed:
            return
        self._closed = True
        if self._chroma_owner is not None:
            await self._chroma_owner.close()


async def build_async_rag_runtime(
    cfg: Any,
    *,
    collection_name: str | None = None,
    enable_cache: bool = True,
    enable_query_expansion: bool = False,
) -> AsyncRAGRuntimeResources:
    """Build the configured async RAG resources for one application lifespan.

    Retrieval startup remains fail-soft, matching the existing API behavior.
    Any async Chroma allocation that cannot produce a usable retriever is
    closed before the degraded bundle is returned. Cancellation is propagated
    after the same cleanup.
    """
    resources = AsyncRAGRuntimeResources(config=cfg)
    chroma_cfg = cfg.chroma
    retrieval_cfg = chroma_cfg.retrieval
    resolved_collection_name = collection_name or chroma_cfg.collections[0].name

    owner: _AsyncChromaClientOwner | None = None
    try:
        owner = await _AsyncChromaClientOwner.open(
            host=str(chroma_cfg.client.host),
            port=int(chroma_cfg.client.port),
        )
        vector_retriever = await ChromaRetriever.create_async(
            client=owner.client,
            collection_name=resolved_collection_name,
            embedding_model=str(chroma_cfg.settings.embedder),
            n_results=int(retrieval_cfg.n_results),
            similarity_threshold=float(retrieval_cfg.similarity_threshold),
            enable_cache=enable_cache,
            enable_query_expansion=enable_query_expansion,
        )
        retriever, bm25_index = await _build_configured_retriever(
            cfg,
            vector_retriever=vector_retriever,
            collection_name=resolved_collection_name,
        )
        resources.chroma_client = owner.client
        resources.collection = vector_retriever.collection
        resources.retriever = retriever
        resources.bm25_index = bm25_index
        resources._chroma_owner = owner
    except asyncio.CancelledError:
        if owner is not None:
            await owner.close()
        raise
    except Exception as exc:
        if owner is not None:
            await owner.close()
        logger.error("Failed to load async RAG retriever: %s", exc)

    try:
        resources.reranker = await asyncio.to_thread(build_reranker_from_config, cfg)
    except asyncio.CancelledError:
        await resources.close()
        raise
    return resources


async def _build_configured_retriever(
    cfg: Any,
    *,
    vector_retriever: ChromaRetriever,
    collection_name: str,
) -> tuple[HybridRetriever | ChromaRetriever, BM25Index | None]:
    """Add the configured BM25 resource without changing fallback semantics."""
    retrieval_cfg = cfg.chroma.retrieval
    if not bool(getattr(retrieval_cfg, "enable_hybrid", False)):
        logger.info("Using ChromaRetriever (vector-only)")
        return vector_retriever, None

    try:
        index_path = Path(str(retrieval_cfg.bm25_index_path))
        bm25 = await asyncio.to_thread(BM25Index, index_path=index_path)
        if bool(getattr(retrieval_cfg, "validate_bm25_sync", False)):
            manifest_path = _resolve_bm25_manifest_path(cfg, index_path)
            synchronized, validation_error = await _validate_bm25_sync_async(
                collection=vector_retriever.collection,
                collection_name=collection_name,
                bm25=bm25,
                index_path=index_path,
                manifest_path=manifest_path,
                batch_size=int(getattr(cfg.chroma.settings, "batch_size", 2500)),
            )
            if not synchronized:
                logger.warning(
                    "BM25 synchronization validation failed (%s); falling back to vector-only retrieval",
                    validation_error,
                )
                return vector_retriever, None
        if len(bm25) > 0:
            retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_index=bm25,
                semantic_candidate_pool=int(getattr(retrieval_cfg, "semantic_candidate_pool", 25)),
                bm25_candidate_pool=int(getattr(retrieval_cfg, "bm25_candidate_pool", 25)),
                reranker_input_limit=int(getattr(retrieval_cfg, "reranker_input_limit", 50)),
            )
            logger.info(
                "Using HybridRetriever (semantic_pool=%s, bm25_pool=%s, union_limit=%s)",
                retriever.semantic_candidate_pool,
                retriever.bm25_candidate_pool,
                retriever.reranker_input_limit,
            )
            return retriever, bm25
        logger.warning("BM25 index empty; falling back to vector-only retrieval")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Failed to initialize hybrid retrieval (%s); falling back to vector-only", exc)
    return vector_retriever, None


async def _validate_bm25_sync_async(
    *,
    collection: Any,
    collection_name: str,
    bm25: BM25Index,
    index_path: Path,
    manifest_path: Path,
    batch_size: int,
) -> tuple[bool, str | None]:
    """Validate BM25 against an async Chroma collection and its manifest."""
    try:
        manifest = await asyncio.to_thread(load_bm25_sync_manifest, manifest_path)
        if manifest.collection_name != collection_name:
            raise BM25SyncError(
                f"manifest collection={manifest.collection_name!r} does not match {collection_name!r}"
            )
        if Path(manifest.bm25_path) != index_path:
            raise BM25SyncError(
                f"manifest BM25 path={manifest.bm25_path!r} does not match {str(index_path)!r}"
            )

        chunk_ids = await _read_collection_ids_async(collection, batch_size=batch_size)
        chunk_ids_hash = compute_chunk_ids_sha256(chunk_ids)
        if len(chunk_ids) != manifest.record_count:
            raise BM25SyncError(
                f"Chroma count={len(chunk_ids)} does not match manifest count={manifest.record_count}"
            )
        if chunk_ids_hash != manifest.chunk_ids_sha256:
            raise BM25SyncError("Chroma chunk-ID hash does not match the synchronization manifest")

        bm25_ids = [str(document.get("id", "")) for document in bm25.documents]
        if len(bm25_ids) != manifest.record_count:
            raise BM25SyncError(
                f"BM25 count={len(bm25_ids)} does not match expected count={manifest.record_count}"
            )
        if any(not chunk_id for chunk_id in bm25_ids):
            raise BM25SyncError("BM25 contains a document without a chunk ID")
        if compute_chunk_ids_sha256(bm25_ids) != manifest.chunk_ids_sha256:
            raise BM25SyncError("BM25 chunk-ID hash does not match the Chroma snapshot")
        return True, None
    except BM25SyncError as exc:
        return False, str(exc)


async def _read_collection_ids_async(collection: Any, *, batch_size: int) -> list[str]:
    """Read all IDs from an async Chroma collection in bounded batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    expected_count = int(await collection.count())
    chunk_ids: list[str] = []
    for offset in range(0, expected_count, batch_size):
        batch = await collection.get(
            limit=min(batch_size, expected_count - offset),
            offset=offset,
            include=[],
        )
        chunk_ids.extend(str(chunk_id) for chunk_id in (batch.get("ids") or []))
    if len(chunk_ids) != expected_count:
        raise BM25SyncError(
            f"Chroma returned {len(chunk_ids)} IDs while reporting {expected_count} records"
        )
    return chunk_ids


async def _cleanup_new_chroma_systems(previous_identifiers: set[str]) -> None:
    """Clean systems allocated by an AsyncHttpClient call that did not return."""
    new_identifiers = set(SharedSystemClient._identifier_to_system) - previous_identifiers
    for identifier in new_identifiers:
        system = SharedSystemClient._identifier_to_system.pop(identifier)
        server = system.instance(AsyncFastAPI)
        try:
            await server._cleanup()
        finally:
            await asyncio.to_thread(system.stop)
