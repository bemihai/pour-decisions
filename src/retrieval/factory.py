"""Factory helpers for configuring retrieval resources from app config."""

from pathlib import Path
from typing import Any

from src.utils import initialize_chroma_client, logger

from .hybrid_retriever import HybridRetriever
from .keyword_search import BM25Index
from .reranker import DocumentReranker
from .vector_retriever import ChromaRetriever
from .web_fallback import WebSearchEngine, WebSearchFallback


def build_retriever_from_config(
    cfg: Any,
    *,
    collection_name: str | None = None,
    enable_cache: bool = True,
    enable_query_expansion: bool = False,
) -> HybridRetriever | ChromaRetriever:
    """Build the configured vector or hybrid retriever stack.

    Args:
        cfg: Application OmegaConf config.
        collection_name: Optional Chroma collection name override. If omitted,
            the first configured collection is used.
        enable_cache: Whether to enable retriever-level caching.
        enable_query_expansion: Whether to enable query expansion in the
            underlying vector retriever.

    Returns:
        A ``HybridRetriever`` when hybrid search is enabled and BM25 is
        available; otherwise a ``ChromaRetriever``.

    Raises:
        Exception: Propagates Chroma connection or vector retriever initialization failures.
    """
    chroma_cfg = cfg.chroma
    retrieval_cfg = chroma_cfg.retrieval
    resolved_collection_name = collection_name or chroma_cfg.collections[0].name

    chroma_client = initialize_chroma_client(
        host=chroma_cfg.client.host,
        port=chroma_cfg.client.port,
    )

    vector_retriever = ChromaRetriever(
        client=chroma_client,
        collection_name=resolved_collection_name,
        embedding_model=chroma_cfg.settings.embedder,
        n_results=int(retrieval_cfg.n_results),
        similarity_threshold=float(retrieval_cfg.similarity_threshold),
        enable_cache=enable_cache,
        enable_query_expansion=enable_query_expansion,
    )

    if bool(getattr(retrieval_cfg, "enable_hybrid", False)):
        try:
            bm25_index_path = Path(str(retrieval_cfg.bm25_index_path))
            bm25 = BM25Index(index_path=bm25_index_path)
            if bool(getattr(retrieval_cfg, "validate_bm25_sync", False)):
                from src.chroma.bm25_builder import validate_bm25_sync

                manifest_path = _resolve_bm25_manifest_path(cfg, bm25_index_path)
                is_synchronized, validation_error = validate_bm25_sync(
                    collection=vector_retriever.collection,
                    collection_name=resolved_collection_name,
                    bm25=bm25,
                    index_path=bm25_index_path,
                    manifest_path=manifest_path,
                    batch_size=int(getattr(chroma_cfg.settings, "batch_size", 2500)),
                )
                if not is_synchronized:
                    logger.warning(
                        "BM25 synchronization validation failed (%s); falling back to vector-only retrieval",
                        validation_error,
                    )
                    return vector_retriever
            if len(bm25) > 0:
                return HybridRetriever(
                    vector_retriever=vector_retriever,
                    bm25_index=bm25,
                    semantic_candidate_pool=int(getattr(retrieval_cfg, "semantic_candidate_pool", 25)),
                    bm25_candidate_pool=int(getattr(retrieval_cfg, "bm25_candidate_pool", 25)),
                    reranker_input_limit=int(getattr(retrieval_cfg, "reranker_input_limit", 50)),
                )
            logger.warning("BM25 index empty; falling back to vector-only retrieval")
        except Exception as exc:
            logger.warning("Failed to initialize hybrid retrieval (%s); falling back to vector-only", exc)

    return vector_retriever


def _resolve_bm25_manifest_path(cfg: Any, index_path: Path) -> Path:
    """Resolve the configured sidecar path with a legacy-safe default."""
    indexing_cfg = getattr(cfg.chroma, "indexing", None)
    bm25_cfg = getattr(indexing_cfg, "bm25", None)
    configured_path = getattr(bm25_cfg, "sync_manifest_path", None)
    if configured_path:
        return Path(str(configured_path))
    return index_path.with_name(f"{index_path.stem}.meta.json")


def build_reranker_from_config(cfg: Any) -> DocumentReranker | None:
    """Build the configured production reranker, if enabled.

    Args:
        cfg: Application OmegaConf config.

    Returns:
        Configured reranker, or ``None`` when disabled or unavailable.
    """
    retrieval_cfg = cfg.chroma.retrieval
    if not bool(getattr(retrieval_cfg, "enable_reranking", False)):
        logger.info("Reranking disabled in config")
        return None

    model_name = str(
        getattr(
            retrieval_cfg,
            "reranker_model",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
    )
    try:
        reranker = DocumentReranker(model_name=model_name)
        logger.info("Loaded reranker: %s", model_name)
        return reranker
    except Exception as exc:
        logger.error("Failed to load reranker: %s", exc)
        return None


def build_web_fallback_from_config(
    cfg: Any,
    *,
    engine: WebSearchEngine | None = None,
) -> WebSearchFallback:
    """Build the lightweight automatic web-fallback adapter.

    The provider client remains lazy and is constructed only if an enabled,
    low-confidence request actually triggers fallback.

    Args:
        cfg: Application configuration.
        engine: Optional injected search engine for tests.

    Returns:
        Configured fallback adapter, disabled when the setting is absent.
    """
    web_search_cfg = getattr(cfg, "web_search", None)
    enabled = bool(getattr(web_search_cfg, "auto_fallback", False))
    return WebSearchFallback(enabled=enabled, engine=engine)
