"""Factory helpers for configuring retrieval resources from app config."""

from typing import Any

from src.utils import initialize_chroma_client, logger

from .hybrid_retriever import HybridRetriever
from .keyword_search import BM25Index
from .vector_retriever import ChromaRetriever


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
            bm25 = BM25Index(index_path=str(retrieval_cfg.bm25_index_path))
            if len(bm25) > 0:
                return HybridRetriever(
                    vector_retriever=vector_retriever,
                    bm25_index=bm25,
                    vector_weight=float(retrieval_cfg.hybrid_vector_weight),
                    keyword_weight=float(retrieval_cfg.hybrid_keyword_weight),
                )
            logger.warning("BM25 index empty; falling back to vector-only retrieval")
        except Exception as exc:
            logger.warning("Failed to initialize hybrid retrieval (%s); falling back to vector-only", exc)

    return vector_retriever
