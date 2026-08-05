"""Reranking module for improving retrieval precision."""
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

from src.utils import logger


# Module-level cache for reranker models
_reranker_cache: Dict[str, CrossEncoder] = {}


def _get_reranker(model_name: str) -> CrossEncoder:
    """Get or create cached reranker instance."""
    if model_name not in _reranker_cache:
        _reranker_cache[model_name] = CrossEncoder(model_name)
        logger.info(f"Loaded cross-encoder model: {model_name}")
    return _reranker_cache[model_name]


class DocumentReranker:
    """
    Rerank retrieved documents using a cross-encoder model.

    Cross-encoders are more accurate than bi-encoders for relevance scoring
    because they process query and document together. Use this as a second
    stage after initial retrieval to improve precision.

    Args:
        model_name: HuggingFace cross-encoder model name.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = _get_reranker(model_name)

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents by query-document relevance.

        Args:
            query: User's query string.
            documents: List of retrieved documents with 'document' key containing text.
            top_k: Number of top documents to return after reranking.

        Returns:
            Reranked list of documents with 'rerank_score' added.
        """
        if not documents:
            return []

        # Prepare query-document pairs for cross-encoder
        pairs = [(query, doc.get('document', '')) for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)

        # Sort by rerank score (descending)
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

        logger.debug(
            f"Reranked {len(documents)} docs, returning top {top_k}. "
            f"Score range: [{reranked[-1]['rerank_score']:.3f}, {reranked[0]['rerank_score']:.3f}]"
        )

        return reranked[:top_k]

    def rerank_with_threshold(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        threshold: float = 0.0,
        top_k: int | None = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank and filter documents by relevance threshold.

        The shared production RAG service selects this method when
        ``chroma.retrieval.rerank_threshold`` is numeric. The calibrated default is ``0.0``;
        setting the value to ``null`` selects ``rerank()`` and restores rank-only behavior.

        Args:
            query: User's query string.
            documents: List of retrieved documents.
            threshold: Minimum rerank score to include (default: 0.0).
            top_k: Optional max number of results (None = no limit).

        Returns:
            Filtered and reranked list of documents.
        """
        if not documents:
            return []

        # Prepare and score
        pairs = [(query, doc.get('document', '')) for doc in documents]
        scores = self.model.predict(pairs)

        # Filter by threshold and add scores
        results = []
        for doc, score in zip(documents, scores):
            if score >= threshold:
                doc_copy = doc.copy()
                doc_copy['rerank_score'] = float(score)
                results.append(doc_copy)

        # Sort by score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        logger.debug(
            f"Reranked with threshold {threshold}: "
            f"{len(documents)} -> {len(results)} docs"
        )

        return results
