"""Tests for shared production RAG threshold and confidence behavior."""

from types import SimpleNamespace

import pytest

from src.retrieval.rag_service import RAGExecutionResult, execute_production_rag


class _StaticRetriever:
    """Return isolated copies of a fixed document set."""

    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents

    def retrieve(self, query: str, n_results: int) -> list[dict]:
        """Return the configured documents without sharing mutable score state."""
        return [dict(document) for document in self.documents[:n_results]]


class _ScoredReranker:
    """Apply deterministic logits through either reranker entry point."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, float | None, int | None]] = []

    def rerank(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
        """Return rank-only results, including negative scores."""
        self.calls.append(("rerank", None, top_k))
        return self._score(documents)[:top_k]

    def rerank_with_threshold(
        self,
        query: str,
        documents: list[dict],
        threshold: float,
        top_k: int | None,
    ) -> list[dict]:
        """Return only results meeting the numeric logit threshold."""
        self.calls.append(("rerank_with_threshold", threshold, top_k))
        scored = [document for document in self._score(documents) if document["rerank_score"] >= threshold]
        return scored if top_k is None else scored[:top_k]

    def _score(self, documents: list[dict]) -> list[dict]:
        """Attach configured scores and sort from highest to lowest."""
        scored = [dict(document, rerank_score=self.scores[document["id"]]) for document in documents]
        return sorted(scored, key=lambda document: document["rerank_score"], reverse=True)


def _config(*, threshold: float | None, min_confidence: float = 0.3) -> SimpleNamespace:
    """Build the configuration fields consumed by the shared service."""
    return SimpleNamespace(
        chroma=SimpleNamespace(
            retrieval=SimpleNamespace(
                n_results=2,
                enable_metadata_boost=False,
                rerank_top_k=2,
                rerank_threshold=threshold,
                min_retrieval_confidence=min_confidence,
                use_deduplication=False,
                enable_compression=False,
            ),
            chunking=SimpleNamespace(enable_small_to_big=False),
            settings=SimpleNamespace(embedder="test-embedder"),
        )
    )


def _documents() -> list[dict]:
    """Return two minimal retrieval documents."""
    return [
        {"id": "first", "document": "First wine passage.", "metadata": {}},
        {"id": "second", "document": "Second wine passage.", "metadata": {}},
    ]


def _execute(
    *,
    threshold: float | None,
    reranker: object | None,
    min_confidence: float = 0.3,
) -> RAGExecutionResult:
    """Execute retrieval without generation for one threshold scenario."""
    return execute_production_rag(
        prompt="Explain the wine.",
        config=_config(threshold=threshold, min_confidence=min_confidence),
        model=None,
        retriever=_StaticRetriever(_documents()),
        reranker=reranker,
        message_history=[],
        generation_enabled=False,
    )


def test_null_threshold_preserves_rank_only_results_including_negative_scores() -> None:
    """Null must retain the existing rerank path and all top-k results."""
    reranker = _ScoredReranker({"first": -1.0, "second": -2.0})

    result = _execute(threshold=None, reranker=reranker)

    assert reranker.calls == [("rerank", None, 2)]
    assert [chunk.id for chunk in result.context_chunks] == ["first", "second"]
    assert [chunk.rerank_score for chunk in result.context_chunks] == [-1.0, -2.0]
    assert result.retrieval_confidence == pytest.approx(0.2689414213699951)
    assert result.low_confidence is True
    assert result.rerank_threshold is None
    assert result.feature_usage.reranking is True
    assert result.feature_usage.rerank_thresholding is False


def test_zero_threshold_filters_negative_scores_and_reports_active_threshold() -> None:
    """A numeric zero threshold should filter logits below zero."""
    reranker = _ScoredReranker({"first": 0.2, "second": -0.2})

    result = _execute(threshold=0.0, reranker=reranker)

    assert reranker.calls == [("rerank_with_threshold", 0.0, 2)]
    assert [chunk.id for chunk in result.context_chunks] == ["first"]
    assert result.retrieval_confidence == pytest.approx(0.549833997312478)
    assert result.low_confidence is False
    assert result.rerank_threshold == 0.0
    assert result.feature_usage.rerank_thresholding is True


def test_positive_threshold_can_produce_empty_low_confidence_context() -> None:
    """Filtering every candidate should remain a successful empty retrieval."""
    reranker = _ScoredReranker({"first": 0.4, "second": -0.2})

    result = _execute(threshold=0.5, reranker=reranker)

    assert result.context == ""
    assert result.context_chunks == []
    assert result.sources == []
    assert result.retrieval_error is None
    assert result.retrieval_confidence == 0.0
    assert result.low_confidence is True
    assert result.rerank_threshold == 0.5
    assert result.feature_usage.reranking is True
    assert result.feature_usage.rerank_thresholding is True


def test_numeric_threshold_can_retain_a_low_confidence_result() -> None:
    """Confidence classification should use the final thresholded result set."""
    reranker = _ScoredReranker({"first": -5.0, "second": -8.0})

    result = _execute(threshold=-10.0, reranker=reranker)

    assert [chunk.id for chunk in result.context_chunks] == ["first", "second"]
    assert result.retrieval_confidence == pytest.approx(0.006692850924284856)
    assert result.low_confidence is True
    assert result.rerank_threshold == -10.0


def test_no_reranker_preserves_explicit_unscored_state() -> None:
    """Without reranking there are no logits from which to infer confidence."""
    result = _execute(threshold=0.0, reranker=None)

    assert [chunk.id for chunk in result.context_chunks] == ["first", "second"]
    assert result.retrieval_confidence is None
    assert result.low_confidence is False
    assert result.rerank_threshold is None
    assert result.feature_usage.reranking is False
    assert result.feature_usage.rerank_thresholding is False
