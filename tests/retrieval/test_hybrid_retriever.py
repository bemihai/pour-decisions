"""Tests for balanced dense/sparse union retrieval."""

from types import SimpleNamespace

import pytest

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_analyzer import build_retrieval_query_plan
from src.retrieval.rag_service import execute_production_rag


class _DenseRetriever:
    """Return deterministic dense candidates and record channel inputs."""

    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, n_results: int, **_kwargs: object) -> list[dict]:
        """Return the configured dense pool."""
        self.calls.append((query, n_results))
        return [dict(document) for document in self.documents[:n_results]]


class _SparseIndex:
    """Return deterministic sparse candidates and record channel inputs."""

    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[dict]:
        """Return the configured sparse pool."""
        self.calls.append((query, top_k))
        return [dict(document) for document in self.documents[:top_k]]


class _WinnerReranker:
    """Select a configured sparse-only candidate from the full union."""

    def __init__(self, winner_id: str) -> None:
        self.winner_id = winner_id
        self.seen_ids: list[str] = []

    def rerank(self, _query: str, documents: list[dict], top_k: int) -> list[dict]:
        """Record every candidate and rank the exact sparse hit first."""
        self.seen_ids = [str(document["id"]) for document in documents]
        scored = [
            {**document, "rerank_score": 5.0 if document["id"] == self.winner_id else -1.0}
            for document in documents
        ]
        return sorted(scored, key=lambda document: document["rerank_score"], reverse=True)[:top_k]


def _doc(document_id: str, *, similarity: float | None = None, bm25_score: float | None = None) -> dict:
    """Build one minimal channel result."""
    document = {"id": document_id, "document": f"Evidence for {document_id}.", "metadata": {}}
    if similarity is not None:
        document["similarity"] = similarity
    if bm25_score is not None:
        document["bm25_score"] = bm25_score
    return document


def _config() -> SimpleNamespace:
    """Build production fields used after candidate union."""
    return SimpleNamespace(
        chroma=SimpleNamespace(
            retrieval=SimpleNamespace(
                n_results=1,
                enable_metadata_boost=False,
                rerank_top_k=1,
                rerank_threshold=None,
                min_retrieval_confidence=0.3,
                use_deduplication=False,
                enable_compression=False,
            ),
            chunking=SimpleNamespace(enable_small_to_big=False),
            settings=SimpleNamespace(embedder="test"),
        )
    )


def test_complete_pools_are_deduplicated_with_channel_provenance() -> None:
    """The union should retain both ranks and scores for shared candidates."""
    dense = _DenseRetriever([_doc("dense", similarity=0.9), _doc("shared", similarity=0.8)])
    sparse = _SparseIndex([_doc("sparse", bm25_score=4.0), _doc("shared", bm25_score=3.0)])
    retriever = HybridRetriever(dense, sparse, semantic_candidate_pool=2, bm25_candidate_pool=2)
    plan = build_retrieval_query_plan("What are the primary flavour characteristics of Nebbiolo?")

    results = retriever.retrieve("ignored", n_results=1, query_plan=plan, use_rrf_fallback=False)

    assert dense.calls == [("nebbiolo aroma flavor taste sensory profile tannin acidity body", 2)]
    assert sparse.calls == [("nebbiolo aroma taste tannin acidity body", 2)]
    assert [result["id"] for result in results] == ["dense", "sparse", "shared"]
    shared = results[2]
    assert shared["retrieval_channels"] == ["dense", "sparse"]
    assert shared["dense_rank"] == 2
    assert shared["sparse_rank"] == 2
    assert shared["dense_similarity"] == 0.8
    assert shared["bm25_score"] == 3.0


def test_default_pools_admit_fifty_unique_candidates_before_reranking() -> None:
    """Default 25+25 pools should reach the bounded union without early top-k loss."""
    dense = _DenseRetriever([_doc(f"dense-{index}", similarity=0.9) for index in range(25)])
    sparse = _SparseIndex([_doc(f"sparse-{index}", bm25_score=4.0) for index in range(25)])
    retriever = HybridRetriever(dense, sparse)

    results = retriever.retrieve("Nebbiolo", n_results=5, use_rrf_fallback=False)

    assert len(results) == 50
    assert results[0]["retrieval_diagnostics"]["dense_candidates"] == 25
    assert results[0]["retrieval_diagnostics"]["sparse_candidates"] == 25
    assert results[0]["retrieval_diagnostics"]["reranker_candidates"] == 50


def test_sparse_only_candidate_reaches_reranker_and_can_win() -> None:
    """The exact sparse hit must survive union construction and final selection."""
    dense = _DenseRetriever([_doc("burgundy-1", similarity=0.91), _doc("burgundy-2", similarity=0.90)])
    sparse = _SparseIndex([_doc("nebbiolo-exact", bm25_score=8.0)])
    hybrid = HybridRetriever(dense, sparse, semantic_candidate_pool=2, bm25_candidate_pool=1)
    reranker = _WinnerReranker("nebbiolo-exact")

    result = execute_production_rag(
        prompt="What are the primary flavour characteristics of Nebbiolo?",
        config=_config(),
        model=None,
        retriever=hybrid,
        reranker=reranker,
        message_history=[],
        generation_enabled=False,
    )

    assert reranker.seen_ids == ["burgundy-1", "nebbiolo-exact", "burgundy-2"]
    assert [chunk.id for chunk in result.context_chunks] == ["nebbiolo-exact"]
    sparse_artifact = next(chunk for chunk in result.raw_retrieved_chunks if chunk.id == "nebbiolo-exact")
    assert sparse_artifact.retrieval_channels == ["sparse"]
    assert sparse_artifact.sparse_rank == 1
    assert sparse_artifact.dense_rank is None


def test_reranker_unavailable_uses_unweighted_rrf_and_keeps_both_channels() -> None:
    """Fallback should use equal reciprocal-rank contributions without channel weights."""
    dense = _DenseRetriever([_doc("dense", similarity=0.9), _doc("shared", similarity=0.8)])
    sparse = _SparseIndex([_doc("sparse", bm25_score=4.0), _doc("shared", bm25_score=3.0)])
    retriever = HybridRetriever(dense, sparse, semantic_candidate_pool=2, bm25_candidate_pool=2)

    results = retriever.retrieve("Nebbiolo", n_results=3, use_rrf_fallback=True)

    assert [result["id"] for result in results] == ["shared", "dense", "sparse"]
    assert results[0]["rrf_score"] == pytest.approx(2 / 62)
    assert results[1]["rrf_score"] == pytest.approx(results[2]["rrf_score"])
    assert {channel for result in results for channel in result["retrieval_channels"]} == {"dense", "sparse"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"semantic_candidate_pool": 0},
        {"bm25_candidate_pool": 0},
        {"reranker_input_limit": 0},
    ],
)
def test_invalid_pool_limits_fail_explicitly(kwargs: dict[str, int]) -> None:
    """Misconfigured candidate bounds should fail during resource construction."""
    with pytest.raises(ValueError):
        HybridRetriever(_DenseRetriever([]), _SparseIndex([]), **kwargs)
