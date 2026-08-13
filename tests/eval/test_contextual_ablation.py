"""Tests for the M3 Phase 2 search-representation ablation."""

from typing import Any

import pytest

from src.eval.contextual_ablation import (
    SearchRepresentationReranker,
    build_ablation_documents,
    build_ablation_search_text,
    validate_aligned_ablation_documents,
)
from src.eval.scripts.contextual_enrichment_ablation import (
    build_comparison,
    materialize_variant_collection,
)


def _source_record() -> dict[str, Any]:
    return {
        "id": "chunk-1",
        "document": "Nebbiolo has high tannin and acidity.",
        "metadata": {
            "document_title": "Grapes and Wines",
            "chapter": "Nebbiolo",
            "section": "Piedmont",
            "structural_role": "prose",
        },
    }


def test_build_ablation_search_text_selects_explicit_representation() -> None:
    """Body and contextual variants use their reviewed representations."""
    source = _source_record()

    body = build_ablation_search_text(source["document"], source["metadata"], "body_only")
    contextual = build_ablation_search_text(source["document"], source["metadata"], "contextual")

    assert body == source["document"]
    assert contextual == "Grapes and Wines > Nebbiolo > Piedmont\n\nNebbiolo has high tannin and acidity."


def test_build_ablation_documents_changes_only_search_text() -> None:
    """Paired records retain IDs, clean bodies, and metadata exactly."""
    body = build_ablation_documents([_source_record()], "body_only")
    contextual = build_ablation_documents([_source_record()], "contextual")

    validate_aligned_ablation_documents(body, contextual)
    assert body[0]["search_text"] != contextual[0]["search_text"]
    assert {key: body[0][key] for key in ("id", "document", "metadata")} == {
        key: contextual[0][key] for key in ("id", "document", "metadata")
    }


def test_build_ablation_search_text_rejects_unknown_representation() -> None:
    """Variant typos fail instead of silently selecting production behavior."""
    source = _source_record()

    with pytest.raises(ValueError, match="Unsupported search representation"):
        build_ablation_search_text(source["document"], source["metadata"], "unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "documents,error",
    [
        ([{"id": "", "document": "text", "metadata": {}}], "missing an id"),
        ([{"id": "one", "document": "", "metadata": {}}], "empty document text"),
        (
            [
                {"id": "one", "document": "first", "metadata": {}},
                {"id": "one", "document": "second", "metadata": {}},
            ],
            "Duplicate ablation source id",
        ),
    ],
)
def test_build_ablation_documents_rejects_invalid_source_records(
    documents: list[dict[str, Any]],
    error: str,
) -> None:
    """Malformed corpora fail before embedding or temporary index writes."""
    with pytest.raises(ValueError, match=error):
        build_ablation_documents(documents, "body_only")


def test_validate_aligned_ablation_documents_rejects_non_representation_drift() -> None:
    """Differences outside search text invalidate the comparison."""
    body = build_ablation_documents([_source_record()], "body_only")
    contextual = build_ablation_documents([_source_record()], "contextual")
    contextual[0]["metadata"] = {"chapter": "Different"}

    with pytest.raises(ValueError, match="metadata"):
        validate_aligned_ablation_documents(body, contextual)


class _PredictModel:
    """Capture cross-encoder pairs and return stable scores."""

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs = pairs
        return [0.2, 0.8]


class _BaseReranker:
    def __init__(self) -> None:
        self.model = _PredictModel()


class _Embedder:
    """Return stable tiny vectors while recording indexed text."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]


def test_materialize_variant_collection_embeds_search_text_and_stores_clean_body(
    in_memory_chroma_client: Any,
) -> None:
    """The disposable index never replaces user-visible evidence with its prefix."""
    documents = build_ablation_documents([_source_record()], "contextual")
    embedder = _Embedder()

    collection = materialize_variant_collection(
        client=in_memory_chroma_client,
        collection_name="m3b-materialization-test",
        collection_metadata={"hnsw:space": "cosine"},
        documents=documents,
        embedder=embedder,
        batch_size=1,
    )
    stored = collection.get(ids=["chunk-1"], include=["documents", "metadatas"])

    assert embedder.texts == [documents[0]["search_text"]]
    assert stored["documents"] == ["Nebbiolo has high tannin and acidity."]
    assert stored["metadatas"][0]["chapter"] == "Nebbiolo"


def test_search_representation_reranker_uses_body_only_pairs() -> None:
    """The body control does not leak contextual headings into reranker pairs."""
    base = _BaseReranker()
    reranker = SearchRepresentationReranker(base, "body_only")
    documents = [
        _source_record(),
        {**_source_record(), "id": "chunk-2", "document": "Barolo is from Piedmont."},
    ]

    results = reranker.rerank_with_threshold("Nebbiolo", documents, threshold=0.0, top_k=2)

    assert base.model.pairs == [
        ("Nebbiolo", "Nebbiolo has high tannin and acidity."),
        ("Nebbiolo", "Barolo is from Piedmont."),
    ]
    assert [result["id"] for result in results] == ["chunk-2", "chunk-1"]


def test_build_comparison_reports_contextual_minus_body_deltas() -> None:
    """The artifact exposes directional metric and latency differences."""
    variants = {
        "body_only": {
            "aggregate_metrics": {"mrr": 0.5, "precision_at_3": 0.4, "precision_at_5": 0.3},
            "mean_retrieval_latency_ms": 100.0,
        },
        "contextual": {
            "aggregate_metrics": {"mrr": 0.7, "precision_at_3": 0.5, "precision_at_5": 0.3},
            "mean_retrieval_latency_ms": 110.0,
        },
    }

    comparison = build_comparison(variants)

    assert comparison == {
        "metric_deltas": {
            "mrr": pytest.approx(0.2),
            "precision_at_3": pytest.approx(0.1),
            "precision_at_5": pytest.approx(0.0),
        },
        "mean_retrieval_latency_delta_ms": pytest.approx(10.0),
    }
