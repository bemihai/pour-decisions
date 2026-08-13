"""Tests for the M3 Phase 2 search-representation ablation."""

from typing import Any

import pytest

from src.eval.contextual_ablation import (
    SearchRepresentationReranker,
    build_ablation_documents,
    build_ablation_search_text,
    validate_aligned_ablation_documents,
)
from src.eval.models import GoldenSample
from src.eval.scripts.contextual_enrichment_ablation import (
    build_acceptance_decision,
    build_comparison,
    materialize_variant_collection,
    score_context_metrics,
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
            "cohort_aggregate_metrics": {"mrr": 0.4, "precision_at_3": 0.3, "precision_at_5": 0.2},
            "mean_retrieval_latency_ms": 100.0,
        },
        "contextual": {
            "aggregate_metrics": {"mrr": 0.7, "precision_at_3": 0.5, "precision_at_5": 0.3},
            "cohort_aggregate_metrics": {"mrr": 0.6, "precision_at_3": 0.4, "precision_at_5": 0.2},
            "mean_retrieval_latency_ms": 110.0,
        },
    }

    comparison = build_comparison(variants)

    assert comparison == {
        "global_metric_deltas": {
            "mrr": pytest.approx(0.2),
            "precision_at_3": pytest.approx(0.1),
            "precision_at_5": pytest.approx(0.0),
        },
        "cohort_metric_deltas": {
            "mrr": pytest.approx(0.2),
            "precision_at_3": pytest.approx(0.1),
            "precision_at_5": pytest.approx(0.0),
        },
        "mean_retrieval_latency_delta_ms": pytest.approx(10.0),
    }


def test_score_context_metrics_compares_only_common_scored_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """Judge deltas exclude asymmetric failures rather than comparing mismatched samples."""
    cohort_samples = [
        GoldenSample(
            id=f"rag_only_00{index}",
            question=f"Question {index}",
            category="rag_only",
            difficulty="medium",
            expected_facts=[f"Fact {index}"],
            ground_truth=f"Reference {index}",
            tags=["rag"],
        )
        for index in (1, 2)
    ]
    variants = {
        representation: {
            "per_sample": [
                {
                    "sample_id": sample.id,
                    "question": sample.question,
                    "is_cohort_sample": True,
                    "contexts": [f"{representation} context {index}"],
                    "retrieved_chunk_ids": [f"chunk-{index}"],
                }
                for index, sample in enumerate(cohort_samples, start=1)
            ]
        }
        for representation in ("body_only", "contextual")
    }

    class _FakeRagasScorer:
        def __init__(self) -> None:
            self.metric_names: list[str] = []

        def score(self, results: list[Any]) -> None:
            is_contextual = results[0].contexts[0].startswith("contextual")
            for index, result in enumerate(results):
                result.scores["context_precision"] = 0.7 if is_contextual else 0.5
                if not (is_contextual and index == 1):
                    result.scores["context_recall"] = 0.9 if is_contextual else 0.8
                else:
                    result.metric_errors["context_recall"] = "judge_error"

    monkeypatch.setattr("src.eval.ragas_scorer.RagasScorer", _FakeRagasScorer)

    result = score_context_metrics(variants, cohort_samples)

    precision = result["common_sample_comparison"]["context_precision"]
    recall = result["common_sample_comparison"]["context_recall"]
    assert precision["common_sample_count"] == 2
    assert precision["contextual_minus_body"] == pytest.approx(0.2)
    assert recall["common_sample_ids"] == ["rag_only_001"]
    assert recall["contextual_minus_body"] == pytest.approx(0.1)


def test_build_acceptance_decision_requires_precision_improvement() -> None:
    """Improved deterministic retrieval cannot hide a failed context-precision gate."""
    artifact = {
        "comparison": {
            "global_metric_deltas": {
                "mrr": 0.09,
                "precision_at_3": 0.14,
                "precision_at_5": 0.09,
            }
        },
        "context_judge": {
            "common_sample_comparison": {
                "context_precision": {"contextual_minus_body": -0.19},
                "context_recall": {"contextual_minus_body": 0.10},
            }
        },
    }

    decision = build_acceptance_decision(artifact)

    assert decision["decision"] == "revise"
    assert decision["failed_checks"] == ["cohort_context_precision_improves"]
