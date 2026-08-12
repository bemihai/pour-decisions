"""Tests for deterministic local retrieval query planning."""

import pytest

from src.retrieval.query_analyzer import QueryAnalysis, build_retrieval_query_plan


def test_nebbiolo_flavour_query_builds_channel_specific_plan() -> None:
    """The known failing question should preserve its entity and focus dense intent."""
    query = "What are the primary flavour characteristics of Nebbiolo?"

    plan = build_retrieval_query_plan(query)

    assert plan.original_query == query
    assert plan.normalized_query == query.lower()
    assert plan.intent == "flavour"
    assert plan.grapes == ("nebbiolo",)
    assert plan.semantic_query == "nebbiolo aroma flavor taste sensory profile tannin acidity body"
    assert plan.sparse_query == "nebbiolo aroma taste tannin acidity body"


@pytest.mark.parametrize(
    ("query", "intent"),
    [
        ("How long should Barolo DOCG age?", "aging"),
        ("Where is Sancerre grown?", "region"),
        ("Explain the Chianti DOCG classification requirements.", "classification"),
        ("What food should I pair with Riesling?", "pairing"),
    ],
)
def test_supported_intents_are_mapped_locally(query: str, intent: str) -> None:
    """Supported intents should be explicit and deterministic without an LLM."""
    plan = build_retrieval_query_plan(query)

    assert plan.intent == intent
    assert plan.semantic_query
    assert plan.sparse_query


def test_misspelling_is_normalized_before_entity_analysis() -> None:
    """Existing terminology correction should feed both retrieval channels."""
    plan = build_retrieval_query_plan("Describe nebiolo aromas")

    assert plan.normalized_query == "describe nebbiolo aromas"
    assert plan.grapes == ("nebbiolo",)
    assert plan.semantic_query.startswith("nebbiolo ")
    assert plan.sparse_query == "nebbiolo aroma taste tannin acidity body"


def test_unknown_intent_and_no_entity_preserve_normalized_question() -> None:
    """Unsupported intent must remain observable rather than using a hidden fallback."""
    query = "Compare traditional and modern winemaking methods."

    first = build_retrieval_query_plan(query)
    second = build_retrieval_query_plan(query)

    assert first == second
    assert first.intent == "unknown"
    assert first.semantic_query == first.normalized_query
    assert first.grapes == ()


def test_query_analysis_filters_include_classification_and_producer() -> None:
    """Newly exposed entity fields should participate in filtering and boosting."""
    analysis = QueryAnalysis(
        original_query="Barolo from Domaine Test",
        classifications=["DOCG"],
        producers=["Domaine Test"],
    )

    assert analysis.to_chroma_filter() == {
        "$or": [
            {"classifications": {"$contains": "DOCG"}},
            {"producers": {"$contains": "Domaine Test"}},
        ]
    }
    assert analysis.get_boost_terms() == ["DOCG", "Domaine Test"]
