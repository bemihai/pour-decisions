"""Tests for approved deterministic corrective-retrieval primitives."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from omegaconf import OmegaConf

from src.retrieval.correction import (
    CorrectionAttemptBudget,
    RAGCorrectionDiagnostic,
    build_correction_query,
    correction_trace_attributes,
    correction_trigger_reason,
    load_correction_config,
    select_correction_result,
)
from src.retrieval.query_analyzer import RetrievalQueryPlan


def _plan(*, original: str, intent: str, regions: tuple[str, ...] = ()) -> RetrievalQueryPlan:
    """Build a minimal immutable plan for correction tests."""
    return RetrievalQueryPlan(
        original_query=original,
        normalized_query=original.casefold(),
        semantic_query=original,
        sparse_query=original,
        intent=intent,
        regions=regions,
    )


def test_config_defaults_disabled_and_validates_values() -> None:
    """Correction cannot activate without an explicit valid opt-in."""
    config = OmegaConf.create({"chroma": {"retrieval": {}}})

    assert load_correction_config(config).enabled is False
    assert load_correction_config(config).timeout_seconds == 3.0

    for invalid in (0, -1, True, "3"):
        config = OmegaConf.create(
            {"chroma": {"retrieval": {"correction": {"enabled": False, "timeout_seconds": invalid}}}}
        )
        with pytest.raises(ValueError, match="timeout_seconds"):
            load_correction_config(config)


def test_trigger_and_query_match_approved_templates() -> None:
    """Only the two evidence-backed shapes receive bounded alternate queries."""
    aging = _plan(
        original="What are the ageing classifications for Rioja wines?",
        intent="aging",
        regions=("rioja",),
    )
    dated = _plan(
        original="How does the Médoc 1855 classification rank Bordeaux châteaux?",
        intent="classification",
        regions=("bordeaux", "medoc"),
    )
    control = _plan(original="What is the aging requirement for Barolo?", intent="aging")

    assert correction_trigger_reason(aging) == "aging_classifications"
    assert build_correction_query(aging).query == (
        "rioja aging classification categories minimum requirements oak bottle"
    )
    assert correction_trigger_reason(dated) == "dated_classification"
    assert build_correction_query(dated).query == (
        "Médoc 1855 Bordeaux château classification ranking tiers growths"
    )
    assert correction_trigger_reason(control) is None
    assert build_correction_query(control) is None


def test_attempt_budget_allows_one_reservation_under_contention() -> None:
    """Concurrent callers share one request-wide deterministic reservation."""
    budget = CorrectionAttemptBudget()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: budget.reserve(), range(16)))

    assert results.count(True) == 1
    assert budget.attempt_count == 1


def test_selection_requires_successful_nonempty_novel_chunks() -> None:
    """Failure, empty results, and ties preserve the valid first pass."""
    first = [{"id": "one", "document": "First"}]

    failed = select_correction_result(first, [{"id": "two"}], correction_succeeded=False)
    empty = select_correction_result(first, [], correction_succeeded=True)
    tied = select_correction_result(first, [{"id": "one"}], correction_succeeded=True)
    novel = select_correction_result(first, [{"id": "two"}], correction_succeeded=True)

    assert (failed.selected_result, failed.selection_reason) == ("first_pass", "correction_failed")
    assert (empty.selected_result, empty.selection_reason) == ("first_pass", "corrected_empty")
    assert (tied.selected_result, tied.selection_reason) == ("first_pass", "no_novel_chunks")
    assert novel.selected_result == "corrected"
    assert novel.selection_reason == "corrected_novel_chunks"
    assert novel.novel_corrected_chunk_count == 1


def test_diagnostic_invariants_and_trace_redaction() -> None:
    """Diagnostics remain bounded and traces omit query hashes and chunk identities."""
    diagnostic = RAGCorrectionDiagnostic(
        enabled=True,
        eligible=True,
        trigger_reason="dated_classification",
        attempt_reserved=True,
        attempt_count=1,
        mode="deterministic",
        alternate_query_id="dated_classification_v1",
        alternate_query_sha256="secret-hash",
        status="completed",
        selected_result="corrected",
        selection_reason="corrected_novel_chunks",
        first_pass_chunk_count=5,
        corrected_chunk_count=4,
        novel_corrected_chunk_count=4,
        added_latency_ms=900.0,
    )

    attributes = correction_trace_attributes(diagnostic)

    assert attributes["correction_alternate_query_id"] == "dated_classification_v1"
    assert "alternate_query_sha256" not in attributes
    assert "secret-hash" not in attributes.values()
    with pytest.raises(ValueError, match="attempt_count"):
        RAGCorrectionDiagnostic(attempt_count=2)
