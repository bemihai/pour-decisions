"""Validation tests for the frozen Phase 5 web-fallback cohort."""

from pathlib import Path

from src.eval import load_golden_dataset


COHORT_PATH = Path(__file__).resolve().parents[2] / "src" / "eval" / "m3e_web_fallback_cohort.jsonl"


def test_web_fallback_cohort_is_frozen_and_current() -> None:
    """The cohort must contain reviewed current-information questions only."""
    samples = load_golden_dataset(COHORT_PATH)

    assert len(samples) == 5
    assert len({sample.id for sample in samples}) == len(samples)
    assert all(sample.category == "rag_only" for sample in samples)
    assert all("requires_current_information" in sample.tags for sample in samples)
    assert all(not sample.ground_truth_chunk_ids for sample in samples)
    assert all(sample.notes and "Reviewed 2026-08-13" in sample.notes for sample in samples)
