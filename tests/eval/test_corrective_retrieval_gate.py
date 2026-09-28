"""Tests for the deterministic corrective-retrieval Gate 0 logic."""

from pathlib import Path

from src.eval.scripts.corrective_retrieval_gate import build_corrective_query, load_cohort, should_trigger


def test_frozen_cohort_has_three_unique_targets_and_controls() -> None:
    """The reviewed evidence gate keeps the required balanced cohort."""
    cohort = load_cohort(Path("src/eval/m12_gate0_cohort.json"))

    target_ids = {entry["sample_id"] for entry in cohort["targets"]}
    control_ids = {entry["sample_id"] for entry in cohort["controls"]}

    assert len(target_ids) == 3
    assert len(control_ids) == 3
    assert target_ids.isdisjoint(control_ids)


def test_trigger_selects_structured_gap_signals() -> None:
    """The trigger uses deterministic intent plus query wording."""
    assert should_trigger(
        {"original_query": "What are the ageing classifications for Rioja wines?", "intent": "aging"}
    )
    assert should_trigger(
        {"original_query": "How does the Médoc 1855 classification rank châteaux?", "intent": "classification"}
    )
    assert not should_trigger(
        {"original_query": "What is the minimum aging requirement for Barolo?", "intent": "aging"}
    )


def test_corrective_query_preserves_entities_and_numeric_anchors() -> None:
    """The alternate query retains plan entities and explicit year anchors."""
    query = build_corrective_query(
        {
            "original_query": "How does the Médoc 1855 classification rank Bordeaux châteaux?",
            "intent": "classification",
            "entities": {"regions": ["bordeaux", "medoc"]},
        }
    )

    assert query == "Médoc 1855 Bordeaux château classification ranking tiers growths"
