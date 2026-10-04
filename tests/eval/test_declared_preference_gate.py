"""Tests for the M13B declared-preference Gate 0 evidence contract."""

from copy import deepcopy

import pytest

from src.eval.scripts.declared_preference_gate import evaluate_gate, load_cohort, validate_cohort


def test_gate0_cohort_is_complete_and_locally_reproducible() -> None:
    """The frozen inputs cover every required target, lifecycle case, and control."""
    cohort = load_cohort()

    result = evaluate_gate(cohort)

    assert result["decision"] in {"ready_for_approval", "pass"}
    assert all(result["checks"].values())
    assert result["counts"] == {
        "samples": 11,
        "targets": 6,
        "lifecycle": 3,
        "controls": 2,
        "retained_dimensions": 6,
    }
    assert result["external_calls"] == 0


def test_gate0_requires_each_retained_dimension_to_change_a_product_outcome() -> None:
    """A dimension cannot remain frozen without a matching visible target case."""
    cohort = load_cohort()
    altered = deepcopy(cohort)
    altered["samples"] = [
        sample for sample in altered["samples"] if sample["sample_id"] != "m13b_pref_006"
    ]

    with pytest.raises(ValueError, match="retained preference dimension"):
        validate_cohort(altered)


def test_gate0_thresholds_are_numerical_and_approval_is_explicit() -> None:
    """Promotion is impossible without an explicit approval state."""
    cohort = load_cohort()

    assert cohort["approval"]["status"] in {"pending", "approved"}
    assert all(type(value) in {int, float} for value in cohort["promotion_thresholds"].values())
    assert all(type(value) in {int, float} for value in cohort["rollback_thresholds"].values())
    assert evaluate_gate(cohort)["decision"] == (
        "pass" if cohort["approval"]["status"] == "approved" else "ready_for_approval"
    )
