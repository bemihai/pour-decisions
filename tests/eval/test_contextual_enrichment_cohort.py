"""Validation for the frozen M3 Phase 2 contextual-enrichment cohort."""

import json
from pathlib import Path
from typing import Any

from src.eval import load_golden_dataset
from src.utils import compute_file_hash


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COHORT_PATH = PROJECT_ROOT / "src" / "eval" / "m3b_contextual_enrichment_cohort.json"


def _load_cohort() -> dict[str, Any]:
    """Load the checked-in cohort manifest."""
    return json.loads(COHORT_PATH.read_text(encoding="utf-8"))


def test_contextual_enrichment_cohort_is_frozen_against_dataset_hash() -> None:
    """Dataset edits must trigger an explicit cohort review and hash update."""
    cohort = _load_cohort()
    dataset_path = PROJECT_ROOT / cohort["dataset_path"]

    assert compute_file_hash(dataset_path) == cohort["dataset_content_hash"]


def test_contextual_enrichment_cohort_has_valid_unique_scorable_samples() -> None:
    """Every selected sample must exist and support deterministic retrieval scoring."""
    cohort = _load_cohort()
    dataset_path = PROJECT_ROOT / cohort["dataset_path"]
    golden_by_id = {sample.id: sample for sample in load_golden_dataset(dataset_path)}
    entries = cohort["samples"]
    sample_ids = [entry["sample_id"] for entry in entries]

    assert len(entries) >= cohort["selection_policy"]["minimum_sample_count"]
    assert len(sample_ids) == len(set(sample_ids))
    assert set(sample_ids).issubset(golden_by_id)
    for sample_id in sample_ids:
        sample = golden_by_id[sample_id]
        assert sample.category == cohort["selection_policy"]["category"]
        assert sample.ground_truth_chunk_ids


def test_contextual_enrichment_cohort_declares_region_and_grape_concepts() -> None:
    """Each sample must exercise the combined context targeted by Phase 2."""
    cohort = _load_cohort()

    for entry in cohort["samples"]:
        assert entry["region_concepts"]
        assert entry["grape_concepts"]
        assert entry["evaluation_focus"].strip()


def test_contextual_enrichment_cohort_covers_multiple_regions_and_grapes() -> None:
    """The cohort must not overfit one famous grape or wine region."""
    cohort = _load_cohort()
    regions = {concept for entry in cohort["samples"] for concept in entry["region_concepts"]}
    grapes = {concept for entry in cohort["samples"] for concept in entry["grape_concepts"]}

    assert len(regions) >= 8
    assert len(grapes) >= 7
