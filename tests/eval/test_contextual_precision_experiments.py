"""Tests for focused contextual precision-recovery experiments."""

import pytest
from omegaconf import OmegaConf

from src.eval.scripts.contextual_precision_experiments import (
    common_judge_comparison,
    compare_with_control,
    config_with_top_k,
)


def test_config_with_top_k_changes_only_copy() -> None:
    """Experiment result counts do not mutate the production config object."""
    config = OmegaConf.create({"chroma": {"retrieval": {"rerank_top_k": 5}}})

    copied = config_with_top_k(config, 3)

    assert config.chroma.retrieval.rerank_top_k == 5
    assert copied.chroma.retrieval.rerank_top_k == 3


def test_common_judge_comparison_uses_scored_intersection() -> None:
    """Asymmetric judge failures cannot bias the candidate delta."""
    control = {
        "per_sample": [
            {"sample_id": "one", "scores": {"context_precision": 0.5, "context_recall": 0.6}},
            {"sample_id": "two", "scores": {"context_precision": 0.7, "context_recall": 0.8}},
        ]
    }
    candidate = {
        "per_sample": [
            {"sample_id": "one", "scores": {"context_precision": 0.8, "context_recall": 0.7}},
            {"sample_id": "two", "scores": {"context_precision": 0.9}},
        ]
    }

    comparison = common_judge_comparison(control, candidate)

    assert comparison["context_precision"]["common_sample_count"] == 2
    assert comparison["context_precision"]["candidate_minus_control"] == pytest.approx(0.25)
    assert comparison["context_recall"]["common_sample_ids"] == ["one"]
    assert comparison["context_recall"]["candidate_minus_control"] == pytest.approx(0.1)


def test_compare_with_control_rejects_precision_at_5_regression() -> None:
    """A smaller result set must still pass the fixed-cutoff global metric gate."""
    control_variant = {
        "aggregate_metrics": {"mrr": 0.7, "precision_at_3": 0.5, "precision_at_5": 0.5},
        "mean_retrieval_latency_ms": 100.0,
    }
    candidate_variant = {
        "aggregate_metrics": {"mrr": 0.8, "precision_at_3": 0.6, "precision_at_5": 0.4},
        "mean_retrieval_latency_ms": 90.0,
    }
    control_judge = {
        "per_sample": [
            {"sample_id": "one", "scores": {"context_precision": 0.5, "context_recall": 0.6}}
        ]
    }
    candidate_judge = {
        "per_sample": [
            {"sample_id": "one", "scores": {"context_precision": 0.7, "context_recall": 0.6}}
        ]
    }

    comparison = compare_with_control(
        control_variant,
        control_judge,
        candidate_variant,
        candidate_judge,
    )

    assert comparison["decision"] == "fail"
    assert comparison["failed_checks"] == ["precision_at_5_within_tolerance"]
