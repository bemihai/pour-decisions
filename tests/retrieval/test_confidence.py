"""Tests for normalized cross-encoder retrieval confidence."""

import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.retrieval.confidence import RetrievalResult, compute_confidence


def test_high_rerank_score_produces_high_confidence() -> None:
    """A strong positive logit should normalize close to one."""
    documents = [{"id": "strong", "rerank_score": 4.0}]

    result = compute_confidence(documents, min_confidence=0.3)

    assert isinstance(result, RetrievalResult)
    assert result.documents is documents
    assert result.confidence == pytest.approx(0.9820137900379085)
    assert result.low_confidence is False
    assert result.web_fallback_used is False


def test_empty_documents_are_always_low_confidence() -> None:
    """No retrieval evidence should be low-confidence even at a zero cutoff."""
    result = compute_confidence([], min_confidence=0.0)

    assert result.confidence == 0.0
    assert result.low_confidence is True


def test_zero_logit_is_not_low_confidence_at_initial_cutoff() -> None:
    """Sigmoid zero is 0.5, above the initial 0.3 candidate."""
    result = compute_confidence([{"rerank_score": 0.0}], min_confidence=0.3)

    assert result.confidence == 0.5
    assert result.low_confidence is False


def test_negative_logits_are_low_confidence() -> None:
    """The maximum logit controls classification for the full result set."""
    result = compute_confidence(
        [{"rerank_score": -8.0}, {"rerank_score": -5.0}],
        min_confidence=0.3,
    )

    assert result.confidence == pytest.approx(0.006692850924284856)
    assert result.low_confidence is True


def test_confidence_equal_to_cutoff_is_not_low() -> None:
    """The low-confidence comparison should be strict and boundary-safe."""
    cutoff = 0.3
    cutoff_logit = math.log(cutoff / (1.0 - cutoff))

    result = compute_confidence([{"rerank_score": cutoff_logit}], min_confidence=cutoff)

    assert result.confidence == pytest.approx(cutoff)
    assert result.low_confidence is False


@pytest.mark.parametrize(
    ("score", "expected"),
    [(1000.0, 1.0), (-1000.0, 0.0)],
)
def test_sigmoid_is_stable_for_extreme_logits(score: float, expected: float) -> None:
    """Extreme cross-encoder logits should not overflow."""
    result = compute_confidence([{"rerank_score": score}], min_confidence=0.3)

    assert result.confidence == expected


@pytest.mark.parametrize("cutoff", [-0.1, 1.1, math.nan, math.inf, True, "0.3"])
def test_invalid_confidence_cutoff_is_rejected(cutoff: object) -> None:
    """The normalized cutoff must be a finite number inside the unit interval."""
    with pytest.raises(ValueError, match="min_confidence"):
        compute_confidence([{"rerank_score": 0.0}], min_confidence=cutoff)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "document",
    [{}, {"rerank_score": "high"}, {"rerank_score": math.nan}, {"rerank_score": True}],
)
def test_invalid_rerank_score_is_rejected(document: dict[str, object]) -> None:
    """Missing or invalid logits should fail explicitly before calibration."""
    with pytest.raises(ValueError, match="rerank_score"):
        compute_confidence([document], min_confidence=0.3)


def test_approved_calibration_configuration_defaults() -> None:
    """Approved config should activate filtering while leaving fallback disabled."""
    config_path = Path(__file__).resolve().parents[2] / "app_config.yml"
    config = OmegaConf.load(config_path)

    assert config.chroma.retrieval.rerank_threshold == 0.0
    assert config.chroma.retrieval.min_retrieval_confidence == 0.3
    assert config.web_search.auto_fallback is False
