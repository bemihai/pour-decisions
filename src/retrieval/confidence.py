"""Confidence calculation for cross-encoder reranking results."""

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalResult:
    """Wrap reranked documents with an aggregate confidence signal."""

    documents: list[dict[str, Any]]
    confidence: float
    low_confidence: bool
    web_fallback_used: bool = False


def compute_confidence(
    documents: list[dict[str, Any]],
    min_confidence: float,
) -> RetrievalResult:
    """Normalize the maximum reranker logit and classify retrieval confidence.

    Args:
        documents: Reranked documents containing numeric ``rerank_score`` values.
        min_confidence: Normalized cutoff in the inclusive range 0.0–1.0.

    Returns:
        The original documents plus normalized confidence and classification.

    Raises:
        ValueError: If the cutoff or any document score is missing, non-numeric,
            or non-finite.
    """
    _validate_min_confidence(min_confidence)
    if not documents:
        return RetrievalResult(
            documents=documents,
            confidence=0.0,
            low_confidence=True,
        )

    rerank_scores = [_rerank_score(document, index) for index, document in enumerate(documents)]
    confidence = _sigmoid(max(rerank_scores))
    return RetrievalResult(
        documents=documents,
        confidence=confidence,
        low_confidence=confidence < min_confidence,
    )


def _validate_min_confidence(min_confidence: float) -> None:
    """Require one finite normalized confidence cutoff."""
    if not isinstance(min_confidence, (int, float)) or isinstance(min_confidence, bool):
        raise ValueError("min_confidence must be a numeric value between 0.0 and 1.0")
    if not math.isfinite(float(min_confidence)) or not 0.0 <= float(min_confidence) <= 1.0:
        raise ValueError("min_confidence must be a finite value between 0.0 and 1.0")


def _rerank_score(document: dict[str, Any], index: int) -> float:
    """Read one finite reranker logit with a useful validation error."""
    try:
        score = document["rerank_score"]
    except KeyError as exc:
        raise ValueError(f"Document at index {index} is missing rerank_score") from exc
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise ValueError(f"Document at index {index} has a non-numeric rerank_score")

    numeric_score = float(score)
    if not math.isfinite(numeric_score):
        raise ValueError(f"Document at index {index} has a non-finite rerank_score")
    return numeric_score


def _sigmoid(value: float) -> float:
    """Calculate sigmoid without overflowing for large positive or negative logits."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)
