"""Unit tests for local retrieval metrics (Phase 3)."""

from __future__ import annotations

import pytest

from src.eval.metrics import (
    mean_precision_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    reciprocal_rank,
)


class TestPrecisionAtK:
    """Tests for precision_at_k()."""

    def test_precision_at_k_known_example(self) -> None:
        """Precision@k matches the reference example from the design spec."""
        retrieved = ["A", "B", "C", "D"]
        relevant = ["A", "C"]

        score = precision_at_k(retrieved_ids=retrieved, relevant_ids=relevant, k=3)

        assert score == pytest.approx(2 / 3)

    def test_precision_at_k_empty_retrieved_returns_zero(self) -> None:
        """Empty retrieval list returns 0.0."""
        score = precision_at_k(retrieved_ids=[], relevant_ids=["A"], k=3)
        assert score == 0.0

    def test_precision_at_k_no_relevant_returns_zero(self) -> None:
        """Empty relevant set returns 0.0."""
        score = precision_at_k(retrieved_ids=["A", "B"], relevant_ids=[], k=3)
        assert score == 0.0

    def test_precision_at_k_all_relevant_returns_one(self) -> None:
        """All top-k results relevant yields 1.0."""
        score = precision_at_k(
            retrieved_ids=["A", "B", "C", "D"],
            relevant_ids=["A", "B", "C"],
            k=3,
        )
        assert score == 1.0

    def test_precision_at_k_k_greater_than_list_length_penalizes_missing(self) -> None:
        """Fixed-k denominator penalizes short retrieved lists."""
        score = precision_at_k(
            retrieved_ids=["A", "B"],
            relevant_ids=["A", "B", "C"],
            k=5,
        )
        assert score == pytest.approx(2 / 5)

    @pytest.mark.parametrize("k", [0, -1])
    def test_precision_at_k_invalid_k_raises(self, k: int) -> None:
        """Non-positive k raises ValueError."""
        with pytest.raises(ValueError, match="k must be > 0"):
            precision_at_k(retrieved_ids=["A"], relevant_ids=["A"], k=k)


class TestReciprocalRank:
    """Tests for reciprocal_rank()."""

    def test_reciprocal_rank_first_relevant_at_position_two(self) -> None:
        """RR is 0.5 when first relevant item appears in rank position 2."""
        retrieved = ["X", "A", "B", "C"]
        relevant = ["A", "C"]

        score = reciprocal_rank(retrieved_ids=retrieved, relevant_ids=relevant)

        assert score == 0.5

    def test_reciprocal_rank_first_item_relevant_returns_one(self) -> None:
        """RR is 1.0 when the first item is relevant."""
        score = reciprocal_rank(retrieved_ids=["A", "B"], relevant_ids=["A"])
        assert score == 1.0

    def test_reciprocal_rank_no_overlap_returns_zero(self) -> None:
        """RR is 0.0 when no relevant item is retrieved."""
        score = reciprocal_rank(retrieved_ids=["X", "Y"], relevant_ids=["A", "B"])
        assert score == 0.0

    def test_reciprocal_rank_empty_retrieved_returns_zero(self) -> None:
        """RR is 0.0 for empty retrieved list."""
        score = reciprocal_rank(retrieved_ids=[], relevant_ids=["A"])
        assert score == 0.0

    def test_reciprocal_rank_empty_relevant_returns_zero(self) -> None:
        """RR is 0.0 for empty relevant set."""
        score = reciprocal_rank(retrieved_ids=["A", "B"], relevant_ids=[])
        assert score == 0.0


class TestAggregateMetrics:
    """Tests for mean metrics over multiple queries."""

    def test_mean_reciprocal_rank_known_example(self) -> None:
        """MRR equals arithmetic mean of per-query RR values."""
        results = [
            (["A", "B", "C"], ["A"]),  # RR=1.0
            (["X", "A", "B"], ["A"]),  # RR=0.5
            (["X", "Y", "Z"], ["A"]),  # RR=0.0
        ]

        mrr = mean_reciprocal_rank(results)

        assert mrr == pytest.approx((1.0 + 0.5 + 0.0) / 3)

    def test_mean_reciprocal_rank_empty_results_returns_zero(self) -> None:
        """Empty result list returns 0.0 for MRR."""
        assert mean_reciprocal_rank([]) == 0.0

    def test_mean_precision_at_k_known_values(self) -> None:
        """Mean precision@k averages per-query precision@k correctly."""
        results = [
            (["A", "B", "C"], ["A", "C"]),  # p@3 = 2/3
            (["X", "Y", "Z"], ["A", "B"]),  # p@3 = 0
            (["A", "B", "X"], ["A", "B", "C"]),  # p@3 = 2/3
        ]

        mp3 = mean_precision_at_k(results=results, k=3)

        assert mp3 == pytest.approx(((2 / 3) + 0.0 + (2 / 3)) / 3)

    def test_mean_precision_at_k_empty_results_returns_zero(self) -> None:
        """Empty result list returns 0.0 for mean precision@k."""
        assert mean_precision_at_k(results=[], k=3) == 0.0

    @pytest.mark.parametrize("k", [0, -2])
    def test_mean_precision_at_k_invalid_k_raises(self, k: int) -> None:
        """Non-positive k raises ValueError."""
        with pytest.raises(ValueError, match="k must be > 0"):
            mean_precision_at_k(results=[(["A"], ["A"])], k=k)

