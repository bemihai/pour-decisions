"""Integration-style tests for Ragas scorer (Phase 5).

These tests are marked with `eval` because they require evaluation dependencies and,
for end-to-end runs, a live API key.
"""

from __future__ import annotations

import os

import pytest

from src.eval.models import SampleResult
from src.eval.ragas_scorer import RagasScorer


pytestmark = pytest.mark.eval


@pytest.fixture()
def ragas_available() -> None:
    """Skip test module when ragas is not installed in the environment."""
    pytest.importorskip("ragas")


@pytest.fixture()
def scorer(ragas_available: None) -> RagasScorer:
    """Build a scorer instance or skip if API key is unavailable."""
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY is required for eval-marked Ragas tests")
    return RagasScorer()


def test_ragas_scorer_scores_two_synthetic_results(scorer: RagasScorer) -> None:
    """Ragas scorer writes expected metric keys for scoreable samples."""
    results = [
        SampleResult(
            id="rag_only_001",
            question="What is Barolo minimum aging?",
            answer="Barolo requires at least 38 months from harvest, 18 in oak.",
            ground_truth="Barolo requires at least 38 months from harvest, with 18 months in oak.",
            contexts=[
                "Barolo requires at least 38 months aging from harvest, with 18 months in oak.",
                "Barolo Riserva requires at least 62 months aging.",
            ],
            latency_ms=120.0,
        ),
        SampleResult(
            id="rag_only_002",
            question="What grape is used for Sancerre?",
            answer="Sancerre is primarily made from Sauvignon Blanc.",
            ground_truth="Sancerre is produced from Sauvignon Blanc grapes.",
            contexts=[
                "Sancerre is an appellation in the Loire producing Sauvignon Blanc wines.",
            ],
            latency_ms=95.0,
        ),
    ]

    scored = scorer.score(results)

    expected_metric_keys = {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    }

    assert len(scored) == 2
    for sample in scored:
        assert expected_metric_keys.issubset(sample.scores.keys())
        for metric_name in expected_metric_keys:
            assert 0.0 <= sample.scores[metric_name] <= 1.0


def test_ragas_scorer_skips_samples_with_error(scorer: RagasScorer) -> None:
    """Samples with existing errors are skipped and keep empty score map."""
    results = [
        SampleResult(
            id="rag_only_003",
            question="What is terroir?",
            answer="",
            contexts=[],
            error="retrieval failure",
            latency_ms=10.0,
        ),
        SampleResult(
            id="rag_only_004",
            question="What is Vinho Verde style?",
            answer="Vinho Verde is light and high-acid.",
            ground_truth="Vinho Verde wines are light-bodied and high in acidity.",
            contexts=["Vinho Verde wines are light, fresh, and high in acidity."],
            latency_ms=15.0,
        ),
    ]

    scored = scorer.score(results)

    assert scored[0].scores == {}
    assert scored[0].error == "retrieval failure"
    assert scored[1].scores


def test_ragas_scorer_skips_samples_with_empty_contexts(scorer: RagasScorer) -> None:
    """Samples without contexts are skipped and do not receive Ragas metrics."""
    results = [
        SampleResult(
            id="rag_only_005",
            question="What is Chablis?",
            answer="Chablis is Chardonnay from northern Burgundy.",
            ground_truth="Chablis is a Burgundy appellation producing Chardonnay wines.",
            contexts=[],
            latency_ms=20.0,
        ),
        SampleResult(
            id="rag_only_006",
            question="What is malolactic fermentation?",
            answer="MLF converts malic acid to lactic acid.",
            ground_truth="Malolactic fermentation converts malic acid into lactic acid.",
            contexts=["Malolactic fermentation converts malic acid into lactic acid."],
            latency_ms=30.0,
        ),
    ]

    scored = scorer.score(results)

    assert scored[0].scores == {}
    assert scored[1].scores


