"""Unit tests for RagasScorer model resolution and configuration.

These tests verify that RagasScorer resolves the evaluator LLM from config
and uses the configured provider (local Ollama by default) without implicit
cloud fallback behavior.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest


def _make_config(
    provider: str = "google",
    name: str = "gemini-2.5-flash",
    execution_provider: str = "ollama",
    execution_model: str = "llama3.2:3b",
    evaluator_provider: str = "ollama",
    evaluator_model: str = "llama3.2:3b",
) -> object:
    """Build a minimal config object that mirrors OmegaConf structure."""
    ragas_cfg = types.SimpleNamespace(
        evaluator_provider=evaluator_provider,
        evaluator_model=evaluator_model,
        metrics=["faithfulness"],
    )
    eval_cfg = types.SimpleNamespace(
        execution_provider=execution_provider,
        execution_model=execution_model,
        ollama=types.SimpleNamespace(base_url="http://localhost:11434"),
        ragas=ragas_cfg,
    )
    model_cfg = types.SimpleNamespace(
        provider=provider,
        name=name,
        ollama=types.SimpleNamespace(base_url="http://localhost:11434"),
    )
    return types.SimpleNamespace(model=model_cfg, eval=eval_cfg)


class TestRagasScorerProviderResolution:
    """Verify that RagasScorer resolves the evaluator LLM from config."""

    @patch("src.eval.ragas_scorer.get_embedder")
    @patch("src.eval.ragas_scorer.load_eval_model")
    @patch("src.eval.ragas_scorer.get_config")
    def test_uses_explicit_evaluator_config(
        self, mock_cfg, mock_load, mock_embedder
    ) -> None:
        """Ragas scorer should use explicit evaluator config, not app model config."""
        from src.eval.ragas_scorer import RagasScorer

        mock_cfg.return_value = _make_config(
            provider="google",
            name="gemini-2.5-flash",
            evaluator_provider="ollama",
            evaluator_model="llama3.2:3b",
        )
        cfg = mock_cfg.return_value
        mock_load.return_value = MagicMock()
        mock_embedder.return_value = MagicMock()

        scorer = RagasScorer()

        assert scorer.evaluator_provider == "ollama"
        assert scorer.evaluator_model == "llama3.2:3b"
        mock_load.assert_called_once_with(cfg)

    @patch("src.eval.ragas_scorer.get_embedder")
    @patch("src.eval.ragas_scorer.load_eval_model")
    @patch("src.eval.ragas_scorer.get_config")
    def test_uses_explicit_evaluator_config_when_set(
        self, mock_cfg, mock_load, mock_embedder
    ) -> None:
        """Explicit evaluator provider/model overrides main model config."""
        from src.eval.ragas_scorer import RagasScorer

        mock_cfg.return_value = _make_config(
            provider="google",
            name="gemini-2.5-flash",
            evaluator_provider="ollama",
            evaluator_model="phi3:mini",
        )
        cfg = mock_cfg.return_value
        mock_load.return_value = MagicMock()
        mock_embedder.return_value = MagicMock()

        scorer = RagasScorer()

        assert scorer.evaluator_provider == "ollama"
        assert scorer.evaluator_model == "phi3:mini"
        mock_load.assert_called_once_with(cfg)
        assert scorer.metric_names == ["faithfulness"]

    @patch("src.eval.ragas_scorer.get_embedder")
    @patch("src.eval.ragas_scorer.load_eval_model")
    @patch("src.eval.ragas_scorer.get_config")
    def test_injected_llm_bypasses_load(
        self, mock_cfg, mock_load, mock_embedder
    ) -> None:
        """Explicitly injected LLM skips model loading."""
        from src.eval.ragas_scorer import RagasScorer

        mock_cfg.return_value = _make_config()
        mock_embedder.return_value = MagicMock()
        injected = MagicMock()

        scorer = RagasScorer(llm=injected)

        mock_load.assert_not_called()
        assert scorer.llm is injected


class TestScorerSkippingBehaviour:
    """Verify score() correctly skips samples with errors or empty contexts."""

    def _make_scorer_with_mock_evaluate(self):
        """Return a scorer whose _evaluate_rows is patched to return dummy scores."""
        with (
            patch("src.eval.ragas_scorer.get_config") as mock_cfg,
            patch("src.eval.ragas_scorer.load_eval_model", return_value=MagicMock()),
            patch("src.eval.ragas_scorer.get_embedder", return_value=MagicMock()),
        ):
            mock_cfg.return_value = _make_config()
            from src.eval.ragas_scorer import RagasScorer
            scorer = RagasScorer()

        scorer._evaluate_rows = MagicMock(  # type: ignore[assignment]
            return_value=[{"faithfulness": 0.9, "answer_relevancy": 0.8}]
        )
        return scorer

    def test_samples_with_errors_are_skipped(self) -> None:
        """Samples that already have an error field should not be scored."""
        from src.eval.models import SampleResult

        scorer = self._make_scorer_with_mock_evaluate()
        results = [
            SampleResult(
                id="s1",
                question="q1",
                contexts=["ctx"],
                error="retrieval failed",
                latency_ms=0.0,
            ),
        ]
        scored = scorer.score(results)
        assert scored[0].scores == {}
        scorer._evaluate_rows.assert_not_called()

    def test_samples_without_contexts_are_skipped(self) -> None:
        """Samples with no retrieved contexts should not be scored."""
        from src.eval.models import SampleResult

        scorer = self._make_scorer_with_mock_evaluate()
        results = [
            SampleResult(
                id="s2",
                question="q2",
                answer="ans",
                contexts=[],
                latency_ms=10.0,
            ),
        ]
        scored = scorer.score(results)
        assert scored[0].scores == {}
        scorer._evaluate_rows.assert_not_called()

    def test_valid_sample_receives_scores(self) -> None:
        """Samples with contexts and no error should receive Ragas scores."""
        from src.eval.models import SampleResult

        scorer = self._make_scorer_with_mock_evaluate()
        results = [
            SampleResult(
                id="s3",
                question="q3",
                answer="ans",
                ground_truth="truth",
                contexts=["ctx1"],
                latency_ms=10.0,
            ),
        ]
        scored = scorer.score(results)
        assert scored[0].scores.get("faithfulness") == pytest.approx(0.9)
        assert scored[0].scores.get("answer_relevancy") == pytest.approx(0.8)


class TestRagasMetricSelection:
    """Verify configured metric selection is respected."""

    @patch("src.eval.ragas_scorer.get_embedder")
    @patch("src.eval.ragas_scorer.load_eval_model")
    @patch("src.eval.ragas_scorer.get_config")
    def test_uses_configured_metric_subset(self, mock_cfg, mock_load, mock_embedder) -> None:
        """Configured eval.ragas.metrics should control the scorer metric list."""
        from src.eval.ragas_scorer import RagasScorer

        cfg = _make_config()
        cfg.eval.ragas.metrics = ["faithfulness", "context_recall"]
        mock_cfg.return_value = cfg
        mock_load.return_value = MagicMock()
        mock_embedder.return_value = MagicMock()

        scorer = RagasScorer()

        assert scorer.metric_names == ["faithfulness", "context_recall"]

    @patch("src.eval.ragas_scorer.get_embedder")
    @patch("src.eval.ragas_scorer.load_eval_model")
    @patch("src.eval.ragas_scorer.get_config")
    def test_rejects_unknown_metric_names(self, mock_cfg, mock_load, mock_embedder) -> None:
        """Unknown metric names should fail fast during scorer construction."""
        from src.eval.ragas_scorer import RagasScorer

        cfg = _make_config()
        cfg.eval.ragas.metrics = ["faithfulness", "made_up_metric"]
        mock_cfg.return_value = cfg
        mock_load.return_value = MagicMock()
        mock_embedder.return_value = MagicMock()

        with pytest.raises(ValueError, match="Unsupported eval.ragas.metrics values"):
            RagasScorer()
