"""Unit tests for RagasScorer model resolution and configuration.

These tests verify that RagasScorer correctly resolves the evaluator LLM from config
and uses the configured provider (local Ollama by default) rather than hardcoding Google.
No live LLM or Ragas evaluation is performed; all external calls are mocked.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    provider: str = "ollama",
    name: str = "gemma2:2b",
    fallback_provider: str = "google",
    fallback_name: str = "gemini-2.5-flash",
    evaluator_provider: str = "",
    evaluator_model: str = "",
) -> object:
    """Build a minimal config object that mirrors OmegaConf structure."""

    ragas_cfg = types.SimpleNamespace(
        evaluator_provider=evaluator_provider,
        evaluator_model=evaluator_model,
        metrics=["faithfulness"],
    )
    eval_cfg = types.SimpleNamespace(ragas=ragas_cfg)
    model_cfg = types.SimpleNamespace(
        provider=provider,
        name=name,
        fallback_provider=fallback_provider,
        fallback_name=fallback_name,
        ollama=types.SimpleNamespace(base_url="http://localhost:11434"),
    )
    return types.SimpleNamespace(model=model_cfg, eval=eval_cfg)


# ---------------------------------------------------------------------------
# Tests: model provider resolution
# ---------------------------------------------------------------------------

class TestRagarsScorerProviderResolution:
    """Verify that RagasScorer resolves the evaluator LLM from config."""

    @patch("src.eval.ragas_scorer.get_embedder")
    @patch("src.eval.ragas_scorer.load_model_with_fallback")
    @patch("src.eval.ragas_scorer.get_config")
    def test_uses_main_model_provider_when_eval_config_is_empty(
        self, mock_cfg, mock_load, mock_embedder
    ) -> None:
        """When evaluator_provider/evaluator_model are empty, use model.provider/model.name."""
        from src.eval.ragas_scorer import RagasScorer

        mock_cfg.return_value = _make_config(provider="ollama", name="gemma2:2b")
        mock_load.return_value = MagicMock()
        mock_embedder.return_value = MagicMock()

        scorer = RagasScorer()

        assert scorer.evaluator_provider == "ollama"
        assert scorer.evaluator_model == "gemma2:2b"
        mock_load.assert_called_once_with(
            "ollama", "gemma2:2b",
            fallback_provider="google",
            fallback_name="gemini-2.5-flash",
        )

    @patch("src.eval.ragas_scorer.get_embedder")
    @patch("src.eval.ragas_scorer.load_model_with_fallback")
    @patch("src.eval.ragas_scorer.get_config")
    def test_uses_explicit_evaluator_config_when_set(
        self, mock_cfg, mock_load, mock_embedder
    ) -> None:
        """Explicit evaluator_provider/evaluator_model override main model config."""
        from src.eval.ragas_scorer import RagasScorer

        mock_cfg.return_value = _make_config(
            provider="ollama",
            name="gemma2:2b",
            evaluator_provider="google",
            evaluator_model="gemini-2.5-flash",
        )
        mock_load.return_value = MagicMock()
        mock_embedder.return_value = MagicMock()

        scorer = RagasScorer()

        assert scorer.evaluator_provider == "google"
        assert scorer.evaluator_model == "gemini-2.5-flash"
        mock_load.assert_called_once_with(
            "google", "gemini-2.5-flash",
            fallback_provider="google",
            fallback_name="gemini-2.5-flash",
        )

    @patch("src.eval.ragas_scorer.get_embedder")
    @patch("src.eval.ragas_scorer.load_model_with_fallback")
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
        assert scorer._llm_auto_loaded is False

    @patch("src.eval.ragas_scorer.get_embedder")
    @patch("src.eval.ragas_scorer.load_model_with_fallback")
    @patch("src.eval.ragas_scorer.get_config")
    def test_fallback_provider_forwarded_correctly(
        self, mock_cfg, mock_load, mock_embedder
    ) -> None:
        """Fallback provider and name are taken from model.fallback_* config."""
        from src.eval.ragas_scorer import RagasScorer

        mock_cfg.return_value = _make_config(
            provider="ollama",
            name="gemma2:2b",
            fallback_provider="google",
            fallback_name="gemini-2.5-flash",
        )
        mock_load.return_value = MagicMock()
        mock_embedder.return_value = MagicMock()

        scorer = RagasScorer()

        assert scorer._fallback_provider == "google"
        assert scorer._fallback_name == "gemini-2.5-flash"

    @patch("src.eval.ragas_scorer.get_embedder")
    @patch("src.eval.ragas_scorer.load_model_with_fallback")
    @patch("src.eval.ragas_scorer.get_config")
    def test_no_fallback_when_not_configured(
        self, mock_cfg, mock_load, mock_embedder
    ) -> None:
        """Missing fallback config results in None fallback fields."""
        from src.eval.ragas_scorer import RagasScorer

        mock_cfg.return_value = _make_config(fallback_provider="", fallback_name="")
        mock_load.return_value = MagicMock()
        mock_embedder.return_value = MagicMock()

        scorer = RagasScorer()

        assert scorer._fallback_provider is None
        assert scorer._fallback_name is None
        mock_load.assert_called_once_with(
            "ollama", "gemma2:2b",
            fallback_provider=None,
            fallback_name=None,
        )


# ---------------------------------------------------------------------------
# Tests: should_retry_with_fallback
# ---------------------------------------------------------------------------

class TestShouldRetryWithFallback:
    """Verify retry-with-fallback decision logic."""

    def _make_scorer(
        self, fallback_provider: str | None = "google", fallback_name: str | None = "gemini-2.5-flash"
    ):
        """Build a scorer with mocked LLM and embedder, no real loading."""
        with (
            patch("src.eval.ragas_scorer.get_config") as mock_cfg,
            patch("src.eval.ragas_scorer.load_model_with_fallback", return_value=MagicMock()),
            patch("src.eval.ragas_scorer.get_embedder", return_value=MagicMock()),
        ):
            mock_cfg.return_value = _make_config(
                fallback_provider=fallback_provider or "",
                fallback_name=fallback_name or "",
            )
            from src.eval.ragas_scorer import RagasScorer
            return RagasScorer()

    def test_no_retry_when_llm_externally_supplied(self) -> None:
        """Externally-injected LLM should never trigger a retry."""
        with (
            patch("src.eval.ragas_scorer.get_config") as mock_cfg,
            patch("src.eval.ragas_scorer.get_embedder", return_value=MagicMock()),
        ):
            mock_cfg.return_value = _make_config()
            from src.eval.ragas_scorer import RagasScorer
            scorer = RagasScorer(llm=MagicMock())

        assert scorer._should_retry_with_fallback([]) is False

    def test_no_retry_when_no_fallback_configured(self) -> None:
        """Missing fallback config should prevent retry."""
        scorer = self._make_scorer(fallback_provider=None, fallback_name=None)
        assert scorer._should_retry_with_fallback([]) is False

    def test_retry_on_empty_scores(self) -> None:
        """Empty scores list should trigger a retry."""
        scorer = self._make_scorer()
        assert scorer._should_retry_with_fallback([]) is True

    def test_retry_when_all_scores_nan(self) -> None:
        """All-NaN score rows should trigger a retry."""
        import math

        scorer = self._make_scorer()
        all_nan = [{"faithfulness": float("nan"), "answer_relevancy": float("nan")}]
        assert scorer._should_retry_with_fallback(all_nan) is True

    def test_no_retry_when_any_valid_score(self) -> None:
        """At least one valid numeric score should suppress retry."""
        scorer = self._make_scorer()
        mixed = [{"faithfulness": 0.8, "answer_relevancy": float("nan")}]
        assert scorer._should_retry_with_fallback(mixed) is False


# ---------------------------------------------------------------------------
# Tests: score() skipping behaviour (no live LLM needed)
# ---------------------------------------------------------------------------

class TestScorerSkippingBehaviour:
    """Verify score() correctly skips samples with errors or empty contexts."""

    def _make_scorer_with_mock_evaluate(self):
        """Return a scorer whose _evaluate_rows is patched to return dummy scores."""
        with (
            patch("src.eval.ragas_scorer.get_config") as mock_cfg,
            patch("src.eval.ragas_scorer.load_model_with_fallback", return_value=MagicMock()),
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
        scorer._evaluate_rows.assert_not_called()  # no scoreable samples at all

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

