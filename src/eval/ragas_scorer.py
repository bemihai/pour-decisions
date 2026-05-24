"""Ragas scorer for full evaluation mode.

This module applies Ragas metrics to `SampleResult` objects produced by the eval
runner. It is designed to be called only in full mode, where LLM-as-judge scoring
is explicitly enabled.

The evaluator LLM is resolved in priority order:
1. Explicitly injected ``llm`` argument (testing / overrides).
2. ``eval.ragas.evaluator_provider`` / ``eval.ragas.evaluator_model`` in ``app_config.yml``.
3. ``model.provider`` / ``model.name`` (the main pipeline model, local Ollama by default).

A ``model.fallback_provider`` / ``model.fallback_name`` is applied automatically when
the primary provider is unavailable (e.g. Ollama offline during CI).
"""

from __future__ import annotations

import math
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.agents.llm import load_base_model, load_model_with_fallback
from src.eval.models import SampleResult
from src.utils import get_config, get_embedder, logger


class RagasScorer:
    """Score eval sample results with Ragas metrics.

    The scorer evaluates only samples that have no execution error and at least one
    retrieved context. Scores are written in-place into each `SampleResult.scores`.
    """

    def __init__(self, llm: BaseChatModel | None = None, embedder: Embeddings | None = None):
        """Initialize scorer dependencies.

        The evaluator LLM is resolved from config if not explicitly supplied.
        By default it uses the same provider configured for the main pipeline
        (``model.provider`` / ``model.name``), which is local Ollama unless overridden
        in ``app_config.yml`` or environment variables.

        Args:
            llm: Optional evaluator LLM. If not provided, loads the model configured
                at ``eval.ragas.evaluator_provider`` / ``eval.ragas.evaluator_model``
                (or falls back to ``model.provider`` / ``model.name``).
            embedder: Optional evaluator embedder. If not provided, reuses
                ``get_embedder()`` cached local embedder.
        """
        cfg = get_config()

        # Resolve evaluator provider/model: prefer explicit eval config, fallback to main model config.
        configured_provider = str(getattr(cfg.eval.ragas, "evaluator_provider", "")).strip()
        configured_model = str(getattr(cfg.eval.ragas, "evaluator_model", "")).strip()

        self.evaluator_provider = configured_provider or str(cfg.model.provider)
        self.evaluator_model = configured_model or str(cfg.model.name)

        self._fallback_provider = str(getattr(cfg.model, "fallback_provider", "")).strip() or None
        self._fallback_name = str(getattr(cfg.model, "fallback_name", "")).strip() or None
        self._llm_auto_loaded = llm is None

        self.llm = llm or load_model_with_fallback(
            self.evaluator_provider,
            self.evaluator_model,
            fallback_provider=self._fallback_provider,
            fallback_name=self._fallback_name,
        )
        self.embedder = embedder or get_embedder()

    def score(self, results: list[SampleResult]) -> list[SampleResult]:
        """Score sample results using faithfulness/relevancy/context metrics.

        Args:
            results: Runner outputs to score.

        Returns:
            The same list with Ragas metric values merged into each sample's
            ``scores`` dictionary.
        """
        scoreable: list[tuple[int, SampleResult]] = [
            (index, sample)
            for index, sample in enumerate(results)
            if sample.error is None and bool(sample.contexts)
        ]

        if not scoreable:
            logger.info("RagasScorer: no scoreable samples (all failed or missing contexts)")
            return results

        estimated_calls = len(scoreable) * 4 * 3
        logger.info(
            "RagasScorer: scoring %d samples with %s/%s, estimated LLM calls: ~%d",
            len(scoreable),
            self.evaluator_provider,
            self.evaluator_model,
            estimated_calls,
        )

        ragas_payload = [self._to_ragas_row(sample) for _, sample in scoreable]
        ragas_scores = self._evaluate_rows(ragas_payload)

        if self._should_retry_with_fallback(ragas_scores):
            assert self._fallback_provider is not None
            assert self._fallback_name is not None
            logger.warning(
                "RagasScorer: evaluator %s/%s returned invalid scores; retrying with fallback %s/%s",
                self.evaluator_provider,
                self.evaluator_model,
                self._fallback_provider,
                self._fallback_name,
            )
            self.llm = load_base_model(self._fallback_provider, self._fallback_name)
            ragas_scores = self._evaluate_rows(ragas_payload)

        for (original_index, _sample), score_dict in zip(scoreable, ragas_scores):
            for metric_name, value in score_dict.items():
                if value is None:
                    continue
                try:
                    parsed = float(value)
                    if math.isnan(parsed):
                        logger.warning(
                            "RagasScorer: NaN score for metric=%s sample=%s; coercing to 0.0",
                            metric_name,
                            results[original_index].id,
                        )
                        parsed = 0.0
                    results[original_index].scores[metric_name] = parsed
                except (TypeError, ValueError):
                    logger.warning(
                        "RagasScorer: non-numeric score for metric=%s value=%r",
                        metric_name,
                        value,
                    )

        return results

    def _to_ragas_row(self, sample: SampleResult) -> dict[str, Any]:
        """Convert one sample to a Ragas single-turn payload row."""
        return {
            "user_input": sample.question,
            "response": sample.answer,
            "retrieved_contexts": sample.contexts,
            # Context-recall metrics require reference text.
            "reference": sample.ground_truth or sample.answer,
        }

    def _evaluate_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run Ragas evaluation and normalize output to list-of-dicts.

        Args:
            rows: Ragas-compatible rows.

        Returns:
            List of metric dictionaries aligned with input row order.
        """
        try:
            from ragas import EvaluationDataset, evaluate
        except ImportError:
            from ragas import evaluate  # type: ignore
            from ragas.dataset_schema import EvaluationDataset  # type: ignore

        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        try:
            from ragas.metrics import AnswerRelevancy
        except ImportError:
            from ragas.metrics import ResponseRelevancy as AnswerRelevancy  # type: ignore

        from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness

        evaluation_dataset = EvaluationDataset.from_list(rows)
        evaluator_llm = LangchainLLMWrapper(self.llm)
        evaluator_embeddings = LangchainEmbeddingsWrapper(self.embedder)

        evaluation_result = evaluate(
            dataset=evaluation_dataset,
            metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

        if hasattr(evaluation_result, "scores") and isinstance(evaluation_result.scores, list):
            return [dict(item) for item in evaluation_result.scores]

        if hasattr(evaluation_result, "to_pandas"):
            frame = evaluation_result.to_pandas()
            return [
                {str(col): frame.iloc[idx][col] for col in frame.columns}
                for idx in range(len(frame))
            ]

        raise RuntimeError("Unsupported Ragas evaluation result format")

    def _should_retry_with_fallback(self, ragas_scores: list[dict[str, Any]]) -> bool:
        """Decide whether to retry scoring with fallback model.

        A retry is triggered only when:
        - LLM was auto-loaded by this scorer (not externally supplied),
        - a fallback provider/model is configured,
        - all score rows appear invalid (empty/None/NaN values).
        """
        if not self._llm_auto_loaded or not self._fallback_provider or not self._fallback_name:
            return False
        if not ragas_scores:
            return True

        def _row_has_any_valid_number(row: dict[str, Any]) -> bool:
            for value in row.values():
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isnan(parsed):
                    return True
            return False

        return not any(_row_has_any_valid_number(row) for row in ragas_scores)

