"""Ragas scorer for full evaluation mode.

This module applies Ragas metrics to ``SampleResult`` objects produced by the
Eval runner. It is designed to be called only in full mode, where
LLM-as-judge scoring is explicitly enabled.

The evaluator LLM is resolved in priority order:
1. Explicitly injected ``llm`` argument (testing / overrides).
2. ``eval.ragas.evaluator_provider`` / ``eval.ragas.evaluator_model`` in
   ``app_config.yml``.
3. ``model.provider`` / ``model.name`` (local Ollama by default).

No implicit provider fallback is applied by this module.
"""

from __future__ import annotations

import math
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.eval.models import SampleResult
from src.eval.utils import load_eval_model, resolve_eval_model_config
from src.utils import get_config, get_embedder, logger


class RagasScorer:
    """Score eval sample results with Ragas metrics.

    The scorer evaluates only samples that have no execution error and at least
    one retrieved context. Scores are written in-place into each
    ``SampleResult.scores``.
    """

    def __init__(self, llm: BaseChatModel | None = None, embedder: Embeddings | None = None):
        """Initialize scorer dependencies.

        Args:
            llm: Optional evaluator LLM. If not provided, loads the model
                configured at ``eval.ragas.evaluator_provider`` /
                ``eval.ragas.evaluator_model`` (or falls back to
                ``model.provider`` / ``model.name``).
            embedder: Optional evaluator embedder. If not provided, reuses
                ``get_embedder()`` cached local embedder.
        """
        cfg = get_config()

        self.evaluator_provider, self.evaluator_model, _ = resolve_eval_model_config(cfg)
        self.llm = llm or load_eval_model(cfg)
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
