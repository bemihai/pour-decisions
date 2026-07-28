"""Ragas scorer for full evaluation mode.

This module applies Ragas metrics to ``SampleResult`` objects produced by the
EvalRunner. It is designed to be called only in full mode, where
LLM-as-judge scoring is explicitly enabled.
"""

import logging
import math
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.eval.models import SampleResult
from src.eval.utils import load_eval_model, resolve_eval_model_config
from src.utils import get_config, get_embedder, logger

_SUPPORTED_RAGAS_METRICS = frozenset(
    {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
)
_CONTEXT_DEPENDENT_METRICS = frozenset(
    {"faithfulness", "context_precision", "context_recall"}
)
_ANSWER_ONLY_METRICS = frozenset({"answer_relevancy"})
_ANSWER_CORRECTNESS_METRIC = "answer_correctness"


class _RagasJobErrorCapture(logging.Handler):
    """Capture Ragas executor failures without suppressing its normal logs."""

    def __init__(self) -> None:
        """Initialize an empty job-indexed failure map."""
        super().__init__()
        self.failures: dict[int, str] = {}

    def emit(self, record: logging.LogRecord) -> None:
        """Store the stable exception type and bounded diagnostic message."""
        if record.msg != "Exception raised in Job[%s]: %s(%s)":
            return
        if not isinstance(record.args, tuple) or len(record.args) != 3:
            return

        job_index, exception_name, exception_message = record.args
        try:
            parsed_job_index = int(job_index)
        except (TypeError, ValueError):
            return

        exception_name = str(exception_name)
        if exception_name == "TimeoutError":
            reason = "ragas_timeout"
        else:
            normalized_message = " ".join(str(exception_message).split())
            detail = f":{normalized_message[:240]}" if normalized_message else ""
            reason = f"ragas_exception:{exception_name}{detail}"
        self.failures[parsed_job_index] = reason


class RagasScorer:
    """Score eval sample results with Ragas metrics.

    Context-dependent metrics evaluate only samples with RAG evidence. Answer-only
    metrics may evaluate any successful response. Scores are written in-place.
    """

    def __init__(self, llm: BaseChatModel | None = None, embedder: Embeddings | None = None):
        """Initialize scorer dependencies.

        Args:
            llm: Optional evaluator LLM. If not provided, loads the model specified in config.
            embedder: Optional evaluator embedder. If not provided, reuses the cached local embedder.
        """
        cfg = get_config()

        self.evaluator_provider, self.evaluator_model, _ = resolve_eval_model_config(cfg)
        configured_metrics = getattr(cfg.eval.ragas, "metrics", None)
        if configured_metrics is None:
            configured_metric_names = list(_SUPPORTED_RAGAS_METRICS)
        else:
            configured_metric_names = [
                str(metric).strip()
                for metric in configured_metrics
                if str(metric).strip()
            ]
        unsupported_metrics = sorted({name for name in configured_metric_names if name not in _SUPPORTED_RAGAS_METRICS})
        if unsupported_metrics:
            raise ValueError(
                "Unsupported eval.ragas.metrics values: "
                f"{', '.join(unsupported_metrics)}. "
                f"Supported metrics: {', '.join(sorted(_SUPPORTED_RAGAS_METRICS))}."
            )
        self.metric_names = configured_metric_names
        self.ragas_timeout_seconds = float(getattr(cfg.eval.ragas, "timeout_seconds", 60) or 0)
        self.ragas_max_retries = max(1, int(getattr(cfg.eval.ragas, "max_retries", 1)))
        self.llm = llm or load_eval_model(cfg)
        self.embedder = embedder or get_embedder()

    def score(self, results: list[SampleResult]) -> list[SampleResult]:
        """Score sample results using faithfulness/relevancy/context metrics.

        Args:
            results: Runner outputs to score.

        Returns:
            The same list with Ragas metric values merged into each sample's ``scores`` dictionary.
        """
        passed_samples: list[tuple[int, SampleResult]] = [
            (index, sample)
            for index, sample in enumerate(results)
            if sample.status == "passed"
        ]
        passed_with_answers = [
            (index, sample)
            for index, sample in passed_samples
            if bool(sample.answer.strip())
        ]
        context_scoreable = [
            (index, sample)
            for index, sample in passed_with_answers
            if bool(sample.contexts)
        ]

        context_metric_names = [
            name for name in self.metric_names if name in _CONTEXT_DEPENDENT_METRICS
        ]
        answer_metric_names = [
            name for name in self.metric_names if name in _ANSWER_ONLY_METRICS
        ]

        context_scoreable_ids = {index for index, _ in context_scoreable}
        for index, sample in passed_samples:
            if index in context_scoreable_ids:
                continue
            for metric_name in context_metric_names:
                sample.unsupported_metrics[metric_name] = "no_rag_evidence"

        if not passed_with_answers:
            logger.info("RagasScorer: no successful answers to score")
            return results

        estimated_calls = (
            len(passed_with_answers) * len(answer_metric_names) * 3
            + len(context_scoreable) * len(context_metric_names) * 3
        )
        logger.info(
            "RagasScorer: scoring %d answers and %d RAG contexts with %s/%s, estimated LLM calls: ~%d",
            len(passed_with_answers),
            len(context_scoreable),
            self.evaluator_provider,
            self.evaluator_model,
            estimated_calls,
        )

        if answer_metric_names:
            self._score_rows(
                results=results,
                scoreable=passed_with_answers,
                metric_names=answer_metric_names,
            )
        if context_metric_names and context_scoreable:
            self._score_rows(
                results=results,
                scoreable=context_scoreable,
                metric_names=context_metric_names,
            )

        return results

    def score_agent_answers(self, results: list[SampleResult]) -> list[SampleResult]:
        """Score complete agent answers against ground truth and expected facts."""
        scoreable = [
            (index, sample)
            for index, sample in enumerate(results)
            if sample.status == "passed"
            and bool(sample.answer.strip())
            and bool(sample.ground_truth or sample.expected_facts)
        ]
        if not scoreable:
            logger.info("RagasScorer: no agent answers with reference facts to score")
            return results

        logger.info(
            "RagasScorer: scoring answer correctness for %d agent samples with %s/%s",
            len(scoreable),
            self.evaluator_provider,
            self.evaluator_model,
        )
        rows = [
            self._to_ragas_row(sample, reference=self._build_answer_reference(sample))
            for _, sample in scoreable
        ]
        score_dicts, error_dicts = self._evaluate_rows(rows, [_ANSWER_CORRECTNESS_METRIC])
        self._merge_scores(
            results,
            scoreable,
            score_dicts,
            metric_names=[_ANSWER_CORRECTNESS_METRIC],
            error_dicts=error_dicts,
        )
        return results

    def _score_rows(
        self,
        results: list[SampleResult],
        scoreable: list[tuple[int, SampleResult]],
        metric_names: list[str],
    ) -> None:
        """Evaluate and merge one support-compatible metric batch."""
        rows = [self._to_ragas_row(sample) for _, sample in scoreable]
        score_dicts, error_dicts = self._evaluate_rows(rows, metric_names)
        score_dicts = [
            {
                metric_name: value
                for metric_name, value in score_dict.items()
                if metric_name in metric_names
            }
            for score_dict in score_dicts
        ]
        self._merge_scores(
            results,
            scoreable,
            score_dicts,
            metric_names=metric_names,
            error_dicts=error_dicts,
        )

    def _merge_scores(
        self,
        results: list[SampleResult],
        scoreable: list[tuple[int, SampleResult]],
        score_dicts: list[dict[str, Any]],
        metric_names: list[str],
        error_dicts: list[dict[str, str]] | None = None,
    ) -> None:
        """Merge normalized Ragas values into their original samples."""
        if len(score_dicts) != len(scoreable):
            raise RuntimeError(
                "Ragas result row count mismatch for metrics "
                f"{', '.join(metric_names)}: submitted {len(scoreable)} rows, "
                f"received {len(score_dicts)} rows."
            )

        normalized_error_dicts = error_dicts or [{} for _ in scoreable]
        if len(normalized_error_dicts) != len(scoreable):
            raise RuntimeError(
                "Ragas error row count mismatch for metrics "
                f"{', '.join(metric_names)}: submitted {len(scoreable)} rows, "
                f"received {len(normalized_error_dicts)} rows."
            )

        for (original_index, _), score_dict, error_dict in zip(
            scoreable,
            score_dicts,
            normalized_error_dicts,
        ):
            for metric_name, value in score_dict.items():
                if value is None:
                    continue
                try:
                    parsed = float(value)
                    if math.isnan(parsed):
                        error_reason = error_dict.get(metric_name, "ragas_returned_nan")
                        logger.warning(
                            "RagasScorer: NaN score for metric=%s sample=%s; recording metric error=%s",
                            metric_name,
                            results[original_index].id,
                            error_reason,
                        )
                        results[original_index].metric_errors[metric_name] = error_reason
                        continue
                    results[original_index].scores[metric_name] = parsed
                except (TypeError, ValueError):
                    logger.warning(
                        "RagasScorer: non-numeric score for metric=%s value=%r", metric_name, value
                    )
                    results[original_index].metric_errors[metric_name] = (
                        f"ragas_returned_non_numeric:{type(value).__name__}"
                    )

    def _to_ragas_row(
        self,
        sample: SampleResult,
        reference: str | None = None,
    ) -> dict[str, Any]:
        """Convert one sample to a Ragas single-turn payload row."""
        return {
            "user_input": sample.question,
            "response": sample.answer,
            "retrieved_contexts": sample.contexts,
            "reference": reference or sample.ground_truth or sample.answer,
        }

    def _build_answer_reference(self, sample: SampleResult) -> str:
        """Combine reference prose and explicit expected facts for judging."""
        parts: list[str] = []
        if sample.ground_truth:
            parts.append(sample.ground_truth)
        if sample.expected_facts:
            parts.append("Expected facts:\n" + "\n".join(f"- {fact}" for fact in sample.expected_facts))
        return "\n\n".join(parts)

    def _evaluate_rows(
        self,
        rows: list[dict[str, Any]],
        metric_names: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Run Ragas evaluation and normalize output to list-of-dicts.

        Args:
            rows: Ragas-compatible rows.

        Returns:
            Score and error dictionaries aligned with input row order.
        """
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.run_config import RunConfig
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        metric_classes = {
            "faithfulness": Faithfulness,
            "answer_relevancy": AnswerRelevancy,
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall,
            _ANSWER_CORRECTNESS_METRIC: AnswerCorrectness,
        }
        selected_metric_names = metric_names or self.metric_names

        evaluation_dataset = EvaluationDataset.from_list(rows)
        evaluator_llm = LangchainLLMWrapper(self.llm)
        evaluator_embeddings = LangchainEmbeddingsWrapper(self.embedder)
        run_config = RunConfig(
            timeout=max(1, int(self.ragas_timeout_seconds)),
            max_retries=self.ragas_max_retries,
        )
        error_capture = _RagasJobErrorCapture()
        ragas_executor_logger = logging.getLogger("ragas.executor")
        ragas_executor_logger.addHandler(error_capture)
        try:
            evaluation_result = evaluate(
                dataset=evaluation_dataset,
                metrics=[metric_classes[name]() for name in selected_metric_names],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                run_config=run_config,
            )
        finally:
            ragas_executor_logger.removeHandler(error_capture)

        error_dicts: list[dict[str, str]] = [{} for _ in rows]
        metric_count = len(selected_metric_names)
        for job_index, reason in error_capture.failures.items():
            if metric_count == 0:
                break
            row_index, metric_index = divmod(job_index, metric_count)
            if row_index < len(error_dicts) and metric_index < metric_count:
                error_dicts[row_index][selected_metric_names[metric_index]] = reason

        if hasattr(evaluation_result, "scores") and isinstance(evaluation_result.scores, list):
            return [dict(item) for item in evaluation_result.scores], error_dicts

        if hasattr(evaluation_result, "to_pandas"):
            frame = evaluation_result.to_pandas()
            return (
                [
                    {str(col): frame.iloc[idx][col] for col in frame.columns}
                    for idx in range(len(frame))
                ],
                error_dicts,
            )

        raise RuntimeError("Unsupported Ragas evaluation result format")
