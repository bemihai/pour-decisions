"""Reporter utilities for building and saving eval run outputs."""

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from src.eval.agent_metrics import TOOL_TRAJECTORY_METRICS
from src.eval.models import (
    EvalRunResult,
    GoldenSample,
    MetricCoverage,
    MetricOutcome,
    MetricSupportCounts,
    SampleResult,
)
from src.utils import logger

_RETRIEVAL_METRIC_PREFIXES = ("mrr", "precision_at_")
_RAG_JUDGE_METRICS = frozenset(
    {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
)
_CONTEXT_DEPENDENT_METRICS = frozenset(
    {"faithfulness", "context_precision", "context_recall"}
)
_AGENT_ANSWER_METRICS = frozenset({"answer_correctness"})


class EvalReporter:
    """Build, persist, and render aggregated eval run results."""

    @staticmethod
    def _has_status(result: SampleResult, status: str) -> bool:
        """Return whether a sample has the given status."""
        return result.status == status

    @staticmethod
    def _is_success(result: SampleResult) -> bool:
        """Return whether a sample completed successfully."""
        return result.status == "passed"

    def build(
        self,
        results: list[SampleResult],
        samples: list[GoldenSample],
        mode: Literal["retrieval", "full"],
        backend: Literal["rag", "retriever", "agent"],
        config_snapshot: dict[str, Any],
        git_sha: str = "unknown",
        run_metadata: dict[str, Any] | None = None,
    ) -> EvalRunResult:
        """Build an aggregated eval result payload from per-sample outputs.

        Args:
            results: Per-sample runtime and scoring outputs.
            samples: Original golden samples aligned by ``id``.
            mode: Eval mode (``retrieval`` or ``full``).
            backend: Backend under test (production ``rag``, low-level
                ``retriever``, or ``agent``).
            config_snapshot: Serializable config subset for reproducibility.
            git_sha: Optional short git SHA for traceability.
            run_metadata: Optional structured metadata about dataset identity, filters,
                and execution settings.

        Returns:
            An ``EvalRunResult`` containing aggregate metrics, category breakdown, and summary counters.
        """
        now = datetime.now(UTC)
        run_id = now.strftime("%Y%m%dT%H%M%S")
        timestamp = now.isoformat().replace("+00:00", "Z")

        categories_by_id = {sample.id: sample.category for sample in samples}
        samples_by_id = {sample.id: sample for sample in samples}
        active_metric_names = self._active_metric_names(
            mode=mode,
            backend=backend,
            config_snapshot=config_snapshot,
        )
        self._annotate_metric_outcomes(
            results=results,
            samples_by_id=samples_by_id,
            active_metric_names=active_metric_names,
        )
        aggregate_metrics = self._mean_metrics(results)
        metric_groups = self._build_metric_groups(results)
        metric_coverage = self._build_metric_coverage(
            results=results,
            categories_by_id=categories_by_id,
            active_metric_names=active_metric_names,
        )

        metrics_by_category: dict[str, dict[str, float]] = {}
        grouped: dict[str, list[SampleResult]] = defaultdict(list)
        for result in results:
            category = categories_by_id.get(result.id)
            if category:
                grouped[category].append(result)

        for category, category_results in grouped.items():
            category_metrics = self._mean_metrics(category_results)
            if category_metrics:
                metrics_by_category[category] = category_metrics

        skipped = sum(1 for result in results if self._has_status(result, "skipped"))
        timeouts = sum(1 for result in results if self._has_status(result, "timeout"))
        errors = sum(1 for result in results if self._has_status(result, "failed"))
        evaluated = sum(1 for result in results if self._is_success(result))
        total_latency_ms = round(sum(result.latency_ms for result in results), 3)

        summary = {
            "total_samples": len(samples),
            "evaluated": evaluated,
            "skipped": skipped,
            "timeouts": timeouts,
            "errors": errors,
            "total_latency_ms": total_latency_ms,
            "evaluation_target": {
                "rag": "production_rag",
                "retriever": "retriever_benchmark",
                "agent": "agent",
            }[backend],
        }
        summary.update(
            self._estimate_llm_call_breakdown(
                results=results,
                mode=mode,
                backend=backend,
                config_snapshot=config_snapshot,
            )
        )
        if run_metadata:
            summary.update(run_metadata)

        return EvalRunResult(
            run_id=run_id,
            timestamp=timestamp,
            mode=mode,
            backend=backend,
            git_sha=git_sha,
            config_snapshot=config_snapshot,
            aggregate_metrics=aggregate_metrics,
            metrics_by_category=metrics_by_category,
            metric_groups=metric_groups,
            metric_coverage=metric_coverage,
            per_sample=results,
            summary=summary,
        )

    def save(self, result: EvalRunResult, output_dir: Path) -> Path:
        """Save one eval run result as JSON.

        Args:
            result: Built run result payload.
            output_dir: Directory where result files are stored.

        Returns:
            Path of the written JSON file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{result.run_id}_{result.mode}_{result.backend}.json"
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved eval report to %s", output_path)
        return output_path

    def print_summary(self, result: EvalRunResult) -> None:
        """Log a compact summary table for the eval run.

        Args:
            result: Eval run result to render.
        """
        lines: list[str] = [
            f"=== Eval Run {result.run_id} ===",
            (
                f"Backend: {result.backend} | Mode: {result.mode} | "
                f"Samples: {result.summary.get('evaluated', 0)}/{result.summary.get('total_samples', 0)}"
            ),
            "",
            "Aggregate Metrics:",
        ]

        for metric_name, value in sorted(result.aggregate_metrics.items()):
            lines.append(f"  {metric_name:<18} {value:.4f}")

        if result.metric_groups:
            lines.append("")
            lines.append("Metric Groups:")
            for group_name, group_scores in result.metric_groups.items():
                lines.append(f"  {group_name}:")
                for metric_name, value in sorted(group_scores.items()):
                    lines.append(f"    {metric_name:<18} {value:.4f}")

        if result.metric_coverage:
            lines.append("")
            lines.append("Metric Coverage:")
            for metric_name, coverage in sorted(result.metric_coverage.items()):
                lines.append(
                    f"  {metric_name:<18} scored={coverage.scored} "
                    f"unsupported={coverage.unsupported} skipped={coverage.skipped} "
                    f"errored={coverage.errored}"
                )

        if result.metrics_by_category:
            lines.append("")
            lines.append("By Category:")
            for category in sorted(result.metrics_by_category):
                category_scores = result.metrics_by_category[category]
                metric_chunks = [f"{name}={score:.4f}" for name, score in sorted(category_scores.items())]
                lines.append(f"  {category:<10} {'  '.join(metric_chunks)}")

        logger.info("\n%s", "\n".join(lines))

    def _mean_metrics(self, results: list[SampleResult]) -> dict[str, float]:
        """Compute mean metric values across samples with available scores."""
        values_by_metric: dict[str, list[float]] = defaultdict(list)
        for result in results:
            if result.status != "passed":
                continue
            for metric_name, metric_value in result.scores.items():
                values_by_metric[metric_name].append(float(metric_value))

        return {
            metric_name: sum(values) / len(values)
            for metric_name, values in values_by_metric.items()
            if values
        }

    def _build_metric_groups(
        self,
        results: list[SampleResult],
    ) -> dict[str, dict[str, float]]:
        """Group aggregate values by stable metric family."""
        aggregate = self._mean_metrics(results)
        return {
            "retrieval": {
                name: value
                for name, value in aggregate.items()
                if self._is_retrieval_metric(name)
            },
            "rag_judge": {
                name: value
                for name, value in aggregate.items()
                if name in _RAG_JUDGE_METRICS
            },
            "agent_tool": {
                name: aggregate[name]
                for name in TOOL_TRAJECTORY_METRICS
                if name in aggregate
            },
            "agent_answer": {
                name: value
                for name, value in aggregate.items()
                if name in _AGENT_ANSWER_METRICS
            },
            "operational": self._operational_metrics(results),
        }

    def _active_metric_names(
        self,
        mode: str,
        backend: str,
        config_snapshot: dict[str, Any],
    ) -> list[str]:
        """Resolve the complete metric set active for one run."""
        metric_names: list[str] = []
        eval_snapshot = config_snapshot.get("eval", {})

        if backend in {"rag", "retriever"}:
            k_values = eval_snapshot.get("retrieval_k_values", [3, 5])
            metric_names.extend(["mrr", *(f"precision_at_{int(k)}" for k in k_values)])

        if mode == "full":
            metric_names.extend(
                str(name)
                for name in eval_snapshot.get(
                    "ragas_metrics",
                    ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
                )
            )

        if backend == "agent":
            metric_names.extend(TOOL_TRAJECTORY_METRICS)
            if mode == "full":
                metric_names.append("answer_correctness")

        return list(dict.fromkeys(metric_names))

    def _annotate_metric_outcomes(
        self,
        results: list[SampleResult],
        samples_by_id: dict[str, GoldenSample],
        active_metric_names: list[str],
    ) -> None:
        """Explain the outcome of every active metric for every sample."""
        for result in results:
            sample = samples_by_id.get(result.id)
            result.metric_outcomes = {
                metric_name: self._metric_outcome(
                    result=result,
                    sample=sample,
                    metric_name=metric_name,
                )
                for metric_name in active_metric_names
            }

    def _metric_outcome(
        self,
        result: SampleResult,
        sample: GoldenSample | None,
        metric_name: str,
    ) -> MetricOutcome:
        """Resolve one metric's sample-level outcome and reason."""
        if result.status == "skipped":
            return MetricOutcome(status="skipped", reason=result.error or "sample_skipped")
        if result.status in {"failed", "timeout"}:
            return MetricOutcome(status="errored", reason=result.error or f"sample_{result.status}")
        if result.status == "unsupported":
            return MetricOutcome(status="unsupported", reason=result.error or "sample_unsupported")

        if metric_name in result.scores:
            return MetricOutcome(status="scored")

        unsupported_reason = result.unsupported_metrics.get(metric_name)
        if unsupported_reason:
            return MetricOutcome(status="unsupported", reason=unsupported_reason)

        if self._is_retrieval_metric(metric_name) and (
            sample is None or not sample.ground_truth_chunk_ids
        ):
            return MetricOutcome(status="unsupported", reason="no_ground_truth_chunk_ids")
        if metric_name in _CONTEXT_DEPENDENT_METRICS and not result.contexts:
            return MetricOutcome(status="unsupported", reason="no_rag_evidence")
        if metric_name == "answer_relevancy" and not result.answer.strip():
            return MetricOutcome(status="unsupported", reason="no_answer")
        if metric_name in TOOL_TRAJECTORY_METRICS and (
            sample is None or not sample.expected_tool_calls
        ):
            return MetricOutcome(status="unsupported", reason="no_expected_tool_calls")
        if metric_name == "answer_correctness" and not result.answer.strip():
            return MetricOutcome(status="unsupported", reason="no_answer")
        if metric_name == "answer_correctness" and (
            sample is None or not (sample.ground_truth or sample.expected_facts)
        ):
            return MetricOutcome(status="unsupported", reason="no_answer_reference")

        return MetricOutcome(status="errored", reason="score_not_returned")

    def _build_metric_coverage(
        self,
        results: list[SampleResult],
        categories_by_id: dict[str, str],
        active_metric_names: list[str],
    ) -> dict[str, MetricCoverage]:
        """Count metric outcomes overall and by sample category."""
        coverage: dict[str, MetricCoverage] = {}
        for metric_name in active_metric_names:
            overall_counts = self._count_metric_outcomes(results, metric_name)
            category_results: dict[str, list[SampleResult]] = defaultdict(list)
            for result in results:
                category = categories_by_id.get(result.id, "unknown")
                category_results[category].append(result)
            coverage[metric_name] = MetricCoverage(
                **overall_counts.model_dump(),
                by_category={
                    category: self._count_metric_outcomes(grouped_results, metric_name)
                    for category, grouped_results in sorted(category_results.items())
                },
            )
        return coverage

    def _count_metric_outcomes(
        self,
        results: list[SampleResult],
        metric_name: str,
    ) -> MetricSupportCounts:
        """Count outcome statuses for one metric over a result slice."""
        counts = {"scored": 0, "unsupported": 0, "skipped": 0, "errored": 0}
        for result in results:
            outcome = result.metric_outcomes[metric_name]
            counts[outcome.status] += 1
        return MetricSupportCounts(**counts)

    def _operational_metrics(self, results: list[SampleResult]) -> dict[str, float]:
        """Compute run-level latency and outcome rates."""
        total = len(results)
        if total == 0:
            return {
                "mean_latency_ms": 0.0,
                "success_rate": 0.0,
                "error_rate": 0.0,
                "timeout_rate": 0.0,
                "skip_rate": 0.0,
            }

        passed = sum(result.status == "passed" for result in results)
        failed = sum(result.status == "failed" for result in results)
        timed_out = sum(result.status == "timeout" for result in results)
        skipped = sum(result.status == "skipped" for result in results)
        return {
            "mean_latency_ms": sum(result.latency_ms for result in results) / total,
            "success_rate": passed / total,
            "error_rate": (failed + timed_out) / total,
            "timeout_rate": timed_out / total,
            "skip_rate": skipped / total,
        }

    @staticmethod
    def _is_retrieval_metric(metric_name: str) -> bool:
        """Return whether a metric belongs to the retrieval family."""
        return metric_name == _RETRIEVAL_METRIC_PREFIXES[0] or metric_name.startswith(
            _RETRIEVAL_METRIC_PREFIXES[1]
        )

    def _estimate_llm_call_breakdown(
        self,
        results: list[SampleResult],
        mode: str,
        backend: str,
        config_snapshot: dict[str, Any],
    ) -> dict[str, int]:
        """Estimate generation and judge LLM calls for this run.

        The estimate is intentionally simple and stable for longitudinal comparison.
        """
        successful = [result for result in results if self._is_success(result)]

        if backend == "retriever" or (mode == "retrieval" and backend == "rag"):
            backend_calls_per_sample = 0
        else:
            backend_calls_per_sample = 1 if backend == "rag" else 3
        estimated_generation = len(successful) * backend_calls_per_sample
        estimated_judge = 0

        if mode == "full":
            configured_metrics = config_snapshot.get("eval", {}).get(
                "ragas_metrics",
                ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
            )
            context_metrics = {
                "faithfulness",
                "context_precision",
                "context_recall",
            }
            answer_metrics = {"answer_relevancy"}
            context_scoreable = [result for result in successful if result.contexts]
            estimated_judge += (
                len(context_scoreable)
                * len(context_metrics.intersection(configured_metrics))
                * 3
            )
            estimated_judge += (
                len(successful)
                * len(answer_metrics.intersection(configured_metrics))
                * 3
            )
            if backend == "agent":
                estimated_judge += len(successful) * 3

        return {
            "estimated_generation_llm_calls": estimated_generation,
            "estimated_judge_llm_calls": estimated_judge,
            "estimated_llm_calls": estimated_generation + estimated_judge,
        }
