"""Reporter utilities for building and saving eval run outputs."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from src.eval.models import EvalRunResult, GoldenSample, SampleResult
from src.utils import logger


class EvalReporter:
    """Build, persist, and render aggregated eval run results."""

    def build(
        self,
        results: list[SampleResult],
        samples: list[GoldenSample],
        mode: Literal["retrieval", "full"],
        backend: Literal["rag", "agent"],
        config_snapshot: dict[str, Any],
        git_sha: str = "unknown",
    ) -> EvalRunResult:
        """Build an aggregated eval result payload from per-sample outputs.

        Args:
            results: Per-sample runtime and scoring outputs.
            samples: Original golden samples aligned by ``id``.
            mode: Eval mode (``retrieval`` or ``full``).
            backend: Backend under test (``rag`` or ``agent``).
            config_snapshot: Serializable config subset for reproducibility.
            git_sha: Optional short git SHA for traceability.

        Returns:
            An ``EvalRunResult`` containing aggregate metrics, category breakdown,
            and summary counters.
        """
        now = datetime.now(UTC)
        run_id = now.strftime("%Y%m%dT%H%M%S")
        timestamp = now.isoformat().replace("+00:00", "Z")

        categories_by_id = {sample.id: sample.category for sample in samples}
        aggregate_metrics = self._mean_metrics(results)

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

        skipped = sum(1 for result in results if (result.error or "").startswith("skipped:"))
        errors = sum(1 for result in results if result.error and not result.error.startswith("skipped:"))
        evaluated = sum(1 for result in results if result.error is None)
        total_latency_ms = round(sum(result.latency_ms for result in results), 3)

        summary = {
            "total_samples": len(samples),
            "evaluated": evaluated,
            "skipped": skipped,
            "errors": errors,
            "total_llm_calls": self._estimate_llm_calls(results=results, mode=mode, backend=backend),
            "total_latency_ms": total_latency_ms,
        }

        return EvalRunResult(
            run_id=run_id,
            timestamp=timestamp,
            mode=mode,
            backend=backend,
            git_sha=git_sha,
            config_snapshot=config_snapshot,
            aggregate_metrics=aggregate_metrics,
            metrics_by_category=metrics_by_category,
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
            for metric_name, metric_value in result.scores.items():
                values_by_metric[metric_name].append(float(metric_value))

        return {
            metric_name: sum(values) / len(values)
            for metric_name, values in values_by_metric.items()
            if values
        }

    def _estimate_llm_calls(self, results: list[SampleResult], mode: str, backend: str) -> int:
        """Estimate total LLM calls for this run.

        The estimate is intentionally simple and stable for longitudinal comparison.
        """
        successful = [result for result in results if result.error is None]

        backend_calls_per_sample = 1 if backend == "rag" else 3
        estimated = len(successful) * backend_calls_per_sample

        if mode == "full":
            ragas_scoreable = [result for result in successful if result.contexts]
            estimated += len(ragas_scoreable) * 4 * 3

        return estimated


