"""Compare aggregate metrics across recent eval result files."""

import argparse
from dataclasses import dataclass
import json
import math
import sys
from pathlib import Path
from typing import Any, Literal

from src.eval.models import CURRENT_EVAL_RESULT_SCHEMA_VERSION, SUPPORTED_EVAL_RESULT_SCHEMA_VERSIONS
from src.utils import get_config, logger

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

ScalarValue = str | int | float | bool | None


@dataclass(frozen=True)
class ConfigLeafDiff:
    """One deterministic scalar-leaf comparison."""

    path: str
    status: Literal["added", "removed", "changed", "unchanged"]
    previous: ScalarValue = None
    latest: ScalarValue = None


@dataclass(frozen=True)
class PairedMetricDelta:
    """One metric delta for a sample scored in both reports."""

    metric: str
    sample_id: str
    previous: float
    latest: float

    @property
    def delta(self) -> float:
        """Return latest minus previous."""
        return self.latest - self.previous


@dataclass(frozen=True)
class RegressionGateResult:
    """One validated aggregate quality-metric gate result."""

    metric: str
    threshold: float
    previous: float
    latest: float

    @property
    def delta(self) -> float:
        """Return latest minus previous."""
        return self.latest - self.previous

    @property
    def regressed(self) -> bool:
        """Return whether the quality drop exceeds the configured threshold."""
        boundary = -self.threshold
        return self.delta < boundary and not math.isclose(
            self.delta,
            boundary,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


class ComparisonInputError(ValueError):
    """Raised when regression-gate inputs are invalid or incomparable."""


_HIGHER_IS_BETTER_METRICS = frozenset(
    {
        "answer_correctness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "faithfulness",
        "mrr",
        "tool_exact_match",
        "tool_ordered_match",
        "tool_precision",
        "tool_recall",
    }
)


def _sorted_result_files(results_dir: Path) -> list[Path]:
    """Return eval result JSON files ordered by mtime descending."""
    if not results_dir.exists():
        return []
    files = [path for path in results_dir.glob("*.json") if path.is_file()]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _load_result(path: Path) -> dict[str, Any]:
    """Load one eval result JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    schema_version = int(payload.get("schema_version", 1))
    if schema_version not in SUPPORTED_EVAL_RESULT_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported eval result schema_version={schema_version} in {path.name}. "
            f"Supported versions: {sorted(SUPPORTED_EVAL_RESULT_SCHEMA_VERSIONS)}. "
            f"Current writer version: {CURRENT_EVAL_RESULT_SCHEMA_VERSION}."
        )
    payload["schema_version"] = schema_version
    return payload


def _render_delta(delta: float, use_colors: bool) -> str:
    """Render a signed delta with optional terminal colors."""
    sign = "+" if delta >= 0 else ""
    formatted = f"{sign}{delta:.4f}"
    if not use_colors:
        return formatted
    if delta > 0:
        return f"{GREEN}{formatted}{RESET}"
    if delta < 0:
        return f"{RED}{formatted}{RESET}"
    return formatted


def _flatten_scalar_leaves(
    value: Any,
    *,
    path: str = "",
) -> dict[str, ScalarValue]:
    """Flatten JSON-compatible containers into deterministic scalar paths."""
    if isinstance(value, dict):
        leaves: dict[str, ScalarValue] = {}
        for key in sorted(value):
            child_path = f"{path}.{key}" if path else str(key)
            leaves.update(_flatten_scalar_leaves(value[key], path=child_path))
        return leaves
    if isinstance(value, list):
        leaves = {}
        for index, item in enumerate(value):
            leaves.update(_flatten_scalar_leaves(item, path=f"{path}[{index}]"))
        return leaves
    if value is None or isinstance(value, (str, int, float, bool)):
        return {path or "<root>": value}
    raise TypeError(f"Unsupported config snapshot value at {path or '<root>'}: {type(value).__name__}")


def _config_leaf_differences(
    previous: dict[str, Any],
    latest: dict[str, Any],
) -> list[ConfigLeafDiff]:
    """Compare all scalar config leaves in stable path order."""
    previous_leaves = _flatten_scalar_leaves(previous)
    latest_leaves = _flatten_scalar_leaves(latest)
    differences: list[ConfigLeafDiff] = []
    for path in sorted(set(previous_leaves) | set(latest_leaves)):
        if path not in previous_leaves:
            differences.append(
                ConfigLeafDiff(
                    path=path,
                    status="added",
                    latest=latest_leaves[path],
                )
            )
        elif path not in latest_leaves:
            differences.append(
                ConfigLeafDiff(
                    path=path,
                    status="removed",
                    previous=previous_leaves[path],
                )
            )
        elif previous_leaves[path] != latest_leaves[path]:
            differences.append(
                ConfigLeafDiff(
                    path=path,
                    status="changed",
                    previous=previous_leaves[path],
                    latest=latest_leaves[path],
                )
            )
        else:
            differences.append(
                ConfigLeafDiff(
                    path=path,
                    status="unchanged",
                    previous=previous_leaves[path],
                    latest=latest_leaves[path],
                )
            )
    return differences


def _render_value(value: ScalarValue) -> str:
    """Render one JSON scalar consistently for comparison output."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _config_comparison_lines(
    previous: dict[str, Any],
    latest: dict[str, Any],
) -> list[str]:
    """Render recursive configuration leaf differences."""
    differences = _config_leaf_differences(previous, latest)
    if not differences:
        return ["No configuration scalar leaves found."]

    lines = ["Configuration leaves (A -> B):"]
    for difference in differences:
        if difference.status == "added":
            detail = f"n/a -> {_render_value(difference.latest)}"
        elif difference.status == "removed":
            detail = f"{_render_value(difference.previous)} -> n/a"
        else:
            detail = (
                f"{_render_value(difference.previous)} -> "
                f"{_render_value(difference.latest)}"
            )
        lines.append(f"  {difference.status:<9} {difference.path}: {detail}")
    return lines


def _sample_score_index(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Index finite numeric sample scores by the persisted SampleResult.id field."""
    indexed: dict[str, dict[str, float]] = {}
    samples = result.get("per_sample", [])
    if not isinstance(samples, list):
        return indexed
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            continue
        if sample_id in indexed:
            raise ComparisonInputError(
                f"Duplicate SampleResult.id in eval report: {sample_id}"
            )
        numeric_scores: dict[str, float] = {}
        scores = sample.get("scores", {})
        if isinstance(scores, dict):
            for metric, value in scores.items():
                if (
                    isinstance(metric, str)
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    numeric_scores[metric] = float(value)
        indexed[sample_id] = numeric_scores
    return indexed


def _paired_metric_deltas(
    previous: dict[str, Any],
    latest: dict[str, Any],
) -> list[PairedMetricDelta]:
    """Return stable per-sample deltas for metrics scored in both reports."""
    previous_scores = _sample_score_index(previous)
    latest_scores = _sample_score_index(latest)
    deltas: list[PairedMetricDelta] = []
    for sample_id in sorted(set(previous_scores) & set(latest_scores)):
        for metric in sorted(
            set(previous_scores[sample_id]) & set(latest_scores[sample_id])
        ):
            deltas.append(
                PairedMetricDelta(
                    metric=metric,
                    sample_id=sample_id,
                    previous=previous_scores[sample_id][metric],
                    latest=latest_scores[sample_id][metric],
                )
            )
    return sorted(deltas, key=lambda item: (item.metric, item.sample_id))


def _paired_comparison_lines(
    deltas: list[PairedMetricDelta],
    *,
    use_colors: bool,
) -> list[str]:
    """Render all mutually scored per-sample metric deltas."""
    if not deltas:
        return ["No metrics were scored for the same sample IDs in both reports."]
    lines = [
        "Paired sample metrics (A -> B):",
        "Metric                Sample ID                 A        B      Delta",
        "-----------------------------------------------------------------------",
    ]
    for item in deltas:
        lines.append(
            f"{item.metric:<20} {item.sample_id:<24} "
            f"{item.previous:>7.4f}  {item.latest:>7.4f}  "
            f"{_render_delta(item.delta, use_colors)}"
        )
    return lines


def _largest_regression_lines(
    deltas: list[PairedMetricDelta],
    *,
    limit: int,
    use_colors: bool,
) -> list[str]:
    """Render the largest negative paired deltas across all metrics."""
    regressions = sorted(
        (item for item in deltas if item.delta < 0),
        key=lambda item: (item.delta, item.metric, item.sample_id),
    )[:limit]
    if not regressions:
        return ["No paired metric regressions found."]
    lines = [f"Largest paired regressions (top {limit}):"]
    for item in regressions:
        lines.append(
            f"  {item.metric} / {item.sample_id}: "
            f"{_render_delta(item.delta, use_colors)}"
        )
    return lines


def _parse_thresholds(values: list[str]) -> dict[str, float]:
    """Parse repeated METRIC=VALUE thresholds with strict validation."""
    thresholds: dict[str, float] = {}
    for raw_value in values:
        metric, separator, threshold_text = raw_value.partition("=")
        metric = metric.strip()
        threshold_text = threshold_text.strip()
        if not separator or not metric or not threshold_text:
            raise ComparisonInputError(
                f"Invalid threshold {raw_value!r}; expected METRIC=VALUE"
            )
        if metric in thresholds:
            raise ComparisonInputError(f"Duplicate threshold for metric {metric!r}")
        try:
            threshold = float(threshold_text)
        except ValueError as exc:
            raise ComparisonInputError(
                f"Invalid threshold value for {metric!r}: {threshold_text!r}"
            ) from exc
        if not math.isfinite(threshold) or threshold < 0:
            raise ComparisonInputError(
                f"Threshold for {metric!r} must be a finite non-negative number"
            )
        thresholds[metric] = threshold
    return thresholds


def _normalized_filters(result: dict[str, Any]) -> dict[str, Any]:
    """Return order-insensitive active filters required for gate comparison."""
    summary = result.get("summary")
    filters = summary.get("filters") if isinstance(summary, dict) else None
    if not isinstance(filters, dict):
        raise ComparisonInputError("Both reports must contain summary.filters")
    normalized: dict[str, Any] = {}
    for key in sorted(filters):
        value = filters[key]
        normalized[key] = sorted(value) if isinstance(value, list) else value
    return normalized


def _dataset_content_hash(result: dict[str, Any]) -> str:
    """Return the required dataset content identity for a gate comparison."""
    summary = result.get("summary")
    dataset = summary.get("dataset") if isinstance(summary, dict) else None
    content_hash = dataset.get("content_hash") if isinstance(dataset, dict) else None
    if not isinstance(content_hash, str) or not content_hash:
        raise ComparisonInputError(
            "Both reports must contain summary.dataset.content_hash"
        )
    return content_hash


def _validate_comparable_runs(
    previous: dict[str, Any],
    latest: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Validate run, dataset, filter, and selected-sample identity."""
    for field in ("backend", "mode"):
        previous_value = previous.get(field)
        latest_value = latest.get(field)
        if not isinstance(previous_value, str) or not isinstance(latest_value, str):
            raise ComparisonInputError(f"Both reports must contain {field}")
        if previous_value != latest_value:
            raise ComparisonInputError(
                f"Reports have different {field}: {previous_value!r} != {latest_value!r}"
            )

    if _dataset_content_hash(previous) != _dataset_content_hash(latest):
        raise ComparisonInputError("Reports use different dataset content hashes")
    if _normalized_filters(previous) != _normalized_filters(latest):
        raise ComparisonInputError("Reports use different active filters")

    previous_scores = _sample_score_index(previous)
    latest_scores = _sample_score_index(latest)
    if set(previous_scores) != set(latest_scores):
        raise ComparisonInputError("Reports contain different selected sample IDs")
    return previous_scores, latest_scores


def _is_higher_better_metric(metric: str) -> bool:
    """Return whether a reviewed quality metric has a supported gate direction."""
    return metric in _HIGHER_IS_BETTER_METRICS or metric.startswith("precision_at_")


def _evaluate_regression_gates(
    previous: dict[str, Any],
    latest: dict[str, Any],
    thresholds: dict[str, float],
) -> list[RegressionGateResult]:
    """Validate comparable support and evaluate opt-in quality gates."""
    previous_scores, latest_scores = _validate_comparable_runs(previous, latest)
    results: list[RegressionGateResult] = []
    for metric, threshold in sorted(thresholds.items()):
        if not _is_higher_better_metric(metric):
            raise ComparisonInputError(
                f"Unsupported regression direction for metric {metric!r}"
            )
        previous_ids = {
            sample_id
            for sample_id, scores in previous_scores.items()
            if metric in scores
        }
        latest_ids = {
            sample_id
            for sample_id, scores in latest_scores.items()
            if metric in scores
        }
        if not previous_ids or previous_ids != latest_ids:
            raise ComparisonInputError(
                f"Metric {metric!r} must have the same non-empty scored sample-ID set"
            )
        previous_mean = sum(
            previous_scores[sample_id][metric] for sample_id in previous_ids
        ) / len(previous_ids)
        latest_mean = sum(
            latest_scores[sample_id][metric] for sample_id in latest_ids
        ) / len(latest_ids)
        results.append(
            RegressionGateResult(
                metric=metric,
                threshold=threshold,
                previous=previous_mean,
                latest=latest_mean,
            )
        )
    return results


def _regression_gate_lines(
    results: list[RegressionGateResult],
    *,
    use_colors: bool,
) -> list[str]:
    """Render validated regression-gate outcomes."""
    lines = ["Regression gates (higher is better):"]
    for result in results:
        status = "FAIL" if result.regressed else "PASS"
        lines.append(
            f"  {status} {result.metric}: delta="
            f"{_render_delta(result.delta, use_colors)} threshold={result.threshold:.4f}"
        )
    return lines


def _comparison_lines(latest: dict[str, Any], previous: dict[str, Any], use_colors: bool) -> list[str]:
    """Build human-readable metric comparison lines."""
    latest_metrics = latest.get("aggregate_metrics", {})
    previous_metrics = previous.get("aggregate_metrics", {})
    latest_coverage = latest.get("metric_coverage", {})
    previous_coverage = previous.get("metric_coverage", {})

    metric_names = sorted(
        set(latest_metrics)
        | set(previous_metrics)
        | set(latest_coverage)
        | set(previous_coverage)
    )
    if not metric_names:
        return ["No aggregate metrics found in result files."]

    lines = [
        "Metric                Previous   Latest     Delta      Support",
        "---------------------------------------------------------------",
    ]
    for metric in metric_names:
        previous_value = previous_metrics.get(metric)
        latest_value = latest_metrics.get(metric)
        if previous_value is None or latest_value is None:
            previous_text = "n/a" if previous_value is None else f"{float(previous_value):.4f}"
            latest_text = "n/a" if latest_value is None else f"{float(latest_value):.4f}"
            delta_text = "n/a"
        else:
            previous_float = float(previous_value)
            latest_float = float(latest_value)
            previous_text = f"{previous_float:.4f}"
            latest_text = f"{latest_float:.4f}"
            delta_text = _render_delta(latest_float - previous_float, use_colors)

        previous_support = _scored_support(previous_coverage, metric)
        latest_support = _scored_support(latest_coverage, metric)
        support_text = f"{previous_support}->{latest_support}"
        lines.append(
            f"{metric:<20} {previous_text:>8}   {latest_text:>8}   "
            f"{delta_text:<10} {support_text}"
        )

    return lines


def _scored_support(coverage: dict[str, Any], metric_name: str) -> str:
    """Render scored/total support for one metric, or ``n/a`` for legacy reports."""
    metric_coverage = coverage.get(metric_name)
    if not isinstance(metric_coverage, dict):
        return "n/a"
    scored = int(metric_coverage.get("scored", 0))
    total = sum(
        int(metric_coverage.get(status, 0))
        for status in ("scored", "unsupported", "skipped", "errored")
    )
    return f"{scored}/{total}"


def main() -> int:
    """Run eval result comparison CLI.

    Returns:
        Process exit code (0 on success).
    """
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Compare recent eval runs")
    parser.add_argument("--latest", type=int, default=2, help="Number of latest runs to load")
    parser.add_argument("--a", type=Path, help="Baseline eval result path")
    parser.add_argument("--b", type=Path, help="Candidate eval result path")
    parser.add_argument(
        "--largest-regressions",
        type=int,
        default=0,
        metavar="N",
        help="Also list the N largest negative paired sample deltas",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Return 1 when a configured quality-metric threshold is exceeded",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Repeatable non-negative quality regression threshold",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(str(cfg.eval.results_dir)),
        help="Directory containing eval JSON outputs",
    )
    args = parser.parse_args()

    if (args.a is None) != (args.b is None):
        parser.error("--a and --b must be supplied together")
    if args.latest <= 0:
        parser.error(f"--latest must be > 0, got {args.latest}")
    if args.largest_regressions < 0:
        parser.error(
            f"--largest-regressions must be >= 0, got {args.largest_regressions}"
        )
    if args.fail_on_regression and not args.threshold:
        parser.error("--fail-on-regression requires at least one --threshold")
    if args.threshold and not args.fail_on_regression:
        parser.error("--threshold requires --fail-on-regression")
    try:
        thresholds = _parse_thresholds(args.threshold)
    except ComparisonInputError as exc:
        parser.error(str(exc))

    files = (
        [args.b, args.a]
        if args.a is not None and args.b is not None
        else _sorted_result_files(args.results_dir)[: args.latest]
    )
    if not files:
        if args.fail_on_regression:
            logger.error("Regression gating requires two comparable eval reports")
            return 2
        logger.info("No eval result files found in %s", args.results_dir)
        return 0

    if len(files) == 1:
        if args.fail_on_regression:
            logger.error("Regression gating requires two comparable eval reports")
            return 2
        only = _load_result(files[0])
        logger.info(
            "Only one eval result found (%s). Aggregate metrics: %s",
            files[0].name,
            only.get("aggregate_metrics", {}),
        )
        return 0

    try:
        latest_result = _load_result(files[0])
        previous_result = _load_result(files[1])
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.error("Could not load eval reports: %s", exc)
        return 2

    logger.info("Comparing A=%s to B=%s", files[1], files[0])
    use_colors = sys.stdout.isatty()
    try:
        lines = _comparison_lines(
            latest=latest_result,
            previous=previous_result,
            use_colors=use_colors,
        )
        lines.extend([""])
        lines.extend(
            _config_comparison_lines(
                previous=previous_result.get("config_snapshot", {}),
                latest=latest_result.get("config_snapshot", {}),
            )
        )
        paired_deltas = _paired_metric_deltas(previous_result, latest_result)
    except (ComparisonInputError, TypeError, ValueError) as exc:
        logger.error("Could not compare eval reports: %s", exc)
        return 2
    lines.extend([""])
    lines.extend(_paired_comparison_lines(paired_deltas, use_colors=use_colors))
    if args.largest_regressions:
        lines.extend([""])
        lines.extend(
            _largest_regression_lines(
                paired_deltas,
                limit=args.largest_regressions,
                use_colors=use_colors,
            )
        )
    logger.info("\n%s", "\n".join(lines))
    if args.fail_on_regression:
        try:
            gate_results = _evaluate_regression_gates(
                previous_result,
                latest_result,
                thresholds,
            )
        except ComparisonInputError as exc:
            logger.error("Cannot apply regression gates: %s", exc)
            return 2
        logger.info(
            "\n%s",
            "\n".join(
                _regression_gate_lines(gate_results, use_colors=use_colors)
            ),
        )
        if any(result.regressed for result in gate_results):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
