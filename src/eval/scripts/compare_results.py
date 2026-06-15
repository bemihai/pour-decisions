"""Compare aggregate metrics across recent eval result files."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.eval.models import CURRENT_EVAL_RESULT_SCHEMA_VERSION, SUPPORTED_EVAL_RESULT_SCHEMA_VERSIONS
from src.utils import get_config, logger

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


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


def _comparison_lines(latest: dict[str, Any], previous: dict[str, Any], use_colors: bool) -> list[str]:
    """Build human-readable metric comparison lines."""
    latest_metrics = latest.get("aggregate_metrics", {})
    previous_metrics = previous.get("aggregate_metrics", {})

    metric_names = sorted(set(latest_metrics) | set(previous_metrics))
    if not metric_names:
        return ["No aggregate metrics found in result files."]

    lines = [
        "Metric                Previous   Latest     Delta",
        "------------------------------------------------",
    ]
    for metric in metric_names:
        prev = float(previous_metrics.get(metric, 0.0))
        curr = float(latest_metrics.get(metric, 0.0))
        delta = curr - prev
        lines.append(f"{metric:<20} {prev:>8.4f}   {curr:>8.4f}   {_render_delta(delta, use_colors)}")

    return lines


def main() -> int:
    """Run eval result comparison CLI.

    Returns:
        Process exit code (0 on success).
    """
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Compare recent eval runs")
    parser.add_argument("--latest", type=int, default=2, help="Number of latest runs to load")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(str(cfg.eval.results_dir)),
        help="Directory containing eval JSON outputs",
    )
    args = parser.parse_args()

    if args.latest <= 0:
        raise ValueError(f"--latest must be > 0, got {args.latest}")

    files = _sorted_result_files(args.results_dir)[: args.latest]
    if not files:
        logger.info("No eval result files found in %s", args.results_dir)
        return 0

    if len(files) == 1:
        only = _load_result(files[0])
        logger.info(
            "Only one eval result found (%s). Aggregate metrics: %s",
            files[0].name,
            only.get("aggregate_metrics", {}),
        )
        return 0

    latest_result = _load_result(files[0])
    previous_result = _load_result(files[1])

    logger.info("Comparing %s (latest) vs %s", files[0].name, files[1].name)
    lines = _comparison_lines(
        latest=latest_result,
        previous=previous_result,
        use_colors=sys.stdout.isatty(),
    )
    logger.info("\n%s", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
