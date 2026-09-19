"""Capture and validate the frozen M10 Phase 0 planning baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.eval.planning_baseline import (
    DEFAULT_MANIFEST_PATH,
    apply_adjudications,
    assert_comparable_artifacts,
    build_planning_baseline_agent,
    capture_planning_baseline,
    estimate_run_cost,
    load_planning_cohort,
    validate_gate0_artifact,
)
from src.utils import logger


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded planning-baseline command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("estimate", help="Report logical call bounds without provider calls")

    capture = subparsers.add_parser("capture", help="Run the frozen cohort sequentially")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--repetitions", type=int, default=None)

    adjudicate = subparsers.add_parser("adjudicate", help="Apply explicit answer decisions")
    adjudicate.add_argument("--artifact", type=Path, required=True)
    adjudicate.add_argument("--decisions", type=Path, required=True)
    adjudicate.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="Validate Gate 0 completeness")
    validate.add_argument("--artifact", type=Path, required=True)
    validate.add_argument("--report", type=Path, required=True)

    compare = subparsers.add_parser("compare", help="Validate paired artifact identity")
    compare.add_argument("--a", type=Path, required=True)
    compare.add_argument("--b", type=Path, required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    """Read one object-shaped JSON file."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one stable local JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    """Execute the selected bounded Phase 0 operation."""
    parser = build_parser()
    args = parser.parse_args()
    manifest, _ = load_planning_cohort(args.manifest)

    if args.command == "estimate":
        estimate = estimate_run_cost(manifest, max_calls_per_sample=5)
        logger.info("M10 Phase 0 cost bound: %s", json.dumps(estimate, sort_keys=True))
        return 0

    if args.command == "capture":
        agent = build_planning_baseline_agent()
        artifact = capture_planning_baseline(
            output_path=args.output,
            manifest_path=args.manifest,
            repetitions=args.repetitions,
            agent=agent,
        )
        logger.info("Captured %d planning executions in %s", len(artifact["executions"]), args.output)
        return 0

    if args.command == "adjudicate":
        artifact = _read_json(args.artifact)
        decisions = _read_json(args.decisions)
        _write_json(args.output, apply_adjudications(artifact, decisions))
        logger.info("Applied explicit planning adjudications to %s", args.output)
        return 0

    if args.command == "validate":
        report = validate_gate0_artifact(_read_json(args.artifact), manifest)
        _write_json(args.report, report)
        logger.info("Gate 0 validation passed: %s", args.report)
        return 0

    assert_comparable_artifacts(_read_json(args.a), _read_json(args.b))
    logger.info("Planning artifacts are comparable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
