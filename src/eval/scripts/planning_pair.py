"""Capture and assess one private, same-commit M10 Phase 2 prompt comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.eval.planning_baseline import (
    DEFAULT_MANIFEST_PATH,
    apply_adjudications,
    build_planning_baseline_agent,
    capture_planning_baseline,
    estimate_run_cost,
    load_planning_cohort,
)
from src.eval.planning_pair import assess_gate2, assert_paired_artifacts, load_prior_prompt_registry
from src.eval.utils import get_git_metadata
from src.utils import logger


def _read_object(path: Path) -> dict[str, Any]:
    """Read one local JSON object without emitting private details to logs."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded paired-capture command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("estimate", help="Report the two-arm logical-call bound without provider calls")

    capture = subparsers.add_parser("capture", help="Capture both arms using current code and a pinned prior prompt")
    capture.add_argument("--baseline-ref", required=True)
    capture.add_argument("--baseline-output", required=True, type=Path)
    capture.add_argument("--treatment-output", required=True, type=Path)

    assess = subparsers.add_parser("assess", help="Validate paired artifacts and explicit answer decisions")
    assess.add_argument("--baseline", required=True, type=Path)
    assess.add_argument("--treatment", required=True, type=Path)
    assess.add_argument("--baseline-decisions", required=True, type=Path)
    assess.add_argument("--treatment-decisions", required=True, type=Path)
    assess.add_argument("--report", required=True, type=Path)
    return parser


def main() -> int:
    """Run a local preflight, the approved external cohort, or local gate assessment."""
    args = build_parser().parse_args()
    manifest, _ = load_planning_cohort(args.manifest)

    if args.command == "estimate":
        estimate = estimate_run_cost(manifest, max_calls_per_sample=5)
        estimate["paired_execution_count"] = 2 * estimate["execution_count"]
        estimate["paired_maximum_logical_model_calls"] = 2 * estimate["maximum_logical_model_calls"]
        logger.info("M10 Phase 2 paired logical-call bound: %s", json.dumps(estimate, sort_keys=True))
        return 0

    if args.command == "capture":
        if args.baseline_output == args.treatment_output:
            raise ValueError("Baseline and treatment output paths must differ")
        if get_git_metadata()["is_dirty"] is not False:
            raise ValueError("Commit Phase 2 code before the paired capture")
        baseline_registry, baseline_commit = load_prior_prompt_registry(args.baseline_ref)
        baseline_agent = build_planning_baseline_agent(prompt_registry=baseline_registry)
        treatment_agent = build_planning_baseline_agent()
        if baseline_agent.execution_provenance.tools != treatment_agent.execution_provenance.tools:
            raise ValueError("Baseline and treatment selected-tool snapshots differ")
        logger.info("M10 Phase 2 baseline prompt ref: %s", baseline_commit)
        baseline = capture_planning_baseline(
            args.baseline_output,
            manifest_path=args.manifest,
            agent=baseline_agent,
            phase=2,
            arm="baseline",
        )
        treatment = capture_planning_baseline(
            args.treatment_output,
            manifest_path=args.manifest,
            agent=treatment_agent,
            phase=2,
            arm="treatment",
        )
        assert_paired_artifacts(baseline, treatment, manifest)
        logger.info("Captured %d comparable paired planning executions", len(baseline["executions"]) * 2)
        return 0

    baseline = _read_object(args.baseline)
    treatment = _read_object(args.treatment)
    baseline_decisions = _read_object(args.baseline_decisions)
    treatment_decisions = _read_object(args.treatment_decisions)
    report = assess_gate2(
        apply_adjudications(baseline, baseline_decisions),
        apply_adjudications(treatment, treatment_decisions),
        manifest,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("M10 Phase 2 Gate 2 assessment: passed=%s report=%s", report["passed"], args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
