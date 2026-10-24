"""Estimate, capture, and assess the bounded food-pairing prompt comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.food_pairing_grounding import (
    DEFAULT_MANIFEST_PATH,
    DecisionFile,
    assess_comparison,
    capture_comparison,
    estimate_comparison,
    load_manifest,
    read_json_object,
)
from src.utils import logger


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded study command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("estimate", help="Report the run bound without provider calls")
    capture = subparsers.add_parser("capture", help="Capture baseline and candidate arms")
    capture.add_argument("--output", required=True, type=Path)
    assess = subparsers.add_parser("assess", help="Apply the manual claim-level audit")
    assess.add_argument("--capture", required=True, type=Path)
    assess.add_argument("--decisions", required=True, type=Path)
    assess.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    """Run one local study command."""
    args = build_parser().parse_args()
    manifest = load_manifest(args.manifest)
    if args.command == "estimate":
        logger.info("Food-pairing comparison estimate: %s", json.dumps(estimate_comparison(manifest)))
        return 0
    if args.command == "capture":
        artifact = capture_comparison(args.output, args.manifest)
        logger.info("Captured %d food-pairing executions in %s", len(artifact["executions"]), args.output)
        return 0

    decisions = DecisionFile.model_validate(read_json_object(args.decisions))
    report = assess_comparison(read_json_object(args.capture), decisions, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Food-pairing comparison passed=%s report=%s", report["passed"], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
