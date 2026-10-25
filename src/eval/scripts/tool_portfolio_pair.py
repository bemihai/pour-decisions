"""Run the bounded full-versus-core tool portfolio experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.cloud_call_meter import CloudCallMeter
from src.eval.tool_portfolio import (
    DEFAULT_PORTFOLIO_COHORT_PATH,
    build_portfolio_agents,
    capture_tool_portfolio_pair,
    estimate_tool_portfolio_pair,
    load_tool_portfolio_cohort,
)
from src.utils import logger


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded portfolio experiment command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_PORTFOLIO_COHORT_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("estimate", help="Report bounds without provider calls")
    capture = subparsers.add_parser("capture", help="Run both arms with local stub tools")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--sdk-meter", type=Path, required=True)
    capture.add_argument("--sdk-call-cap", type=int, required=True)
    return parser


def main() -> int:
    """Estimate or execute the paired portfolio experiment."""
    args = build_parser().parse_args()
    cohort = load_tool_portfolio_cohort(args.cohort)
    if args.command == "estimate":
        logger.info(
            "Tool portfolio pair estimate: %s",
            json.dumps(estimate_tool_portfolio_pair(cohort), sort_keys=True),
        )
        return 0

    full_agent, lean_agent = build_portfolio_agents()
    meter = CloudCallMeter(args.sdk_meter, args.sdk_call_cap)
    with meter.count_sdk_chat():
        artifact = capture_tool_portfolio_pair(cohort, full_agent, lean_agent)
    artifact["sdk_call_meter"] = meter.snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Captured portfolio pair: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
