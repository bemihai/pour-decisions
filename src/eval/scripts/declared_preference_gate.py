"""Validate the frozen M13B declared-preference product-value cohort.

This Gate 0 check is deterministic and local. It verifies the released baseline,
cohort coverage, and numerical decision contract without executing an agent,
calling a model, or changing runtime behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_COHORT_PATH = Path("src/eval/m13b_gate0_cohort.json")
REQUIRED_DIMENSIONS = {"grape", "region", "producer", "wine_style", "price_ceiling"}
REQUIRED_LIFECYCLE_OPERATIONS = {"delete", "reset", "cross_thread"}
REQUIRED_CONTROLS = {"observed_only", "empty_profile"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT_PATH)
    return parser.parse_args()


def load_cohort(path: Path = DEFAULT_COHORT_PATH) -> dict[str, Any]:
    """Load and validate the proposed or approved Gate 0 cohort."""
    cohort = json.loads(path.read_text(encoding="utf-8"))
    validate_cohort(cohort)
    return cohort


def validate_cohort(cohort: dict[str, Any]) -> None:
    """Validate one in-memory Gate 0 cohort contract."""
    if cohort.get("schema_version") != 1 or cohort.get("milestone") != "m13b" or cohort.get("gate") != 0:
        raise ValueError("Expected the M13B Gate 0 schema")

    samples = cohort.get("samples", [])
    sample_ids = [sample.get("sample_id") for sample in samples]
    if not samples or None in sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Gate 0 sample IDs must be present and unique")

    target_dimensions = {
        sample.get("dimension") for sample in samples if sample.get("cohort_role") == "target"
    }
    retained_dimensions = set(cohort.get("retained_dimensions", []))
    if retained_dimensions != REQUIRED_DIMENSIONS or target_dimensions != retained_dimensions:
        raise ValueError("Every retained preference dimension needs exactly represented product value")

    lifecycle_operations = {
        sample.get("operation") for sample in samples if sample.get("cohort_role") == "lifecycle"
    }
    if lifecycle_operations != REQUIRED_LIFECYCLE_OPERATIONS:
        raise ValueError("Gate 0 must cover delete, reset, and cross-thread persistence")

    controls = {sample.get("control_kind") for sample in samples if sample.get("cohort_role") == "control"}
    if controls != REQUIRED_CONTROLS:
        raise ValueError("Gate 0 must include observed-only and empty-profile controls")

    for sample in samples:
        for field in ("question", "expected_tool", "expected_visible_effect", "required_evidence", "prohibited_facts"):
            if not sample.get(field):
                raise ValueError(f"{sample['sample_id']} is missing {field}")

    for contract_name in ("promotion_thresholds", "rollback_thresholds"):
        contract = cohort.get(contract_name, {})
        invalid_values = any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in contract.values()
        )
        if not contract or invalid_values:
            raise ValueError(f"{contract_name} must contain only numerical thresholds")


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_gate(cohort: dict[str, Any]) -> dict[str, Any]:
    """Evaluate all evidence that can be decided before user approval."""
    samples = cohort["samples"]
    targets = [sample for sample in samples if sample["cohort_role"] == "target"]
    lifecycle = [sample for sample in samples if sample["cohort_role"] == "lifecycle"]
    controls = [sample for sample in samples if sample["cohort_role"] == "control"]
    source_matches = {
        source: Path(source).is_file() and _sha256(Path(source)) == expected
        for source, expected in cohort["baseline"]["source_sha256"].items()
    }
    baseline_tools = cohort["baseline"]["tools"]
    checks = {
        "baseline_sources_match": all(source_matches.values()),
        "released_baseline_has_no_declared_preference_storage": not cohort["baseline"][
            "declared_preference_storage_supported"
        ],
        "released_taste_tools_do_not_support_declared_preferences": all(
            not tool["declared_preferences_supported"] for tool in baseline_tools
        ),
        "all_retained_dimensions_have_visible_targets": len(targets) == len(REQUIRED_DIMENSIONS),
        "lifecycle_coverage_complete": len(lifecycle) == len(REQUIRED_LIFECYCLE_OPERATIONS),
        "control_coverage_complete": len(controls) == len(REQUIRED_CONTROLS),
        "external_calls_are_zero": cohort["baseline"]["external_calls"] == 0,
        "numerical_contract_present": bool(cohort["promotion_thresholds"] and cohort["rollback_thresholds"]),
    }
    evidence_ready = all(checks.values())
    approved = cohort["approval"]["status"] == "approved"
    if evidence_ready and approved:
        decision = "pass"
    elif evidence_ready:
        decision = "ready_for_approval"
    else:
        decision = "defer"
    return {
        "decision": decision,
        "approval_status": cohort["approval"]["status"],
        "checks": checks,
        "counts": {
            "samples": len(samples),
            "targets": len(targets),
            "lifecycle": len(lifecycle),
            "controls": len(controls),
            "retained_dimensions": len(cohort["retained_dimensions"]),
        },
        "source_matches": source_matches,
        "external_calls": 0,
    }


def main() -> int:
    """Print the deterministic Gate 0 readiness result."""
    args = parse_args()
    cohort = load_cohort(args.cohort)
    result = evaluate_gate(cohort)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] != "defer" else 1


if __name__ == "__main__":
    raise SystemExit(main())
