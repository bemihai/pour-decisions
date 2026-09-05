"""Unit tests for eval result comparison helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.eval.scripts import compare_results
from src.eval.scripts.compare_results import (
    _comparison_lines,
    _config_leaf_differences,
    _evaluate_regression_gates,
    _largest_regression_lines,
    _load_result,
    _paired_metric_deltas,
    _parse_thresholds,
)


def test_load_result_defaults_missing_schema_version_to_legacy_v1(tmp_path: Path) -> None:
    """Older result files without schema_version should be treated as schema version 1."""
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"run_id": "20260501T120000", "aggregate_metrics": {"mrr": 0.5}}), encoding="utf-8")

    payload = _load_result(path)

    assert payload["schema_version"] == 1


def test_load_result_rejects_unknown_schema_version(tmp_path: Path) -> None:
    """Unknown result schema versions should fail with a clear error."""
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema_version": 999, "run_id": "20260501T120000"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported eval result schema_version=999"):
        _load_result(path)


@pytest.mark.parametrize("schema_version", [1, 2, 3, 4, 5, 6, 7])
def test_load_result_accepts_all_supported_schema_versions(
    tmp_path: Path,
    schema_version: int,
) -> None:
    """Comparison tooling should read every historical schema plus current v7."""
    path = tmp_path / f"schema-{schema_version}.json"
    path.write_text(
        json.dumps({"schema_version": schema_version, "run_id": "compatible"}),
        encoding="utf-8",
    )

    assert _load_result(path)["schema_version"] == schema_version


def test_comparison_lines_show_metric_deltas() -> None:
    """Comparison output should include metrics from both runs and their deltas."""
    latest = {"aggregate_metrics": {"mrr": 0.8, "precision_at_3": 0.6}}
    previous = {"aggregate_metrics": {"mrr": 0.5, "precision_at_3": 0.4}}

    lines = _comparison_lines(latest=latest, previous=previous, use_colors=False)

    assert any("mrr" in line and "+0.3000" in line for line in lines)
    assert any("precision_at_3" in line and "+0.2000" in line for line in lines)


def test_comparison_lines_do_not_treat_missing_metrics_as_zero() -> None:
    """Unavailable metrics render as n/a and include scored support counts."""
    latest = {
        "aggregate_metrics": {"mrr": 0.8, "answer_correctness": 0.7},
        "metric_coverage": {
            "mrr": {"scored": 8, "unsupported": 2, "skipped": 0, "errored": 0},
            "answer_correctness": {
                "scored": 5,
                "unsupported": 5,
                "skipped": 0,
                "errored": 0,
            },
            "faithfulness": {
                "scored": 0,
                "unsupported": 10,
                "skipped": 0,
                "errored": 0,
            },
        },
    }
    previous = {
        "aggregate_metrics": {"mrr": 0.5},
        "metric_coverage": {
            "mrr": {"scored": 10, "unsupported": 0, "skipped": 0, "errored": 0}
        },
    }

    lines = _comparison_lines(latest=latest, previous=previous, use_colors=False)

    assert any("mrr" in line and "10/10->8/10" in line for line in lines)
    assert any(
        "answer_correctness" in line and "n/a" in line and "n/a->5/10" in line
        for line in lines
    )
    assert any(
        "faithfulness" in line and line.count("n/a") >= 3 and "n/a->0/10" in line
        for line in lines
    )


def test_main_handles_single_result_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The report command should exit cleanly before a second baseline exists."""
    result_path = tmp_path / "20260727T080000_retrieval_rag.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 6,
                "run_id": "20260727T080000",
                "aggregate_metrics": {"mrr": 0.75},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_results", "--results-dir", str(tmp_path), "--latest", "2"],
    )

    assert compare_results.main() == 0


def test_config_diff_reports_all_leaf_statuses_in_path_order() -> None:
    """Recursive config comparison should classify scalar leaves deterministically."""
    previous = {
        "execution": {
            "prompt_bundle_hash": "sha256:old",
            "models": {"generation": {"name": "same-model"}},
            "tools": {
                "contract_hash": "sha256:old-tools",
                "selected_names": ["first", "removed"],
            },
            "agent_policy": {"hash": "sha256:old-policy"},
        }
    }
    latest = {
        "execution": {
            "prompt_bundle_hash": "sha256:new",
            "models": {
                "generation": {"name": "same-model"},
                "planning": {"provider": "google"},
            },
            "tools": {
                "contract_hash": "sha256:new-tools",
                "selected_names": ["first"],
            },
            "agent_policy": {"hash": "sha256:new-policy"},
        }
    }

    differences = _config_leaf_differences(previous, latest)

    assert [item.path for item in differences] == sorted(item.path for item in differences)
    assert {(item.path, item.status) for item in differences} == {
        ("execution.agent_policy.hash", "changed"),
        ("execution.models.generation.name", "unchanged"),
        ("execution.models.planning.provider", "added"),
        ("execution.prompt_bundle_hash", "changed"),
        ("execution.tools.contract_hash", "changed"),
        ("execution.tools.selected_names[0]", "unchanged"),
        ("execution.tools.selected_names[1]", "removed"),
    }


def test_paired_deltas_use_sample_result_id_and_mutual_score_support() -> None:
    """Pairing should use id and include only metrics scored in both reports."""
    previous = {
        "per_sample": [
            {
                "id": "rag_only_001",
                "query_id": "obsolete-a",
                "scores": {"mrr": 0.8, "previous_only": 1.0},
            },
            {"id": "previous_only", "scores": {"mrr": 0.5}},
        ]
    }
    latest = {
        "per_sample": [
            {
                "id": "rag_only_001",
                "query_id": "obsolete-b",
                "scores": {"mrr": 0.6, "latest_only": 1.0},
            },
            {"id": "latest_only", "scores": {"mrr": 0.9}},
        ]
    }

    deltas = _paired_metric_deltas(previous, latest)

    assert len(deltas) == 1
    assert deltas[0].sample_id == "rag_only_001"
    assert deltas[0].metric == "mrr"
    assert deltas[0].delta == pytest.approx(-0.2)


def test_largest_regressions_are_sorted_by_most_negative_delta() -> None:
    """Optional regression output should present the largest drops first."""
    previous = {
        "per_sample": [
            {"id": "one", "scores": {"mrr": 0.9}},
            {"id": "two", "scores": {"mrr": 0.8}},
        ]
    }
    latest = {
        "per_sample": [
            {"id": "one", "scores": {"mrr": 0.4}},
            {"id": "two", "scores": {"mrr": 0.6}},
        ]
    }

    lines = _largest_regression_lines(
        _paired_metric_deltas(previous, latest),
        limit=1,
        use_colors=False,
    )

    assert "one" in lines[1]
    assert "-0.5000" in lines[1]


def test_main_accepts_explicit_report_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit A/B paths should bypass latest-file discovery and compare A to B."""
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(
        json.dumps({"schema_version": 7, "aggregate_metrics": {"mrr": 0.4}}),
        encoding="utf-8",
    )
    b_path.write_text(
        json.dumps({"schema_version": 7, "aggregate_metrics": {"mrr": 0.7}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_results", "--a", str(a_path), "--b", str(b_path)],
    )

    assert compare_results.main() == 0
    assert f"Comparing A={a_path} to B={b_path}" in caplog.text
    assert "+0.3000" in caplog.text


def test_main_requires_explicit_paths_as_a_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supplying only one explicit path should be a CLI usage error."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_results", "--a", str(tmp_path / "a.json")],
    )

    with pytest.raises(SystemExit) as exc_info:
        compare_results.main()

    assert exc_info.value.code == 2


def _gate_report(
    *,
    scores: dict[str, dict[str, float]],
    backend: str = "rag",
    mode: str = "full",
    dataset_hash: str = "sha256:dataset",
    filters: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a compact comparable report for regression-gate tests."""
    return {
        "schema_version": 7,
        "backend": backend,
        "mode": mode,
        "summary": {
            "dataset": {"content_hash": dataset_hash},
            "filters": filters
            or {
                "categories": ["rag_only"],
                "difficulties": None,
                "tags": None,
                "sample_ids": None,
            },
        },
        "per_sample": [
            {"id": sample_id, "scores": sample_scores}
            for sample_id, sample_scores in scores.items()
        ],
    }


def test_parse_thresholds_rejects_duplicates_and_invalid_values() -> None:
    """Gate thresholds should be explicit, unique, finite, and non-negative."""
    assert _parse_thresholds(["mrr=0.05", "faithfulness=0"]) == {
        "mrr": 0.05,
        "faithfulness": 0.0,
    }
    with pytest.raises(compare_results.ComparisonInputError, match="Duplicate"):
        _parse_thresholds(["mrr=0.1", "mrr=0.2"])
    with pytest.raises(compare_results.ComparisonInputError, match="finite non-negative"):
        _parse_thresholds(["mrr=-0.1"])


def test_regression_gate_returns_failure_only_beyond_threshold() -> None:
    """Quality metrics should regress only when B minus A is below minus threshold."""
    previous = _gate_report(
        scores={
            "one": {"mrr": 0.9},
            "two": {"mrr": 0.7},
        }
    )
    latest = _gate_report(
        scores={
            "one": {"mrr": 0.6},
            "two": {"mrr": 0.6},
        }
    )

    result = _evaluate_regression_gates(previous, latest, {"mrr": 0.1})[0]

    assert result.delta == pytest.approx(-0.2)
    assert result.regressed is True
    assert _evaluate_regression_gates(previous, latest, {"mrr": 0.2})[0].regressed is False


@pytest.mark.parametrize(
    ("latest_updates", "message"),
    [
        ({"backend": "agent"}, "different backend"),
        ({"mode": "retrieval"}, "different mode"),
        ({"summary": {"dataset": {"content_hash": "other"}, "filters": {}}}, "dataset content"),
    ],
)
def test_regression_gate_rejects_incomparable_run_identity(
    latest_updates: dict[str, object],
    message: str,
) -> None:
    """Backend, mode, dataset, and filter identity must be comparable."""
    previous = _gate_report(scores={"one": {"mrr": 0.8}})
    latest = _gate_report(scores={"one": {"mrr": 0.8}})
    latest.update(latest_updates)

    with pytest.raises(compare_results.ComparisonInputError, match=message):
        _evaluate_regression_gates(previous, latest, {"mrr": 0.1})


def test_regression_gate_rejects_sample_and_metric_support_mismatches() -> None:
    """Selected IDs and each gated metric's scored IDs must match exactly."""
    previous = _gate_report(
        scores={"one": {"mrr": 0.8}, "two": {"mrr": 0.6}}
    )
    different_samples = _gate_report(scores={"one": {"mrr": 0.8}})
    with pytest.raises(compare_results.ComparisonInputError, match="selected sample IDs"):
        _evaluate_regression_gates(previous, different_samples, {"mrr": 0.1})

    different_support = _gate_report(
        scores={"one": {"mrr": 0.8}, "two": {"faithfulness": 0.6}}
    )
    with pytest.raises(compare_results.ComparisonInputError, match="scored sample-ID set"):
        _evaluate_regression_gates(previous, different_support, {"mrr": 0.1})


def test_regression_gate_rejects_different_active_filters() -> None:
    """Equivalent data with different active filters should not be gate-comparable."""
    previous = _gate_report(
        scores={"one": {"mrr": 0.8}},
        filters={"categories": ["rag_only"], "sample_ids": None},
    )
    latest = _gate_report(
        scores={"one": {"mrr": 0.8}},
        filters={"categories": ["cellar"], "sample_ids": None},
    )

    with pytest.raises(compare_results.ComparisonInputError, match="different active filters"):
        _evaluate_regression_gates(previous, latest, {"mrr": 0.1})


def test_regression_gate_rejects_unknown_metric_direction() -> None:
    """Operational or unknown metrics should not be assigned a guessed direction."""
    previous = _gate_report(scores={"one": {"latency_ms": 100.0}})
    latest = _gate_report(scores={"one": {"latency_ms": 90.0}})

    with pytest.raises(compare_results.ComparisonInputError, match="Unsupported regression direction"):
        _evaluate_regression_gates(previous, latest, {"latency_ms": 5.0})


def test_main_regression_gate_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should return 1 for regressions and 0 when the configured gate passes."""
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(
        json.dumps(_gate_report(scores={"one": {"mrr": 0.8}})),
        encoding="utf-8",
    )
    b_path.write_text(
        json.dumps(_gate_report(scores={"one": {"mrr": 0.6}})),
        encoding="utf-8",
    )
    base_args = [
        "compare_results",
        "--a",
        str(a_path),
        "--b",
        str(b_path),
        "--fail-on-regression",
    ]

    monkeypatch.setattr(sys, "argv", [*base_args, "--threshold", "mrr=0.1"])
    assert compare_results.main() == 1

    monkeypatch.setattr(sys, "argv", [*base_args, "--threshold", "mrr=0.3"])
    assert compare_results.main() == 0


def test_main_gating_without_threshold_is_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in gating without an explicit metric threshold should return usage error 2."""
    monkeypatch.setattr(sys, "argv", ["compare_results", "--fail-on-regression"])

    with pytest.raises(SystemExit) as exc_info:
        compare_results.main()

    assert exc_info.value.code == 2


def test_main_returns_two_for_incomparable_gate_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gate request across different backends should return comparison error 2."""
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(
        json.dumps(_gate_report(scores={"one": {"mrr": 0.8}})),
        encoding="utf-8",
    )
    b_path.write_text(
        json.dumps(
            _gate_report(
                scores={"one": {"mrr": 0.8}},
                backend="agent",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_results",
            "--a",
            str(a_path),
            "--b",
            str(b_path),
            "--fail-on-regression",
            "--threshold",
            "mrr=0.1",
        ],
    )

    assert compare_results.main() == 2
