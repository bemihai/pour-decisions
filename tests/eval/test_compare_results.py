"""Unit tests for eval result comparison helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.eval.scripts import compare_results
from src.eval.scripts.compare_results import _comparison_lines, _load_result


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
