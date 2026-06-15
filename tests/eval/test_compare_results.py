"""Unit tests for eval result comparison helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
