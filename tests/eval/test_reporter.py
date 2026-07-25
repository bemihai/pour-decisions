"""Unit tests for eval reporter (Phase 6)."""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.models import EvalRunResult, GoldenSample, SampleResult
from src.eval.reporter import EvalReporter


def _sample(sample_id: str, category: str, ground_truth_chunk_ids: list[str] | None = None) -> GoldenSample:
    """Build a minimal golden sample for reporter tests."""
    return GoldenSample(
        id=sample_id,
        question=f"Question for {sample_id}",
        category=category,
        difficulty="easy",
        expected_facts=["fact"],
        expected_tool_calls=[],
        ground_truth="Ground truth",
        ground_truth_chunk_ids=ground_truth_chunk_ids or [],
        tags=[category],
    )


def test_build_computes_aggregate_and_category_means() -> None:
    """build() computes mean metrics for overall and per-category views."""
    reporter = EvalReporter()
    samples = [
        _sample("rag_only_001", "rag_only", ["a"]),
        _sample("rag_only_002", "rag_only", ["b"]),
        _sample("pairing_001", "pairing"),
    ]
    results = [
        SampleResult(
            id="rag_only_001",
            question="Q1",
            answer="A1",
            contexts=["ctx"],
            latency_ms=100,
            scores={"mrr": 1.0, "precision_at_3": 0.67},
        ),
        SampleResult(
            id="rag_only_002",
            question="Q2",
            answer="A2",
            contexts=["ctx"],
            latency_ms=200,
            scores={"mrr": 0.5, "precision_at_3": 0.33},
        ),
        SampleResult(
            id="pairing_001",
            question="Q3",
            answer="A3",
            contexts=[],
            latency_ms=150,
            error="skipped: cellar DB is empty",
            scores={},
        ),
    ]

    run = reporter.build(
        results=results,
        samples=samples,
        mode="retrieval",
        backend="rag",
        config_snapshot={"model": "gemma3:4b"},
        git_sha="abc123",
    )

    assert isinstance(run, EvalRunResult)
    assert run.aggregate_metrics["mrr"] == 0.75
    assert run.aggregate_metrics["precision_at_3"] == 0.5
    assert run.metrics_by_category["rag_only"]["mrr"] == 0.75
    assert "pairing" not in run.metrics_by_category
    assert run.summary["total_samples"] == 3
    assert run.summary["evaluated"] == 2
    assert run.summary["skipped"] == 1
    assert run.summary["timeouts"] == 0
    assert run.summary["errors"] == 0
    assert run.summary["evaluation_target"] == "production_rag"
    assert run.summary["estimated_generation_llm_calls"] == 0
    assert run.summary["estimated_judge_llm_calls"] == 0
    assert run.summary["estimated_llm_calls"] == 0


def test_build_identifies_low_level_retriever_benchmark() -> None:
    """Retriever reports should be explicitly separated from production RAG runs."""
    sample = _sample("rag_only_001", "rag_only", ["chunk-1"])
    result = SampleResult(
        id=sample.id,
        question=sample.question,
        contexts=["raw context"],
        retrieved_chunk_ids=["chunk-1"],
        scores={"mrr": 1.0},
    )

    run = EvalReporter().build(
        results=[result],
        samples=[sample],
        mode="retrieval",
        backend="retriever",
        config_snapshot={},
    )

    assert run.backend == "retriever"
    assert run.summary["evaluation_target"] == "retriever_benchmark"
    assert run.summary["estimated_llm_calls"] == 0


def test_sample_result_infers_status_from_legacy_error_field() -> None:
    """Legacy payloads without status still infer the correct outcome state."""
    passed = SampleResult(id="rag_only_001", question="Q1", answer="A1")
    skipped = SampleResult(id="rag_only_002", question="Q2", error="skipped: cellar DB is empty")
    timeout = SampleResult(id="rag_only_003", question="Q3", error="timeout: sample exceeded 30.00s")
    failed = SampleResult(id="rag_only_004", question="Q4", error="retrieval failure")

    assert passed.status == "passed"
    assert skipped.status == "skipped"
    assert timeout.status == "timeout"
    assert failed.status == "failed"


def test_build_tracks_timeouts_separately_from_errors() -> None:
    """Timeout results are counted separately from ordinary execution errors."""
    reporter = EvalReporter()
    samples = [
        _sample("rag_only_001", "rag_only", ["a"]),
        _sample("rag_only_002", "rag_only", ["b"]),
        _sample("rag_only_003", "rag_only", ["c"]),
    ]
    results = [
        SampleResult(
            id="rag_only_001",
            question="Q1",
            answer="A1",
            contexts=["ctx"],
            latency_ms=100,
            scores={"mrr": 1.0},
        ),
        SampleResult(
            id="rag_only_002",
            question="Q2",
            latency_ms=10,
            error="timeout: sample exceeded 30.00s",
        ),
        SampleResult(
            id="rag_only_003",
            question="Q3",
            latency_ms=10,
            error="retrieval failure",
        ),
    ]

    run = reporter.build(
        results=results,
        samples=samples,
        mode="retrieval",
        backend="rag",
        config_snapshot={"model": "gemma3:4b"},
        git_sha="abc123",
    )

    assert run.summary["evaluated"] == 1
    assert run.summary["skipped"] == 0
    assert run.summary["timeouts"] == 1
    assert run.summary["errors"] == 1


def test_build_uses_explicit_status_over_error_parsing() -> None:
    """Reporter summary counters should be driven by explicit sample status."""
    reporter = EvalReporter()
    samples = [
        _sample("rag_only_001", "rag_only", ["a"]),
        _sample("rag_only_002", "rag_only", ["b"]),
        _sample("rag_only_003", "rag_only", ["c"]),
        _sample("rag_only_004", "rag_only", ["d"]),
    ]
    results = [
        SampleResult(id="rag_only_001", question="Q1", answer="A1", status="passed", contexts=["ctx"]),
        SampleResult(id="rag_only_002", question="Q2", status="skipped", error="custom skip reason"),
        SampleResult(id="rag_only_003", question="Q3", status="timeout", error="request budget exceeded"),
        SampleResult(id="rag_only_004", question="Q4", status="failed", error="retrieval failure"),
    ]

    run = reporter.build(
        results=results,
        samples=samples,
        mode="retrieval",
        backend="rag",
        config_snapshot={"model": "gemma3:4b"},
        git_sha="abc123",
    )

    assert run.summary["evaluated"] == 1
    assert run.summary["skipped"] == 1
    assert run.summary["timeouts"] == 1
    assert run.summary["errors"] == 1


def test_build_merges_run_metadata_into_summary() -> None:
    """Reporter should persist structured dataset and execution metadata."""
    reporter = EvalReporter()
    samples = [_sample("rag_only_001", "rag_only", ["a"])]
    results = [
        SampleResult(
            id="rag_only_001",
            question="Q1",
            answer="A1",
            contexts=["ctx"],
            latency_ms=100,
            scores={"mrr": 1.0},
        ),
    ]

    run = reporter.build(
        results=results,
        samples=samples,
        mode="retrieval",
        backend="rag",
        config_snapshot={"model": "gemma3:4b"},
        git_sha="abc123",
        run_metadata={
            "dataset": {
                "path": "/tmp/golden.jsonl",
                "content_hash": "abc123",
                "total_sample_count": 10,
                "selected_sample_count": 1,
            },
            "filters": {"categories": ["rag_only"], "difficulties": None, "tags": ["barolo"]},
            "execution": {"mode": "retrieval", "backend": "rag", "max_concurrency": 2},
            "git": {"sha": "abc123", "branch": "main", "is_dirty": True},
        },
    )

    assert run.summary["dataset"]["path"] == "/tmp/golden.jsonl"
    assert run.summary["dataset"]["content_hash"] == "abc123"
    assert run.summary["dataset"]["total_sample_count"] == 10
    assert run.summary["dataset"]["selected_sample_count"] == 1
    assert run.summary["filters"]["categories"] == ["rag_only"]
    assert run.summary["filters"]["tags"] == ["barolo"]
    assert run.summary["execution"]["max_concurrency"] == 2
    assert run.summary["git"]["branch"] == "main"
    assert run.summary["git"]["is_dirty"] is True


def test_save_writes_valid_json_file(tmp_path: Path) -> None:
    """save() creates a JSON file with the expected top-level keys."""
    reporter = EvalReporter()
    run = EvalRunResult(
        run_id="20260503T120000",
        timestamp="2026-05-03T12:00:00Z",
        mode="retrieval",
        backend="rag",
        git_sha="abc123",
        config_snapshot={"model": "gemma3:4b"},
        aggregate_metrics={"mrr": 0.5},
        metrics_by_category={"rag_only": {"mrr": 0.5}},
        per_sample=[SampleResult(id="rag_only_001", question="Q", answer="A")],
        summary={"total_samples": 1, "evaluated": 1, "skipped": 0, "timeouts": 0, "errors": 0},
    )

    output_path = reporter.save(run, output_dir=tmp_path)

    assert output_path.exists()
    assert output_path.name == "20260503T120000_retrieval_rag.json"

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["run_id"] == "20260503T120000"
    assert payload["aggregate_metrics"]["mrr"] == 0.5


def test_print_summary_does_not_raise() -> None:
    """print_summary() logs a compact table without raising exceptions."""
    reporter = EvalReporter()
    run = EvalRunResult(
        run_id="20260503T120000",
        timestamp="2026-05-03T12:00:00Z",
        mode="full",
        backend="rag",
        git_sha="abc123",
        config_snapshot={"model": "gemma3:4b"},
        aggregate_metrics={"faithfulness": 0.8, "mrr": 0.6},
        metrics_by_category={"rag_only": {"faithfulness": 0.8}},
        per_sample=[SampleResult(id="rag_only_001", question="Q", answer="A")],
        summary={"total_samples": 1, "evaluated": 1, "skipped": 0, "timeouts": 0, "errors": 0},
    )

    reporter.print_summary(run)
