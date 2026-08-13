"""Tests for Phase 1 chunk-quality calibration diagnostics."""

from src.chroma.chunk_filter import ChunkQualityFilter
from src.eval.scripts.chunk_quality_calibration import build_calibration_report


def _chunk(chunk_id: str, text: str, **metadata: object) -> dict[str, object]:
    """Build a minimal loader-shaped chunk fixture."""
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {"file_path": "books/test.pdf", **metadata},
    }


def test_report_aggregates_quality_decisions_deterministically() -> None:
    """Counts and keys should be stable for downstream artifact comparison."""
    prose = " ".join(["Nebbiolo offers roses, cherries, acidity, and firm tannins."] * 15)
    chunks = [
        _chunk("prose", prose, section="Nebbiolo"),
        _chunk("toc", "Chapter One ........ 12\nChapter Two ........ 34", section="Contents"),
        _chunk("fragment", "THE BEST WINES"),
    ]

    report = build_calibration_report(chunks, ChunkQualityFilter(mode="enforce", min_score=0.4))

    assert report["candidate_count"] == 3
    assert report["retained_count"] == 1
    assert report["rejected_count"] == 2
    assert report["retained_rate"] == 1 / 3
    assert report["rejected_rate"] == 2 / 3
    assert list(report["structural_role_counts"]) == sorted(report["structural_role_counts"])
    assert list(report["rejection_reason_counts"]) == sorted(report["rejection_reason_counts"])
    assert list(report["quality_score_buckets"]) == [
        "0.00-0.19",
        "0.20-0.39",
        "0.40-0.59",
        "0.60-0.79",
        "0.80-1.00",
    ]


def test_report_bounds_samples_per_decision_and_role() -> None:
    """Large corpora should not produce unbounded copied text in the report."""
    chunks = [
        _chunk(f"toc-{index}", f"Chapter {index} ........ {index + 10}", section="Contents")
        for index in range(4)
    ]

    report = build_calibration_report(
        chunks,
        ChunkQualityFilter(mode="enforce", min_score=0.4),
        sample_limit=2,
    )

    samples = report["samples"]["rejected"]["toc"]
    assert len(samples) == 2
    assert [sample["chunk_id"] for sample in samples] == ["toc-0", "toc-1"]
    assert all(len(sample["text_preview"]) <= 240 for sample in samples)


def test_zero_sample_limit_keeps_aggregate_evidence() -> None:
    """Sampling can be disabled without disabling aggregate calibration."""
    report = build_calibration_report(
        [_chunk("fragment", "THE BEST WINES")],
        ChunkQualityFilter(mode="audit", min_score=0.4),
        sample_limit=0,
    )

    assert report["candidate_count"] == 1
    assert report["rejected_count"] == 1
    assert report["samples"] == {"retained": {}, "rejected": {}}
