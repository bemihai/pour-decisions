"""Generate deterministic Phase 1 chunk-quality calibration diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from src.chroma.chunk_filter import ChunkQualityAssessment, ChunkQualityFilter
from src.chroma.ingestion_pipeline import DocumentChunkingPipeline, DocumentExtractionPipeline, assemble_chroma_chunks
from src.utils import get_config, logger
from src.utils.env import load_env


DEFAULT_SAMPLE_LIMIT = 5
SUPPORTED_EXTENSIONS = {".epub", ".pdf"}
SCORE_BUCKETS = (
    ("0.00-0.19", 0.0, 0.2),
    ("0.20-0.39", 0.2, 0.4),
    ("0.40-0.59", 0.4, 0.6),
    ("0.60-0.79", 0.6, 0.8),
    ("0.80-1.00", 0.8, 1.01),
)


@dataclass(frozen=True)
class CalibrationSample:
    """Bounded evidence sample for one quality decision."""

    chunk_id: str
    source: str
    structural_role: str
    quality_score: float
    rejection_reasons: tuple[str, ...]
    text_preview: str


def build_calibration_report(
    chunks: Iterable[Mapping[str, Any]],
    quality_filter: ChunkQualityFilter,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> dict[str, Any]:
    """Aggregate deterministic quality decisions for raw ingestion candidates.

    Args:
        chunks: Loader-shaped chunks produced before quality enforcement.
        quality_filter: Authoritative filter used by production indexing.
        sample_limit: Maximum retained and rejected examples per structural role.

    Returns:
        Machine-readable calibration counts and bounded representative samples.
    """
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")

    role_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    score_bucket_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    samples: dict[str, dict[str, list[CalibrationSample]]] = {
        "retained": defaultdict(list),
        "rejected": defaultdict(list),
    }

    for chunk in chunks:
        metadata = dict(chunk.get("metadata") or {})
        assessment = quality_filter.assess(str(chunk.get("text") or ""), metadata)
        decision = "rejected" if assessment.should_reject else "retained"
        decision_counts[decision] += 1
        role_counts[assessment.structural_role] += 1
        reason_counts.update(assessment.rejection_reasons)
        score_bucket_counts[_score_bucket(assessment.quality_score)] += 1

        if sample_limit:
            role_samples = samples[decision][assessment.structural_role]
            if len(role_samples) < sample_limit:
                role_samples.append(_sample_from_chunk(chunk, assessment))

    total = sum(decision_counts.values())
    rejected = decision_counts["rejected"]
    retained = decision_counts["retained"]
    return {
        "candidate_count": total,
        "retained_count": retained,
        "rejected_count": rejected,
        "retained_rate": _rate(retained, total),
        "rejected_rate": _rate(rejected, total),
        "structural_role_counts": dict(sorted(role_counts.items())),
        "quality_score_buckets": {
            name: score_bucket_counts[name]
            for name, _, _ in SCORE_BUCKETS
        },
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "samples": {
            decision: {
                role: [asdict(sample) for sample in role_samples]
                for role, role_samples in sorted(samples[decision].items())
            }
            for decision in ("retained", "rejected")
        },
    }


def calibrate_directory(
    data_path: Path,
    *,
    extraction_config: Mapping[str, Any] | Any,
    chunking_config: Mapping[str, Any] | Any,
    indexing_config: Mapping[str, Any] | Any,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> dict[str, Any]:
    """Assess every supported source file without mutating either index."""
    source_path = Path(data_path)
    if not source_path.is_dir():
        raise ValueError(f"Data directory {source_path} does not exist or is not a directory")

    extraction_pipeline = DocumentExtractionPipeline(extraction_config)
    chunking_pipeline = DocumentChunkingPipeline(chunking_config)
    quality_filter = ChunkQualityFilter.from_config(indexing_config)
    source_files = sorted(
        path
        for path in source_path.rglob("*")
        if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
    )
    if not source_files:
        raise ValueError(f"No supported PDF or EPUB files found in {source_path}")

    source_candidate_counts: dict[str, int] = {}

    def iter_source_chunks() -> Iterable[dict[str, Any]]:
        """Yield one source at a time so corpus replay has bounded memory."""
        for file_path in source_files:
            logger.info("Calibrating chunk quality for %s", file_path.name)
            elements = extraction_pipeline.extract(file_path)
            candidates = chunking_pipeline.chunk(elements)
            source_chunks = assemble_chroma_chunks(candidates, extract_metadata=False)
            source_candidate_counts[str(file_path)] = len(source_chunks)
            yield from source_chunks

    report = build_calibration_report(iter_source_chunks(), quality_filter, sample_limit=sample_limit)
    report.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "data_path": str(source_path),
            "source_count": len(source_files),
            "source_candidate_counts": dict(sorted(source_candidate_counts.items())),
            "quality_filter": {
                "mode": quality_filter.mode,
                "min_score": quality_filter.min_score,
            },
        }
    )
    return report


def _sample_from_chunk(
    chunk: Mapping[str, Any],
    assessment: ChunkQualityAssessment,
) -> CalibrationSample:
    """Build a compact sample without copying a full copyrighted passage."""
    metadata = dict(chunk.get("metadata") or {})
    preview = " ".join(str(chunk.get("text") or "").split())[:240]
    return CalibrationSample(
        chunk_id=str(chunk.get("id") or metadata.get("chunk_id") or ""),
        source=str(metadata.get("file_path") or metadata.get("filename") or ""),
        structural_role=assessment.structural_role,
        quality_score=assessment.quality_score,
        rejection_reasons=assessment.rejection_reasons,
        text_preview=preview,
    )


def _score_bucket(score: float) -> str:
    """Return the stable reporting bucket for a normalized score."""
    for name, lower, upper in SCORE_BUCKETS:
        if lower <= score < upper:
            return name
    raise ValueError(f"Quality score outside normalized range: {score}")


def _rate(count: int, total: int) -> float:
    """Return a zero-safe fraction."""
    return count / total if total else 0.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate Phase 1 chunk-quality calibration diagnostics")
    parser.add_argument("--data-path", type=Path, default=None, help="Override the configured corpus directory")
    parser.add_argument("--output", type=Path, required=True, help="Write the JSON report to this path")
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    return parser.parse_args()


def main() -> int:
    """Generate and persist one calibration report."""
    args = parse_args()
    load_env()
    config = get_config().chroma
    collection = config.collections[0]
    report = calibrate_directory(
        args.data_path or Path(collection.local_data_path),
        extraction_config=config.extraction,
        chunking_config=config.chunking,
        indexing_config=config.indexing,
        sample_limit=args.sample_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info("Chunk-quality calibration artifact: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
