"""Compatibility entry point for provider-neutral document ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils import get_config, logger

from .ingestion_pipeline import (
    ChunkMetadata,
    DocumentChunkingPipeline,
    DocumentExtractionPipeline,
    assemble_chroma_chunks,
)

__all__ = ["ChunkMetadata", "split_file"]


def split_file(
    filepath: str | Path,
    strategy: str | None = None,
    chunk_size: int | None = None,
    overlap_size: int | None = None,
    embedding_model: str | None = None,
    extract_metadata: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run provider-neutral ingestion while preserving the legacy return shape.

    Only the reviewed ``section_recursive`` and ``section_semantic`` strategy
    names are accepted. The legacy ``basic``, ``by_title``, and ``semantic``
    names depended on ``unstructured`` and are intentionally unsupported.
    """
    config = get_config().chroma
    configured_strategy = strategy or str(config.chunking.strategy)
    chunking_config = {
        "strategy": configured_strategy,
        "chunk_size": chunk_size if chunk_size is not None else int(config.chunking.chunk_size),
        "chunk_overlap": overlap_size if overlap_size is not None else int(config.chunking.chunk_overlap),
        "min_chunk_chars": int(getattr(config.chunking, "min_chunk_chars", 200)),
        "semantic": {
            "enabled": bool(getattr(config.chunking.semantic, "enabled", False)),
            "embedding_model": embedding_model,
            "breakpoint_threshold_type": kwargs.get(
                "breakpoint_threshold_type",
                str(config.chunking.semantic.breakpoint_threshold_type),
            ),
            "breakpoint_threshold_amount": kwargs.get(
                "breakpoint_threshold_amount",
                float(config.chunking.semantic.breakpoint_threshold_amount),
            ),
        },
    }

    source_path = Path(filepath)
    logger.info("Processing file: %s", source_path.name)
    extraction_pipeline = DocumentExtractionPipeline(config.extraction)
    chunking_pipeline = DocumentChunkingPipeline(chunking_config)
    candidates = chunking_pipeline.chunk(extraction_pipeline.extract(source_path))
    chunks = assemble_chroma_chunks(candidates, extract_metadata=extract_metadata)
    logger.info("Generated %d chunks from %s", len(chunks), source_path.name)
    return chunks
