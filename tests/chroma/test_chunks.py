"""Tests for the legacy-shaped provider-neutral ingestion entry point."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.chroma.chunking import ChunkCandidate
from src.chroma.chunks import ChunkMetadata, split_file


def _config() -> SimpleNamespace:
    """Build the config fields consumed by the compatibility wrapper."""
    return SimpleNamespace(
        chroma=SimpleNamespace(
            extraction=SimpleNamespace(
                pdf_provider="pdfplumber",
                epub_provider="ebooklib",
                fail_on_unsupported_file=False,
                strip_repeated_headers=True,
                strip_repeated_footers=True,
            ),
            chunking=SimpleNamespace(
                strategy="section_recursive",
                chunk_size=1024,
                chunk_overlap=256,
                min_chunk_chars=200,
                semantic=SimpleNamespace(
                    enabled=False,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=95.0,
                ),
            ),
        )
    )


def test_chunk_metadata_keeps_legacy_and_traceability_defaults() -> None:
    """Legacy metadata fields should remain readable before the mandatory reindex."""
    metadata = ChunkMetadata(
        filename="test.pdf",
        file_path="/books/test.pdf",
        file_type=".pdf",
        chunk_index=0,
        chunk_id="test_0_abc123",
        content_hash="abc123",
    )

    assert metadata.page_number == -1
    assert metadata.language == "unknown"
    assert metadata.grapes == ""
    assert metadata.extraction_provider == ""
    assert metadata.chunking_strategy == ""
    assert metadata.start_page == -1
    assert metadata.end_page == -1


@patch("src.chroma.chunks.assemble_chroma_chunks")
@patch("src.chroma.chunks.DocumentChunkingPipeline")
@patch("src.chroma.chunks.DocumentExtractionPipeline")
@patch("src.chroma.chunks.get_config")
def test_split_file_runs_new_pipelines_and_preserves_return_shape(
    mock_get_config: Mock,
    mock_extraction_pipeline: Mock,
    mock_chunking_pipeline: Mock,
    mock_assemble: Mock,
) -> None:
    """The compatibility wrapper should delegate to normalized orchestration."""
    mock_get_config.return_value = _config()
    elements = [Mock()]
    candidates = [
        ChunkCandidate(
            text="Burgundy produces Pinot Noir.",
            source_path="books/france.pdf",
            file_type="pdf",
            chunk_index=0,
        )
    ]
    expected = [{"id": "chunk", "text": candidates[0].text, "metadata": {}, "importance_score": 1.0}]
    mock_extraction_pipeline.return_value.extract.return_value = elements
    mock_chunking_pipeline.return_value.chunk.return_value = candidates
    mock_assemble.return_value = expected

    result = split_file(Path("books/france.pdf"), extract_metadata=False)

    assert result == expected
    mock_extraction_pipeline.assert_called_once_with(mock_get_config.return_value.chroma.extraction)
    mock_extraction_pipeline.return_value.extract.assert_called_once_with(Path("books/france.pdf"))
    mock_chunking_pipeline.return_value.chunk.assert_called_once_with(elements)
    mock_assemble.assert_called_once_with(candidates, extract_metadata=False)


@patch("src.chroma.chunks.get_config", return_value=_config())
def test_split_file_rejects_removed_legacy_strategy(_mock_get_config: Mock) -> None:
    """Removed strategy names should fail explicitly instead of changing behavior silently."""
    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        split_file(Path("books/france.pdf"), strategy="by_title")
