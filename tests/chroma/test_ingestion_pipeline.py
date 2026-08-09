"""Integration tests for provider-neutral extraction and chunk assembly."""

from pathlib import Path

from ebooklib import epub
import pytest

from src.chroma.chunking import ChunkCandidate, ChunkerRegistry, SectionRecursiveChunker
from src.chroma.extraction import UnsupportedDocumentTypeError
from src.chroma.ingestion_pipeline import (
    DocumentChunkingPipeline,
    DocumentExtractionPipeline,
    assemble_chroma_chunks,
)


def _extraction_config(*, fail_on_unsupported_file: bool = False) -> dict[str, object]:
    return {
        "pdf_provider": "pdfplumber",
        "epub_provider": "ebooklib",
        "fail_on_unsupported_file": fail_on_unsupported_file,
        "strip_repeated_headers": True,
        "strip_repeated_footers": True,
    }


def _chunking_config() -> dict[str, object]:
    return {
        "strategy": "section_recursive",
        "chunk_size": 256,
        "chunk_overlap": 32,
        "min_chunk_chars": 0,
        "semantic": {
            "enabled": False,
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": 95.0,
        },
    }


@pytest.fixture
def ingestion_epub(tmp_path: Path) -> Path:
    """Create a small structured EPUB for end-to-end ingestion."""
    book = epub.EpubBook()
    book.set_identifier("pipeline-test")
    book.set_title("Wine Guide")
    book.set_language("en")
    chapter = epub.EpubHtml(title="Burgundy", file_name="burgundy.xhtml", lang="en")
    chapter.content = """
    <html><body>
      <h1>Wine Guide</h1>
      <h2>France</h2>
      <h3>Burgundy</h3>
      <p>Pinot Noir and Chardonnay define this region and its wines.</p>
    </body></html>
    """
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    epub_path = tmp_path / "wine-guide.epub"
    epub.write_epub(str(epub_path), book)
    return epub_path


@pytest.mark.parametrize(
    ("fixture_name", "provider", "file_type"),
    [
        ("test_wine_pdf", "pdfplumber", ".pdf"),
        ("ingestion_epub", "ebooklib", ".epub"),
    ],
)
def test_supported_formats_flow_through_extraction_chunking_and_assembly(
    request: pytest.FixtureRequest,
    fixture_name: str,
    provider: str,
    file_type: str,
) -> None:
    """PDF and EPUB inputs should produce loader-shaped traceable chunks."""
    source_path = request.getfixturevalue(fixture_name)
    extraction_pipeline = DocumentExtractionPipeline(_extraction_config())
    chunking_pipeline = DocumentChunkingPipeline(_chunking_config())

    elements = extraction_pipeline.extract(source_path)
    candidates = chunking_pipeline.chunk(elements)
    chunks = assemble_chroma_chunks(candidates)

    assert elements
    assert candidates
    assert chunks
    assert all(set(chunk) == {"id", "text", "metadata", "importance_score"} for chunk in chunks)
    assert all(chunk["metadata"]["extraction_provider"] == provider for chunk in chunks)
    assert all(chunk["metadata"]["chunking_strategy"] == "section_recursive" for chunk in chunks)
    assert all(chunk["metadata"]["file_type"] == file_type for chunk in chunks)
    assert all("heading_path" in chunk["metadata"] for chunk in chunks)
    assert all("start_page" in chunk["metadata"] and "end_page" in chunk["metadata"] for chunk in chunks)


def test_chunker_registry_resolves_reviewed_default() -> None:
    """The configured Phase 0 default should resolve explicitly."""
    chunker = ChunkerRegistry.resolve_from_config(_chunking_config())

    assert isinstance(chunker, SectionRecursiveChunker)


def test_chunker_registry_rejects_unknown_strategy() -> None:
    """Unknown strategies should fail before indexing starts."""
    config = {**_chunking_config(), "strategy": "by_title"}

    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        ChunkerRegistry.resolve_from_config(config)


def test_unsupported_input_is_skipped_or_raised_by_config(tmp_path: Path) -> None:
    """Unsupported source handling should follow the reviewed strictness flag."""
    source_path = tmp_path / "notes.txt"
    source_path.write_text("Wine notes")

    assert DocumentExtractionPipeline(_extraction_config()).extract(source_path) == []
    with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported document type"):
        DocumentExtractionPipeline(_extraction_config(fail_on_unsupported_file=True)).extract(source_path)


def test_assembly_preserves_metadata_contract_and_optional_wine_extraction() -> None:
    """Assembly should retain legacy fields and add deterministic traceability."""
    candidate = ChunkCandidate(
        text="Barolo DOCG is made from Nebbiolo.",
        source_path="books/italy.pdf",
        file_type="pdf",
        chunk_index=2,
        page_number=14,
        start_page=14,
        end_page=15,
        document_title="Italian Wine",
        chapter="Piedmont",
        section="Barolo",
        heading_path="Italian Wine > Piedmont > Barolo",
        extraction_provider="pdfplumber",
        chunking_strategy="section_recursive",
    )

    chunk = assemble_chroma_chunks([candidate])[0]
    without_wine_metadata = assemble_chroma_chunks([candidate], extract_metadata=False)[0]

    assert chunk["id"].startswith("italy_2_")
    assert chunk["metadata"]["page_number"] == 14
    assert chunk["metadata"]["start_page"] == 14
    assert chunk["metadata"]["end_page"] == 15
    assert chunk["metadata"]["heading_path"] == "Italian Wine > Piedmont > Barolo"
    assert chunk["metadata"]["grapes"] == "nebbiolo"
    assert chunk["metadata"]["classifications"] == "DOCG"
    assert without_wine_metadata["metadata"]["grapes"] == ""
