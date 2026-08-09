"""Tests for optional section-bounded semantic chunking."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from src.chroma.chunking import ChunkCandidate, SectionSemanticChunker
from src.chroma.extraction import DocumentElement


def _element(
    text: str,
    order_index: int,
    *,
    chapter: str = "Burgundy",
    section: str = "Cote de Nuits",
    page_number: int | None = 1,
) -> DocumentElement:
    return DocumentElement(
        text=text,
        source_path="books/france.pdf",
        file_type="pdf",
        order_index=order_index,
        page_number=page_number,
        document_title="The Wines of France",
        chapter=chapter,
        section=section,
        metadata={"extraction_provider": "pdfplumber"},
    )


@patch("src.chroma.chunking.section_semantic.get_embedder")
def test_disabled_semantic_chunking_uses_no_embedder(mock_get_embedder: Mock) -> None:
    """The default disabled setting should stay on the zero-embedding path."""
    chunker = SectionSemanticChunker.from_config(
        {
            "chunk_size": 200,
            "chunk_overlap": 20,
            "min_chunk_chars": 0,
            "semantic": {"enabled": False},
        }
    )

    chunks = chunker.chunk([_element("Pinot Noir defines the local red wines.", 0)])

    mock_get_embedder.assert_not_called()
    assert len(chunks) == 1
    assert chunks[0].chunking_strategy == "section_recursive"


@patch("src.chroma.chunking.section_semantic.SemanticChunker")
@patch("src.chroma.chunking.section_semantic.get_embedder")
def test_semantic_chunking_uses_cached_embedder_and_isolates_sections(
    mock_get_embedder: Mock,
    mock_semantic_chunker: Mock,
) -> None:
    """Each semantic call should receive content from exactly one section."""
    embedder = Mock(name="cached_local_embedder")
    mock_get_embedder.return_value = embedder
    splitter = mock_semantic_chunker.return_value
    splitter.create_documents.side_effect = lambda texts: [Document(page_content=texts[0])]
    elements = [
        _element("Pinot Noir defines Burgundy.", 0),
        _element("Cabernet Sauvignon defines the Medoc.", 1, chapter="Bordeaux", section="Medoc"),
    ]

    chunks = SectionSemanticChunker(enabled=True, min_chunk_chars=0).chunk(elements)

    mock_get_embedder.assert_called_once_with(None)
    assert [call.args[0] for call in splitter.create_documents.call_args_list] == [
        ["Pinot Noir defines Burgundy."],
        ["Cabernet Sauvignon defines the Medoc."],
    ]
    assert [chunk.section for chunk in chunks] == ["Cote de Nuits", "Medoc"]
    assert all(chunk.chunking_strategy == "section_semantic" for chunk in chunks)


@patch("src.chroma.chunking.section_semantic.SemanticChunker")
@patch("src.chroma.chunking.section_semantic.get_embedder")
def test_semantic_output_uses_shared_candidate_contract_and_lineage(
    mock_get_embedder: Mock,
    mock_semantic_chunker: Mock,
) -> None:
    """Semantic candidates should preserve common context and source metadata."""
    mock_get_embedder.return_value = Mock()
    mock_semantic_chunker.return_value.create_documents.return_value = [
        Document(page_content="Pinot Noir grows here."),
        Document(page_content="Limestone shapes the wines."),
    ]
    elements = [
        _element("Pinot Noir grows here.", 0, page_number=8),
        _element("Limestone shapes the wines.", 1, page_number=9),
    ]

    chunks = SectionSemanticChunker(enabled=True, min_chunk_chars=0).chunk(elements)

    assert all(isinstance(chunk, ChunkCandidate) for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert [chunk.page_number for chunk in chunks] == [8, 9]
    assert chunks[0].heading_path == "The Wines of France > Burgundy > Cote de Nuits"
    assert chunks[0].extraction_provider == "pdfplumber"


@patch("src.chroma.chunking.section_semantic.SemanticChunker")
@patch("src.chroma.chunking.section_semantic.get_embedder")
def test_config_wires_reviewed_semantic_thresholds(
    mock_get_embedder: Mock,
    mock_semantic_chunker: Mock,
) -> None:
    """Config construction should pass reviewed controls to the semantic splitter."""
    mock_get_embedder.return_value = Mock()
    mock_semantic_chunker.return_value.create_documents.return_value = [Document(page_content="Wine content.")]
    config = SimpleNamespace(
        chunk_size=800,
        chunk_overlap=100,
        min_chunk_chars=0,
        semantic=SimpleNamespace(
            enabled=True,
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=2.5,
        ),
    )

    SectionSemanticChunker.from_config(config).chunk([_element("Wine content.", 0)])

    mock_semantic_chunker.assert_called_once_with(
        embeddings=mock_get_embedder.return_value,
        add_start_index=False,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=2.5,
    )


@patch("src.chroma.chunking.section_semantic.SemanticChunker")
@patch("src.chroma.chunking.section_semantic.get_embedder")
def test_failed_section_falls_back_without_affecting_later_sections(
    mock_get_embedder: Mock,
    mock_semantic_chunker: Mock,
) -> None:
    """A section-local semantic failure should retain content deterministically."""
    mock_get_embedder.return_value = Mock()
    splitter = mock_semantic_chunker.return_value
    splitter.create_documents.side_effect = [
        RuntimeError("embedding failure"),
        [Document(page_content="Cabernet Sauvignon defines the Medoc.")],
    ]
    elements = [
        _element("Pinot Noir defines Burgundy.", 0),
        _element("Cabernet Sauvignon defines the Medoc.", 1, chapter="Bordeaux", section="Medoc"),
    ]

    chunks = SectionSemanticChunker(enabled=True, min_chunk_chars=0).chunk(elements)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert [chunk.chunking_strategy for chunk in chunks] == ["section_recursive", "section_semantic"]
    assert [chunk.section for chunk in chunks] == ["Cote de Nuits", "Medoc"]


@patch("src.chroma.chunking.section_semantic.get_embedder", side_effect=RuntimeError("model unavailable"))
def test_unavailable_embedder_falls_back_for_the_document(mock_get_embedder: Mock) -> None:
    """Embedder startup failure should retain content through recursive chunking."""
    chunks = SectionSemanticChunker(enabled=True, min_chunk_chars=0).chunk(
        [_element("Pinot Noir defines Burgundy.", 0)]
    )

    mock_get_embedder.assert_called_once_with(None)
    assert len(chunks) == 1
    assert chunks[0].chunking_strategy == "section_recursive"
