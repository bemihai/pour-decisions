"""Tests for deterministic section-aware recursive chunking."""

import pytest

from src.chroma.chunking import ChunkCandidate, DocumentChunker, SectionRecursiveChunker
from src.chroma.extraction import DocumentElement


def _element(
    text: str,
    order_index: int,
    *,
    chapter: str = "Burgundy",
    section: str = "Cote de Nuits",
    page_number: int | None = 1,
    element_type: str = "paragraph",
    metadata: dict[str, str | int | float | bool] | None = None,
) -> DocumentElement:
    return DocumentElement(
        text=text,
        source_path="books/france.pdf",
        file_type="pdf",
        order_index=order_index,
        page_number=page_number,
        element_type=element_type,
        document_title="The Wines of France",
        chapter=chapter,
        section=section,
        metadata=metadata or {},
    )


def test_chunk_candidate_normalizes_text_and_document_chunker_is_abstract() -> None:
    """The shared contract should be immutable, normalized, and abstract."""
    candidate = ChunkCandidate(
        text="  Burgundy produces Pinot Noir.  ",
        source_path="books/france.pdf",
        file_type="pdf",
        chunk_index=0,
    )

    assert candidate.text == "Burgundy produces Pinot Noir."
    with pytest.raises(TypeError, match="abstract method chunk"):
        DocumentChunker()


def test_chunks_do_not_cross_chapter_or_section_boundaries() -> None:
    """Structural groups should remain isolated even when both would fit."""
    elements = [
        _element("Pinot Noir defines many red wines here.", 1),
        _element("Chardonnay is central to the white wines.", 3, chapter="Chablis", section="Grand Crus"),
    ]

    chunks = SectionRecursiveChunker(chunk_size=200, chunk_overlap=40, min_chunk_chars=0).chunk(elements)

    assert len(chunks) == 2
    assert chunks[0].chapter == "Burgundy"
    assert chunks[1].chapter == "Chablis"
    assert "Chardonnay" not in chunks[0].text
    assert "Pinot Noir" not in chunks[1].text


def test_oversized_section_is_split_at_recursive_boundaries() -> None:
    """Long prose should split below the configured character limit."""
    sentences = [f"Sentence {index} describes the vineyard and its soils." for index in range(12)]
    elements = [_element(" ".join(sentences), 0, page_number=7)]

    chunks = SectionRecursiveChunker(chunk_size=120, chunk_overlap=20, min_chunk_chars=0).chunk(elements)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert all(chunk.start_page == 7 and chunk.end_page == 7 for chunk in chunks)


def test_heading_context_is_metadata_not_standalone_text() -> None:
    """Heading paths should be retained without creating heading-only chunks."""
    elements = [
        _element("Cote de Nuits", 0, element_type="heading"),
        _element("The slope is planted primarily to Pinot Noir.", 1, page_number=11),
    ]

    chunks = SectionRecursiveChunker(chunk_size=200, chunk_overlap=0, min_chunk_chars=0).chunk(elements)

    assert len(chunks) == 1
    assert chunks[0].text == "The slope is planted primarily to Pinot Noir."
    assert chunks[0].heading_path == "The Wines of France > Burgundy > Cote de Nuits"
    assert chunks[0].page_number == 11
    assert chunks[0].chunking_strategy == "section_recursive"


def test_small_low_signal_fragment_is_dropped_but_wine_signal_is_retained() -> None:
    """The minimum length rule should retain useful wine entity fragments."""
    elements = [
        _element("General introduction.", 0, section="Preface"),
        _element("Barolo DOCG: Nebbiolo.", 1, section="Classification"),
    ]

    chunks = SectionRecursiveChunker(chunk_size=200, chunk_overlap=0, min_chunk_chars=50).chunk(elements)

    assert [chunk.text for chunk in chunks] == ["Barolo DOCG: Nebbiolo."]


def test_explicit_high_signal_element_metadata_retains_short_fragment() -> None:
    """Upstream entity metadata should also override the minimum length rule."""
    element = _element("Estate overview.", 0, metadata={"producer": "Domaine Test"})

    chunks = SectionRecursiveChunker(chunk_size=200, chunk_overlap=0, min_chunk_chars=100).chunk([element])

    assert len(chunks) == 1


def test_overlap_remains_inside_each_section() -> None:
    """Overlap should repeat local text only and never leak adjacent sections."""
    first_text = " ".join(f"Burgundy{index}" for index in range(30))
    second_text = " ".join(f"Bordeaux{index}" for index in range(30))
    elements = [
        _element(first_text, 0, section="Burgundy detail"),
        _element(second_text, 1, chapter="Bordeaux", section="Bordeaux detail"),
    ]

    chunks = SectionRecursiveChunker(chunk_size=100, chunk_overlap=25, min_chunk_chars=0).chunk(elements)

    burgundy_chunks = [chunk for chunk in chunks if chunk.section == "Burgundy detail"]
    bordeaux_chunks = [chunk for chunk in chunks if chunk.section == "Bordeaux detail"]
    assert len(burgundy_chunks) > 1
    assert len(bordeaux_chunks) > 1
    assert all("Bordeaux" not in chunk.text for chunk in burgundy_chunks)
    assert all("Burgundy" not in chunk.text for chunk in bordeaux_chunks)


def test_chunk_page_lineage_spans_only_touched_elements() -> None:
    """Candidate page bounds should reflect elements included in each chunk."""
    elements = [
        _element("Pinot Noir grows on the upper slope.", 0, page_number=4),
        _element("Limestone and marl shape the resulting wines.", 1, page_number=5),
    ]

    chunks = SectionRecursiveChunker(chunk_size=200, chunk_overlap=0, min_chunk_chars=0).chunk(elements)

    assert len(chunks) == 1
    assert chunks[0].start_page == 4
    assert chunks[0].end_page == 5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_size": 0}, "chunk_size"),
        ({"chunk_size": 100, "chunk_overlap": 100}, "chunk_overlap"),
        ({"chunk_overlap": -1}, "chunk_overlap"),
        ({"min_chunk_chars": -1}, "min_chunk_chars"),
    ],
)
def test_invalid_chunking_limits_are_rejected(kwargs: dict[str, int], message: str) -> None:
    """Invalid limits should fail before any document processing begins."""
    with pytest.raises(ValueError, match=message):
        SectionRecursiveChunker(**kwargs)
