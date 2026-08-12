"""Tests for provider-neutral EPUB extraction with EbookLib."""

from pathlib import Path

from ebooklib import epub
import pytest

from src.chroma.extraction import EbookLibExtractor, UnsupportedDocumentTypeError


@pytest.fixture
def structured_epub(tmp_path: Path) -> Path:
    """Create a small EPUB with explicit spine and heading structure."""
    book = epub.EpubBook()
    book.set_identifier("m03-test-book")
    book.set_title("Wine Guide")
    book.set_language("en")

    chapter = epub.EpubHtml(title="Burgundy", file_name="burgundy.xhtml", lang="en")
    chapter.content = """
    <html><body>
      <h1>Wine Guide</h1>
      <p>Introduction to the world's wine regions.</p>
      <h2 id="france">France</h2>
      <h3>Burgundy</h3>
      <p>Pinot Noir and Chardonnay define the region.</p>
      <ul><li>Côte de Nuits</li><li>Côte de Beaune</li></ul>
      <table><tr><th>Village</th><th>Grape</th></tr><tr><td>Gevrey</td><td>Pinot Noir</td></tr></table>
    </body></html>
    """
    unsupported_image = epub.EpubItem(
        uid="label-image",
        file_name="images/label.png",
        media_type="image/png",
        content=b"not-a-real-image",
    )
    book.add_item(chapter)
    book.add_item(unsupported_image)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.toc = [epub.Link("burgundy.xhtml#france", "France", "france")]
    book.spine = ["nav", chapter]

    epub_path = tmp_path / "wine-guide.epub"
    epub.write_epub(str(epub_path), book)
    return epub_path


@pytest.fixture
def dictionary_style_epub(tmp_path: Path) -> Path:
    """Create an EPUB where peer grape entries reuse subsection heading levels."""
    book = epub.EpubBook()
    book.set_identifier("m03-entry-boundary-test")
    book.set_title("Grape Guide")
    book.set_language("en")

    chapter = epub.EpubHtml(title="Grape entries", file_name="entries.xhtml", lang="en")
    chapter.content = """
    <html><body>
      <h2 class="h2c">NEBBIOLO</h2>
      <p>Nebbiolo introduction.</p>
      <h2 class="h2">ENJOYING NEBBIOLO</h2>
      <h3 class="h3">The taste of Nebbiolo</h3>
      <p>Tar, roses, cherries, high tannin, and high acidity.</p>
      <h3 class="h3">Maturity charts</h3>
      <p>Barolo develops over decades.</p>
      <h3 class="h3">NEGOSKA</h3>
      <p>Negoska is a Greek grape.</p>
      <h3 class="h3">NEGRA MOLE</h3>
      <p>Negra Mole is associated with Madeira.</p>
      <h3 class="h3">NEGRAMOLL</h3>
      <p>Negramoll is grown in the Canary Islands.</p>
    </body></html>
    """
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]

    epub_path = tmp_path / "entry-boundaries.epub"
    epub.write_epub(str(epub_path), book)
    return epub_path


def test_epub_extractor_preserves_spine_body_order(structured_epub: Path) -> None:
    """Supported XHTML blocks should be emitted in stable document order."""
    elements = EbookLibExtractor().extract(structured_epub)

    assert [element.text for element in elements] == [
        "Wine Guide",
        "Introduction to the world's wine regions.",
        "France",
        "Burgundy",
        "Pinot Noir and Chardonnay define the region.",
        "Côte de Nuits",
        "Côte de Beaune",
        "Village Grape Gevrey Pinot Noir",
    ]
    assert [element.order_index for element in elements] == list(range(len(elements)))
    assert all(element.page_number is None for element in elements)
    assert all(element.metadata["item_name"] == "burgundy.xhtml" for element in elements)


def test_epub_extractor_maps_headings_and_context(structured_epub: Path) -> None:
    """XHTML heading levels should update title, chapter, and section lineage."""
    elements = EbookLibExtractor().extract(structured_epub)
    by_text = {element.text: element for element in elements}

    assert by_text["Wine Guide"].heading_level == 1
    assert by_text["Wine Guide"].document_title == "Wine Guide"
    assert by_text["France"].heading_level == 2
    assert by_text["France"].chapter == "France"
    assert by_text["Burgundy"].heading_level == 3
    assert by_text["Burgundy"].section == "Burgundy"
    assert by_text["Pinot Noir and Chardonnay define the region."].section == "Burgundy"
    assert by_text["France"].metadata["entry_boundary_evidence"] == "navigation_anchor"
    assert by_text["France"].metadata["element_id"] == "france"
    assert by_text["France"].metadata["nav_title"] == "France"


def test_epub_extractor_classifies_supported_blocks_and_skips_other_items(structured_epub: Path) -> None:
    """Lists and tables should be explicit while non-document manifest items are skipped."""
    elements = EbookLibExtractor().extract(structured_epub)

    assert [element.element_type for element in elements].count("heading") == 3
    assert [element.element_type for element in elements].count("list_item") == 2
    assert [element.element_type for element in elements].count("table") == 1
    assert all("label.png" not in str(element.metadata) for element in elements)


def test_epub_extractor_resets_dictionary_peer_entry_context(dictionary_style_epub: Path) -> None:
    """A run of peer grape headings must end the preceding Nebbiolo entry context."""
    elements = EbookLibExtractor().extract(dictionary_style_epub)
    by_text = {element.text: element for element in elements}

    taste = by_text["Tar, roses, cherries, high tannin, and high acidity."]
    assert taste.chapter == "NEBBIOLO"
    assert taste.section == "The taste of Nebbiolo"
    assert taste.metadata["entry_boundary_evidence"] == "css_entry_class"

    assert by_text["NEGOSKA"].chapter == "NEGOSKA"
    assert by_text["NEGRA MOLE"].chapter == "NEGRA MOLE"
    assert by_text["NEGRAMOLL"].chapter == "NEGRAMOLL"
    assert by_text["Negra Mole is associated with Madeira."].chapter == "NEGRA MOLE"
    assert by_text["NEGRA MOLE"].metadata["entry_boundary_evidence"] == "peer_heading_run"
    assert all(
        element.chapter != "NEBBIOLO" or element.section not in {"NEGOSKA", "NEGRA MOLE", "NEGRAMOLL"}
        for element in elements
    )


def test_epub_extractor_preserves_structural_evidence_metadata(dictionary_style_epub: Path) -> None:
    """CSS classes and entry-boundary decisions should remain auditable."""
    elements = EbookLibExtractor().extract(dictionary_style_epub)
    nebbiolo = next(element for element in elements if element.text == "NEBBIOLO")
    taste = next(element for element in elements if element.text == "The taste of Nebbiolo")

    assert nebbiolo.metadata["css_class"] == "h2c"
    assert nebbiolo.metadata["entry_boundary"] is True
    assert nebbiolo.metadata["context_update_evidence"] == "css_entry_class"
    assert taste.metadata["entry_boundary"] is False
    assert taste.metadata["context_update_evidence"] == "heading_level_3"


def test_epub_extractor_rejects_non_epub_path() -> None:
    """Direct provider use should reject unsupported input formats explicitly."""
    with pytest.raises(UnsupportedDocumentTypeError, match="does not support"):
        EbookLibExtractor().extract(Path("book.pdf"))
