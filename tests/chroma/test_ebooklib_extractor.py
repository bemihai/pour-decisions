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
      <h2>France</h2>
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
    book.spine = ["nav", chapter]

    epub_path = tmp_path / "wine-guide.epub"
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


def test_epub_extractor_classifies_supported_blocks_and_skips_other_items(structured_epub: Path) -> None:
    """Lists and tables should be explicit while non-document manifest items are skipped."""
    elements = EbookLibExtractor().extract(structured_epub)

    assert [element.element_type for element in elements].count("heading") == 3
    assert [element.element_type for element in elements].count("list_item") == 2
    assert [element.element_type for element in elements].count("table") == 1
    assert all("label.png" not in str(element.metadata) for element in elements)


def test_epub_extractor_rejects_non_epub_path() -> None:
    """Direct provider use should reject unsupported input formats explicitly."""
    with pytest.raises(UnsupportedDocumentTypeError, match="does not support"):
        EbookLibExtractor().extract(Path("book.pdf"))
