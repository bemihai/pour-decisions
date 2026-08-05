"""Tests for provider-neutral document extraction contracts."""

from pathlib import Path

import pytest

from src.chroma.extraction import DocumentElement, DocumentExtractor, UnsupportedDocumentTypeError


def test_document_element_represents_pdf_content() -> None:
    """PDF elements should preserve page and structural context without native objects."""
    element = DocumentElement(
        text="  Barolo is made from Nebbiolo.  ",
        source_path="books/italy.pdf",
        file_type="pdf",
        order_index=4,
        page_number=17,
        element_type="paragraph",
        document_title="The Wines of Italy",
        chapter="Piedmont",
        section="Barolo",
        metadata={"font_size": 11.5, "is_bold": False},
    )

    assert element.text == "Barolo is made from Nebbiolo."
    assert element.page_number == 17
    assert element.metadata == {"font_size": 11.5, "is_bold": False}


def test_document_element_represents_epub_content_without_page_number() -> None:
    """EPUB elements should use the same contract without page-based metadata."""
    element = DocumentElement(
        text="Burgundy classifications",
        source_path="books/france.epub",
        file_type="epub",
        order_index=2,
        element_type="heading",
        heading_level=2,
        chapter="Burgundy",
    )

    assert element.page_number is None
    assert element.heading_level == 2
    assert element.metadata == {}


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_document_element_rejects_empty_text(text: str) -> None:
    """Extractors must not be able to return empty normalized content."""
    with pytest.raises(ValueError, match="text must not be empty"):
        DocumentElement(
            text=text,
            source_path="books/wine.pdf",
            file_type="pdf",
            order_index=0,
        )


@pytest.mark.parametrize(
    ("field_name", "overrides"),
    [
        ("source_path", {"source_path": " "}),
        ("file_type", {"file_type": ""}),
    ],
)
def test_document_element_rejects_blank_required_fields(field_name: str, overrides: dict[str, str]) -> None:
    """Required source identifiers should contain meaningful values."""
    values = {
        "text": "Wine content",
        "source_path": "books/wine.pdf",
        "file_type": "pdf",
        "order_index": 0,
        **overrides,
    }

    with pytest.raises(ValueError, match=f"{field_name} must not be empty"):
        DocumentElement(**values)


def test_document_extractor_requires_extract_implementation() -> None:
    """The extraction interface should remain abstract until implemented."""
    with pytest.raises(TypeError, match="abstract method extract"):
        DocumentExtractor()


def test_document_extractor_implementation_returns_elements() -> None:
    """A concrete provider should implement the typed extraction boundary."""

    class TestExtractor(DocumentExtractor):
        def extract(self, path: Path) -> list[DocumentElement]:
            return [
                DocumentElement(
                    text="Extracted wine content",
                    source_path=str(path),
                    file_type=path.suffix.removeprefix("."),
                    order_index=0,
                )
            ]

    elements = TestExtractor().extract(Path("books/wine.pdf"))

    assert len(elements) == 1
    assert elements[0].source_path == "books/wine.pdf"


def test_unsupported_document_type_error_is_a_value_error() -> None:
    """Unsupported format failures should be explicit input errors."""
    error = UnsupportedDocumentTypeError("Unsupported extension: .docx")

    assert isinstance(error, ValueError)
