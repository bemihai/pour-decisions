"""Tests for provider-neutral PDF extraction with pdfplumber."""

from pathlib import Path
from typing import Any

import pytest

from src.chroma.extraction import PdfPlumberExtractor, UnsupportedDocumentTypeError


class _FakePage:
    """Minimal pdfplumber page used to exercise layout heuristics."""

    def __init__(self, page_number: int, lines: list[dict[str, Any]], *, height: float = 1000.0) -> None:
        self.page_number = page_number
        self.height = height
        self._lines = lines

    def extract_text_lines(self, *, strip: bool, return_chars: bool) -> list[dict[str, Any]]:
        assert strip is True
        assert return_chars is True
        return self._lines


class _FakePdf:
    """Context-managed pdfplumber document fixture."""

    def __init__(self, pages: list[_FakePage], *, title: str = "") -> None:
        self.pages = pages
        self.metadata = {"Title": title} if title else {}

    def __enter__(self) -> "_FakePdf":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _line(
    text: str,
    *,
    top: float,
    bottom: float,
    size: float = 12.0,
    bold: bool = False,
) -> dict[str, Any]:
    """Build a pdfplumber-like line dictionary."""
    font_name = "Times-Bold" if bold else "Times-Roman"
    return {
        "text": text,
        "top": top,
        "bottom": bottom,
        "chars": [{"size": size, "fontname": font_name} for _ in text],
    }


def test_pdf_extractor_smoke_preserves_order_and_page_numbers(test_wine_pdf: Path) -> None:
    """The repository PDF should produce ordered non-empty page-based elements."""
    elements = PdfPlumberExtractor().extract(test_wine_pdf)

    assert elements
    assert [element.order_index for element in elements] == list(range(len(elements)))
    assert all(element.text.strip() for element in elements)
    assert all(element.file_type == "pdf" for element in elements)
    assert all(element.page_number is not None and element.page_number >= 1 for element in elements)
    assert {element.page_number for element in elements} >= {1, 2}


def test_pdf_extractor_strips_repeated_margin_lines_when_enabled(mocker) -> None:
    """Repeated headers and changing page-number footers should be structural noise."""
    pages = [
        _FakePage(
            page_number,
            [
                _line("THE WINE GUIDE", top=20, bottom=35),
                _line(f"Body content from page {page_number}.", top=200, bottom=215),
                _line(f"Page {page_number}", top=970, bottom=985),
            ],
        )
        for page_number in range(1, 4)
    ]
    mocker.patch("src.chroma.extraction.pdfplumber_extractor.pdfplumber.open", return_value=_FakePdf(pages))

    elements = PdfPlumberExtractor().extract(Path("book.pdf"))

    assert [element.text for element in elements] == [
        "Body content from page 1.",
        "Body content from page 2.",
        "Body content from page 3.",
    ]


def test_pdf_extractor_retains_margin_lines_when_stripping_disabled(mocker) -> None:
    """Disabled structural stripping should preserve auditable header/footer elements."""
    pages = [
        _FakePage(
            page_number,
            [
                _line("THE WINE GUIDE", top=20, bottom=35),
                _line("Wine prose.", top=200, bottom=215),
                _line(f"Page {page_number}", top=970, bottom=985),
            ],
        )
        for page_number in range(1, 3)
    ]
    mocker.patch("src.chroma.extraction.pdfplumber_extractor.pdfplumber.open", return_value=_FakePdf(pages))

    elements = PdfPlumberExtractor(
        strip_repeated_headers=False,
        strip_repeated_footers=False,
    ).extract(Path("book.pdf"))

    assert [element.text for element in elements].count("THE WINE GUIDE") == 2
    assert [element.element_type for element in elements if element.text.startswith("Page ")] == ["footer", "footer"]
    assert [element.metadata["structural_position"] for element in elements[:1]] == ["header"]


def test_pdf_extractor_tracks_heading_context_and_element_types(mocker) -> None:
    """Font and text heuristics should emit normalized structure and context."""
    page = _FakePage(
        1,
        [
            _line("Wine Atlas", top=100, bottom=130, size=24, bold=True),
            _line("France", top=160, bottom=185, size=18, bold=True),
            _line("Burgundy", top=210, bottom=230, size=14, bold=True),
            _line("Burgundy produces Pinot Noir and Chardonnay in a continental climate.", top=260, bottom=275),
            _line("• Côte de Nuits", top=300, bottom=315),
            _line("Village   Main grape   Classification", top=340, bottom=355),
            _line("   ", top=380, bottom=395),
        ],
    )
    mocker.patch("src.chroma.extraction.pdfplumber_extractor.pdfplumber.open", return_value=_FakePdf([page]))

    elements = PdfPlumberExtractor().extract(Path("atlas.pdf"))

    assert [element.element_type for element in elements] == [
        "heading",
        "heading",
        "heading",
        "paragraph",
        "list_item",
        "table",
    ]
    assert elements[0].document_title == "Wine Atlas"
    assert elements[1].chapter == "France"
    assert elements[2].section == "Burgundy"
    assert elements[-1].section == "Burgundy"


def test_pdf_extractor_rejects_non_pdf_path() -> None:
    """Direct provider use should reject unsupported input formats explicitly."""
    with pytest.raises(UnsupportedDocumentTypeError, match="does not support"):
        PdfPlumberExtractor().extract(Path("book.epub"))
