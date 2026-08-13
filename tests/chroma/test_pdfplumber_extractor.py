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


class _CroppedFakePage:
    """Cropped page view with precomputed extraction lines."""

    def __init__(self, lines: list[dict[str, Any]]) -> None:
        self._lines = lines

    def extract_text_lines(self, *, strip: bool, return_chars: bool) -> list[dict[str, Any]]:
        assert strip is True
        assert return_chars is True
        return self._lines


class _LayoutFakePage(_FakePage):
    """Fake two-column page whose crops expose independent column lines."""

    def __init__(
        self,
        page_number: int,
        lines: list[dict[str, Any]],
        *,
        left_lines: list[dict[str, Any]],
        right_lines: list[dict[str, Any]],
        width: float = 600.0,
        height: float = 1000.0,
        x0: float = 0.0,
        top: float = 0.0,
    ) -> None:
        super().__init__(page_number, lines, height=height)
        self.width = width
        self.bbox = (x0, top, x0 + width, top + height)
        self._left_lines = left_lines
        self._right_lines = right_lines
        self.crop_boxes: list[tuple[float, float, float, float]] = []

    def crop(self, bbox: tuple[float, float, float, float]) -> _CroppedFakePage:
        """Return the requested precomputed column at the detected midpoint."""
        self.crop_boxes.append(bbox)
        return _CroppedFakePage(self._left_lines if bbox[0] == self.bbox[0] else self._right_lines)


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


def _positioned_line(
    text: str,
    *,
    x0: float,
    x1: float,
    top: float,
    bottom: float,
    size: float = 12.0,
    bold: bool = False,
) -> dict[str, Any]:
    """Build a line with enough character geometry for column detection."""
    font_name = "Times-Bold" if bold else "Times-Roman"
    visible_chars = [char for char in text if not char.isspace()]
    char_width = (x1 - x0) / max(1, len(visible_chars))
    chars = []
    cursor = x0
    for char in visible_chars:
        chars.append(
            {
                "text": char,
                "x0": cursor,
                "x1": cursor + char_width,
                "size": size,
                "fontname": font_name,
            }
        )
        cursor += char_width
    return {
        "text": text,
        "x0": x0,
        "x1": x1,
        "top": top,
        "bottom": bottom,
        "chars": chars,
    }


def _merged_positioned_line(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    """Model pdfplumber joining same-height text from two columns."""
    return {
        "text": f"{left['text']} {right['text']}",
        "x0": left["x0"],
        "x1": right["x1"],
        "top": min(left["top"], right["top"]),
        "bottom": max(left["bottom"], right["bottom"]),
        "chars": [*left["chars"], *right["chars"]],
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


def test_pdf_extractor_reads_two_columns_without_interleaving(mocker) -> None:
    """A persistent gutter should produce column-first blocks, not row-interleaved text."""
    heading = _positioned_line(
        "Nebbiolo Reference",
        x0=220,
        x1=380,
        top=100,
        bottom=120,
        size=24,
        bold=True,
    )
    left_lines = [
        _positioned_line("Left column is concen-", x0=50, x1=285, top=200, bottom=212),
        _positioned_line("trated and aromatic.", x0=50, x1=250, top=220, bottom=232),
        _positioned_line("Left column finishes here.", x0=50, x1=275, top=240, bottom=252),
    ]
    right_lines = [
        _positioned_line("Right column starts separately.", x0=315, x1=550, top=200, bottom=212),
        _positioned_line("It discusses another wine.", x0=315, x1=530, top=220, bottom=232),
        _positioned_line("Right column finishes here.", x0=315, x1=545, top=240, bottom=252),
    ]
    raw_lines = [
        heading,
        *[
            _merged_positioned_line(left, right)
            for left, right in zip(left_lines, right_lines, strict=True)
        ],
    ]
    page = _LayoutFakePage(1, raw_lines, left_lines=[heading, *left_lines], right_lines=[heading, *right_lines])
    mocker.patch("src.chroma.extraction.pdfplumber_extractor.pdfplumber.open", return_value=_FakePdf([page]))

    elements = PdfPlumberExtractor().extract(Path("columns.pdf"))

    assert [element.element_type for element in elements] == ["heading", "paragraph", "paragraph"]
    assert elements[1].text == (
        "Left column is concentrated and aromatic. Left column finishes here."
    )
    assert elements[2].text == (
        "Right column starts separately. It discusses another wine. Right column finishes here."
    )
    assert elements[1].metadata["column_id"] == 0
    assert elements[2].metadata["column_id"] == 1
    assert elements[1].metadata["layout_mode"] == "two_column"
    assert elements[1].metadata["reading_order_confidence"] >= 0.7
    assert "Right column" not in elements[1].text
    assert "Left column" not in elements[2].text


def test_pdf_extractor_crops_two_columns_with_offset_page_bbox(mocker) -> None:
    """Column crops must use the page's absolute box instead of assuming a zero origin."""
    page_x0 = 54.0
    page_top = -54.0
    heading = _positioned_line(
        "Nebbiolo Reference",
        x0=page_x0 + 220,
        x1=page_x0 + 380,
        top=100,
        bottom=120,
        size=24,
        bold=True,
    )
    left_lines = [
        _positioned_line("Left column first line.", x0=page_x0 + 50, x1=page_x0 + 285, top=200, bottom=212),
        _positioned_line("Left column second line.", x0=page_x0 + 50, x1=page_x0 + 275, top=220, bottom=232),
        _positioned_line("Left column third line.", x0=page_x0 + 50, x1=page_x0 + 270, top=240, bottom=252),
    ]
    right_lines = [
        _positioned_line("Right column first line.", x0=page_x0 + 315, x1=page_x0 + 550, top=200, bottom=212),
        _positioned_line("Right column second line.", x0=page_x0 + 315, x1=page_x0 + 545, top=220, bottom=232),
        _positioned_line("Right column third line.", x0=page_x0 + 315, x1=page_x0 + 540, top=240, bottom=252),
    ]
    raw_lines = [
        heading,
        *[
            _merged_positioned_line(left, right)
            for left, right in zip(left_lines, right_lines, strict=True)
        ],
    ]
    page = _LayoutFakePage(
        1,
        raw_lines,
        left_lines=[heading, *left_lines],
        right_lines=[heading, *right_lines],
        width=600.0,
        height=720.0,
        x0=page_x0,
        top=page_top,
    )
    mocker.patch("src.chroma.extraction.pdfplumber_extractor.pdfplumber.open", return_value=_FakePdf([page]))

    elements = PdfPlumberExtractor().extract(Path("offset-columns.pdf"))

    assert [element.element_type for element in elements] == ["heading", "paragraph", "paragraph"]
    assert elements[1].text == "Left column first line. Left column second line. Left column third line."
    assert elements[2].text == "Right column first line. Right column second line. Right column third line."
    assert page.crop_boxes[0][0:2] == (page_x0, page_top)
    assert page.crop_boxes[1][1:] == (page_top, page_x0 + 600.0, page_top + 720.0)


def test_pdf_extractor_detects_columns_below_full_width_prose(mocker) -> None:
    """Regional gutter evidence must survive full-width prose elsewhere on the page."""
    full_width_lines = [
        _positioned_line(
            f"Full-width introductory sentence number {index} crosses the center of the page.",
            x0=50,
            x1=550,
            top=100 + index * 20,
            bottom=112 + index * 20,
        )
        for index in range(2)
    ]
    left_lines = [
        _positioned_line("Nebbiolo has tar and roses.", x0=50, x1=280, top=280, bottom=292),
        _positioned_line("Its tannins are firm.", x0=50, x1=270, top=300, bottom=312),
        _positioned_line("Its acidity is high.", x0=50, x1=265, top=320, bottom=332),
    ]
    right_lines = [
        _positioned_line("Aged cheese is savory.", x0=320, x1=550, top=280, bottom=292),
        _positioned_line("Truffles echo the wine.", x0=320, x1=540, top=300, bottom=312),
        _positioned_line("The pairing is intense.", x0=320, x1=545, top=320, bottom=332),
    ]
    raw_lines = [
        *full_width_lines,
        *[
            _merged_positioned_line(left, right)
            for left, right in zip(left_lines, right_lines, strict=True)
        ],
    ]
    page = _LayoutFakePage(
        1,
        raw_lines,
        left_lines=[*full_width_lines, *left_lines],
        right_lines=[*full_width_lines, *right_lines],
    )
    mocker.patch("src.chroma.extraction.pdfplumber_extractor.pdfplumber.open", return_value=_FakePdf([page]))

    elements = PdfPlumberExtractor().extract(Path("mixed-regions.pdf"))
    left = next(element for element in elements if "Nebbiolo has tar" in element.text)
    right = next(element for element in elements if "Aged cheese" in element.text)

    assert left.metadata["layout_mode"] == "two_column"
    assert left.metadata["column_id"] == 0
    assert right.metadata["column_id"] == 1
    assert "Aged cheese" not in left.text
    assert "Nebbiolo" not in right.text


def test_pdf_extractor_marks_out_of_bounds_geometry_for_audit(mocker) -> None:
    """Pages whose text lives outside their declared canvas must not be trusted."""
    lines = [
        _positioned_line(
            f"Duplicated off-canvas wine line {index}.",
            x0=-480,
            x1=-40,
            top=180 + index * 20,
            bottom=192 + index * 20,
        )
        for index in range(4)
    ]
    page = _LayoutFakePage(1, lines, left_lines=[], right_lines=[], width=660.0, height=805.0)
    mocker.patch("src.chroma.extraction.pdfplumber_extractor.pdfplumber.open", return_value=_FakePdf([page]))

    elements = PdfPlumberExtractor().extract(Path("invalid-geometry.pdf"))

    assert elements
    assert all(element.metadata["layout_mode"] == "invalid_geometry" for element in elements)
    assert all(element.metadata["layout_audit_required"] is True for element in elements)
    assert all(element.metadata["reading_order_confidence"] == 0.0 for element in elements)


def test_pdf_extractor_does_not_promote_layout_artifacts_to_headings(mocker) -> None:
    """OCR placeholders, single letters, scales, and rating rows must not change context."""
    page = _FakePage(
        1,
        [
            _line("Wine Atlas", top=100, bottom=130, size=24, bold=True),
            _line("Piedmont", top=160, bottom=185, size=18, bold=True),
            _line("(cid:1)(cid:1) (cid:2)(cid:2)", top=210, bottom=230, bold=True),
            _line("S", top=240, bottom=255, bold=True),
            _line("0 5 10", top=270, bottom=285, bold=True),
            _line("NOW TO 2020 17.5", top=300, bottom=315, bold=True),
            _line("Nebbiolo has high tannin and acidity.", top=340, bottom=355),
        ],
    )
    mocker.patch("src.chroma.extraction.pdfplumber_extractor.pdfplumber.open", return_value=_FakePdf([page]))

    elements = PdfPlumberExtractor().extract(Path("artifacts.pdf"))
    prose = next(element for element in elements if element.text.startswith("Nebbiolo"))

    assert prose.chapter == "Piedmont"
    assert prose.section == ""
    assert all(
        element.element_type != "heading"
        for element in elements
        if element.text in {"(cid:1)(cid:1) (cid:2)(cid:2)", "S", "0 5 10", "NOW TO 2020 17.5"}
    )


def test_pdf_extractor_rejects_non_pdf_path() -> None:
    """Direct provider use should reject unsupported input formats explicitly."""
    with pytest.raises(UnsupportedDocumentTypeError, match="does not support"):
        PdfPlumberExtractor().extract(Path("book.epub"))
