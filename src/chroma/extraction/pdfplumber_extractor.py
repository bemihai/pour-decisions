"""PDF extraction backed by pdfplumber's page-local layout data."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import re
from statistics import median
from typing import Any

import pdfplumber

from .base import DocumentElement, DocumentExtractor, UnsupportedDocumentTypeError


_LIST_ITEM_PATTERN = re.compile(r"^(?:[-*•▪◦]|\d+[.)]|[A-Za-z][.)])\s+")
_TABLE_SEPARATOR_PATTERN = re.compile(r"(?:\s{3,}|\t|\s*\|\s*)")
_DIGIT_PATTERN = re.compile(r"\d+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class _PdfLine:
    """Normalized layout details for one extracted PDF line."""

    text: str
    page_number: int
    page_height: float
    line_index: int
    top: float
    bottom: float
    font_size: float
    bold_fraction: float

    @property
    def structural_position(self) -> str:
        """Classify a line by its position in the page margins."""
        if self.page_height <= 0:
            return "body"
        if self.top <= self.page_height * 0.08:
            return "header"
        if self.bottom >= self.page_height * 0.92:
            return "footer"
        return "body"


@dataclass
class _DocumentContext:
    """Mutable heading context applied to emitted immutable elements."""

    document_title: str = ""
    chapter: str = ""
    section: str = ""

    def apply_heading(self, text: str, level: int) -> None:
        """Update document context from one inferred heading."""
        if level == 1:
            if not self.document_title:
                self.document_title = text
            elif _normalized_line_key(text) != _normalized_line_key(self.document_title):
                self.chapter = text
                self.section = ""
        elif level == 2:
            self.chapter = text
            self.section = ""
        else:
            self.section = text


class PdfPlumberExtractor(DocumentExtractor):
    """Extract ordered provider-neutral elements from a PDF file."""

    def __init__(
        self,
        *,
        strip_repeated_headers: bool = True,
        strip_repeated_footers: bool = True,
    ) -> None:
        """Configure structural margin-line removal."""
        self._strip_repeated_headers = strip_repeated_headers
        self._strip_repeated_footers = strip_repeated_footers

    def extract(self, path: Path) -> list[DocumentElement]:
        """Extract ordered text and layout context from one PDF."""
        source_path = Path(path)
        if source_path.suffix.lower() != ".pdf":
            unsupported_suffix = source_path.suffix or "no suffix"
            raise UnsupportedDocumentTypeError(f"PdfPlumberExtractor does not support {unsupported_suffix}")

        with pdfplumber.open(source_path) as pdf:
            lines = [line for page in pdf.pages for line in _extract_page_lines(page)]
            metadata_title = str((pdf.metadata or {}).get("Title") or "").strip()

        if not lines:
            return []

        repeated_headers, repeated_footers = _repeated_margin_line_keys(lines)
        body_font_size = _body_font_size(lines)
        context = _DocumentContext(document_title=metadata_title)
        elements: list[DocumentElement] = []

        for line in lines:
            normalized_key = _normalized_line_key(line.text)
            if self._strip_repeated_headers and normalized_key in repeated_headers:
                continue
            if self._strip_repeated_footers and normalized_key in repeated_footers:
                continue

            heading_level = _infer_heading_level(line, body_font_size)
            element_type = _infer_element_type(line, heading_level)
            if heading_level is not None:
                context.apply_heading(line.text, heading_level)

            elements.append(
                DocumentElement(
                    text=line.text,
                    source_path=str(source_path),
                    file_type="pdf",
                    order_index=len(elements),
                    page_number=line.page_number,
                    element_type=element_type,
                    heading_level=heading_level,
                    document_title=context.document_title,
                    chapter=context.chapter,
                    section=context.section,
                    metadata={
                        "extraction_provider": "pdfplumber",
                        "line_index": line.line_index,
                        "top": round(line.top, 3),
                        "bottom": round(line.bottom, 3),
                        "font_size": round(line.font_size, 3),
                        "is_bold": line.bold_fraction >= 0.6,
                        "structural_position": line.structural_position,
                    },
                )
            )
        return elements


def _extract_page_lines(page: Any) -> list[_PdfLine]:
    """Convert pdfplumber line dictionaries into normalized line records."""
    extracted_lines = page.extract_text_lines(strip=True, return_chars=True) or []
    page_number = int(page.page_number)
    page_height = float(page.height)
    lines: list[_PdfLine] = []

    for line_index, raw_line in enumerate(extracted_lines):
        text = str(raw_line.get("text") or "").strip()
        if not text:
            continue

        chars = list(raw_line.get("chars") or [])
        font_sizes = [float(char["size"]) for char in chars if _is_positive_number(char.get("size"))]
        bold_chars = sum("bold" in str(char.get("fontname") or "").casefold() for char in chars)
        lines.append(
            _PdfLine(
                text=text,
                page_number=page_number,
                page_height=page_height,
                line_index=line_index,
                top=float(raw_line.get("top") or 0.0),
                bottom=float(raw_line.get("bottom") or raw_line.get("top") or 0.0),
                font_size=float(median(font_sizes)) if font_sizes else 0.0,
                bold_fraction=bold_chars / len(chars) if chars else 0.0,
            )
        )
    return lines


def _repeated_margin_line_keys(lines: list[_PdfLine]) -> tuple[set[str], set[str]]:
    """Find normalized header/footer text repeated across enough pages."""
    page_numbers = {line.page_number for line in lines}
    if len(page_numbers) < 2:
        return set(), set()

    minimum_pages = max(2, math.ceil(len(page_numbers) * 0.5))
    header_pages: dict[str, set[int]] = defaultdict(set)
    footer_pages: dict[str, set[int]] = defaultdict(set)
    for line in lines:
        key = _normalized_line_key(line.text)
        if not key:
            continue
        if line.structural_position == "header":
            header_pages[key].add(line.page_number)
        elif line.structural_position == "footer":
            footer_pages[key].add(line.page_number)

    repeated_headers = {key for key, pages in header_pages.items() if len(pages) >= minimum_pages}
    repeated_footers = {key for key, pages in footer_pages.items() if len(pages) >= minimum_pages}
    return repeated_headers, repeated_footers


def _body_font_size(lines: list[_PdfLine]) -> float:
    """Estimate the dominant prose size from body-position character counts."""
    weighted_sizes: list[float] = []
    for line in lines:
        if line.structural_position == "body" and line.font_size > 0:
            weighted_sizes.extend([line.font_size] * max(1, len(line.text)))
    return float(median(weighted_sizes)) if weighted_sizes else 0.0


def _infer_heading_level(line: _PdfLine, body_font_size: float) -> int | None:
    """Infer a three-level heading from explicit font and text-shape rules."""
    word_count = len(line.text.split())
    if line.structural_position != "body" or word_count > 15 or len(line.text) > 140:
        return None
    if line.text.endswith((".", "?", "!", ";")):
        return None

    ratio = line.font_size / body_font_size if body_font_size > 0 else 0.0
    if ratio >= 1.6:
        return 1
    if ratio >= 1.3:
        return 2
    if ratio >= 1.12 or line.bold_fraction >= 0.6:
        return 3
    if word_count <= 10 and any(char.isalpha() for char in line.text) and line.text.isupper():
        return 3
    return None


def _infer_element_type(line: _PdfLine, heading_level: int | None) -> str:
    """Classify one line using deterministic structural heuristics."""
    if line.structural_position == "footer":
        return "footer"
    if line.structural_position == "header":
        return "unknown"
    if heading_level is not None:
        return "heading"
    if _LIST_ITEM_PATTERN.match(line.text):
        return "list_item"
    if _looks_like_table_row(line.text):
        return "table"
    if not any(char.isalpha() for char in line.text):
        return "unknown"
    return "paragraph"


def _looks_like_table_row(text: str) -> bool:
    """Return whether text contains at least two explicit column separators."""
    return len([part for part in _TABLE_SEPARATOR_PATTERN.split(text) if part.strip()]) >= 3


def _normalized_line_key(text: str) -> str:
    """Normalize margin text while treating changing page numbers as equivalent."""
    normalized = _WHITESPACE_PATTERN.sub(" ", text).strip().casefold()
    return _DIGIT_PATTERN.sub("#", normalized)


def _is_positive_number(value: Any) -> bool:
    """Return whether a value is a finite positive int or float."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return math.isfinite(float(value)) and float(value) > 0
