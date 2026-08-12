"""PDF extraction backed by pdfplumber's page-local layout data."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import math
from pathlib import Path
import re
from statistics import median
from typing import Any

import pdfplumber

from .base import DocumentElement, DocumentExtractor, UnsupportedDocumentTypeError
from ..structural_roles import classify_structural_role


_LIST_ITEM_PATTERN = re.compile(r"^(?:[-*•▪◦]|\d+[.)]|[A-Za-z][.)])\s+")
_TABLE_SEPARATOR_PATTERN = re.compile(r"(?:\s{3,}|\t|\s*\|\s*)")
_DIGIT_PATTERN = re.compile(r"\d+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_CID_PATTERN = re.compile(r"(?:\(cid:\d+\)\s*)+", re.IGNORECASE)
_NUMERIC_SCALE_PATTERN = re.compile(r"^\d+(?:\s*(?:-{1,3}|\s)\s*\d+){2,}$")
_RATING_ROW_PATTERN = re.compile(
    r"^(?=.*\d)(?=.*\b(?:NOW|DRINK|KEEP|FROM|TO|SOON)\b)[A-Z0-9 .-]+$",
    re.IGNORECASE,
)
_FORM_LABEL_PATTERN = re.compile(r"^[A-Za-z]\s+\d+\s+[^.!?]{0,80}:?")
_PRESERVED_HYPHEN_PREFIXES = {"full", "half", "long", "medium", "off", "semi", "short", "well"}


@dataclass(frozen=True)
class _PageLayout:
    """Detected page-local reading layout."""

    mode: str
    confidence: float
    gutter_start: float | None = None
    gutter_end: float | None = None
    audit_required: bool = False

    @property
    def has_columns(self) -> bool:
        """Return whether the page should be read as two columns."""
        return self.gutter_start is not None and self.gutter_end is not None


@dataclass(frozen=True)
class _PdfLine:
    """Normalized layout details for one extracted PDF line."""

    text: str
    page_number: int
    page_width: float
    page_height: float
    line_index: int
    top: float
    bottom: float
    x0: float
    x1: float
    font_size: float
    bold_fraction: float
    column_id: int
    block_id: int
    layout_mode: str
    reading_order_confidence: float
    layout_audit_required: bool

    @property
    def structural_position(self) -> str:
        """Classify a line by its position in the page margins."""
        if self.page_height <= 0:
            return "body"
        if self.top <= self.page_height * 0.05:
            return "header"
        if self.bottom >= self.page_height * 0.95:
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
        """Extract ordered text blocks and validated layout context from one PDF."""
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
        filtered_lines = [
            line
            for line in lines
            if not (
                self._strip_repeated_headers
                and line.structural_position == "header"
                and _normalized_line_key(line.text) in repeated_headers
            )
            and not (
                self._strip_repeated_footers
                and line.structural_position == "footer"
                and _normalized_line_key(line.text) in repeated_footers
            )
        ]
        return _build_document_elements(
            filtered_lines,
            source_path=source_path,
            metadata_title=metadata_title,
            body_font_size=body_font_size,
        )


def _extract_page_lines(page: Any) -> list[_PdfLine]:
    """Extract page lines in explicit single- or multi-column reading order."""
    raw_lines = page.extract_text_lines(strip=True, return_chars=True) or []
    page_width = float(getattr(page, "width", 0.0) or 0.0)
    page_height = float(getattr(page, "height", 0.0) or 0.0)
    page_bbox = tuple(
        float(coordinate)
        for coordinate in (
            getattr(page, "bbox", None)
            or (0.0, 0.0, page_width, page_height)
        )
    )
    page_x0, page_top, page_x1, page_bottom = page_bbox
    page_number = int(page.page_number)
    layout = _detect_page_layout(raw_lines, page_width=page_width, page_x0=page_x0)

    if not layout.has_columns or not callable(getattr(page, "crop", None)):
        lines = [
            _raw_line_to_pdf_line(
                raw_line,
                page_number=page_number,
                page_width=page_width,
                page_height=page_height,
                line_index=index,
                column_id=0,
                layout=layout,
            )
            for index, raw_line in enumerate(raw_lines)
            if str(raw_line.get("text") or "").strip()
        ]
        return _assign_block_ids(lines)

    gutter_midpoint = (float(layout.gutter_start) + float(layout.gutter_end)) / 2
    left_raw = page.crop((page_x0, page_top, gutter_midpoint, page_bottom)).extract_text_lines(
        strip=True,
        return_chars=True,
    ) or []
    right_raw = page.crop((gutter_midpoint, page_top, page_x1, page_bottom)).extract_text_lines(
        strip=True,
        return_chars=True,
    ) or []
    full_width_raw = [raw_line for raw_line in raw_lines if _line_occupies_gutter(raw_line, layout)]
    full_width_ranges = [
        (float(raw_line.get("top") or 0.0), float(raw_line.get("bottom") or raw_line.get("top") or 0.0))
        for raw_line in full_width_raw
    ]
    left_raw = [raw_line for raw_line in left_raw if not _overlaps_any_range(raw_line, full_width_ranges)]
    right_raw = [raw_line for raw_line in right_raw if not _overlaps_any_range(raw_line, full_width_ranges)]

    ordered_raw = _order_column_regions(left_raw, right_raw, full_width_raw)
    lines = [
        _raw_line_to_pdf_line(
            raw_line,
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            line_index=index,
            column_id=column_id,
            layout=layout,
        )
        for index, (raw_line, column_id) in enumerate(ordered_raw)
        if str(raw_line.get("text") or "").strip()
    ]
    return _assign_block_ids(lines)


def _detect_page_layout(
    raw_lines: list[dict[str, Any]],
    *,
    page_width: float,
    page_x0: float = 0.0,
) -> _PageLayout:
    """Detect a persistent central gutter from positioned character coverage."""
    if page_width <= 0 or len(raw_lines) < 4:
        return _PageLayout(mode="single_column", confidence=1.0)

    positioned_chars = [
        char
        for raw_line in raw_lines
        for char in list(raw_line.get("chars") or [])
        if _is_finite_number(char.get("x0")) and _is_finite_number(char.get("x1"))
    ]
    if len(positioned_chars) < 20:
        return _PageLayout(mode="single_column", confidence=1.0)
    page_x1 = page_x0 + page_width
    outside_chars = [
        char
        for char in positioned_chars
        if float(char["x1"]) < page_x0 - 1.0 or float(char["x0"]) > page_x1 + 1.0
    ]
    if len(outside_chars) / len(positioned_chars) > 0.20:
        return _PageLayout(mode="invalid_geometry", confidence=0.0, audit_required=True)

    bin_count = max(1, int(math.ceil(page_width)))
    coverage = [0] * bin_count
    for char in positioned_chars:
        relative_x0 = float(char["x0"]) - page_x0
        relative_x1 = float(char["x1"]) - page_x0
        start = max(0, min(bin_count - 1, int(math.floor(relative_x0))))
        end = max(start + 1, min(bin_count, int(math.ceil(relative_x1))))
        for index in range(start, end):
            coverage[index] += 1

    font_sizes = [float(char["size"]) for char in positioned_chars if _is_positive_number(char.get("size"))]
    typical_font_size = float(median(font_sizes)) if font_sizes else 10.0
    minimum_gutter_width = max(8.0, typical_font_size * 0.8)
    maximum_coverage = max(1, math.ceil(len(raw_lines) * 0.02))
    search_start = int(page_width * 0.30)
    search_end = min(bin_count, int(page_width * 0.70))
    runs = _merge_nearby_runs(
        _low_coverage_runs(
            coverage,
            start=search_start,
            end=search_end,
            maximum_coverage=maximum_coverage,
        ),
        maximum_gap=max(2, int(math.ceil(typical_font_size * 0.30))),
    )
    candidates = [
        (start, end)
        for start, end in runs
        if end - start >= minimum_gutter_width
        and end - start <= page_width * 0.18
        and page_width * 0.35 <= (start + end) / 2 <= page_width * 0.65
    ]
    if not candidates:
        regional_runs = _regional_gutter_runs(
            raw_lines,
            bin_count=bin_count,
            page_x0=page_x0,
            search_start=search_start,
            search_end=search_end,
            typical_font_size=typical_font_size,
        )
        candidates = [
            (start, end)
            for start, end in regional_runs
            if end - start >= minimum_gutter_width
            and end - start <= page_width * 0.18
            and page_width * 0.35 <= (start + end) / 2 <= page_width * 0.65
        ]
    if not candidates:
        return _PageLayout(mode="single_column", confidence=1.0)

    scored_candidates: list[tuple[float, float, float, float, float]] = []
    for relative_start, relative_end in candidates:
        start = relative_start + page_x0
        end = relative_end + page_x0
        left_chars = [char for char in positioned_chars if float(char["x1"]) <= start]
        right_chars = [char for char in positioned_chars if float(char["x0"]) >= end]
        if not left_chars or not right_chars:
            continue

        balance = min(len(left_chars), len(right_chars)) / max(len(left_chars), len(right_chars))
        supporting_lines = 0
        left_line_count = 0
        right_line_count = 0
        for raw_line in raw_lines:
            chars = list(raw_line.get("chars") or [])
            has_left = any(_is_finite_number(char.get("x1")) and float(char["x1"]) <= start for char in chars)
            has_right = any(_is_finite_number(char.get("x0")) and float(char["x0"]) >= end for char in chars)
            left_line_count += int(has_left)
            right_line_count += int(has_right)
            supporting_lines += int(has_left and has_right and not _line_has_char_between(chars, start, end))

        if min(left_line_count, right_line_count) < 2 or balance < 0.18:
            continue

        support_ratio = supporting_lines / max(1, len(raw_lines))
        if support_ratio < 0.05:
            continue
        relative_midpoint = (relative_start + relative_end) / 2
        distance_from_center = abs(relative_midpoint - (page_width / 2)) / page_width
        score = support_ratio + (balance * 0.25) - distance_from_center
        scored_candidates.append((score, float(start), float(end), support_ratio, balance))

    if not scored_candidates:
        return _PageLayout(mode="single_column", confidence=1.0)

    _, gutter_start, gutter_end, support_ratio, balance = max(scored_candidates, key=lambda item: item[0])
    confidence = min(1.0, 0.55 + support_ratio + (balance * 0.20))
    audit_required = confidence < 0.70
    return _PageLayout(
        mode="ambiguous_two_column" if audit_required else "two_column",
        confidence=confidence,
        gutter_start=gutter_start,
        gutter_end=gutter_end,
        audit_required=audit_required,
    )


def _regional_gutter_runs(
    raw_lines: list[dict[str, Any]],
    *,
    bin_count: int,
    page_x0: float,
    search_start: int,
    search_end: int,
    typical_font_size: float,
) -> list[tuple[int, int]]:
    """Find gutters supported by a page region when full-width text masks them globally."""
    votes = [0] * bin_count
    minimum_gap_width = max(12.0, typical_font_size * 1.5)
    for raw_line in raw_lines:
        chars = sorted(
            (
                char
                for char in list(raw_line.get("chars") or [])
                if _is_finite_number(char.get("x0")) and _is_finite_number(char.get("x1"))
            ),
            key=lambda char: (float(char["x0"]), float(char["x1"])),
        )
        for left, right in zip(chars, chars[1:]):
            gap_start = float(left["x1"]) - page_x0
            gap_end = float(right["x0"]) - page_x0
            if gap_end - gap_start < minimum_gap_width:
                continue
            start = max(search_start, min(search_end, int(math.ceil(gap_start))))
            end = max(start, min(search_end, int(math.floor(gap_end))))
            for index in range(start, end):
                votes[index] += 1

    baseline_support = max(2, math.ceil(len(raw_lines) * 0.08))
    peak_support = max(votes[search_start:search_end], default=0)
    minimum_support = max(baseline_support, math.ceil(peak_support * 0.70))
    return _supported_runs(
        votes,
        start=search_start,
        end=search_end,
        minimum_support=minimum_support,
    )


def _supported_runs(
    votes: list[int],
    *,
    start: int,
    end: int,
    minimum_support: int,
) -> list[tuple[int, int]]:
    """Return half-open runs meeting a minimum independent-line vote count."""
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index in range(start, end):
        if votes[index] >= minimum_support:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            runs.append((run_start, index))
            run_start = None
    if run_start is not None:
        runs.append((run_start, end))
    return runs


def _low_coverage_runs(
    coverage: list[int],
    *,
    start: int,
    end: int,
    maximum_coverage: int,
) -> list[tuple[int, int]]:
    """Return half-open runs whose character coverage stays below a threshold."""
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index in range(start, end):
        if coverage[index] <= maximum_coverage:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            runs.append((run_start, index))
            run_start = None
    if run_start is not None:
        runs.append((run_start, end))
    return runs


def _merge_nearby_runs(runs: list[tuple[int, int]], *, maximum_gap: int) -> list[tuple[int, int]]:
    """Merge low-coverage runs separated only by narrow glyph-rounding spikes."""
    if not runs:
        return []

    merged = [runs[0]]
    for start, end in runs[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end <= maximum_gap:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def _line_has_char_between(chars: list[dict[str, Any]], start: float, end: float) -> bool:
    """Return whether a positioned character occupies the candidate gutter."""
    return any(
        _is_finite_number(char.get("x0"))
        and _is_finite_number(char.get("x1"))
        and float(char["x0"]) < end
        and float(char["x1"]) > start
        for char in chars
    )


def _line_occupies_gutter(raw_line: dict[str, Any], layout: _PageLayout) -> bool:
    """Return whether a raw line crosses the center of the detected gutter."""
    if not layout.has_columns:
        return False
    gutter_midpoint = (float(layout.gutter_start) + float(layout.gutter_end)) / 2
    return _line_has_char_between(
        list(raw_line.get("chars") or []),
        gutter_midpoint - 1.0,
        gutter_midpoint + 1.0,
    )


def _overlaps_any_range(raw_line: dict[str, Any], ranges: list[tuple[float, float]]) -> bool:
    """Return whether a raw line overlaps any full-width vertical range."""
    top = float(raw_line.get("top") or 0.0)
    bottom = float(raw_line.get("bottom") or top)
    return any(top < range_bottom and bottom > range_top for range_top, range_bottom in ranges)


def _order_column_regions(
    left_raw: list[dict[str, Any]],
    right_raw: list[dict[str, Any]],
    full_width_raw: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], int]]:
    """Order columns inside vertical regions separated by full-width lines."""
    left = sorted(left_raw, key=_raw_line_position)
    right = sorted(right_raw, key=_raw_line_position)
    full_width = sorted(full_width_raw, key=_raw_line_position)
    ordered: list[tuple[dict[str, Any], int]] = []
    region_top = -math.inf

    for full_line in full_width:
        full_top = float(full_line.get("top") or 0.0)
        ordered.extend((line, 0) for line in left if region_top <= _raw_line_center(line) < full_top)
        ordered.extend((line, 1) for line in right if region_top <= _raw_line_center(line) < full_top)
        ordered.append((full_line, -1))
        region_top = float(full_line.get("bottom") or full_top)

    ordered.extend((line, 0) for line in left if _raw_line_center(line) >= region_top)
    ordered.extend((line, 1) for line in right if _raw_line_center(line) >= region_top)
    return ordered


def _raw_line_position(raw_line: dict[str, Any]) -> tuple[float, float]:
    """Return deterministic vertical/horizontal ordering for a raw line."""
    return float(raw_line.get("top") or 0.0), float(raw_line.get("x0") or 0.0)


def _raw_line_center(raw_line: dict[str, Any]) -> float:
    """Return the vertical center of one raw line."""
    top = float(raw_line.get("top") or 0.0)
    bottom = float(raw_line.get("bottom") or top)
    return (top + bottom) / 2


def _raw_line_to_pdf_line(
    raw_line: dict[str, Any],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    line_index: int,
    column_id: int,
    layout: _PageLayout,
) -> _PdfLine:
    """Convert one provider line dictionary into a normalized line record."""
    chars = list(raw_line.get("chars") or [])
    font_sizes = [float(char["size"]) for char in chars if _is_positive_number(char.get("size"))]
    bold_chars = sum("bold" in str(char.get("fontname") or "").casefold() for char in chars)
    x0_values = [float(char["x0"]) for char in chars if _is_finite_number(char.get("x0"))]
    x1_values = [float(char["x1"]) for char in chars if _is_finite_number(char.get("x1"))]
    raw_x0 = raw_line.get("x0")
    raw_x1 = raw_line.get("x1")
    return _PdfLine(
        text=str(raw_line.get("text") or "").strip(),
        page_number=page_number,
        page_width=page_width,
        page_height=page_height,
        line_index=line_index,
        top=float(raw_line.get("top") or 0.0),
        bottom=float(raw_line.get("bottom") or raw_line.get("top") or 0.0),
        x0=float(raw_x0) if _is_finite_number(raw_x0) else min(x0_values, default=0.0),
        x1=float(raw_x1) if _is_finite_number(raw_x1) else max(x1_values, default=0.0),
        font_size=float(median(font_sizes)) if font_sizes else 0.0,
        bold_fraction=bold_chars / len(chars) if chars else 0.0,
        column_id=column_id,
        block_id=0,
        layout_mode=layout.mode,
        reading_order_confidence=layout.confidence,
        layout_audit_required=layout.audit_required,
    )


def _assign_block_ids(lines: list[_PdfLine]) -> list[_PdfLine]:
    """Assign page-local block IDs without crossing columns or large layout gaps."""
    assigned: list[_PdfLine] = []
    block_id = -1
    previous: _PdfLine | None = None
    for line in lines:
        starts_new_block = previous is None or _starts_new_block(previous, line)
        if starts_new_block:
            block_id += 1
        assigned.append(replace(line, block_id=block_id))
        previous = line
    return assigned


def _starts_new_block(previous: _PdfLine, current: _PdfLine) -> bool:
    """Return whether two adjacent reading-order lines belong to different blocks."""
    if previous.page_number != current.page_number or previous.column_id != current.column_id:
        return True
    if previous.structural_position != current.structural_position:
        return True
    if current.top < previous.top:
        return True
    typical_font_size = max(previous.font_size, current.font_size, 8.0)
    if current.top - previous.bottom > max(6.0, typical_font_size * 0.75):
        return True
    return (previous.bold_fraction >= 0.6) != (current.bold_fraction >= 0.6)


def _build_document_elements(
    lines: list[_PdfLine],
    *,
    source_path: Path,
    metadata_title: str,
    body_font_size: float,
) -> list[DocumentElement]:
    """Build elements while merging only lines from the same validated layout block."""
    context = _DocumentContext(document_title=metadata_title)
    elements: list[DocumentElement] = []
    pending_lines: list[_PdfLine] = []
    pending_type = ""
    pending_context = ("", "", "")
    pending_key: tuple[Any, ...] | None = None

    def flush_pending() -> None:
        if not pending_lines:
            return
        elements.append(
            _document_element_from_lines(
                pending_lines,
                source_path=source_path,
                order_index=len(elements),
                element_type=pending_type,
                heading_level=None,
                context=pending_context,
            )
        )
        pending_lines.clear()

    for line in lines:
        heading_level = _infer_heading_level(line, body_font_size)
        element_type = _infer_element_type(line, heading_level)
        if heading_level is not None:
            flush_pending()
            pending_key = None
            context.apply_heading(line.text, heading_level)
            elements.append(
                _document_element_from_lines(
                    [line],
                    source_path=source_path,
                    order_index=len(elements),
                    element_type=element_type,
                    heading_level=heading_level,
                    context=(context.document_title, context.chapter, context.section),
                )
            )
            continue

        current_context = (context.document_title, context.chapter, context.section)
        key = (line.page_number, line.block_id, line.column_id, element_type, current_context)
        if key != pending_key:
            flush_pending()
            pending_key = key
            pending_type = element_type
            pending_context = current_context
        pending_lines.append(line)

    flush_pending()
    return _apply_page_structural_roles(elements)


def _apply_page_structural_roles(elements: list[DocumentElement]) -> list[DocumentElement]:
    """Propagate page-wide form/ToC/index roles across fragmented layout elements."""
    elements_by_page: dict[int | None, list[DocumentElement]] = defaultdict(list)
    for element in elements:
        elements_by_page[element.page_number].append(element)

    page_roles: dict[int | None, str] = {}
    for page_number, page_elements in elements_by_page.items():
        page_text = "\n".join(
            element.text
            for element in page_elements
            if element.element_type not in {"heading", "header", "footer"}
        )
        assessment = classify_structural_role(page_text, {})
        if assessment.role in {"worksheet", "toc", "index"} and assessment.confidence >= 0.95:
            page_roles[page_number] = assessment.role

    if not page_roles:
        return elements
    return [
        replace(
            element,
            structural_role=page_roles[element.page_number],
            metadata={**element.metadata, "page_structural_role": page_roles[element.page_number]},
        )
        if element.page_number in page_roles and element.element_type not in {"heading", "header", "footer"}
        else element
        for element in elements
    ]


def _document_element_from_lines(
    lines: list[_PdfLine],
    *,
    source_path: Path,
    order_index: int,
    element_type: str,
    heading_level: int | None,
    context: tuple[str, str, str],
) -> DocumentElement:
    """Create one immutable document element from a validated layout block."""
    first = lines[0]
    join_with_newline = element_type in {"list_item", "table", "unknown"}
    text = _join_block_lines([line.text for line in lines], join_with_newline=join_with_newline)
    structural_role = classify_structural_role(
        text,
        {"chapter": context[1], "section": context[2]},
        element_type=element_type,
    ).role
    return DocumentElement(
        text=text,
        source_path=str(source_path),
        file_type="pdf",
        order_index=order_index,
        page_number=first.page_number,
        element_type=element_type,
        heading_level=heading_level,
        document_title=context[0],
        chapter=context[1],
        section=context[2],
        structural_role=structural_role,
        metadata={
            "extraction_provider": "pdfplumber",
            "line_index": first.line_index,
            "line_count": len(lines),
            "top": round(min(line.top for line in lines), 3),
            "bottom": round(max(line.bottom for line in lines), 3),
            "x0": round(min(line.x0 for line in lines), 3),
            "x1": round(max(line.x1 for line in lines), 3),
            "font_size": round(float(median([line.font_size for line in lines])), 3),
            "is_bold": sum(line.bold_fraction for line in lines) / len(lines) >= 0.6,
            "structural_position": first.structural_position,
            "block_id": first.block_id,
            "column_id": first.column_id,
            "layout_mode": first.layout_mode,
            "reading_order_confidence": round(min(line.reading_order_confidence for line in lines), 3),
            "layout_audit_required": any(line.layout_audit_required for line in lines),
        },
    )


def _join_block_lines(lines: list[str], *, join_with_newline: bool) -> str:
    """Join line wraps without crossing the already validated block boundary."""
    if join_with_newline:
        return "\n".join(lines)

    text = lines[0]
    for line in lines[1:]:
        previous_word = text.rsplit(maxsplit=1)[-1].removesuffix("-").casefold()
        if text.endswith("-") and line[:1].islower() and previous_word not in _PRESERVED_HYPHEN_PREFIXES:
            text = f"{text[:-1]}{line}"
        else:
            text = f"{text} {line}"
    return text


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
    if line.text.endswith((".", "?", "!", ";")) or _is_heading_noise(line.text):
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


def _is_heading_noise(text: str) -> bool:
    """Reject common OCR, form, numeric, and tasting-rating heading impostors."""
    normalized = _WHITESPACE_PATTERN.sub(" ", text).strip()
    if not normalized or _CID_PATTERN.fullmatch(normalized):
        return True
    if len(normalized) == 1 and normalized.isalpha():
        return True
    if _NUMERIC_SCALE_PATTERN.fullmatch(normalized) or _RATING_ROW_PATTERN.fullmatch(normalized):
        return True
    return bool(_FORM_LABEL_PATTERN.match(normalized))


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
    if not any(char.isalpha() for char in line.text) or _is_heading_noise(line.text):
        return "unknown"
    return "paragraph"


def _looks_like_table_row(text: str) -> bool:
    """Return whether text contains at least two explicit column separators."""
    return len([part for part in _TABLE_SEPARATOR_PATTERN.split(text) if part.strip()]) >= 3


def _normalized_line_key(text: str) -> str:
    """Normalize margin text while treating changing page numbers as equivalent."""
    normalized = _WHITESPACE_PATTERN.sub(" ", text).strip().casefold()
    return _DIGIT_PATTERN.sub("#", normalized)


def _is_finite_number(value: Any) -> bool:
    """Return whether a value is a finite int or float."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_positive_number(value: Any) -> bool:
    """Return whether a value is a finite positive int or float."""
    return _is_finite_number(value) and float(value) > 0
