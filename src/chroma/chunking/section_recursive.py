"""Deterministic block-aware chunking within document section boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chroma.extraction.base import DocumentElement
from src.chroma.metadata_extractor import extract_wine_metadata
from src.chroma.structural_roles import REJECTED_STRUCTURAL_ROLES, classify_structural_role

from .base import ChunkCandidate, DocumentChunker


_CONTENT_ELEMENT_TYPES = {"paragraph", "list_item", "table", "unknown"}
_HIGH_SIGNAL_METADATA_FIELDS = {
    "grape",
    "grapes",
    "appellation",
    "appellations",
    "classification",
    "classifications",
    "producer",
    "producers",
}
_OVERSIZED_BLOCK_SEPARATORS = [r"\n+", r"(?<=[.!?])\s+", r"\s+", ""]
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass
class _SectionGroup:
    """Ordered compatible blocks belonging to one structural section."""

    source_path: str
    file_type: str
    document_title: str
    chapter: str
    section: str
    entry_title: str
    structural_role: str
    column_id: str
    elements: list[DocumentElement] = field(default_factory=list)

    @property
    def heading_path(self) -> str:
        """Return a stable, de-duplicated structural heading path."""
        headings: list[str] = []
        for heading in (self.document_title, self.chapter, self.entry_title, self.section):
            normalized_heading = heading.strip()
            if normalized_heading and normalized_heading not in headings:
                headings.append(normalized_heading)
        return " > ".join(headings)


@dataclass(frozen=True)
class _ElementSpan:
    """Character range and page lineage for one section element."""

    start: int
    end: int
    element: DocumentElement


@dataclass(frozen=True)
class _TextUnit:
    """One complete source block or bounded fragment of an oversized block."""

    text: str
    element: DocumentElement


class SectionRecursiveChunker(DocumentChunker):
    """Pack compatible source blocks without crossing validated boundaries."""

    strategy = "section_recursive"

    def __init__(
        self,
        *,
        chunk_size: int = 1024,
        chunk_overlap: int = 256,
        min_chunk_chars: int = 200,
    ) -> None:
        """Configure deterministic character limits for section-local chunks."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if min_chunk_chars < 0:
            raise ValueError("min_chunk_chars must not be negative")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_chars = min_chunk_chars

    def chunk(self, elements: list[DocumentElement]) -> list[ChunkCandidate]:
        """Build block-aware chunks in deterministic document order."""
        candidates: list[ChunkCandidate] = []
        for group in _group_elements(elements):
            for units in _pack_group_elements(
                group.elements,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            ):
                packed_elements = [unit.element for unit in units]
                chunk_text, spans = _build_unit_text(units)
                candidate = _build_candidate(
                    group=group,
                    spans=spans,
                    text=chunk_text,
                    start=0,
                    end=len(chunk_text),
                    chunk_index=len(candidates),
                    strategy=self.strategy,
                    min_chunk_chars=self._min_chunk_chars,
                    chunk_elements=packed_elements,
                )
                if candidate is not None:
                    candidates.append(candidate)
        return candidates


def _group_elements(elements: list[DocumentElement]) -> list[_SectionGroup]:
    """Group contiguous content by every validated structural boundary."""
    groups: list[_SectionGroup] = []
    current_key: tuple[str, str, str, str, str, str, str, str] | None = None
    current_group: _SectionGroup | None = None

    for element in sorted(elements, key=lambda item: (item.source_path, item.order_index)):
        if element.element_type == "heading" or element.element_type not in _CONTENT_ELEMENT_TYPES:
            continue

        structural_role = _element_structural_role(element)
        if structural_role in REJECTED_STRUCTURAL_ROLES:
            current_key = None
            current_group = None
            continue

        entry_title = str(element.metadata.get("entry_title", "")).strip()
        column_id = _column_boundary(element)
        key = (
            element.source_path,
            element.file_type,
            element.document_title,
            element.chapter,
            element.section,
            entry_title,
            structural_role,
            column_id,
        )
        if key != current_key:
            current_group = _SectionGroup(
                source_path=element.source_path,
                file_type=element.file_type,
                document_title=element.document_title,
                chapter=element.chapter,
                section=element.section,
                entry_title=entry_title,
                structural_role=structural_role,
                column_id=column_id,
            )
            groups.append(current_group)
            current_key = key
        current_group.elements.append(element)

    return groups


def _element_structural_role(element: DocumentElement) -> str:
    """Return an extractor role or classify the isolated source block."""
    if element.structural_role != "unknown":
        return element.structural_role
    return classify_structural_role(
        element.text,
        {
            "document_title": element.document_title,
            "chapter": element.chapter,
            "section": element.section,
        },
        element_type=element.element_type,
    ).role


def _column_boundary(element: DocumentElement) -> str:
    """Return a stable column token only when layout extraction supplied one."""
    if "column_id" not in element.metadata:
        return ""
    return str(element.metadata["column_id"])


def _pack_group_elements(
    elements: list[DocumentElement],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[list[_TextUnit]]:
    """Pack whole blocks, using complete-unit overlap between bounded chunks."""
    units = [unit for element in elements for unit in _element_units(element, chunk_size)]
    chunks: list[list[_TextUnit]] = []
    current: list[_TextUnit] = []

    for unit in units:
        if current and _joined_length([*current, unit]) > chunk_size:
            chunks.append(current)
            current = _trailing_complete_units(current, chunk_overlap)
            while current and _joined_length([*current, unit]) > chunk_size:
                current.pop(0)
        current.append(unit)

    if current:
        chunks.append(current)
    return chunks


def _element_units(element: DocumentElement, chunk_size: int) -> list[_TextUnit]:
    """Keep normal blocks intact and split only a block that exceeds the limit."""
    if len(element.text) <= chunk_size:
        return [_TextUnit(text=element.text, element=element)]

    sentence_units = [part.strip() for part in _SENTENCE_BOUNDARY.split(element.text) if part.strip()]
    bounded_units: list[_TextUnit] = []
    for sentence in sentence_units:
        if len(sentence) <= chunk_size:
            bounded_units.append(_TextUnit(text=sentence, element=element))
            continue
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=_OVERSIZED_BLOCK_SEPARATORS,
            keep_separator=True,
            is_separator_regex=True,
        )
        bounded_units.extend(
            _TextUnit(text=part.strip(), element=element)
            for part in splitter.split_text(sentence)
            if part.strip()
        )
    return bounded_units


def _trailing_complete_units(units: list[_TextUnit], overlap: int) -> list[_TextUnit]:
    """Select complete trailing paragraphs or sentences within the overlap budget."""
    if overlap == 0:
        return []
    trailing: list[_TextUnit] = []
    for unit in reversed(units):
        candidate = [unit, *trailing]
        if _joined_length(candidate) > overlap:
            break
        trailing = candidate
    return trailing


def _joined_length(units: list[_TextUnit]) -> int:
    """Return the exact joined size for a list of text units."""
    return sum(len(unit.text) for unit in units) + max(0, len(units) - 1) * 2


def _build_unit_text(units: list[_TextUnit]) -> tuple[str, list[_ElementSpan]]:
    """Join packed units and preserve lineage for repeated oversized blocks."""
    parts: list[str] = []
    spans: list[_ElementSpan] = []
    cursor = 0
    for unit in units:
        if parts:
            cursor += 2
        start = cursor
        parts.append(unit.text)
        cursor += len(unit.text)
        spans.append(_ElementSpan(start=start, end=cursor, element=unit.element))
    return "\n\n".join(parts), spans


def _build_section_text(elements: list[DocumentElement]) -> tuple[str, list[_ElementSpan]]:
    """Join section elements for the optional semantic strategy."""
    return _build_unit_text([_TextUnit(text=element.text, element=element) for element in elements])


def _elements_for_range(spans: list[_ElementSpan], start: int, end: int) -> list[DocumentElement]:
    """Return source elements touched by a chunk character range."""
    return [span.element for span in spans if span.start < end and span.end > start]


def _build_candidate(
    *,
    group: _SectionGroup,
    spans: list[_ElementSpan],
    text: str,
    start: int,
    end: int,
    chunk_index: int,
    strategy: str,
    min_chunk_chars: int,
    chunk_elements: list[DocumentElement] | None = None,
) -> ChunkCandidate | None:
    """Build one candidate with shared filtering and source lineage."""
    chunk_elements = chunk_elements or _elements_for_range(spans, start, end)
    if len(text) < min_chunk_chars and not _is_high_signal(text, chunk_elements):
        return None

    pages = [element.page_number for element in chunk_elements if element.page_number is not None]
    start_page = min(pages) if pages else None
    end_page = max(pages) if pages else None
    element_types = {element.element_type for element in chunk_elements}
    element_type = next(iter(element_types)) if len(element_types) == 1 else ""
    role_assessment = classify_structural_role(
        text,
        {
            "document_title": group.document_title,
            "chapter": group.chapter,
            "section": group.section,
            "heading_path": group.heading_path,
            "structural_role": group.structural_role,
        },
        element_type=element_type,
    )
    return ChunkCandidate(
        text=text,
        source_path=group.source_path,
        file_type=group.file_type,
        chunk_index=chunk_index,
        page_number=start_page,
        start_page=start_page,
        end_page=end_page,
        document_title=group.document_title,
        chapter=group.chapter,
        section=group.section,
        heading_path=group.heading_path,
        chunking_strategy=strategy,
        extraction_provider=_extraction_provider(group.elements),
        structural_role=role_assessment.role,
        metadata=_lineage_metadata(group, chunk_elements),
    )


def _lineage_metadata(group: _SectionGroup, elements: list[DocumentElement]) -> dict[str, str | int | float | bool]:
    """Preserve scalar entry, column, and block lineage for indexed metadata."""
    metadata: dict[str, str | int | float | bool] = {}
    if group.entry_title:
        metadata["entry_title"] = group.entry_title
    if group.column_id:
        metadata["column_id"] = int(group.column_id) if group.column_id.isdigit() else group.column_id
    block_ids = [int(element.metadata["block_id"]) for element in elements if "block_id" in element.metadata]
    if block_ids:
        metadata["start_block_id"] = min(block_ids)
        metadata["end_block_id"] = max(block_ids)
    if any(bool(element.metadata.get("layout_audit_required")) for element in elements):
        metadata["layout_audit_required"] = True
    confidence_values = [
        float(element.metadata["reading_order_confidence"])
        for element in elements
        if isinstance(element.metadata.get("reading_order_confidence"), (int, float))
    ]
    if confidence_values:
        metadata["reading_order_confidence"] = min(confidence_values)
    return metadata


def _is_high_signal(text: str, elements: list[DocumentElement]) -> bool:
    """Identify short chunks carrying wine entities worth retaining."""
    wine_metadata = extract_wine_metadata(text)
    if wine_metadata.grapes or wine_metadata.appellations or wine_metadata.classifications or wine_metadata.producers:
        return True

    return any(
        key.casefold() in _HIGH_SIGNAL_METADATA_FIELDS and bool(value)
        for element in elements
        for key, value in element.metadata.items()
    )


def _extraction_provider(elements: list[DocumentElement]) -> str:
    """Preserve a provider name when extractors supplied one explicitly."""
    providers = {
        str(element.metadata.get("extraction_provider", "")).strip()
        for element in elements
        if element.metadata.get("extraction_provider")
    }
    return next(iter(providers)) if len(providers) == 1 else ""
