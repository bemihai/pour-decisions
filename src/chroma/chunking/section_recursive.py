"""Deterministic recursive chunking within document section boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chroma.extraction.base import DocumentElement
from src.chroma.metadata_extractor import extract_wine_metadata

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
_SPLIT_SEPARATORS = [r"\n\n+", r"(?<=[.!?])\s+", r"\s+", ""]


@dataclass
class _SectionGroup:
    """Ordered content and context belonging to one structural section."""

    source_path: str
    file_type: str
    document_title: str
    chapter: str
    section: str
    elements: list[DocumentElement] = field(default_factory=list)

    @property
    def heading_path(self) -> str:
        """Return a stable, de-duplicated structural heading path."""
        headings: list[str] = []
        for heading in (self.document_title, self.chapter, self.section):
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


class SectionRecursiveChunker(DocumentChunker):
    """Split document content recursively without crossing section boundaries."""

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

        self._min_chunk_chars = min_chunk_chars
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=_SPLIT_SEPARATORS,
            keep_separator=True,
            is_separator_regex=True,
            add_start_index=True,
        )

    def chunk(self, elements: list[DocumentElement]) -> list[ChunkCandidate]:
        """Build section-local chunks in deterministic document order."""
        candidates: list[ChunkCandidate] = []
        for group in _group_elements(elements):
            section_text, spans = _build_section_text(group.elements)
            if not section_text:
                continue

            for document in self._splitter.create_documents([section_text]):
                chunk_text = document.page_content.strip()
                start_index = int(document.metadata.get("start_index", 0))
                candidate = _build_candidate(
                    group=group,
                    spans=spans,
                    text=chunk_text,
                    start=start_index,
                    end=start_index + len(chunk_text),
                    chunk_index=len(candidates),
                    strategy=self.strategy,
                    min_chunk_chars=self._min_chunk_chars,
                )
                if candidate is not None:
                    candidates.append(candidate)
        return candidates


def _group_elements(elements: list[DocumentElement]) -> list[_SectionGroup]:
    """Group ordered content by source and structural heading context."""
    groups: list[_SectionGroup] = []
    current_key: tuple[str, str, str, str, str] | None = None
    current_group: _SectionGroup | None = None

    for element in sorted(elements, key=lambda item: (item.source_path, item.order_index)):
        if element.element_type == "heading" or element.element_type not in _CONTENT_ELEMENT_TYPES:
            continue

        key = (
            element.source_path,
            element.file_type,
            element.document_title,
            element.chapter,
            element.section,
        )
        if key != current_key:
            current_group = _SectionGroup(
                source_path=element.source_path,
                file_type=element.file_type,
                document_title=element.document_title,
                chapter=element.chapter,
                section=element.section,
            )
            groups.append(current_group)
            current_key = key
        current_group.elements.append(element)

    return groups


def _build_section_text(elements: list[DocumentElement]) -> tuple[str, list[_ElementSpan]]:
    """Join section elements with paragraph boundaries and record their spans."""
    parts: list[str] = []
    spans: list[_ElementSpan] = []
    cursor = 0
    for element in elements:
        if parts:
            cursor += 2
        start = cursor
        parts.append(element.text)
        cursor += len(element.text)
        spans.append(_ElementSpan(start=start, end=cursor, element=element))
    return "\n\n".join(parts), spans


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
) -> ChunkCandidate | None:
    """Build one candidate with shared filtering and source lineage."""
    chunk_elements = _elements_for_range(spans, start, end)
    if len(text) < min_chunk_chars and not _is_high_signal(text, chunk_elements):
        return None

    pages = [element.page_number for element in chunk_elements if element.page_number is not None]
    start_page = min(pages) if pages else None
    end_page = max(pages) if pages else None
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
    )


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
