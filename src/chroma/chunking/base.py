"""Provider-neutral contracts for building retrieval chunks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.chroma.extraction.base import DocumentElement


@dataclass(frozen=True)
class ChunkCandidate:
    """One normalized retrieval chunk produced from document elements."""

    text: str
    source_path: str
    file_type: str
    chunk_index: int
    page_number: int | None = None
    start_page: int | None = None
    end_page: int | None = None
    document_title: str = ""
    chapter: str = ""
    section: str = ""
    heading_path: str = ""
    chunking_strategy: str = ""
    extraction_provider: str = ""
    structural_role: str = "unknown"
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize chunk text and reject blank required string fields."""
        normalized_text = _require_non_empty_string(self.text, field_name="text")
        _require_non_empty_string(self.source_path, field_name="source_path")
        _require_non_empty_string(self.file_type, field_name="file_type")
        object.__setattr__(self, "text", normalized_text)


class DocumentChunker(ABC):
    """Build normalized retrieval chunks from extracted document elements."""

    @abstractmethod
    def chunk(self, elements: list[DocumentElement]) -> list[ChunkCandidate]:
        """Build retrieval chunks from normalized document elements."""


def _require_non_empty_string(value: str, *, field_name: str) -> str:
    """Return a stripped required string or raise a clear validation error."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must not be empty")
    return normalized_value
