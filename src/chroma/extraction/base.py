"""Provider-neutral contracts for extracting structured document content."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


class UnsupportedDocumentTypeError(ValueError):
    """Raised when no configured extractor supports a document type."""


@dataclass(frozen=True)
class DocumentElement:
    """One normalized content element emitted by a document extractor.

    Provider-native values belong in ``metadata`` so downstream chunking and
    indexing code can depend only on this stable contract.
    """

    text: str
    source_path: str
    file_type: str
    order_index: int
    page_number: int | None = None
    element_type: str = "paragraph"
    heading_level: int | None = None
    document_title: str = ""
    chapter: str = ""
    section: str = ""
    structural_role: str = "unknown"
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize text and reject blank required string fields."""
        normalized_text = _require_non_empty_string(self.text, field_name="text")
        _require_non_empty_string(self.source_path, field_name="source_path")
        _require_non_empty_string(self.file_type, field_name="file_type")
        object.__setattr__(self, "text", normalized_text)


class DocumentExtractor(ABC):
    """Extract normalized elements from one supported document format."""

    @abstractmethod
    def extract(self, path: Path) -> list[DocumentElement]:
        """Extract normalized document elements from one source file."""


def _require_non_empty_string(value: str, *, field_name: str) -> str:
    """Return a stripped required string or raise a clear validation error."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must not be empty")
    return normalized_value
