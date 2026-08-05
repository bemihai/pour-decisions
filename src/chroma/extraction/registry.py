"""Explicit extraction-provider resolution by file type and configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import DocumentExtractor, UnsupportedDocumentTypeError
from .ebooklib_extractor import EbookLibExtractor
from .pdfplumber_extractor import PdfPlumberExtractor


class ExtractorRegistry:
    """Resolve only the extraction providers reviewed for Milestone 3."""

    @classmethod
    def resolve(
        cls,
        file_type: str,
        provider: str,
        *,
        strip_repeated_headers: bool = True,
        strip_repeated_footers: bool = True,
    ) -> DocumentExtractor:
        """Construct one extractor for an explicit file-type/provider pair."""
        normalized_file_type = _normalize_file_type(file_type)
        normalized_provider = provider.strip().casefold()
        if normalized_file_type == "pdf" and normalized_provider == "pdfplumber":
            return PdfPlumberExtractor(
                strip_repeated_headers=strip_repeated_headers,
                strip_repeated_footers=strip_repeated_footers,
            )
        if normalized_file_type == "epub" and normalized_provider == "ebooklib":
            return EbookLibExtractor()
        raise UnsupportedDocumentTypeError(
            f"Unsupported extraction provider {provider!r} for file type {normalized_file_type!r}"
        )

    @classmethod
    def resolve_from_config(
        cls,
        file_type: str,
        extraction_config: Mapping[str, Any] | Any,
    ) -> DocumentExtractor | None:
        """Resolve an extractor from the explicit ``chroma.extraction`` section."""
        normalized_file_type = _normalize_file_type(file_type)
        fail_on_unsupported = bool(_config_value(extraction_config, "fail_on_unsupported_file", False))
        provider_key = {"pdf": "pdf_provider", "epub": "epub_provider"}.get(normalized_file_type)
        if provider_key is None:
            if fail_on_unsupported:
                raise UnsupportedDocumentTypeError(f"Unsupported document type: {normalized_file_type!r}")
            return None

        provider = _config_value(extraction_config, provider_key, None)
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError(f"chroma.extraction.{provider_key} must be a non-empty string")
        return cls.resolve(
            normalized_file_type,
            provider,
            strip_repeated_headers=bool(_config_value(extraction_config, "strip_repeated_headers", True)),
            strip_repeated_footers=bool(_config_value(extraction_config, "strip_repeated_footers", True)),
        )


def _normalize_file_type(file_type: str) -> str:
    """Normalize a suffix or file-type value for registry lookup."""
    if not isinstance(file_type, str):
        raise TypeError("file_type must be a string")
    normalized = file_type.strip().casefold().removeprefix(".")
    if not normalized:
        raise UnsupportedDocumentTypeError("Document type must not be empty")
    return normalized


def _config_value(config: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    """Read one value from a mapping or OmegaConf-like config object."""
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)
