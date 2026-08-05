"""Tests for explicit extraction-provider registry behavior."""

import pytest

from src.chroma.extraction import (
    EbookLibExtractor,
    ExtractorRegistry,
    PdfPlumberExtractor,
    UnsupportedDocumentTypeError,
)
from src.utils import get_config


def test_registry_resolves_reviewed_providers() -> None:
    """Only the reviewed PDF and EPUB providers should resolve."""
    assert isinstance(ExtractorRegistry.resolve(".pdf", "pdfplumber"), PdfPlumberExtractor)
    assert isinstance(ExtractorRegistry.resolve("EPUB", "ebooklib"), EbookLibExtractor)


def test_registry_resolves_application_extraction_config() -> None:
    """The reviewed application defaults should construct both local providers."""
    extraction_config = get_config().chroma.extraction

    assert isinstance(ExtractorRegistry.resolve_from_config("pdf", extraction_config), PdfPlumberExtractor)
    assert isinstance(ExtractorRegistry.resolve_from_config("epub", extraction_config), EbookLibExtractor)


def test_registry_applies_margin_configuration() -> None:
    """PDF structural switches should be passed explicitly to the provider."""
    extractor = ExtractorRegistry.resolve_from_config(
        "pdf",
        {
            "pdf_provider": "pdfplumber",
            "strip_repeated_headers": False,
            "strip_repeated_footers": False,
        },
    )

    assert isinstance(extractor, PdfPlumberExtractor)
    assert extractor._strip_repeated_headers is False
    assert extractor._strip_repeated_footers is False


def test_registry_returns_none_for_unsupported_file_when_configured() -> None:
    """Unsupported files may be skipped only through the explicit config switch."""
    extractor = ExtractorRegistry.resolve_from_config(
        ".docx",
        {"fail_on_unsupported_file": False},
    )

    assert extractor is None


def test_registry_raises_for_unsupported_file_when_configured() -> None:
    """Strict unsupported-file behavior should raise the normalized exception."""
    with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported document type"):
        ExtractorRegistry.resolve_from_config(
            ".docx",
            {"fail_on_unsupported_file": True},
        )


def test_registry_rejects_invalid_provider_without_fallback() -> None:
    """A configured provider typo must not silently select a different adapter."""
    with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported extraction provider"):
        ExtractorRegistry.resolve_from_config(
            "pdf",
            {"pdf_provider": "unknown-provider"},
        )


def test_registry_requires_explicit_provider_setting() -> None:
    """Supported formats should require their provider key in config."""
    with pytest.raises(ValueError, match="pdf_provider"):
        ExtractorRegistry.resolve_from_config("pdf", {})
