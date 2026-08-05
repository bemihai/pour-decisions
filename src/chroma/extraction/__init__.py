"""Provider-neutral document extraction contracts."""

from .base import DocumentElement, DocumentExtractor, UnsupportedDocumentTypeError
from .ebooklib_extractor import EbookLibExtractor
from .pdfplumber_extractor import PdfPlumberExtractor
from .registry import ExtractorRegistry

__all__ = [
    "DocumentElement",
    "DocumentExtractor",
    "EbookLibExtractor",
    "ExtractorRegistry",
    "PdfPlumberExtractor",
    "UnsupportedDocumentTypeError",
]
