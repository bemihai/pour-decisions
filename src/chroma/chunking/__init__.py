"""Provider-neutral document chunking contracts and strategies."""

from .base import ChunkCandidate, DocumentChunker
from .section_recursive import SectionRecursiveChunker

__all__ = ["ChunkCandidate", "DocumentChunker", "SectionRecursiveChunker"]
