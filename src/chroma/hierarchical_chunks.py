"""Small-to-big retrieval: use small chunks for retrieval, return larger context.

This module implements the small-to-big retrieval pattern where:
- Small chunks (e.g., 256 chars) are used for embedding and retrieval
- Larger parent chunks (e.g., 1024 chars) are returned to the LLM

This improves retrieval precision while maintaining context quality.
"""
from typing import Any
from dataclasses import dataclass

from src.utils import logger


@dataclass
class HierarchicalChunk:
    """
    A chunk with both small (retrieval) and large (context) versions.

    Attributes:
        small_text: Small text for embedding/retrieval (high precision).
        large_text: Larger surrounding context for LLM (better understanding).
        chunk_id: Unique identifier for the chunk.
        metadata: Additional metadata about the chunk.
    """
    small_text: str
    large_text: str
    chunk_id: str
    metadata: dict[str, Any]


def create_hierarchical_chunks(
    text: str,
    small_chunk_size: int = 256,
    large_chunk_size: int = 1024,
    overlap: int = 64,
) -> list[HierarchicalChunk]:
    """
    Create hierarchical chunks with small retrieval chunks and larger context chunks.

    Each small chunk is associated with a larger parent chunk that provides
    more context. The small chunk is used for embedding and retrieval,
    while the large chunk is returned to the LLM.

    Args:
        text: Full text to chunk.
        small_chunk_size: Size of small chunks for retrieval (default: 256).
        large_chunk_size: Size of large chunks for context (default: 1024).
        overlap: Overlap between consecutive small chunks (default: 64).

    Returns:
        list of HierarchicalChunk objects.

    Raises:
        ValueError: If overlap >= small_chunk_size (would cause infinite loop).
    """
    if not text or not text.strip():
        return []

    # Prevent infinite loop: overlap must be less than chunk size
    if overlap >= small_chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than small_chunk_size ({small_chunk_size})")

    chunks = []
    text_len = len(text)

    # Calculate the context window around each small chunk
    context_padding = (large_chunk_size - small_chunk_size) // 2

    position = 0
    chunk_index = 0

    while position < text_len:
        # Extract small chunk
        small_end = min(position + small_chunk_size, text_len)
        small_text = text[position:small_end].strip()

        if not small_text:
            position += small_chunk_size - overlap
            continue

        # Extract large chunk (centered around small chunk)
        large_start = max(0, position - context_padding)
        large_end = min(text_len, small_end + context_padding)
        large_text = text[large_start:large_end].strip()

        chunk = HierarchicalChunk(
            small_text=small_text,
            large_text=large_text,
            chunk_id=f"chunk_{chunk_index}",
            metadata={
                "small_start": position,
                "small_end": small_end,
                "large_start": large_start,
                "large_end": large_end,
                "chunk_index": chunk_index,
            }
        )
        chunks.append(chunk)

        chunk_index += 1
        position += small_chunk_size - overlap

    logger.debug(f"Created {len(chunks)} hierarchical chunks")
    return chunks


def expand_to_parent_context(
    retrieved_docs: list[dict[str, Any]],
    use_large_context: bool = True,
) -> list[dict[str, Any]]:
    """
    Expand retrieved documents to their larger parent context.

    For documents that have a 'parent_context' field in metadata,
    replaces the document text with the larger context.

    Args:
        retrieved_docs: list of retrieved documents.
        use_large_context: Whether to use parent context (default: True).

    Returns:
        list of documents with expanded context.
    """

    expanded = []
    for doc in retrieved_docs:
        doc_copy = doc.copy()
        metadata = doc_copy.get("metadata", {})

        # Check if parent context is available
        parent_context = metadata.get("parent_context", "")
        if parent_context and use_large_context:
            doc_copy["document"] = parent_context
            doc_copy["used_parent_context"] = True
        else:
            doc_copy["used_parent_context"] = False

        expanded.append(doc_copy)

    return expanded


def prepare_chunks_for_indexing(
    hierarchical_chunks: list[HierarchicalChunk],
    file_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Prepare hierarchical chunks for indexing into ChromaDB.

    Stores the small chunk as the main document (for embedding) and
    the large chunk in metadata (for retrieval expansion).

    Args:
        hierarchical_chunks: list of HierarchicalChunk objects.
        file_metadata: Base metadata from the source file.

    Returns:
        list of chunk dictionaries ready for indexing.
    """
    chunks_for_index = []

    for chunk in hierarchical_chunks:
        chunk_metadata = {
            **file_metadata,
            **chunk.metadata,
            'parent_context': chunk.large_text,  # Store large context for expansion
            'small_chunk_size': len(chunk.small_text),
            'large_chunk_size': len(chunk.large_text),
        }

        chunks_for_index.append({
            'id': chunk.chunk_id,
            'text': chunk.small_text,  # Index small chunk for retrieval
            'metadata': chunk_metadata,
        })

    return chunks_for_index

