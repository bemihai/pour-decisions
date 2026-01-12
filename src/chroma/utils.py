"""Utility functions for ChromaDB operations."""

from datetime import datetime
from typing import Any

import chromadb as cdb
from chromadb.errors import NotFoundError

from src.utils import logger

__all__ = [
    "create_batches",
    "get_or_create_collection",
    "validate_chunks",
    "get_collection_stats",
    "get_all_stats",
    "split_text_into_sentences",
]


def create_batches(
        ids: list,
        embeddings: list | None = None,
        metadata: list | None = None,
        documents: list | None = None,
        batch_size: int = 1000,
) -> list[tuple]:
    """Create batches for ChromaDB to avoid FastAPI errors with large collections."""
    batches = []
    if len(ids) > batch_size:
        for i in range(0, len(ids), batch_size):
            batches.append(
                (
                    ids[i : i + batch_size],
                    embeddings[i : i + batch_size] if embeddings else None,
                    metadata[i: i + batch_size] if metadata else None,
                    documents[i : i + batch_size] if documents else None,
                )
            )
    else:
        batches.append((ids, embeddings, metadata, documents))

    return batches


def get_or_create_collection(client: cdb.ClientAPI, name: str, metadata: dict | None = None) -> cdb.Collection:
    """
    Get or create a ChromaDB collection.

    Args:
        client: An instance of the ChromaDB client.
        name: The name of the collection.
        metadata: Optional metadata for the collection.

    Returns:
         An instance of the ChromaDB collection.
    """
    try:
        collection = client.get_collection(name)
        logger.info(f"Using existing collection: {name}")
    except NotFoundError as _:
        metadata = metadata or {}
        metadata["created"] = str(datetime.now())
        collection = client.create_collection(name=name, metadata=dict(metadata))
        logger.info(f"Created new collection: {name}")

    return collection


def validate_chunks(chunks: list[dict]) -> list[dict]:
    """
    Validate and filter chunks for quality.

    Args:
        chunks: List of chunk dictionaries with 'text' keys.

    Returns:
        List of validated chunk dictionaries.
    """
    valid_chunks = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()

        # Skip empty or very short chunks
        if len(text) < 10:
            continue

        # Skip chunks with too little actual content
        if len(text.split()) < 3:
            continue

        valid_chunks.append(chunk)

    logger.info(f"Validated {len(valid_chunks)} chunks from {len(chunks)} total")
    return valid_chunks


def get_collection_stats(client: cdb.ClientAPI, collection_name: str) -> dict[str, Any]:
    """
    Get detailed statistics for a ChromaDB collection.

    Args:
        client: ChromaDB client instance.
        collection_name: Name of the collection to analyze.

    Returns:
        Dictionary with collection statistics.
    """
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        return {"error": str(e), "name": collection_name}

    count = collection.count()
    stats = {
        "name": collection_name,
        "record_count": count,
        "metadata": dict(collection.metadata) if collection.metadata else {},
    }

    if count == 0:
        stats["embedding_dimension"] = "N/A (empty collection)"
        return stats

    # Sample records to get embedding dimension and metadata stats
    sample_size = min(100, count)
    sample = collection.get(
        limit=sample_size,
        include=["embeddings", "metadatas", "documents"]
    )

    # Get embedding dimension
    if sample["embeddings"] is not None and len(sample["embeddings"]) > 0:
        stats["embedding_dimension"] = len(sample["embeddings"][0])
    else:
        stats["embedding_dimension"] = "N/A"

    # Document length stats
    if sample["documents"]:
        doc_lengths = [len(doc) for doc in sample["documents"] if doc]
        if doc_lengths:
            stats["avg_document_length"] = sum(doc_lengths) // len(doc_lengths)
            stats["min_document_length"] = min(doc_lengths)
            stats["max_document_length"] = max(doc_lengths)

    # Metadata field analysis
    if sample["metadatas"]:
        metadata_fields = set()
        for meta in sample["metadatas"]:
            if meta is not None:
                metadata_fields = metadata_fields.union(set(meta.keys()))

        stats["metadata_fields"] = metadata_fields

    return stats


def get_all_stats(client: cdb.ClientAPI) -> list[dict[str, Any]]:
    """Get statistics for all collections in ChromaDB."""
    collections = client.list_collections()
    return [get_collection_stats(client, col.name) for col in collections]


def split_text_into_sentences(content: str) -> list[str]:
    """
    Split text content into sentences.

    Args:
        content: The text content to split.

    Returns:
        List of text chunks.
    """
    if not content or not content.strip():
        return [content]

    sentences = content.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > 1000:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
        else:
            current_chunk += sentence + ". "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [content]