"""Context builder utility for formatting retrieved documents."""
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# Module-level embedder cache to avoid re-instantiation
_embedder_cache: Dict[str, HuggingFaceEmbeddings] = {}


def _get_embedder(model_name: str) -> HuggingFaceEmbeddings:
    """Get or create cached embedder instance."""
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = HuggingFaceEmbeddings(model_name=model_name)
    return _embedder_cache[model_name]


def build_context_from_chunks(
    retrieved_docs: List[Dict[str, Any]],
    include_metadata: bool = True,
    include_similarity: bool = False,
    max_chunks: int | None = None,
) -> str:
    """
    Build a formatted context string from retrieved document chunks.

    Args:
        retrieved_docs: List of retrieved documents from ChromaRetriever.
        include_metadata: Whether to include source metadata in context.
        include_similarity: Whether to include similarity scores.
        max_chunks: Optional limit on number of chunks to include.

    Returns:
        Formatted context string ready for LLM prompt.
    """
    if not retrieved_docs:
        return ""

    if max_chunks is not None:
        retrieved_docs = retrieved_docs[:max_chunks]

    context_parts = []
    for idx, doc in enumerate(retrieved_docs, 1):
        chunk_text = doc.get('document', '').strip()
        metadata = doc.get('metadata', {})
        similarity = doc.get('similarity')

        if not chunk_text:
            continue

        chunk_header = f"[Source {idx}"
        if include_metadata and metadata:
            # Extract useful metadata fields
            source = metadata.get('source', metadata.get('filename', ''))
            page = metadata.get('page', metadata.get('page_number'))
            chunk_id = metadata.get('chunk_id')

            if source:
                if '/' in source:
                    source = source.split('/')[-1]
                chunk_header += f" - {source}"

            if page is not None:
                chunk_header += f", Page {page}"

            if chunk_id is not None:
                chunk_header += f", Chunk {chunk_id}"

        if include_similarity and similarity is not None:
            chunk_header += f", Relevance: {similarity:.2f}"

        chunk_header += "]"
        context_parts.append(f"{chunk_header}\n{chunk_text}")

    if context_parts:
        return "\n\n---\n\n".join(context_parts)

    return ""


def build_semantic_context(
    retrieved_docs: List[Dict[str, Any]],
    similarity_threshold: float = 0.9,
    include_metadata: bool = True,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> str:
    """
    Build context while removing near-duplicate chunks using semantic similarity.

    Uses the deduplication module for consistent duplicate removal across
    the pipeline. Performs both hash-based and semantic deduplication.

    Args:
        retrieved_docs: List of retrieved documents from ChromaRetriever.
        similarity_threshold: Threshold for considering chunks as duplicates. Default is 0.9.
        include_metadata: Whether to include source metadata.
        embedding_model: HuggingFace model name for embeddings (should match retrieval's model).

    Returns:
        Formatted context string with duplicates removed.
    """
    if not retrieved_docs:
        return ""

    unique_docs = deduplicate_chunks(
        retrieved_docs,
        similarity_threshold=similarity_threshold,
        embedding_model=embedding_model,
    )

    return build_context_from_chunks(unique_docs, include_metadata=include_metadata)


def deduplicate_chunks(
    retrieved_docs: List[Dict[str, Any]],
    similarity_threshold: float = 0.9,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """Return the chunks retained by production semantic deduplication.

    Args:
        retrieved_docs: Retrieved documents to deduplicate.
        similarity_threshold: Threshold for considering chunks duplicates.
        embedding_model: Embedding model used for semantic similarity.

    Returns:
        Deduplicated document dictionaries in retained order.
    """
    if not retrieved_docs:
        return []

    from src.chroma.deduplication import deduplicate_context

    return deduplicate_context(
        retrieved_docs,
        similarity_threshold=similarity_threshold,
        embedding_model=embedding_model,
        use_hash_first=True,
    )


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score between 0 and 1.
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)

    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def format_sources_for_display(retrieved_docs: List[Dict[str, Any]]) -> list:
    """
    Format retrieved sources for UI display.

    Args:
        retrieved_docs: List of retrieved documents from ChromaRetriever.

    Returns:
        A list of tuples of the form (source name, page number, relevance score).
    """
    sources = []

    for idx, doc in enumerate(retrieved_docs, 1):
        metadata = doc.get('metadata', {})
        similarity = doc.get('similarity')

        source = metadata.get('source', metadata.get('filename', 'Unknown'))
        if '/' in source:
            source = source.split('/')[-1]
        source = Path(source).stem

        page = metadata.get('page', metadata.get('page_number'))
        sources.append((source, page, similarity))

    return sources


if __name__ == "__main__":
    # Example usage
    sample_docs = [
        {
            'document': "This is the content of chunk one.",
            'metadata': {'source': 'doc1.pdf', 'page': 1, 'chunk_id': 0},
            'similarity': 0.95
        },
        {
            'document': "This is the content of chunk two.",
            'metadata': {'source': 'doc1.pdf', 'page': 2, 'chunk_id': 1},
            'similarity': 0.90
        },
        {
            'document': "This is the content of chunk one.",  # Duplicate content
            'metadata': {'source': 'doc2.pdf', 'page': 1, 'chunk_id': 0},
            'similarity': 0.85
        }
    ]

    context = build_semantic_context(sample_docs)
    print("Formatted Context:\n", context)

    sources = format_sources_for_display(sample_docs)
    print("\nFormatted Sources for Display:")
    for src in sources:
        print(src)
