"""Document parsing and chunking strategies for ChromaDB ingestion.

Supports three strategies: basic (fixed-size), by_title (structure-aware),
and semantic (embedding-based breakpoints). Extracts wine-specific metadata
from each chunk when enabled.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import List

from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

from langchain_experimental.text_splitter import SemanticChunker

from .utils import split_text_into_sentences
from .metadata_extractor import extract_wine_metadata, extract_document_context
from src.utils import logger, generate_hash, get_embedder


@dataclass
class ChunkMetadata:
    """
    Dataclass for chunk metadata.

    Includes standard document metadata plus wine-specific fields
    for improved retrieval filtering.
    """
    # Standard document metadata
    filename: str
    file_path: str
    file_type: str
    chunk_index: int
    chunk_id: str
    content_hash: str
    page_number: int = -1
    language: str = "unknown"
    category: str = "unknown"
    topic: str = "unknown"
    summary: str = "none"
    word_count: int = 0
    char_count: int = 0

    # Document context (for contextual retrieval)
    document_title: str = ""
    chapter: str = ""
    section: str = ""

    # Wine-specific metadata
    grapes: str = ""  # Comma-separated list of grape varieties
    regions: str = ""  # Comma-separated list of wine regions
    vintages: str = ""  # Comma-separated list of vintage years
    classifications: str = ""  # Comma-separated list (DOCG, AOC, etc.)
    producers: str = ""  # Comma-separated list of producer/winery names
    appellations: str = ""  # Comma-separated list of wine appellations


def semantic_chunking(
    content: str,
    embedding_model: str | None = None,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0,
) -> List[str]:
    """
    Create semantic chunks using LangChain's SemanticChunker.

    Args:
        content: Text content to chunk.
        embedding_model: HuggingFace model name for embeddings. If not provided explicitly, will use the default
            model from app config.
        breakpoint_threshold_type: How to determine breakpoints. Options:
            - "percentile": Break at percentile threshold of distances
            - "standard_deviation": Break at standard deviation threshold
            - "interquartile": Break at interquartile range threshold
        breakpoint_threshold_amount: Threshold value for breakpoint detection.
            For percentile: 95.0 means break at 95th percentile of distances.

    Returns:
        List of chunk text strings.
    """
    try:
        embedder = get_embedder(embedding_model)
        chunker = SemanticChunker(
            embeddings=embedder,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

        documents = chunker.create_documents([content])
        chunks = [doc.page_content for doc in documents]

        return chunks

    except Exception as e:
        logger.error(f"Error in semantic chunking, falling back to simple split: {e}")
        return split_text_into_sentences(content)



def split_file(
    filepath: str | Path,
    strategy: str = "basic",
    chunk_size: int = 512,
    overlap_size: int = 128,
    embedding_model: str | None = None,
    extract_metadata: bool = True,
    **kwargs,
) -> list[dict]:
    """
    Split a file into chunks using specified strategy and enrich with metadata.

    Args:
        filepath: Path to the file to split.
        strategy: Chunking strategy ("basic", "by_title", "semantic").
        chunk_size: Maximum chunk size (used for basic/by_title strategies).
        overlap_size: Overlap between chunks (used for basic/by_title strategies).
        embedding_model: HuggingFace model for semantic chunking (used for semantic strategy).
            If not provided, will use default from app config.
        extract_metadata: Whether to extract wine-specific metadata from chunks.
        **kwargs: Additional arguments for the chunking function.

    Returns:
        List of enhanced chunk dictionaries with metadata.
    """
    filepath = Path(filepath)
    chunks = []

    try:
        logger.info(f"Processing file: {filepath.name}")
        elements = partition(filename=str(filepath))
        doc_context = extract_document_context(elements)

        if strategy == "semantic":
            full_text = "\n".join([str(elem) for elem in elements])
            semantic_chunks = semantic_chunking(
                content=full_text,
                embedding_model=embedding_model,
                breakpoint_threshold_type=kwargs.get("breakpoint_threshold_type", "percentile"),
                breakpoint_threshold_amount=kwargs.get("breakpoint_threshold_amount", 95.0),
            )

            for i, chunk_text in enumerate(semantic_chunks):
                chunk_id = f"{filepath.stem}_{i}_{generate_hash(chunk_text)[:8]}"
                wine_meta = extract_wine_metadata(chunk_text) if extract_metadata else None

                metadata = ChunkMetadata(
                    filename=filepath.name,
                    file_path=str(filepath),
                    file_type=filepath.suffix.lower(),
                    chunk_index=i,
                    chunk_id=chunk_id,
                    content_hash=generate_hash(chunk_text),
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    document_title=doc_context.get("document_title", ""),
                    chapter=doc_context.get("chapter", ""),
                    section=doc_context.get("section", ""),
                    grapes=",".join(wine_meta.grapes) if wine_meta else "",
                    regions=",".join(wine_meta.regions) if wine_meta else "",
                    vintages=",".join(wine_meta.vintages) if wine_meta else "",
                    classifications=",".join(wine_meta.classifications) if wine_meta else "",
                    producers=",".join(wine_meta.producers) if wine_meta else "",
                    appellations=",".join(wine_meta.appellations) if wine_meta else "",
                )

                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": metadata.__dict__,
                    "importance_score": 1.0
                })
        else:
            if strategy == "basic":
                unstructured_chunks = chunk_elements(
                    elements, max_characters=chunk_size, overlap=overlap_size, **kwargs
                )
            elif strategy == "by_title":
                unstructured_chunks = chunk_by_title(
                    elements, max_characters=chunk_size, overlap=overlap_size, **kwargs
                )
            else:
                raise ValueError(f"Unknown chunking strategy: {strategy}")

            for i, chunk in enumerate(unstructured_chunks):
                chunk_text = str(chunk)
                chunk_id = f"{filepath.stem}_{i}_{generate_hash(chunk_text)[:8]}"

                # Extract metadata from unstructured chunk
                chunk_metadata = {}
                if hasattr(chunk, "metadata") and chunk.metadata:
                    if hasattr(chunk.metadata, "to_dict"):
                        chunk_metadata = chunk.metadata.to_dict()
                    else:
                        chunk_metadata = chunk.metadata.__dict__

                # Extract wine metadata if enabled
                wine_meta = extract_wine_metadata(chunk_text) if extract_metadata else None

                metadata = ChunkMetadata(
                    filename=filepath.name,
                    file_path=str(filepath),
                    file_type=filepath.suffix.lower(),
                    chunk_index=i,
                    chunk_id=chunk_id,
                    content_hash=generate_hash(chunk_text),
                    page_number=chunk_metadata.get("page_number", -1),
                    language=chunk_metadata.get("languages", ["unknown"])[0],
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    document_title=doc_context.get("document_title", ""),
                    chapter=doc_context.get("chapter", ""),
                    section=doc_context.get("section", ""),
                    grapes=",".join(wine_meta.grapes) if wine_meta else "",
                    regions=",".join(wine_meta.regions) if wine_meta else "",
                    vintages=",".join(wine_meta.vintages) if wine_meta else "",
                    classifications=",".join(wine_meta.classifications) if wine_meta else "",
                    producers=",".join(wine_meta.producers) if wine_meta else "",
                    appellations=",".join(wine_meta.appellations) if wine_meta else "",
                )

                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": metadata.__dict__,
                    "importance_score": 1.0  # Default importance
                })

        logger.info(f"Generated {len(chunks)} chunks from {filepath.name}")
        return chunks

    except Exception as e:
        logger.error(f"Error processing file {filepath}: {e}")
        return []
