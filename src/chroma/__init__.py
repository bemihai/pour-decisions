"""Document processing, chunking, and ChromaDB indexing pipeline.

Handles the complete ingestion path from raw documents (PDF, EPUB) to indexed
embeddings in ChromaDB, including wine-specific metadata extraction, incremental
indexing, content deduplication, and collection statistics.
"""

from .chunks import split_file
from .deduplication import deduplicate_by_content_hash, deduplicate_chunks, deduplicate_context
from .hierarchical_chunks import HierarchicalChunk, create_hierarchical_chunks
from .index_tracker import IndexTracker
from .metadata_extractor import extract_wine_metadata, extract_document_context
from .loader import CollectionDataLoader
from .stats import get_collection_stats, get_all_stats
from .utils import *
