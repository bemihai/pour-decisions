"""Retrieval pipeline for querying the ChromaDB vector store.

Provides hybrid search (vector + BM25), cross-encoder reranking, query
analysis with metadata filtering, context compression, and context building.
"""

from .vector_retriever import ChromaRetriever
from .query_utils import normalize_query, expand_query
from .keyword_search import BM25Index
from .hybrid_retriever import HybridRetriever
from .factory import build_retriever_from_config
from .reranker import DocumentReranker
from .query_compression import compress_context
from .query_analyzer import analyze_query, boost_by_metadata_match, QueryAnalysis
from .context_builder import (
    build_context_from_chunks,
    build_semantic_context,
    deduplicate_chunks,
    format_sources_for_display,
)
from .rag_service import (
    RAGChunkArtifact,
    RAGExecutionResult,
    RAGFeatureUsage,
    RAGSourceArtifact,
    execute_production_rag,
)

__all__ = [
    "ChromaRetriever",
    "normalize_query",
    "expand_query",
    "BM25Index",
    "HybridRetriever",
    "build_retriever_from_config",
    "DocumentReranker",
    "compress_context",
    "analyze_query",
    "boost_by_metadata_match",
    "QueryAnalysis",
    "build_context_from_chunks",
    "build_semantic_context",
    "deduplicate_chunks",
    "format_sources_for_display",
    "RAGChunkArtifact",
    "RAGExecutionResult",
    "RAGFeatureUsage",
    "RAGSourceArtifact",
    "execute_production_rag",
]
