"""RAG Pipeline Quickstart - Comprehensive Testing Script

This script tests all components of the Pour Decisions RAG pipeline:
- Query preprocessing (wine terminology normalization, expansion)
- Hybrid search (vector + BM25 with RRF fusion)
- Cross-encoder reranking
- Query analysis for metadata boosting
- Context compression
- Semantic deduplication
- Small-to-big retrieval (if enabled)

Uses a dedicated test collection to avoid interfering with production data.

Usage:
    python scripts/rag_quickstart
    PYTHONPATH=$(pwd) python scripts/rag_quickstart.py

Prerequisites:
    - ChromaDB must be running (make chroma-up)
    - Test data must be indexed first: python scripts/chroma_quickstart
"""
import os
import time
from typing import List, Dict, Any

import chromadb

from src.chroma.hierarchical_chunks import expand_to_parent_context
from src.chroma.bm25_builder import read_collection_documents
from src.utils import get_config, logger
from src.agents.llm import load_base_model

from src.retrieval import (
    ChromaRetriever,
    HybridRetriever,
    DocumentReranker,
    normalize_query,
    expand_query,
    analyze_query,
    boost_by_metadata_match,
    compress_context,
    build_semantic_context,
    build_context_from_chunks,
    BM25Index
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Test collection name - separate from production
TEST_COLLECTION = "wine_test"


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def check_prerequisites():
    """Check if prerequisites are met."""
    print_section("Prerequisites Check")

    # Check ChromaDB connection
    try:
        cfg = get_config()
        client = chromadb.HttpClient(
            host=cfg.chroma.client.host,
            port=cfg.chroma.client.port
        )
        print(f"✓ ChromaDB connected at {cfg.chroma.client.host}:{cfg.chroma.client.port}")
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        print(f"\nPlease start ChromaDB with: make chroma-up")
        return False

    # Check if test collection exists
    try:
        collections = client.list_collections()
        collection_names = [col.name for col in collections]

        if TEST_COLLECTION not in collection_names:
            print(f"❌ Test collection '{TEST_COLLECTION}' not found")
            print(f"\nPlease index test data first:")
            print(f"  python -m src.chroma.chroma_quickstart")
            return False

        # Check collection has data
        collection = client.get_collection(TEST_COLLECTION)
        count = collection.count()

        if count == 0:
            print(f"❌ Test collection '{TEST_COLLECTION}' is empty")
            print(f"\nPlease index test data first:")
            print(f"  python scripts/chroma_quickstart")
            return False

        print(f"✓ Test collection '{TEST_COLLECTION}' found with {count:,} documents")

    except Exception as e:
        print(f"❌ Error checking test collection: {e}")
        return False

    return True


def test_query_preprocessing(query: str):
    """Test query preprocessing with wine terminology."""
    print_section("1. Query Preprocessing")

    print(f"Original query: '{query}'")

    # Normalize
    normalized = normalize_query(query)
    print(f"Normalized:     '{normalized}'")

    # Expand
    expanded = expand_query(normalized)
    print(f"Expanded:       '{expanded}'")

    # Analyze for metadata
    analysis = analyze_query(query)
    if analysis.has_filters:
        print(f"\nDetected entities:")
        if analysis.grapes:
            print(f"  Grapes: {', '.join(analysis.grapes)}")
        if analysis.regions:
            print(f"  Regions: {', '.join(analysis.regions)}")
        if analysis.vintages:
            print(f"  Vintages: {', '.join(analysis.vintages)}")
        if analysis.appellations:
            print(f"  Appellations: {', '.join(analysis.appellations)}")

    return expanded, analysis


def test_retrieval(query: str, cfg, use_hybrid: bool = True):
    """Test retrieval with both vector and hybrid search."""
    print_section(f"2. Retrieval ({'Hybrid' if use_hybrid else 'Vector Only'})")

    # Initialize ChromaDB client
    client = chromadb.HttpClient(
        host=cfg.chroma.client.host,
        port=cfg.chroma.client.port
    )

    # Initialize vector retriever with test collection
    vector_retriever = ChromaRetriever(
        client=client,
        collection_name=TEST_COLLECTION,
        embedding_model=cfg.chroma.settings.embedder,
    )

    # Initialize retrievers
    if use_hybrid:
        print("Using hybrid retrieval (Vector + BM25 with RRF fusion)")

        # Initialize BM25 index
        print("Building BM25 index from test collection...")
        bm25_index = BM25Index()

        # Fetch documents from collection to build BM25 index
        collection = vector_retriever.collection
        documents = read_collection_documents(collection)

        if documents:
            bm25_index.build_index(documents)
            print(f"Built BM25 index with {len(documents)} documents")
        else:
            print("Warning: No documents found to build BM25 index")

        # Initialize hybrid retriever
        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_index=bm25_index,
            semantic_candidate_pool=cfg.chroma.retrieval.semantic_candidate_pool,
            bm25_candidate_pool=cfg.chroma.retrieval.bm25_candidate_pool,
            reranker_input_limit=cfg.chroma.retrieval.reranker_input_limit,
        )
    else:
        print("Using vector-only retrieval")
        retriever = vector_retriever

    # Retrieve
    n_results = 10  # Get more for reranking
    start_time = time.time()
    docs = retriever.retrieve(query, n_results=n_results)
    retrieval_time = time.time() - start_time

    print(f"Retrieved {len(docs)} documents in {retrieval_time:.3f}s")

    if docs:
        print(f"\nTop 3 results:")
        for i, doc in enumerate(docs[:3], 1):
            similarity = doc.get('similarity', 0)
            metadata = doc.get('metadata', {})
            text_preview = doc.get('document', '')[:100]
            print(f"\n  [{i}] Similarity: {similarity:.3f}")
            print(f"      Source: {metadata.get('filename', 'N/A')}, Page: {metadata.get('page_number', 'N/A')}")
            print(f"      Preview: {text_preview}...")

    return docs


def test_metadata_boosting(docs: List[Dict[str, Any]], analysis):
    """Test metadata-based relevance boosting."""
    print_section("3. Metadata Boosting")

    if not analysis.has_filters:
        print("No filterable entities detected in query - skipping boosting")
        return docs

    print(f"Boosting documents that match: {analysis.get_boost_terms()}")

    # Show before scores
    print("\nBefore boosting (top 3):")
    for i, doc in enumerate(docs[:3], 1):
        similarity = doc.get('similarity', 0)
        print(f"  [{i}] Similarity: {similarity:.3f}")

    # Apply boosting
    boost_factor = 0.1
    boosted_docs = boost_by_metadata_match(docs, analysis, boost_factor=boost_factor)

    # Show after scores
    print(f"\nAfter boosting (top 3):")
    for i, doc in enumerate(boosted_docs[:3], 1):
        similarity = doc.get('similarity', 0)
        matches = doc.get('metadata_matches', 0)
        print(f"  [{i}] Similarity: {similarity:.3f} (metadata matches: {matches})")

    return boosted_docs


def test_reranking(query: str, docs: List[Dict[str, Any]], cfg):
    """Test cross-encoder reranking."""
    print_section("4. Cross-Encoder Reranking")

    if not docs:
        print("No documents to rerank")
        return docs

    reranker = DocumentReranker(
        model_name=cfg.chroma.retrieval.reranker_model,
    )

    print(f"Reranking with model: {cfg.chroma.retrieval.reranker_model}")
    print(f"Target top-k: {cfg.chroma.retrieval.rerank_top_k}")

    # Show before reranking
    print(f"\nBefore reranking (top 3):")
    for i, doc in enumerate(docs[:3], 1):
        similarity = doc.get('similarity', 0)
        print(f"  [{i}] Similarity: {similarity:.3f}")

    # Rerank
    start_time = time.time()
    reranked_docs = reranker.rerank(query, docs, top_k=cfg.chroma.retrieval.rerank_top_k)
    rerank_time = time.time() - start_time

    print(f"\nAfter reranking (top {len(reranked_docs)}) - took {rerank_time:.3f}s:")
    for i, doc in enumerate(reranked_docs[:3], 1):
        similarity = doc.get('similarity', 0)
        print(f"  [{i}] Similarity: {similarity:.3f}")

    return reranked_docs


def test_small_to_big(docs: List[Dict[str, Any]], cfg):
    """Test small-to-big context expansion."""
    print_section("5. Small-to-Big Retrieval")

    enable_small_to_big = getattr(cfg.chroma.chunking, 'enable_small_to_big', False)

    if not enable_small_to_big:
        print("Small-to-big retrieval is DISABLED in config")
        print("To enable: set chroma.chunking.enable_small_to_big=true")
        return docs

    print("Small-to-big retrieval is ENABLED")
    print("Expanding documents to parent context...")

    expanded_docs = expand_to_parent_context(docs)

    expanded_count = sum(1 for d in expanded_docs if d.get('used_parent_context', False))
    print(f"Expanded {expanded_count}/{len(expanded_docs)} documents to parent context")

    return expanded_docs


def test_context_building(docs: List[Dict[str, Any]], cfg):
    """Test context building with and without deduplication."""
    print_section("6. Context Building")

    if not docs:
        print("No documents to build context from")
        return ""

    use_dedup = cfg.chroma.retrieval.use_deduplication

    if use_dedup:
        print(f"Using semantic deduplication (threshold: {cfg.chroma.retrieval.deduplication_threshold})")
        context = build_semantic_context(
            docs,
            similarity_threshold=cfg.chroma.retrieval.deduplication_threshold,
            include_metadata=True,
            embedding_model=cfg.chroma.settings.embedder,
        )
    else:
        print("Using standard context building (no deduplication)")
        context = build_context_from_chunks(
            docs,
            include_metadata=True,
            include_similarity=False,
        )

    print(f"Built context: {len(context)} characters from {len(docs)} chunks")

    return context


def test_context_compression(context: str, cfg):
    """Test context compression."""
    print_section("7. Context Compression")

    enable_compression = getattr(cfg.chroma.retrieval, 'enable_compression', False)

    if not enable_compression:
        print("Context compression is DISABLED in config")
        print("To enable: set chroma.retrieval.enable_compression=true")
        return context

    max_chars = getattr(cfg.chroma.retrieval, 'compression_max_chars', 8000)

    print(f"Compression ENABLED (max: {max_chars} chars)")
    print(f"Original context: {len(context)} characters")

    if len(context) <= max_chars:
        print("Context already within limit - no compression needed")
        return context

    compressed = compress_context(context, max_chars=max_chars)

    reduction = ((len(context) - len(compressed)) / len(context)) * 100
    print(f"Compressed context: {len(compressed)} characters ({reduction:.1f}% reduction)")

    return compressed


def test_llm_generation(query: str, context: str, cfg):
    """Test LLM generation with the built context."""
    print_section("8. LLM Generation")

    print(f"Model: {cfg.model.provider}/{cfg.model.name}")
    print(f"Context length: {len(context)} characters")

    # Build prompt
    prompt = f"""You are a knowledgeable wine sommelier assistant.

Use the following wine knowledge to answer the question. Cite sources using [1], [2], etc.

Context:
{context}

Question: {query}

Answer:"""

    print(f"\nPrompt length: {len(prompt)} characters")

    try:
        llm = load_base_model(
            model_provider=cfg.model.provider,
            model_name=cfg.model.name
        )

        print("\nGenerating answer...")
        start_time = time.time()
        response = llm.invoke(prompt)
        generation_time = time.time() - start_time

        print(f"Generation took {generation_time:.3f}s")
        print(f"\n{'─'*70}")
        print("Answer:")
        print(f"{'─'*70}")
        print(response.content)
        print(f"{'─'*70}")

        return response.content

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        print(f"\nError: {e}")
        return None


def main():
    """Main quickstart flow testing all RAG components."""
    print(f"""
{'='*70}
  RAG Pipeline - Comprehensive Test
{'='*70}

This script tests all components of the RAG pipeline:
  1. Query preprocessing (normalization, expansion, entity extraction)
  2. Hybrid retrieval (vector + BM25 with RRF)
  3. Metadata boosting (boost docs matching query entities)
  4. Cross-encoder reranking
  5. Small-to-big retrieval (if enabled)
  6. Context building (with semantic deduplication)
  7. Context compression (if enabled)
  8. LLM generation

Uses test collection: '{TEST_COLLECTION}' (isolated from production)

{'='*70}
""")

    # Check prerequisites
    if not check_prerequisites():
        print("\n" + "="*70)
        print("Prerequisites not met. Please fix the issues above and try again.")
        print("="*70)
        return 1

    # Load config
    cfg = get_config()

    # Test query - contains multiple wine entities for testing metadata boosting
    query = "What are the characteristics of 2015 Barolo wines from Piedmont?"
    print(f"Test Query: '{query}'")

    # ...existing test code...

    # 1. Query preprocessing
    processed_query, analysis = test_query_preprocessing(query)

    # 2. Retrieval (hybrid)
    docs = test_retrieval(processed_query, cfg, use_hybrid=True)

    if not docs:
        print("\nNo documents retrieved - stopping here")
        return

    # 3. Metadata boosting
    docs = test_metadata_boosting(docs, analysis)

    # 4. Reranking
    if cfg.chroma.retrieval.enable_reranking:
        docs = test_reranking(query, docs, cfg)
    else:
        print_section("4. Cross-Encoder Reranking")
        print("Reranking is DISABLED in config")
        print("To enable: set chroma.retrieval.enable_reranking=true")

    # 5. Small-to-big
    docs = test_small_to_big(docs, cfg)

    # 6. Context building
    context = test_context_building(docs, cfg)

    # 7. Context compression
    context = test_context_compression(context, cfg)

    # 8. LLM generation
    answer = test_llm_generation(query, context, cfg)

    # Summary
    print_section("Summary")
    print("Pipeline test complete!")
    print(f"\nComponents tested:")
    print(f"  ✓ Query preprocessing (normalization, expansion, entity extraction)")
    print(f"  ✓ Hybrid retrieval (vector + BM25)")
    print(f"  ✓ Metadata boosting")
    print(f"  {'✓' if cfg.chroma.retrieval.enable_reranking else '✗'} Cross-encoder reranking")
    print(f"  {'✓' if getattr(cfg.chroma.chunking, 'enable_small_to_big', False) else '✗'} Small-to-big retrieval")
    print(f"  ✓ Context building with deduplication")
    print(f"  {'✓' if getattr(cfg.chroma.retrieval, 'enable_compression', False) else '✗'} Context compression")
    print(f"  ✓ LLM generation")

    print(f"\nFinal stats:")
    print(f"  Documents retrieved: {len(docs)}")
    print(f"  Context length: {len(context)} chars")
    print(f"  Answer generated: {'Yes' if answer else 'No'}")


if __name__ == "__main__":
    main()
