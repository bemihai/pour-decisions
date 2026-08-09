# Chroma Module

> **Project version**: 0.7.3 — last verified 2026-08-09.
> Milestone 3 (Phases 1 & 2) will add `ChunkQualityFilter` and `ContextualEnricher` to the
> indexing pipeline and modify `loader.py` and `chunks.py`. Update this README when those
> phases are implemented. See `design/roadmap/agentic-ai/milestones/m03-rag-quality-foundation.md`.

The `chroma` module provides document processing, chunking, and indexing capabilities for ChromaDB vector storage. It handles the complete pipeline from raw documents to indexed embeddings with wine-specific metadata extraction.

## Module Overview

This module is responsible for:
- **Document Processing**: Parse and split documents (PDF, EPUB) into chunks
- **Chunking Strategies**: Multiple strategies for text segmentation
- **Metadata Extraction**: Wine-specific entity extraction (grapes, regions, vintages)
- **Index Management**: Incremental indexing with change detection
- **Deduplication**: Remove duplicate content at indexing and retrieval time
- **Statistics**: Collection diagnostics and monitoring

## Components

| File | Purpose |
|------|---------|
| `extraction/` | Provider-neutral PDF and EPUB extraction |
| `chunking/` | Section-aware recursive and optional semantic chunking |
| `ingestion_pipeline.py` | Extraction, chunking, and loader-contract assembly |
| `chunks.py` | Compatibility entry point for provider-neutral ingestion |
| `loader.py` | Batch document loading into ChromaDB |
| `hierarchical_chunks.py` | Small-to-big retrieval pattern |
| `metadata_extractor.py` | Wine entity extraction |
| `deduplication.py` | Content deduplication utilities |
| `index_tracker.py` | Incremental indexing with manifest tracking |
| `stats.py` | Collection statistics and diagnostics |
| `utils.py` | Helper functions for ChromaDB operations |

## Data Processing Strategies

### Chunking Strategies

PDF files are extracted with `pdfplumber`; EPUB files are extracted from XHTML in spine order with
`ebooklib`. Both providers emit the same `DocumentElement` contract before chunking begins.

#### 1. Section-aware recursive chunking (`strategy="section_recursive"`)

This is the deterministic default. Elements are grouped by document title, chapter, and section.
Paragraphs and list items are packed to the configured size, oversized sections are split at
paragraph/sentence/whitespace boundaries, and overlap never crosses a section boundary.

```python
from src.chroma.chunks import split_file

chunks = split_file(
    filepath="wine_guide.pdf",
    strategy="section_recursive",
    chunk_size=1024,
    overlap_size=256,
)
```

#### 2. Section-bounded semantic chunking (`strategy="section_semantic"`)

This optional strategy uses the cached local embedder to find semantic breakpoints independently
inside each section. It is disabled by default because it adds an embedding pass during indexing.
If the embedder or one semantic split fails, that content falls back to `section_recursive`.

Enable `chroma.chunking.semantic.enabled` and select `section_semantic` only for evaluation. The
`basic`, `by_title`, and `semantic` strategy names were removed with the legacy extraction path.

### Hierarchical Chunks (Small-to-Big Retrieval)

A common RAG challenge: small chunks improve retrieval precision (the embedding matches exactly what you're looking for), but LLMs need larger context to generate good answers. Hierarchical chunking solves this by maintaining two versions of each chunk.

**The pattern**: During indexing, each piece of text is stored with:
1. A **small chunk** (e.g., 256 chars) used for embedding and similarity matching
2. A **large chunk** (e.g., 1024 chars) stored in metadata, returned to the LLM

When a query matches the small chunk, the system returns the larger surrounding context. This gives you precision in retrieval and richness in generation.

**How it works**: The algorithm slides through the text, extracting small chunks at regular intervals. For each small chunk, it captures a larger window centered around it (padding equally before and after). The large chunk is stored in the `parent_context` metadata field.

```python
from src.chroma import create_hierarchical_chunks, HierarchicalChunk

chunks = create_hierarchical_chunks(
    text="Full document text...",
    small_chunk_size=256,   # Used for embedding/retrieval
    large_chunk_size=1024,  # Returned to LLM for context
    overlap=64              # Overlap between consecutive small chunks
)

# Each HierarchicalChunk contains:
# - small_text: precise text for embedding matching
# - large_text: expanded context window for LLM consumption
# - chunk_id: unique identifier
# - metadata: position information (start/end indices)
```

To use hierarchical chunks during retrieval, call `expand_to_parent_context()` on your results to swap the small retrieved text with the larger parent context before sending to the LLM.

### Wine Metadata Extraction

One of the key differentiators of this RAG system is domain-specific metadata extraction. Rather than treating wine documents as generic text, the module identifies wine-specific entities and stores them as structured metadata. This enables powerful filtered retrieval queries like "find all chunks about Nebbiolo from Piedmont."

**How extraction works**: The `extract_wine_metadata()` function uses regex patterns and curated dictionaries from `src/utils/wine_terminology.py` to identify entities. Pattern matching is case-insensitive and handles common variations (e.g., "Cab Sauv" → "Cabernet Sauvignon"). The extracted metadata is stored as comma-separated strings in chunk metadata for ChromaDB filtering.

**Why this matters**: Vector similarity search finds semantically related content, but sometimes you need exact matches. If a user asks about "Piedmont wines," metadata filtering ensures you retrieve chunks that explicitly mention Piedmont, not just semantically similar regions. Combining vector search with metadata filters dramatically improves precision.

```python
from src.chroma import extract_wine_metadata

metadata = extract_wine_metadata("A 2019 Barolo from Piedmont featuring Nebbiolo grapes")
# Returns a WineMetadata dataclass:
# metadata.grapes = {"Nebbiolo"}
# metadata.regions = {"Piedmont"}
# metadata.vintages = {"2019"}
# metadata.appellations = {"Barolo"}
# metadata.classifications = set()  # No classification terms found
# metadata.producers = set()        # No producer patterns found
```

**Extracted entity types:**
- **Grapes**: Over 100 varieties including synonyms (e.g., "Syrah"/"Shiraz" both map to "Syrah")
- **Regions**: Major wine regions worldwide (Bordeaux, Burgundy, Piedmont, Napa Valley, Rioja, etc.)
- **Vintages**: Four-digit years between 1900-2050, filtered to avoid page numbers
- **Classifications**: Quality designations like DOCG, AOC, AVA, Grand Cru, Premier Cru, Reserva
- **Producers**: Pattern-matched names with prefixes (Château, Domaine, Bodega) or suffixes (Winery, Vineyards, Estate)
- **Appellations**: Specific wine names that indicate origin (Barolo, Champagne, Brunello di Montalcino, Châteauneuf-du-Pape)

The `extract_document_context()` function complements this by extracting structural context (document title, chapter, section) from the parsed elements, providing additional filtering dimensions.

> **Known limitation — document context fields are file-level, not chunk-level.** `extract_document_context()`
> scans the parsed elements once per file and returns the *last* `chapter` heading and the *last* short
> `section` title found anywhere in the document. Every chunk from that file carries those same values.
> A chunk from Chapter 3 will have `chapter = "Chapter 12"` if Chapter 12 is the last chapter in the
> book. This means the `chapter` and `section` metadata fields are unreliable for per-chunk context and
> should not be used as precise filters. This is a planned improvement in Milestone 3 (contextual
> enrichment).

## Loading Data into ChromaDB

### CollectionDataLoader

The `CollectionDataLoader` class is the primary interface for processing documents and loading them into ChromaDB. It orchestrates the complete pipeline: parsing documents, chunking, generating embeddings, extracting metadata, and batch inserting into the vector store.

**Initialization parameters:**
- `collection_name`: Name of the ChromaDB collection. If it doesn't exist, it will be created automatically with the provided metadata.
- `collection_metadata`: Optional dictionary stored with the collection (e.g., description, creation info).
- `chroma_host`/`chroma_port`: Connection details for the ChromaDB server.
- `embedding_model`: HuggingFace model identifier for generating embeddings. The same model must be used for both indexing and querying.
- `batch_size`: Number of documents per batch insert (default: 2500). Larger batches are faster but use more memory. ChromaDB's HTTP API has limits on request size.

```python
from src.chroma import CollectionDataLoader

loader = CollectionDataLoader(
    collection_name="wine_knowledge",
    collection_metadata={"description": "Wine reference documents"},
    chroma_host="localhost",
    chroma_port=8000,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=2500
)
```

### Loading a Directory

The `load_directory()` method processes all matching files in a directory with progress tracking. It's designed for batch indexing of document collections and supports incremental updates to avoid reprocessing unchanged files.

**Key features:**
- **File filtering**: Only processes files matching the specified extensions (default: `.pdf`, `.epub`).
- **Incremental mode**: When `incremental=True` (default), the loader checks the index manifest to skip files that haven't changed since last indexing. This is determined by comparing file hashes.
- **Duplicate detection**: When `skip_duplicates=True`, chunks with matching content hashes are skipped even if they come from different files. This prevents redundant content from inflating the collection.
- **Progress tracking**: Uses `tqdm` for real-time progress display showing files processed and chunks added.
- **Error resilience**: Failed files are logged but don't stop the batch. They'll be retried on the next run since they won't be marked in the manifest.

**Return value**: A statistics dictionary with detailed breakdown of processing results, useful for monitoring and debugging.

```python
stats = loader.load_directory(
    data_path="./documents/wine_books",
    file_extensions=[".pdf", ".epub"],
    strategy="section_recursive",
    chunk_size=1024,
    overlap_size=256,
    skip_duplicates=True,      # Skip chunks with matching content hash
    extract_metadata=True,     # Extract wine entities
    incremental=True,          # Only process new/modified files
    force_reindex=False        # Set True to reprocess all files
)

# stats dictionary contains:
# - total_files: Number of files found matching extensions
# - files_processed: Number of files actually processed this run
# - files_skipped: Number of files skipped (already indexed)
# - successful_files: Files processed without errors
# - failed_files: Files that encountered errors
# - total_chunks_generated: Raw chunks before validation/dedup
# - total_chunks_added: Chunks actually inserted into ChromaDB
# - total_chunks_skipped: Duplicates or invalid chunks skipped
# - processing_time: Total time in seconds
# - errors: List of error messages for debugging
```

### Processing a Single File

For more granular control, `process_file()` handles a single document. This is useful for testing chunking strategies, adding individual documents, or building custom processing pipelines.

The method returns detailed statistics for the specific file, including timing information helpful for performance tuning. All the same options available in `load_directory()` apply here.

```python
file_stats = loader.process_file(
    file_path="wine_atlas.pdf",
    strategy="section_semantic",
    chunk_size=1024,
    overlap_size=256,
    skip_duplicates=True,
    extract_metadata=True
)

# file_stats contains:
# - filename: Name of the processed file
# - chunks_generated: Number of chunks created
# - chunks_added: Chunks inserted into ChromaDB
# - chunks_skipped: Duplicates skipped
# - processing_time: Time in seconds
# - errors: List of any errors encountered
```

### Incremental Indexing

Re-indexing entire document collections is expensive. The `IndexTracker` class solves this by maintaining a JSON manifest that tracks which files have been indexed and their content hashes. On subsequent runs, only new or modified files are processed.

**How change detection works**: When a file is indexed, the tracker records its absolute path, MD5 hash of contents, file size, and modification time. On the next run, it compares the current file hash against the stored hash. If they differ, the file is reprocessed. This catches both content changes and file replacements.

**Manifest storage**: By default, manifests are stored in `chroma-data/manifests/{collection_name}_manifest.json`. This location can be customized via the `manifest_path` parameter.

**Handling failures**: If a file fails during processing (e.g., parsing error), it's not marked in the manifest. This means it will be retried on the next run, providing automatic retry behavior for transient failures.

```python
from src.chroma import IndexTracker
from pathlib import Path

tracker = IndexTracker(collection_name="wine_knowledge")

# Check if a specific file needs reindexing
needs_index = not tracker.is_file_indexed(Path("wine_guide.pdf"))

# Get list of files that need processing from a candidate list
files_to_process = tracker.get_files_to_index([
    Path("wine_guide.pdf"),
    Path("tasting_notes.pdf"),
    Path("new_document.pdf")
])

# After successful indexing, mark the file
tracker.mark_indexed(Path("wine_guide.pdf"), chunk_count=42)
tracker.save()  # Persist to disk

# Get overall statistics
stats = tracker.get_stats()
# {"total_files": 10, "total_chunks": 420, "collection_name": "wine_knowledge", "last_updated": "2025-01-10T..."}

# Clear tracking data to force full reindex
tracker.clear()
```

## Data Structure in ChromaDB

Understanding how data is organized in ChromaDB is essential for effective querying and debugging. Each document (chunk) is stored with its text content, a vector embedding, and structured metadata.

### Collection Schema

ChromaDB collections store documents as records with four components:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique chunk identifier. Format: `{filename}_{chunk_index}_{content_hash[:8]}`. The hash suffix ensures uniqueness even for identical chunk indices across files. |
| `document` | string | The actual chunk text content. This is what gets returned in query results and sent to the LLM. |
| `embedding` | float[] | Dense vector representation of the document, generated by the embedding model. Dimension depends on the model (e.g., 384 for MiniLM, 768 for MPNet). |
| `metadata` | object | Structured key-value pairs for filtering and context. Both standard document metadata and wine-specific entities are stored here. |

### Metadata Fields

Metadata enables filtered queries and provides context for retrieved chunks. All fields are stored as strings (ChromaDB requirement for filtering).

**Standard document metadata** (automatically extracted during processing):
- `filename`: Original file name (e.g., "wine_atlas.pdf")
- `file_path`: Absolute path to the source file
- `file_type`: File extension (e.g., ".pdf", ".epub")
- `chunk_index`: Sequential position of this chunk within the source file
- `chunk_id`: Same as the document ID
- `content_hash`: SHA-256 hash of chunk content for deduplication
- `page_number`: Source page number if available (-1 if not)
- `language`: Detected language of the content
- `word_count`: Number of words in the chunk
- `char_count`: Number of characters in the chunk
- `document_title`: Title extracted from document structure
- `chapter`: Chapter heading if detected
- `section`: Section heading if detected

**Wine-specific metadata** (extracted when `extract_metadata=True`):
- `grapes`: Comma-separated list of grape varieties found in the chunk
- `regions`: Comma-separated list of wine regions mentioned
- `vintages`: Comma-separated list of vintage years detected
- `classifications`: Wine quality classifications (DOCG, AOC, AVA, etc.)
- `producers`: Producer/winery names identified
- `appellations`: Wine appellations mentioned (Barolo, Champagne, etc.)

The comma-separated format allows ChromaDB's `$contains` operator to match individual values within the field.

### Example Document

```json
{
  "id": "wine_atlas_42_a1b2c3d4",
  "document": "Barolo is made exclusively from Nebbiolo grapes in the Piedmont region...",
  "metadata": {
    "filename": "wine_atlas.pdf",
    "file_type": ".pdf",
    "chunk_index": 42,
    "content_hash": "sha256...",
    "page_number": 156,
    "word_count": 89,
    "document_title": "The World Atlas of Wine",
    "chapter": "Chapter 12 - Italy",
    "grapes": "Nebbiolo",
    "regions": "Piedmont",
    "appellations": "Barolo",
    "classifications": "DOCG"
  }
}
```

## Querying and Retrieval

While the `retrieval` module handles the query interface, understanding how the chroma module structures data informs effective query patterns. The data organization supports three main retrieval approaches: pure vector similarity, metadata-filtered search, and hybrid (vector + keyword) search.

### Vector Similarity Search

The most common retrieval pattern. The query is embedded using the same model that indexed the documents, then ChromaDB finds the most similar document vectors using cosine distance.

**How it works internally**:
1. Query text is normalized and optionally expanded with wine terminology synonyms
2. The embedding model converts the query to a dense vector
3. ChromaDB performs approximate nearest neighbor search
4. Results are ranked by similarity (1 - distance) and filtered by optional threshold

**Key considerations**:
- Always use the same embedding model for indexing and querying
- The `similarity_threshold` parameter filters out low-confidence matches
- Query caching (enabled by default) avoids redundant embedding computations

```python
from src.retrieval import ChromaRetriever
from src.utils import initialize_chroma_client

client = initialize_chroma_client("localhost", 8000)

retriever = ChromaRetriever(
    client=client,
    collection_name="wine_knowledge",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    n_results=10,
    similarity_threshold=0.5,  # Filter results below 50% similarity
    enable_query_expansion=True,  # Expand with wine synonyms
    enable_cache=True  # Cache query embeddings
)

results = retriever.retrieve("What grapes are used in Barolo?")
# Returns list of dicts with: id, document, metadata, distance, similarity
```

### Metadata Filtering

Combine vector search with exact metadata matches using ChromaDB's `where` clause. This is powerful for domain-specific queries where you know certain constraints.

**Filter operators available**:
- `$eq`: Exact match
- `$ne`: Not equal
- `$contains`: String contains (works with comma-separated values)
- `$gt`, `$gte`, `$lt`, `$lte`: Numeric comparisons
- `$and`, `$or`: Combine multiple conditions

**When to use metadata filters**:
- User specifies a region, grape, or vintage explicitly
- You want to narrow results to a specific source document
- Combining broad semantic queries with precise constraints

```python
# Filter by region - only retrieve chunks mentioning Piedmont
results = retriever.retrieve(
    query="best vintages",
    where={"regions": {"$contains": "Piedmont"}}
)

# Filter by grape variety
results = retriever.retrieve(
    query="tasting notes",
    where={"grapes": {"$contains": "Nebbiolo"}}
)

# Combine multiple filters with $and
results = retriever.retrieve(
    query="food pairing suggestions",
    where={
        "$and": [
            {"regions": {"$contains": "Burgundy"}},
            {"grapes": {"$contains": "Pinot Noir"}}
        ]
    }
)

# Filter by source document
results = retriever.retrieve(
    query="aging potential",
    where={"filename": {"$eq": "wine_atlas.pdf"}}
)
```

### Hybrid Search

Combines the semantic understanding of vector search with the precision of keyword matching using Reciprocal Rank Fusion (RRF). This approach often outperforms either method alone.

**How RRF works**: Each retrieval method (vector and BM25) produces a ranked list. RRF assigns scores based on rank position: `score = weight / (k + rank)` where k=60 by default. Scores are summed across methods, and results are re-ranked by combined score. This rewards documents that appear in both lists while not requiring score normalization.

**When to use hybrid search**:
- Queries mix semantic concepts with specific terms (e.g., "2015 Barolo recommendations")
- You want robustness to query phrasing variations
- Important terms might be missed by vector similarity alone

```python
from src.retrieval import HybridRetriever, ChromaRetriever
from src.retrieval.keyword_search import BM25Index

# Initialize both retrievers
vector_retriever = ChromaRetriever(...)
bm25_index = BM25Index.load("chroma-data/bm25_index.pkl")

hybrid = HybridRetriever(
    vector_retriever=vector_retriever,
    bm25_index=bm25_index,
    vector_weight=0.7,  # 70% weight to semantic similarity
    keyword_weight=0.3  # 30% weight to keyword matching
)

results = hybrid.retrieve("2015 Barolo recommendations", n_results=10)
# Each result includes 'rrf_score' indicating combined relevance
```

## Deduplication

Duplicate content degrades RAG quality in two ways: it wastes context window tokens, and it can bias the LLM toward repeated information. The chroma module provides deduplication at both indexing and retrieval stages.

### At Index Time

During document loading, duplicate chunks are detected using content hashes (SHA-256). When `skip_duplicates=True` (default), the loader queries ChromaDB for existing chunks with the same hash before inserting.

**How it works**: Each chunk's text content is hashed. Before insertion, the loader checks `where={"content_hash": hash_value}`. If a match exists, the chunk is skipped. This prevents identical content from different files (e.g., shared introductions, boilerplate text) from being indexed multiple times.

**Trade-offs**: Hash-based deduplication is fast but only catches exact duplicates. Near-duplicates (same content with minor variations) will still be indexed separately. For more aggressive deduplication, consider semantic deduplication at retrieval time.

### At Retrieval Time

Retrieved results often contain semantically similar chunks, especially when multiple documents cover the same topic. The `deduplication.py` module provides utilities to reduce redundancy before sending context to the LLM.

**Two-stage deduplication pipeline**:
1. **Hash-based (fast)**: Remove exact duplicates by content hash
2. **Semantic (thorough)**: Compute embeddings and remove chunks with similarity above threshold

The `similarity_threshold` parameter controls how aggressively near-duplicates are merged. A threshold of 0.90 means chunks with >90% cosine similarity are considered duplicates, keeping only the highest-ranked one.

```python
from src.chroma import deduplicate_context

# Full deduplication pipeline
unique_chunks = deduplicate_context(
    chunks=retrieved_chunks,
    similarity_threshold=0.90,  # Remove chunks >90% similar
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    use_hash_first=True  # Run fast hash-based dedup before semantic
)

# For just hash-based deduplication (faster, less thorough)
from src.chroma import deduplicate_by_content_hash
unique_chunks = deduplicate_by_content_hash(retrieved_chunks)

# For just semantic deduplication
from src.chroma import deduplicate_chunks
unique_chunks = deduplicate_chunks(
    chunks=retrieved_chunks,
    similarity_threshold=0.85  # More aggressive threshold
)
```

**When to use each approach**:
- **Hash-based only**: Fast processing, exact duplicates are the main concern
- **Semantic only**: Small result sets where embedding cost is acceptable
- **Combined pipeline**: Production use where both speed and quality matter

## Collection Statistics

Monitoring your ChromaDB collections helps identify issues like unbalanced chunk sizes, missing metadata, or unexpectedly low document counts. The `stats.py` module provides both CLI and programmatic access to collection diagnostics.

### CLI Tool

The default stats CLI samples at most 100 records per collection for a quick operational overview.
Sampled output is labeled `statistics_mode: sampled` and must not be used as an acceptance artifact.

```bash
# Display stats for all collections
python -m src.chroma.stats

# Specific collection only
python -m src.chroma.stats --collection wine_knowledge

# Output as JSON for scripting/monitoring
python -m src.chroma.stats --json
```

Use exact mode for milestone checkpoints and corpus comparisons. It reads every record from the
configured collection in bounded batches and records exact document-length, empty/near-empty,
per-source, and sorted chunk-ID hash diagnostics. Near-empty means fewer than 200 characters and
includes empty documents.

```bash
# Save the default dated artifact under eval-results/
make chroma-stats-exact

# Choose an explicit artifact path
make chroma-stats-exact CORPUS_STATS_OUTPUT=eval-results/corpus-before-change.json

# Direct CLI usage with a custom batch size
python -m src.chroma.stats --exact --batch-size 1000 --output eval-results/corpus.json
```

Exact artifacts are labeled `statistics_mode: exact`. The default Gate 0 filename is
`eval-results/m3_gate0_corpus_<YYYYMMDD>.json`.

**Sample output**:
```
============================================================
ChromaDB Statistics
Server: localhost:8000
Total Collections: 2
Total Records: 12,450
============================================================

Collection: wine_knowledge
============================================================
  Records: 10,234
  Embedding Dimension: 384

  Document Length (chars):
    Average: 487
    Min: 52
    Max: 1,024

  Wine Metadata Samples:
    grapes: Nebbiolo, Sangiovese, Pinot Noir
    regions: Piedmont, Tuscany, Burgundy
    vintages: 2015, 2018, 2019
```

### Programmatic Access

For integration with monitoring systems or custom dashboards, use the stats functions directly.

```python
from src.chroma import get_collection_stats, get_all_stats
from src.utils import initialize_chroma_client

client = initialize_chroma_client("localhost", 8000)

# Get detailed stats for a single collection
stats = get_collection_stats(client, "wine_knowledge")
# Returns dict with:
# - name: Collection name
# - record_count: Total documents
# - embedding_dimension: Vector size
# - metadata: Collection-level metadata
# - avg_document_length, min_document_length, max_document_length
# - metadata_fields: Set of all metadata keys found
# - metadata_sample_values: Sample values for each field

# Get stats for all collections in the database
all_stats = get_all_stats(client)
# Returns list of stats dicts, one per collection
```

**Use cases for statistics**:
- **Capacity planning**: Track collection growth over time
- **Quality assurance**: Verify metadata extraction is working (check for empty fields)
- **Debugging**: Identify collections with unexpected chunk sizes
- **Monitoring**: Alert on collection count changes or embedding dimension mismatches

## Best Practices

### Choosing a Chunking Strategy

The right strategy depends on your content and requirements:

| Strategy | Speed | Quality | Best For |
|----------|-------|---------|----------|
| `section_recursive` | Fast | Stable default | Structured PDF/EPUB books |
| `section_semantic` | Slower | Eval-gated | Long prose sections with weak paragraph boundaries |

**Recommendation**: Keep `section_recursive` as the default. Evaluate `section_semantic` against the
retrieval baseline before enabling it because it adds local indexing cost and less stable boundaries.

### Optimizing Chunk Size

Chunk size affects both retrieval precision and LLM context quality:

- **Too small (< 200 chars)**: High precision but fragments context, LLM may lack sufficient information
- **Too large (> 2000 chars)**: Rich context but dilutes embedding signal, retrieval precision suffers
- **Sweet spot (400-800 chars)**: Balances precision and context for most use cases

For wine content, 512 characters with 128 overlap works well. Consider hierarchical chunks if you need both precision and rich context.

### Managing Incremental Indexing

Enable `incremental=True` (the default) to avoid reprocessing unchanged files. This is especially important for large collections where re-indexing is expensive.

**When to force reindex** (`force_reindex=True`):
- After changing the embedding model (vectors are incompatible)
- After modifying chunking strategy or parameters
- After updating metadata extraction logic
- When troubleshooting retrieval quality issues

> **Forced reindex keeps Chroma and BM25 synchronized.** `make chroma-reindex` recreates the
> configured Chroma collection, rebuilds BM25 from exactly those records in bounded batches, and
> atomically replaces `chroma-data/bm25_index.pkl` plus its synchronization manifest. Retrieval
> validates collection count and sorted chunk-ID hash before enabling hybrid search; a missing or
> stale manifest produces an explicit vector-only fallback. Incremental uploads invalidate the
> previous proof when IDs change, so run the verified forced reindex before relying on hybrid search.

### Effective Metadata Filtering

Design queries to combine semantic search with metadata constraints:

1. **Extract entities from user queries**: If a user asks about "Piedmont Nebbiolo", extract "Piedmont" and "Nebbiolo" as filter values
2. **Use `$and` for precision**: Combining multiple filters narrows results effectively
3. **Fall back gracefully**: If filtered search returns few results, retry without filters
4. **Index metadata consistently**: Ensure all documents have metadata populated for even coverage

### Deduplication Guidelines

Apply deduplication at the right stage:

- **Index time** (`skip_duplicates=True`): Always enable. Prevents identical chunks from inflating your collection.
- **Retrieval time**: Apply semantic deduplication when sending context to the LLM. This catches near-duplicates that hash-based methods miss.
- **Threshold tuning**: Start with 0.90 similarity threshold. Lower it (0.85) if you still see redundant content; raise it (0.95) if useful variations are being removed.

### Monitoring and Maintenance

Regular collection health checks catch issues early:

1. **Run stats periodically**: Check record counts, embedding dimensions, metadata coverage
2. **Verify metadata extraction**: Sample chunks should have populated wine entity fields
3. **Monitor manifest files**: Large manifests indicate many indexed files; check for orphaned entries
4. **Track cache hit rates**: Low hit rates may indicate query diversity or cache size issues

## Testing

```bash
# Run all chroma tests
pytest tests/chroma/ -v

# Run with coverage report
pytest tests/chroma/ --cov=src.chroma --cov-report=term-missing --cov-report=html

# View HTML coverage report
open htmlcov/index.html

# Run specific test file
pytest tests/chroma/test_metadata_extractor.py -v

# Run specific test class
pytest tests/chroma/test_loader.py::TestLoadDirectory -v

# Run with make commands
make test              # All tests with coverage
make test-fast         # Quick run without coverage
```
