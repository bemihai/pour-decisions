# Pour Decisions RAG Pipeline

**Last Updated**: December 29, 2025

This document describes the Retrieval-Augmented Generation (RAG) pipeline implemented in Pour Decisions, a wine knowledge chatbot. The pipeline enables the system to answer wine-related questions by retrieving relevant information from a curated collection of wine books and documents.

---

## Overview

The Pour Decisions RAG system follows a three-stage architecture:

1. **Ingestion Pipeline** - Processes wine books and documents into searchable chunks
2. **Retrieval Pipeline** - Finds relevant information for user queries
3. **Generation Pipeline** - Uses an LLM to generate answers from retrieved context

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Pour Decisions RAG Pipeline                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INGESTION                                                               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────────┐ │
│  │  Wine    │──▶│ Chunking │──▶│Embedding │──▶│      ChromaDB        │ │
│  │  Books   │   │          │   │          │   │   (Vector Store)     │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┬───────────┘ │
│                                                           │             │
│  RETRIEVAL                                                ▼             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────────┐ │
│  │  User    │──▶│  Query   │──▶│  Hybrid  │──▶│     Reranker         │ │
│  │  Query   │   │ Preproc  │   │ Retrieval│   │   (Cross-Encoder)    │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┬───────────┘ │
│                                                           │             │
│  GENERATION                                               ▼             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────────┐ │
│  │ Context  │──▶│  Prompt  │──▶│  Gemini  │──▶│      Response        │ │
│  │ Builder  │   │ Template │   │   LLM    │   │                      │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Ingestion Pipeline

The ingestion pipeline transforms raw wine documents (PDFs, EPUBs) into searchable vector embeddings stored in ChromaDB.

### 1.1 Data Sources

- **Wine Books**: Professional wine reference books (e.g., The World Atlas of Wine)
- **File Formats**: PDF, EPUB
- **Storage Location**: `data/` directory

### 1.2 Document Processing

Documents are processed through several layers:

| Layer | Component | Description |
|-------|-----------|-------------|
| **Extraction** | `unstructured.partition` | Extracts text from PDFs and EPUBs |
| **Chunking** | `chunks.py` | Splits documents into manageable pieces |
| **Embedding** | HuggingFace `all-MiniLM-L6-v2` | Converts text to 384-dimensional vectors |
| **Loading** | `CollectionDataLoader` | Stores embeddings and metadata in ChromaDB |

### 1.3 Chunking Strategies

The system supports three chunking strategies:

**Basic Chunking**
- Splits documents into fixed-size chunks with overlap
- Default: 512 characters with 128 character overlap
- Fast but may break semantic boundaries

**Title-Based Chunking**
- Respects document structure (sections, paragraphs)
- Preserves logical groupings
- Better for structured wine reference books

**Semantic Chunking**
- Uses embeddings to detect semantic boundaries
- Splits where topic changes significantly
- No LLM calls required (uses local embeddings)
- Best quality but slower

### 1.4 Chunk Metadata

Each chunk is enriched with metadata for filtering and context:

| Field | Description |
|-------|-------------|
| `filename` | Source document name |
| `file_path` | Full path to source |
| `file_type` | File extension (.pdf, .epub) |
| `chunk_index` | Position within document |
| `chunk_id` | Unique identifier |
| `content_hash` | MD5 hash for deduplication |
| `page_number` | Source page (if available) |
| `word_count` | Number of words in chunk |
| `char_count` | Number of characters |
| `document_title` | Title of source document (contextual retrieval) |
| `chapter` | Chapter heading if detected |
| `section` | Section heading if detected |
| `grapes` | Grape varieties mentioned (comma-separated) |
| `regions` | Wine regions mentioned (comma-separated) |
| `vintages` | Vintage years mentioned (comma-separated) |
| `classifications` | Wine classifications (DOCG, AOC, etc.) |
| `producers` | Producer/winery names detected |
| `appellations` | Wine appellations (Barolo, Champagne, etc.) |

**Wine Metadata Extraction**

During indexing, each chunk is analyzed to extract wine-specific metadata:
- **Grapes**: Detected using the wine terminology dictionary (e.g., "nebbiolo", "cabernet sauvignon")
- **Regions**: Matched against known wine regions and their variations
- **Vintages**: Year patterns between 1945-2025
- **Classifications**: Wine classification abbreviations (DOCG, AOC, AVA, etc.)
- **Producers**: Detected using naming patterns (e.g., "Château X", "Y Winery", "Domaine Z")
- **Appellations**: Famous wine appellations (Barolo, Champagne, Châteauneuf-du-Pape, etc.)

This metadata enables filtered retrieval queries like "show me chunks about Nebbiolo from Piedmont".

### 1.5 Vector Storage

**ChromaDB** serves as the vector database with the following configuration:

| Setting | Value | Purpose |
|---------|-------|---------|
| Distance Metric | Cosine | Measures semantic similarity |
| HNSW Search EF | 100 | Search quality parameter |
| HNSW Construction EF | 200 | Index build quality |
| Batch Size | 2500 | Documents per batch insert |

### 1.6 Loading Data into ChromaDB

The ingestion process is orchestrated through make commands for convenience.

**Prerequisites**
1. ChromaDB must be running (either via Docker or locally)
2. Wine books must be placed in the configured data directory
3. Environment variables configured in `.env`

**Step 1: Start ChromaDB**

```bash
# Start ChromaDB container for local development
make chroma-up
```

This starts ChromaDB on `http://localhost:8000` and waits for the health check.

**Step 2: Populate with Wine Knowledge**

```bash
# Process wine books and load into ChromaDB
make chroma-upload
```

This command executes `src/chroma/load_data.py` which:
1. Reads configuration from `app_config.yml`
2. Iterates through configured collections
3. For each collection:
   - Scans the data directory for PDF and EPUB files
   - Chunks documents using the configured strategy
   - Generates embeddings for each chunk
   - Loads chunks with metadata into ChromaDB
   - Skips duplicates based on content hash

**Configuration in `app_config.yml`**

```yaml
chroma:
  client:
    host: localhost
    port: 8000
  chunking:
    strategy: by_title        # basic, by_title, or semantic
    chunk_size: 1024
    chunk_overlap: 256
  settings:
    batch_size: 2500
    embedder: sentence-transformers/all-MiniLM-L6-v2
  collections:
    - name: wine_books
      local_data_path: data/wine_books
      metadata:
        description: "Professional wine books collection"
```

**Ingestion Output**

The loader provides detailed statistics:
- Files processed (successful/failed)
- Chunks generated and added
- Duplicates skipped
- Processing time per file
- Total processing time

**Other ChromaDB Commands**

| Command | Description |
|---------|-------------|
| `make chroma-up` | Start ChromaDB container |
| `make chroma-down` | Stop ChromaDB container |
| `make chroma-upload` | Index new/modified files (incremental) |
| `make chroma-reindex` | Force reindex all files |
| `make chroma-status` | Show index status (files and chunks) |
| `make chroma-stats` | Show collection statistics (records, embeddings, metadata) |
| `make chroma-health` | Check ChromaDB health status |
| `make chroma-reset` | Reset ChromaDB (clears all data) |
| `make chroma-backup` | Backup ChromaDB data directory |
| `make chroma-restore BACKUP_FILE=path` | Restore from backup |

### 1.8 Incremental Indexing

The loader supports incremental indexing to avoid re-processing unchanged files. This is enabled by default.

**How It Works**

1. A manifest file tracks all indexed files with their content hashes
2. When running `make chroma-upload`, only new or modified files are processed
3. File changes are detected by comparing MD5 hashes of file contents
4. The manifest is stored at `chroma-data/manifests/{collection}_manifest.json`
5. **Resume support**: The manifest is saved after each successful file, so if indexing is interrupted (Ctrl+C, crash), it will resume from where it left off

**Manifest Contents**

For each indexed file, the manifest stores:
- File path and content hash
- File size and modification time
- When the file was indexed
- Number of chunks created
- Collection name

**Commands**

```bash
# Incremental mode (default) - only process new/modified files
make chroma-upload

# Force reindex all files
make chroma-reindex

# Show current index status
make chroma-status
```

**Example Output**

```
📊 Index Status for 'wine_books':
   Files indexed: 5
   Total chunks: 2847
   Last updated: 2025-12-30T10:15:32

   Indexed files:
   - wine_atlas.pdf (892 chunks)
   - world_of_wine.epub (1203 chunks)
   - sommelier_guide.pdf (752 chunks)
```

**When to Force Reindex**

Use `make chroma-reindex` when:
- Chunking strategy has changed
- Embedding model has been upgraded
- Metadata extraction logic has been updated
- You want to rebuild the entire index

### 1.7 ChromaDB Storage Structure

ChromaDB persists data in a hybrid storage architecture combining SQLite for metadata and binary files for vector indices.

**Data Directory Structure**

```
chroma-data/
├── chroma.sqlite3                              # Metadata database
└── {collection-uuid}/                          # Per-collection HNSW index
    ├── data_level0.bin                         # HNSW graph level 0 data
    ├── header.bin                              # Index header information
    ├── index_metadata.pickle                   # Python pickled metadata
    ├── length.bin                              # Vector length data
    └── link_lists.bin                          # HNSW graph connections
```

**Storage Components**

| Component | Format | Contents |
|-----------|--------|----------|
| `chroma.sqlite3` | SQLite | Collections, segments, document text, metadata |
| `data_level0.bin` | Binary | HNSW index vectors at base level |
| `header.bin` | Binary | Index configuration and parameters |
| `index_metadata.pickle` | Pickle | Python metadata for index reconstruction |
| `link_lists.bin` | Binary | HNSW graph edge connections |

**What Gets Stored**

For each chunk indexed in ChromaDB:

| Data Type | Storage Location | Description |
|-----------|------------------|-------------|
| Document ID | SQLite | Unique identifier (e.g., `wine_atlas_42_a1b2c3d4`) |
| Document Text | SQLite | Original chunk text content |
| Embedding Vector | Binary files | 384-dimensional float vector |
| Metadata | SQLite | JSON-like key-value pairs (filename, page, hash, etc.) |

**Docker Volume Mapping**

In `docker-compose.yml`, ChromaDB data is persisted via volume mount:

```yaml
chromadb:
  image: chromadb/chroma:latest
  volumes:
    - ./chroma-data:/data        # Host path : Container path
  environment:
    - IS_PERSISTENT=TRUE         # Enable persistence
```

This ensures data survives container restarts and can be backed up from the host filesystem.

**Collection Configuration**

Each collection is created with HNSW index parameters optimized for wine knowledge retrieval:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `hnsw:space` | cosine | Similarity metric (cosine distance) |
| `hnsw:search_ef` | 100 | Search quality (higher = better but slower) |
| `hnsw:construction_ef` | 200 | Index build quality |
| `hnsw:num_threads` | 8 | Parallel indexing threads |

**Typical Storage Sizes**

| Content | Approximate Size |
|---------|------------------|
| 1,000 chunks | ~50 MB |
| 10,000 chunks | ~200 MB |
| 50,000 chunks | ~800 MB |

Storage is dominated by the HNSW index files, which grow linearly with the number of vectors.

---

## 2. Retrieval Pipeline

The retrieval pipeline finds the most relevant chunks for a user's question using a multi-stage approach. **Hybrid search and reranking are enabled by default** via `app_config.yml`.

### 2.1 Query Preprocessing

Before retrieval, queries are preprocessed to improve matching:

**Wine Terminology Normalization**
- Fixes common misspellings ("chardonay" → "chardonnay")
- Expands abbreviations ("cab sauv" → "cabernet sauvignon")
- Normalizes region names ("burgandy" → "burgundy")
- Handles grape synonyms ("shiraz" ↔ "syrah")

**Query Expansion**
- Adds related terms for better recall
- Example: "barolo" → "barolo nebbiolo piedmont italy docg langhe"
- Domain-specific expansions for wine topics

### 2.2 Hybrid Search

The system combines two retrieval methods for better results:

**Vector Search (Semantic)**
- Embeds query using same model as documents
- Finds chunks with similar meaning
- Handles natural language queries well
- Weight: 70% (configurable)

**BM25 Search (Keyword)**
- Matches exact terms in documents
- Fast and interpretable
- Good for specific wine names, producers
- Weight: 30% (configurable)

**Reciprocal Rank Fusion (RRF)**
- Combines rankings from both methods
- Formula: `score = Σ(weight / (k + rank))`
- k=60 (standard RRF constant)
- Returns unified ranked list

### 2.3 Reranking

After initial retrieval, a cross-encoder reranks results for precision:

| Stage | Model | Speed | Accuracy |
|-------|-------|-------|----------|
| Initial Retrieval | Bi-encoder | Fast | Good |
| Reranking | Cross-encoder | Slower | Better |

**Cross-Encoder Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

The reranker processes query-document pairs together, providing more accurate relevance scores than the initial bi-encoder retrieval.

### 2.4 Query Caching

To reduce latency and embedding costs, results are cached:

- **Cache Type**: LRU (Least Recently Used)
- **Default Size**: 100 queries
- **Cache Key**: Hash of query + parameters
- **Hit Rate Tracking**: Statistics available via `get_cache_stats()`

Repeated identical queries return cached results instantly.

### 2.5 Similarity Filtering

Results below a similarity threshold are filtered out:

- **Default Threshold**: 0.3 (configurable)
- **Score Range**: 0.0 (unrelated) to 1.0 (identical)
- **Purpose**: Prevents irrelevant chunks from reaching the LLM

---

## 3. Generation Pipeline

The generation pipeline constructs prompts and generates answers using an LLM.

### 3.1 Context Building

Retrieved chunks are formatted into a context string:

- Source attribution for each chunk
- Page numbers when available
- Similarity scores (optional)
- Semantic deduplication to remove redundant chunks

### 3.2 Prompt Template

The system uses a structured prompt template:

```
System: You are a knowledgeable wine sommelier assistant.

Context: [Retrieved wine knowledge chunks]

User: [User's question]

Instructions: Answer using only the provided context. If the 
information is not in the context, say "I don't have information 
about that in my wine knowledge base."
```

### 3.3 LLM Configuration

| Setting | Value |
|---------|-------|
| Provider | Google Gemini |
| Model | gemini-1.5-flash (recommended) |
| Temperature | 0.0 (deterministic) |
| Max Retries | 2 |

---

## 4. Wine Domain Optimizations

The RAG pipeline includes several wine-specific enhancements:

### 4.1 Wine Terminology Dictionary

Comprehensive mappings for:
- **Grape Varieties**: 20+ varieties with synonyms
- **Wine Regions**: 20+ regions with variations
- **Classifications**: DOCG, AOC, AVA, etc.
- **Common Misspellings**: Frequently mistyped terms

### 4.2 Query Expansion Rules

Domain-specific expansions that add relevant context:

| Query Contains | Expanded With |
|----------------|---------------|
| "barolo" | nebbiolo, piedmont, italy, docg, langhe |
| "champagne" | sparkling, france, chardonnay, pinot noir |
| "bordeaux red" | cabernet, merlot, medoc, saint emilion |
| "burgundy white" | chardonnay, meursault, puligny |

### 4.3 Metadata Filtering

Queries can filter by metadata:
- Source book
- Page range
- Wine region (if tagged)
- Document type

---

## 5. Component Reference

### 5.1 File Structure

```
src/chroma/                        # Indexing & storage
├── chunks.py                      # Document chunking strategies
├── loader.py                      # CollectionDataLoader (batch upsert)
├── load_data.py                   # CLI for loading data into ChromaDB
├── hierarchical_chunks.py         # Small-to-big retrieval pattern
├── index_tracker.py               # Incremental indexing manifest
├── metadata_extractor.py          # Wine metadata extraction (grapes, regions, etc.)
├── deduplication.py               # Content deduplication utilities
├── stats.py                       # Collection statistics
└── utils.py                       # ChromaDB helper functions

src/retrieval/                     # Querying & post-processing
├── vector_retriever.py            # ChromaRetriever (vector search + cache)
├── keyword_search.py              # BM25Index (keyword-based search)
├── hybrid_retriever.py            # HybridRetriever (RRF fusion)
├── reranker.py                    # DocumentReranker (cross-encoder)
├── query_utils.py                 # Wine term normalization & expansion
├── query_analyzer.py              # Metadata entity extraction & filtering
├── query_compression.py           # TF-IDF context compression
└── context_builder.py             # Context formatting & semantic dedup
```

### 5.2 Key Classes

| Class | Purpose |
|-------|---------|
| `CollectionDataLoader` | Ingests documents into ChromaDB |
| `ChromaRetriever` | Vector similarity search |
| `BM25Index` | Keyword-based search |
| `HybridRetriever` | Combines vector + keyword search |
| `DocumentReranker` | Cross-encoder reranking |

### 5.3 Configuration

Main settings in `app_config.yml`:

```yaml
chroma:
  chunking:
    strategy: by_title
    chunk_size: 1024
    chunk_overlap: 256
  retrieval:
    n_results: 5
    similarity_threshold: 0.3
    # Hybrid search
    enable_hybrid: true
    hybrid_vector_weight: 0.7
    hybrid_keyword_weight: 0.3
    bm25_index_path: "chroma-data/bm25_index.pkl"
    # Reranking
    enable_reranking: true
    reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: 5
  settings:
    embedder: sentence-transformers/all-MiniLM-L6-v2
```

**Retrieval Mode Options**

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_hybrid` | `true` | Enable hybrid search (vector + BM25) |
| `enable_reranking` | `true` | Enable cross-encoder reranking |

When both are enabled, the retrieval flow is:
1. Vector search retrieves `n_results * 2` candidates
2. BM25 search retrieves `n_results * 2` candidates  
3. RRF fusion combines rankings
4. Cross-encoder reranks to `rerank_top_k` final results

---

## 6. Performance Considerations

### 6.1 Latency Breakdown

| Stage | Typical Time |
|-------|--------------|
| Query Preprocessing | <10ms |
| Embedding Generation | 50-100ms |
| Vector Search | 20-50ms |
| BM25 Search | 5-10ms |
| RRF Fusion | <5ms |
| Reranking (if enabled) | 100-200ms |
| **Total (without reranking)** | **~100-200ms** |
| **Total (with reranking)** | **~200-400ms** |

### 6.2 Memory Usage

| Component | Memory |
|-----------|--------|
| Embedding Model | ~100MB |
| Cross-Encoder | ~100MB |
| BM25 Index | ~10MB per 10K docs |
| Query Cache | ~10MB (100 queries) |

### 6.3 Cost Optimization

The pipeline minimizes LLM costs through:
- Local embeddings (no API calls)
- Query caching (avoids repeated processing)
- Semantic chunking without LLM
- Efficient context building

---

## 7. Future Enhancements

Planned improvements documented in `design/rag/rag-improvement-plan.md`:

### Completed
- Contextual retrieval (augment chunks with document context)
- Small-to-big retrieval (small chunks for search, larger for context)
- Wine entity extraction (producers, appellations)
- Context compression (reduce token usage)
- Evaluation framework with P@K, MRR metrics

### Remaining
- Multi-query retrieval for complex questions (requires LLM)
- Knowledge graph integration (low priority)

---

## 8. Testing the Pipeline

### 8.1 Quickstart Script

A comprehensive testing script is available at `scripts/rag_quickstart.py` that tests all pipeline components:

```bash
PYTHONPATH=$(pwd) python scripts/rag_quickstart.py
```

**What it tests:**
1. **Query Preprocessing** - Normalization, expansion, entity extraction
2. **Hybrid Retrieval** - Vector + BM25 with RRF fusion
3. **Metadata Boosting** - Relevance boosting based on wine entities
4. **Cross-Encoder Reranking** - Precision improvement
5. **Small-to-Big Retrieval** - Context expansion (if enabled)
6. **Context Building** - With semantic deduplication
7. **Context Compression** - Token reduction (if enabled)
8. **LLM Generation** - Final answer generation

**Example Output:**

```
======================================================================
  Pour Decisions RAG Pipeline - Comprehensive Test
======================================================================

Test Query: 'What are the characteristics of 2015 Barolo wines from Piedmont?'

======================================================================
  1. Query Preprocessing
======================================================================

Original query: 'What are the characteristics of 2015 Barolo wines from Piedmont?'
Normalized:     'what are the characteristics of 2015 barolo wines from piedmont?'
Expanded:       'what are the characteristics of 2015 barolo wines from piedmont? ...'

Detected entities:
  Vintages: 2015
  Appellations: Barolo
  Regions: piedmont

======================================================================
  2. Retrieval (Hybrid)
======================================================================

Using hybrid retrieval (Vector + BM25 with RRF fusion)
Retrieved 10 documents in 0.234s

Top 3 results:
  [1] Similarity: 0.876
      Source: barolo_guide.pdf, Page: 45
      Preview: The 2015 vintage in Barolo was exceptional, with perfect ripening conditions...

...
```

The script provides detailed output for each stage, showing exactly how the pipeline processes queries.

---

## 9. Troubleshooting

### Common Issues

**No results returned**
- Check similarity threshold (try lowering to 0.2)
- Verify ChromaDB is running and accessible
- Ensure documents were indexed successfully

**Low relevance results**
- Enable query expansion
- Check wine terminology normalization
- Consider reranking for better precision

**Slow retrieval**
- Enable query caching
- Reduce n_results parameter
- Check ChromaDB HNSW settings

### Logging

Enable debug logging to trace retrieval:
- Query preprocessing transformations
- Cache hits/misses
- Similarity scores for each result
- Reranking score changes

---

## References

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain RAG](https://python.langchain.com/docs/tutorials/rag/)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

