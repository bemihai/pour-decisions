# RAG Pipeline Deep Dive

> **Project version**: 0.7.3 — last verified 2026-05-27.
> Traces the indexing and query pipeline as it currently exists. Milestone 3 (noise filtering,
> contextual enrichment, HyDE, reranker threshold, web fallback) will change several sections here.
> See `design/agentic/planning/3-rag-quality-foundation.md` for the planned changes.

This document explains the current RAG pipeline step by step, tracing every component from document
ingestion through to the final LLM answer. It is intended as a reference before working on
Milestone 3 improvements.

---

## Part 1 — Indexing Pipeline

The indexing pipeline converts raw wine books (PDF / EPUB) into searchable vector embeddings stored
in ChromaDB, plus a BM25 keyword index persisted on disk.

### Entry point

`make chroma-upload` runs `python -m src.chroma.load_data`, which calls `main()` in
`src/chroma/load_data.py`. For each collection declared in `app_config.yml` (`wine_books` by
default), it instantiates a `CollectionDataLoader` and calls `load_directory()`.

### Step 1 — File discovery and change detection (`IndexTracker`)

`load_directory()` scans the configured `local_data_path` for `.pdf` and `.epub` files. It then
consults `IndexTracker`, which maintains a JSON manifest at
`chroma-data/manifests/{collection_name}_manifest.json`. Only files whose MD5 hash has changed (or
that are new) are queued for processing. This makes subsequent runs cheap.

Setting `force_reindex=True` bypasses the manifest and reprocesses every file.

### Step 2 — Document parsing (`split_file` in `src/chroma/chunks.py`)

Each file is parsed by the `unstructured` library (`partition()`), which returns a list of typed
elements (Title, NarrativeText, ListItem, etc.). The chunking strategy is then applied. Three
strategies are supported (configured under `chroma.chunking.strategy`):

| Strategy | How it works | Current config |
|----------|-------------|----------------|
| `basic` | Fixed-size character windows with overlap | - |
| `by_title` | Splits at `Title` element boundaries; falls back to character limit | **active** (1024 chars, 256 overlap) |
| `semantic` | Embeds sentences, breaks where cosine distance jumps (percentile threshold) | - |

All strategies return a list of `{"id", "text", "metadata", "importance_score"}` dicts.

### Step 3 — Document context extraction (`extract_document_context`)

Called once per file, before chunking. It scans the first few parsed elements to extract:
- `document_title` — first `Title` element or the first non-empty element
- `chapter` — the last element matching `^(Chapter|Part|Section)\s+\d+`
- `section` — the last short title element (< 100 chars)

These three fields are stored on **every** chunk from the file. They are not per-chunk values;
they reflect the document-level context at the time of the last heading seen. This is a limitation:
chunks from early sections will carry the section heading of the last heading found in the entire
document, not the heading of their own section.

### Step 4 — Wine metadata extraction (`extract_wine_metadata`)

For each chunk text, `metadata_extractor.py` runs regex-based NER using curated term dictionaries
loaded from `src/utils/terminology/`:

| Field | Method |
|-------|--------|
| `grapes` | Regex word-boundary match against `GRAPE_PATTERNS` (100+ varieties + synonyms) |
| `regions` | Regex match against `REGION_PATTERNS` |
| `vintages` | 4-digit years in range 1900–2050 |
| `classifications` | Match against `CLASSIFICATION_PATTERNS` (DOCG, AOC, AVA, etc.) |
| `producers` | Regex for `Château X`, `Domaine Y`, `X Winery`, `Y Vineyards` patterns |
| `appellations` | Exact match against `WINE_APPELLATIONS` list |

Results are stored as comma-separated strings in chunk metadata (ChromaDB requires string values
for filtering). This enables `$contains` filter queries at retrieval time.

### Step 5 — Chunk assembly (`ChunkMetadata`)

A `ChunkMetadata` dataclass is populated with all standard document fields (filename, page number,
word count, etc.) plus the document context and wine entity fields. The dataclass is serialized to
a plain `dict` for storage.

Chunk ID format: `{filename_stem}_{chunk_index}_{content_hash[:8]}`

### Step 6 — Deduplication check (content hash)

Before embedding, `process_file()` queries ChromaDB with `where={"content_hash": hash}` for each
chunk. If a match exists, the chunk is skipped (`chunks_skipped` counter incremented). This
prevents the same content from being indexed twice across different runs or files.

### Step 7 — Embedding generation

The surviving chunks are passed to `embedder.embed_documents(docs)` where `embedder` is a cached
`HuggingFaceEmbeddings` instance loaded via `get_embedder()`. The model is configured in
`app_config.yml` under `chroma.settings.embedder` (set by the `EMBEDDING_MODEL` env var).

The raw chunk text is embedded — there is **no** context-prefix enrichment at this stage. The
embedding vector is context-free: a chunk saying "Premier Cru" from a Burgundy chapter and one
from a Bordeaux chapter produce identical embeddings.

### Step 8 — Batch upsert to ChromaDB

Chunks are inserted in batches of up to 2500 via `collection.add()`. Each record stores:
- `id` — the chunk ID
- `embedding` — the dense vector
- `document` — the raw chunk text
- `metadata` — the full `ChunkMetadata` dict

### Step 9 — BM25 index (separate path)

The BM25 index at `chroma-data/bm25_index.pkl` is **not** rebuilt by `load_data.py`. It is a
pre-built pickle that must be regenerated manually (or by a dedicated make target). The index is
built via `BM25Index.build_index(documents)` from `src/retrieval/keyword_search.py`, which
tokenizes each document with simple lowercase word splitting and constructs a `BM25Okapi` model.

**Gap**: `load_data.py` does not trigger BM25 index construction. Indexing new documents without
rebuilding the BM25 index leaves the keyword search out of sync with ChromaDB.

---

## Part 2 — Query Pipeline

There are two query paths: the **RAG-only path** (used when `agent_mode=rag_only` in the chat
API) and the **agentic path** (LangGraph ReAct agent with tools). This section focuses on the
RAG-only path since it is where the retrieval pipeline runs directly. The agentic path calls
`search_wine_knowledge` and related tools in `rag_tools.py`, which bypass most pipeline features
(see note below).

### Step 1 — Query normalization (`normalize_query` in `src/retrieval/query_utils.py`)

The raw user query is lowercased, then:
1. Common misspellings are corrected (from `misspellings.json`)
2. Grape synonyms are replaced with canonical names (e.g., "Shiraz" → "Syrah")
3. Region variations are canonicalized (e.g., "Bourgogne" → "Burgundy")

This step happens inside `ChromaRetriever._preprocess_query()` before embedding.

### Step 2 — Query expansion (`expand_query`)

If `enable_query_expansion=True` (default), related wine terminology is appended to the query
based on keyword matches in `query_expansions.json`. For example, a query mentioning "tannins"
might have "structure" and "astringency" appended.

### Step 3 — Query entity analysis (`analyze_query` in `src/retrieval/query_analyzer.py`)

The query is analyzed using the same extraction functions as the metadata extractor. The result
is a `QueryAnalysis` dataclass containing lists of detected grapes, regions, vintages, and
appellations. This is used in two ways:
- Building a ChromaDB `where` filter for metadata-scoped retrieval (not used in the current
  RAG-only path — the filter is available but the chat route does not apply it to the retriever
  call)
- Driving metadata boosting after retrieval

### Step 4 — Retrieval (`HybridRetriever` or `ChromaRetriever`)

The retriever is preloaded at FastAPI startup in `app.state`. When `enable_hybrid=true` in config,
a `HybridRetriever` is used; otherwise `ChromaRetriever`.

**Vector retrieval** (`ChromaRetriever.retrieve()`):
1. Embed the preprocessed query using the same HuggingFace model as indexing
2. Query ChromaDB HNSW index with cosine similarity
3. Filter results by `similarity_threshold` (0.3 by default)
4. Return up to `n_results * 2` candidates (doubled when reranking is enabled so reranker has
   more to work with)

**BM25 retrieval** (`BM25Index.search()`):
1. Tokenize query by lowercase word split
2. Score all documents in the BM25 index using BM25Okapi
3. Return top `n_results * 2` documents with positive scores

**Fusion** (`HybridRetriever._reciprocal_rank_fusion()`):
Both result lists are merged using Reciprocal Rank Fusion:
`score = weight / (k + rank)` where `k=60`, `vector_weight=0.7`, `keyword_weight=0.3`.
Results from both lists receive additive scores. Final list is sorted descending by fused score
and trimmed to `n_results`.

**LRU cache**: `ChromaRetriever` caches results keyed by (query, n_results, where, where_document)
using an MD5 hash, with LRU eviction at 100 entries.

### Step 5 — Metadata boosting (`boost_by_metadata_match`)

If `enable_metadata_boost=true` and the query analysis detected entities, each retrieved document
gets a score boost of `metadata_boost_factor * number_of_matching_entity_fields` (default
`+0.1` per match, capped at 1.0). Documents whose metadata contains the exact entities from the
query (e.g., `grapes` field contains "Nebbiolo") are re-ranked higher. The list is re-sorted by
boosted similarity score.

### Step 6 — Cross-encoder reranking (`DocumentReranker.rerank()`)

If `enable_reranking=true`, the `cross-encoder/ms-marco-MiniLM-L-6-v2` model scores every
(query, document) pair jointly. This is more accurate than bi-encoder similarity because query
and document are processed together.

Key detail: the current code calls `reranker.rerank()` — **not** `rerank_with_threshold()`. The
threshold variant exists (`rerank_with_threshold()`, default threshold `0.0`) but is not activated
in the production call site. All retrieved documents pass regardless of their rerank score.

Results are sorted by `rerank_score` descending and trimmed to `rerank_top_k` (default 5).

### Step 7 — Small-to-big expansion (optional, disabled by default)

If `enable_small_to_big=true`, each retrieved chunk is replaced with its `parent_context` stored
in metadata (the larger surrounding window). This feature is disabled in the current config.

### Step 8 — Semantic deduplication (`build_semantic_context`)

If `use_deduplication=true` (default), `build_semantic_context()` is called which runs
`deduplicate_context()` from `src/chroma/deduplication.py`:
1. **Hash-based pass**: remove exact duplicates by `content_hash`
2. **Semantic pass**: embed all surviving chunks, compute pairwise cosine similarity, discard any
   chunk with similarity > `deduplication_threshold` (0.9) to an earlier chunk

The deduplicated chunks are then formatted into a single string, with each chunk prefixed by
`[Source N - filename, Page P, Chunk id]`.

### Step 9 — Context compression (optional, disabled by default)

If `enable_compression=true`, `compress_context()` is applied on the formatted context string:
1. Remove redundant sentences using Jaccard overlap (threshold 0.8)
2. TF-IDF extractive compression (keep top sentences by score)
3. Hard truncation at `compression_max_chars` (8000 chars) as last resort

This feature is disabled by default (`enable_compression: false`).

### Step 10 — LLM answer generation

The formatted context string plus the conversation history are passed to the LLM via
`process_user_prompt()` in `src/agents/llm.py`. The LLM generates an answer grounded in the
retrieved context. The answer is returned to the chat API alongside source citations.

---

## Part 3 — Agentic Path and RAG Tools

When `agent_mode=intelligent`, the pipeline runs through the LangGraph ReAct agent. The agent calls
`@tool`-decorated functions in
`src/agents/tools/rag_tools.py` (`search_wine_knowledge`, `search_wine_region_info`, etc.).

**Important**: the RAG tools bypass most of the pipeline described above. They create a fresh
`ChromaRetriever` directly (not a `HybridRetriever`), call `.retrieve()` on it, and pipe results
straight to `build_context_from_chunks()`. The following pipeline features are **not used** by the
agent RAG tools:

- Hybrid BM25 search
- Cross-encoder reranking
- Metadata boosting
- Semantic deduplication
- Context compression

The agentic path therefore delivers simpler, cheaper retrieval per tool call, relying on the
agent's ability to call tools multiple times or rephrase queries to compensate. The deprecated
keyword-routing agent has been removed; the supported chat modes are `intelligent` and `rag_only`.

---

## Part 4 — Configuration Reference

All retrieval knobs live under `chroma.retrieval` in `app_config.yml`. Current active settings:

| Setting | Value | Effect |
|---------|-------|--------|
| `n_results` | 5 | Final chunks returned to LLM |
| `similarity_threshold` | 0.3 | Drop vector results below this |
| `enable_hybrid` | true | Use HybridRetriever (vector + BM25) |
| `hybrid_vector_weight` | 0.7 | Vector share in RRF |
| `hybrid_keyword_weight` | 0.3 | BM25 share in RRF |
| `bm25_index_path` | `chroma-data/bm25_index.pkl` | Loaded at startup |
| `enable_reranking` | true | Cross-encoder pass after retrieval |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | |
| `rerank_top_k` | 5 | Results after reranking |
| `enable_compression` | false | TF-IDF compression disabled |
| `compression_max_chars` | 8000 | Limit if compression enabled |
| `enable_metadata_boost` | true | Entity-match score boosting |
| `metadata_boost_factor` | 0.1 | Per-entity boost increment |
| `use_deduplication` | true | Semantic dedup before context build |
| `deduplication_threshold` | 0.9 | Similarity above which = duplicate |

---

## Part 5 — Known Deficiencies (Context for Milestone 3)

These are the four concrete problems the Milestone 3 spec is addressing:

1. **No noise filtering at index time** (`src/chroma/chunks.py`): `split_file()` applies no
   quality gate. Table of contents pages, bibliography entries, index pages, and heading-only
   chunks all get embedded and indexed alongside prose content. These structural chunks look
   meaningful to the BM25 index and can surface before actual wine content.

2. **Context-free embeddings** (`src/chroma/loader.py`): The raw chunk text is passed directly to
   `embedder.embed_documents()`. No document, chapter, or section context is prepended. A chunk
   about "Premier Cru" reads identically to the embedding model whether it comes from a Burgundy
   chapter or a Bordeaux chapter.

3. **BM25 index includes noise** (`src/chroma/load_data.py`): The BM25 index is built from all
   chunks without quality filtering, compounding problem 1 for keyword searches.

4. **Reranker threshold is effectively zero** (`src/retrieval/reranker.py`, line 81): The
   production call site uses `reranker.rerank()` (not `rerank_with_threshold()`). The threshold
   method exists and is implemented, but is unused. All candidates pass the reranker regardless
   of how low their cross-encoder score is.
