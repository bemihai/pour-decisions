# Retrieval Module

> **Project version:** 0.7.3 — last verified 2026-08-01.
> Milestone 3 (Phases 3–5) will add `HyDEExpander`, `RetrievalConfidenceSignal`, and
> `WebSearchFallback` to this module, and will modify `vector_retriever.py` and
> `hybrid_retriever.py`. The reranker threshold (currently 0.0) will also be activated.
> Update this README when those phases are implemented.
> See `design/roadmap/agentic-ai/milestones/m03-rag-quality-foundation.md`.

The `retrieval` module implements the query-time pipeline for searching the ChromaDB vector store. It covers query preprocessing, hybrid search, reranking, context compression, and context formatting.

## Components

| File | Class / Function | Purpose |
|------|------------------|---------|
| `vector_retriever.py` | `ChromaRetriever` | Vector similarity search against ChromaDB with query expansion and LRU caching |
| `keyword_search.py` | `BM25Index` | BM25 keyword search, persisted as pickle (`chroma-data/bm25_index.pkl`) |
| `hybrid_retriever.py` | `HybridRetriever` | Reciprocal Rank Fusion of vector (70%) + BM25 (30%) results |
| `reranker.py` | `DocumentReranker` | Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) |
| `query_utils.py` | `normalize_query`, `expand_query` | Wine term normalization (misspellings, synonyms, regions) and expansion |
| `query_analyzer.py` | `analyze_query`, `boost_by_metadata_match`, `QueryAnalysis` | Extract grape/region/vintage/appellation entities and build ChromaDB metadata filters |
| `query_compression.py` | `compress_context` | Local TF-IDF extractive compression to reduce token usage |
| `context_builder.py` | `build_context_from_chunks`, `build_semantic_context`, `format_sources_for_display` | Context formatting, semantic deduplication, and source citation |
| `factory.py` | `build_retriever_from_config`, `build_reranker_from_config` | Construct the configured production retrieval resources |
| `rag_service.py` | `execute_production_rag` | Shared retrieval, context, artifact, and optional generation orchestration for API, eval, and agent tools |

## Query Flow

The full pipeline below is owned by `execute_production_rag()` and is used by the **RAG-only chat
endpoint** (`/api/chat` with `agent_mode=rag_only`), the eval harness, and agent RAG tools.

```
User query
  |
  v
query_utils.normalize_query()              # fix misspellings, canonical synonyms
  |
  v
query_utils.expand_query()                 # add related wine terms
  |
  v
query_analyzer.analyze_query()             # extract entities -> QueryAnalysis
  |
  v
HybridRetriever.retrieve()                 # vector + BM25 via RRF
  |   (or ChromaRetriever if hybrid disabled)
  v
query_analyzer.boost_by_metadata_match()   # score boost for entity matches
  |
  v
DocumentReranker.rerank()                  # cross-encoder precision pass
  |                                        # NOTE: uses rerank(), not rerank_with_threshold()
  |                                        # threshold is effectively 0.0 — all docs pass
  v
context_builder.build_semantic_context()   # semantic dedup + format for LLM
  |   (or build_context_from_chunks if deduplication disabled)
  v
query_compression.compress_context()       # optional TF-IDF compression (disabled by default)
```

### Agentic path — shared retrieval without generation

When the chat endpoint runs in `intelligent` agent mode, wine knowledge queries are
handled by `@tool`-decorated functions in `src/agents/tools/rag_tools.py`
(`search_wine_knowledge`, `search_wine_region_info`, etc.).

These tools build resources through `build_retriever_from_config()` and
`build_reranker_from_config()`, then call `execute_production_rag(generation_enabled=False)`.
They therefore use the same configured normalization, hybrid retrieval, metadata boosting,
reranking, deduplication, compression, context construction, and internal artifacts as API and
eval. Final-answer generation remains disabled inside the tool because the LangGraph agent owns
answer synthesis.

## Usage

### Basic Retrieval

```python
from src.retrieval import ChromaRetriever
from src.utils import initialize_chroma_client, get_config

cfg = get_config()
client = initialize_chroma_client(cfg.chroma.client.host, cfg.chroma.client.port)

retriever = ChromaRetriever(
    client=client,
    collection_name="wine_books",
    embedding_model=cfg.chroma.settings.embedder,
    n_results=5,
    similarity_threshold=0.3,
)

results = retriever.retrieve("What grapes go into Champagne?")
```

### Hybrid Search

```python
from src.retrieval import ChromaRetriever, BM25Index, HybridRetriever

bm25 = BM25Index(index_path="chroma-data/bm25_index.pkl")

hybrid = HybridRetriever(
    vector_retriever=retriever,
    bm25_index=bm25,
    vector_weight=0.7,
    keyword_weight=0.3,
)

results = hybrid.retrieve("Barolo DOCG aging requirements", n_results=10)
```

### Full Pipeline with Reranking

```python
from src.retrieval import (
    HybridRetriever, DocumentReranker,
    analyze_query, boost_by_metadata_match,
    build_semantic_context,
)

results = hybrid.retrieve(query, n_results=10)

reranker = DocumentReranker()
results = reranker.rerank(query, results, top_k=5)

analysis = analyze_query(query)
if analysis.has_filters:
    results = boost_by_metadata_match(results, analysis)

context = build_semantic_context(results, embedding_model=cfg.chroma.settings.embedder)
```

## Configuration

All settings live under `chroma.retrieval` in `app_config.yml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `n_results` | 5 | Chunks to retrieve |
| `similarity_threshold` | 0.3 | Minimum similarity score |
| `enable_hybrid` | true | Use hybrid vector + BM25 search |
| `hybrid_vector_weight` | 0.7 | Vector search weight in RRF |
| `hybrid_keyword_weight` | 0.3 | BM25 weight in RRF |
| `bm25_index_path` | `chroma-data/bm25_index.pkl` | Persisted BM25 index location |
| `enable_reranking` | true | Cross-encoder reranking pass |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `rerank_top_k` | 5 | Results after reranking |
| `enable_compression` | false | TF-IDF context compression |
| `compression_max_chars` | 8000 | Max compressed context length |
| `enable_metadata_boost` | true | Boost results matching query entities |
| `metadata_boost_factor` | 0.1 | Score increment per entity match |
| `use_deduplication` | true | Semantic deduplication |
| `deduplication_threshold` | 0.9 | Cosine similarity threshold for dedup |

## Wine Terminology

Query normalization and expansion rely on JSON dictionaries in `src/utils/terminology/`:

- `grape_synonyms.json` - Maps canonical grape names to synonyms
- `misspellings.json` - Common wine term misspellings
- `region_variations.json` - Region name variations (e.g., "Bourgogne" -> "Burgundy")
- `query_expansions.json` - Keyword expansion mappings
- `classifications.json` - Wine classification systems (AOC, DOCG, etc.)
- `wine_appellations.json` - Known wine appellations

These are loaded by `src/utils/terms.py` and re-exported from `src/utils`.
