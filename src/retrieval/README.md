# Retrieval module

> **Project version**: 0.7.3 — last verified 2026-08-12.

This module implements the shared Milestone 3 retrieval path used by the RAG-only API, evaluation
harness, and agent wine-knowledge tools. It combines deterministic query planning, synchronized
dense and BM25 candidate generation, metadata-aware ordering, local cross-encoder reranking,
confidence reporting, semantic deduplication, optional compression, and source attribution.

## Components

| File | Responsibility |
|---|---|
| `query_utils.py` | Query normalization and the legacy optional expansion helper |
| `query_analyzer.py` | Entity/intent analysis and channel-specific `RetrievalQueryPlan` construction |
| `bm25_analyzer.py` | Unicode tokenization, wine-term canonicalization, and question-filler removal |
| `vector_retriever.py` | Chroma dense search with thresholding and optional LRU caching |
| `keyword_search.py` | Persisted BM25 index using the shared analyzer |
| `hybrid_retriever.py` | Balanced dense/sparse candidate union; unweighted RRF only as fallback |
| `reranker.py` | Cross-encoder reranking over contextual search text |
| `confidence.py` | Normalized retrieval confidence and low-confidence classification |
| `context_builder.py` | Semantic deduplication, context formatting, and display sources |
| `query_compression.py` | Optional local TF-IDF extractive compression |
| `factory.py` | Config-driven, BM25-sync-aware retriever and reranker construction |
| `rag_service.py` | Shared execution path and serializable stage artifacts |

## Production query path

```text
user query
  -> normalize + extract wine entities + detect intent
  -> build semantic_query and sparse_query
  -> dense candidate pool (25) + BM25 candidate pool (25)
  -> alternating, de-duplicated balanced union (maximum 50)
  -> metadata match boost
  -> cross-encoder rerank and score threshold (0.0)
  -> confidence calculation
  -> semantic deduplication
  -> formatted context and source artifacts
  -> optional TF-IDF compression (disabled)
  -> optional answer generation
```

`execute_production_rag()` owns this order. It is called by:

- `/api/chat` in `rag_only` mode, with generation enabled;
- `src.eval` retrieval/full runs, so evaluation exercises production behavior;
- agent tools in `src/agents/tools/rag_tools.py`, with generation disabled because the LangGraph
  agent owns final synthesis.

The service exposes `normalized_query`, the complete query plan, raw retrieved candidates, final
context chunks, per-feature usage, confidence, threshold, sources, and any retrieval error. This
makes ranking regressions inspectable rather than hiding them behind the generated answer.

## Query planning

`build_retrieval_query_plan()` is deterministic and local; it makes no LLM call.

1. Normalize known spelling and terminology variations.
2. Extract grapes, regions, vintages, classifications, producers, and appellations.
3. Detect supported intents: flavour, aging, pairing, classification, or region.
4. When both an entity and intent are explicit, build a compact semantic query from canonical
   entities plus intent vocabulary and a separate sparse query with exact keyword vocabulary.
5. Analyze the sparse text with the same BM25 analyzer used for indexed documents.

For example, the flavour form of a Nebbiolo question keeps `Nebbiolo` and adds sensory concepts
such as aroma, flavour, taste, tannin, acidity, and body. The sparse side removes conversational
filler while retaining discriminative wine terms. Legacy broad query expansion remains available
on `ChromaRetriever`, but the production factory disables it because the typed plan supplies the
reviewed channel-specific inputs.

Entity detection currently supplies metadata boosting, not hard Chroma filtering. This avoids
zero-result failures from incomplete extracted metadata while still preferring explicit matches.

## BM25 and synchronization

BM25 adds value by recovering exact entity/terminology evidence that dense similarity can miss.
Documents and queries share `analyze_bm25_text()`, which:

- normalizes Unicode, apostrophes, hyphens, case, and diacritics;
- applies longest-match grape, region, and misspelling aliases;
- removes common question filler without stemming away wine names.

The persisted index stores the clean `document`, contextual `search_text`, metadata, and stable
chunk ID. `build_retriever_from_config()` enables hybrid retrieval only when the BM25 sidecar
manifest matches the active Chroma collection name, record count, sorted-ID hash, and configured
index path. A missing, empty, invalid, or stale index produces an explicit logged vector-only
fallback.

## Hybrid candidate generation

The primary hybrid path does not blend incomparable vector and BM25 scores with fixed weights.
Each channel contributes its complete configured pool, candidates are de-duplicated by chunk ID,
and alternating channel ranks produce a deterministic union. Every candidate retains:

- `dense_rank` and `dense_similarity`;
- `sparse_rank` and `bm25_score`;
- `retrieval_channels`;
- pool sizes and dense/sparse latency diagnostics.

The local cross-encoder then compares all admitted candidates on one scoring scale. If no reranker
is available, `HybridRetriever` uses standard unweighted reciprocal-rank fusion as an explicit
fallback and returns the requested count. The previously described 70/30 weighted fusion is not
part of the current implementation.

## Reranking, threshold, and confidence

Metadata matches are applied before reranking. `DocumentReranker` scores the normalized user query
against the same contextual text construction used at index time, while final evidence still uses
the clean document body.

The accepted `rerank_threshold` is `0.0`: negative cross-encoder logits are filtered. Confidence is
the normalized maximum reranker score and is compared with the provisional
`min_retrieval_confidence` of `0.3`. The result reports `low_confidence`, but automatic web fallback
remains disabled until a real failure cohort can calibrate that boundary.

HyDE expansion and web fallback flags exist in execution artifacts for later phases but are not
active in the current Phase 0 path.

## Usage

Use factories and the shared service for production-equivalent behavior:

```python
from src.retrieval.factory import build_reranker_from_config, build_retriever_from_config
from src.retrieval.rag_service import execute_production_rag
from src.utils import get_config

cfg = get_config()
retriever = build_retriever_from_config(cfg)
reranker = build_reranker_from_config(cfg)

result = execute_production_rag(
    prompt="What are the primary flavour characteristics of Nebbiolo?",
    config=cfg,
    model=None,
    retriever=retriever,
    reranker=reranker,
    message_history=[],
    generation_enabled=False,
    n_results_override=10,
)

for chunk in result.context_chunks:
    print(chunk.id, chunk.rerank_score, chunk.retrieval_channels)
```

Direct component construction is appropriate for ablations, but it does not by itself reproduce
the production stage order.

## Configuration

Active `chroma.retrieval` defaults:

| Setting | Value | Meaning |
|---|---:|---|
| `n_results` | 5 | Normal final result count |
| `similarity_threshold` | 0.3 | Minimum dense similarity before union |
| `enable_hybrid` | true | Request dense + verified BM25 retrieval |
| `semantic_candidate_pool` | 25 | Dense candidates admitted before union |
| `bm25_candidate_pool` | 25 | Sparse candidates admitted before union |
| `reranker_input_limit` | 50 | Maximum unique candidates sent to reranking |
| `validate_bm25_sync` | true | Require a matching synchronization manifest |
| `enable_reranking` | true | Use the local cross-encoder |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `rerank_top_k` | 5 | Normal post-rerank result count |
| `rerank_threshold` | 0.0 | Reject negative reranker logits |
| `min_retrieval_confidence` | 0.3 | Provisional low-confidence boundary |
| `enable_metadata_boost` | true | Prefer explicit entity matches |
| `metadata_boost_factor` | 0.1 | Pre-rerank increment per entity match |
| `use_deduplication` | true | Remove semantically redundant final chunks |
| `deduplication_threshold` | 0.9 | Duplicate cosine-similarity boundary |
| `enable_compression` | false | Keep full retrieved context by default |

## Accepted Phase 0 evidence

The final 24 scorable retrieval cases (25 total, one intentionally unsupported) produced:

| Metric | Result |
|---|---:|
| MRR | 0.8368 |
| Precision@3 | 0.6250 |
| Precision@5 | 0.5833 |
| Mean latency | 1,351 ms |

The hybrid ablation reached Recall@10 0.9208, versus 0.5854 for vector-only and 0.3781 for
BM25-only. The failing Nebbiolo flavour query returned relevant evidence at rank 1 and 9 of its
first 10 raw candidates were judged relevant, with no contents/reference/interleaved-column/OCR
artifact in that top 10.

Generated reports live under ignored `eval-results/`. The accepted run is
`20260812T133822_retrieval_rag.json`; the exact commands and closure criteria are recorded in
[`docs/m03-phase0-corrective-manual-testing.md`](../../docs/m03-phase0-corrective-manual-testing.md).
