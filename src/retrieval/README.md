# Retrieval module

> **Project version**: 0.8.4 — last verified 2026-09-04.

This module finds and prepares book evidence for a user question. It owns deterministic query
planning, dense and BM25 search, candidate union, reranking, confidence, deduplication, source
formatting, and the shared production execution service. It does not own document extraction or the
agent's final answer synthesis.

For the complete indexing-to-answer explanation, start with
[`docs/pour-decisions-rag-pipeline.md`](../../docs/pour-decisions-rag-pipeline.md).

## What happens here

```text
question
  -> normalized semantic and keyword queries
  -> 25 dense candidates + 25 verified-BM25 candidates
  -> balanced unique union (at most 50)
  -> metadata preference
  -> local cross-encoder reranking and threshold
  -> confidence and semantic deduplication
  -> up to five clean, sourced context passages
```

In plain terms, dense search finds passages with similar meaning, while BM25 finds exact words such
as grape names and regions. The module combines both lists without pretending their scores are
directly comparable, then uses a more precise local model to decide which passages best answer the
question.

## Component map

| File | Responsibility |
|---|---|
| `query_utils.py` | Query normalization and legacy optional expansion helper |
| `query_analyzer.py` | Entity/intent analysis and `RetrievalQueryPlan` construction |
| `bm25_analyzer.py` | Shared Unicode and wine-terminology analyzer |
| `vector_retriever.py` | Dense Chroma search, thresholding, and optional cache |
| `keyword_search.py` | Persisted BM25 index and keyword search |
| `hybrid_retriever.py` | Balanced candidate union and unweighted-RRF fallback |
| `reranker.py` | Cross-encoder pair scoring and thresholding |
| `confidence.py` | Normalized confidence and low-confidence flag |
| `context_builder.py` | Semantic deduplication, context formatting, and sources |
| `query_compression.py` | Optional local TF-IDF compression; disabled |
| `web_fallback.py` | Optional cached web-result adapter; disabled by default |
| `factory.py` | Config-driven construction and BM25 synchronization validation |
| `rag_service.py` | Shared stage order and serializable production artifacts |

## Shared production entry point

`execute_production_rag()` owns the stage order. Use it whenever behavior should match production:

- `/api/chat` uses it with generation enabled in RAG-only mode;
- `src.eval` uses it so measured retrieval is the deployed retrieval path;
- `src/agents/tools/rag_tools.py` uses it with generation disabled because the LangGraph agent
  performs final synthesis.

The returned result includes the normalized query, complete query plan, raw candidates, final
chunks, confidence and threshold, feature usage, sources, timings, and errors. These artifacts make
ranking behavior inspectable before looking at a generated answer.

```python
from src.retrieval.factory import build_reranker_from_config, build_retriever_from_config
from src.retrieval.rag_service import execute_production_rag
from src.utils import get_config

config = get_config()
result = execute_production_rag(
    prompt="What are the primary flavour characteristics of Nebbiolo?",
    config=config,
    model=None,
    retriever=build_retriever_from_config(config),
    reranker=build_reranker_from_config(config),
    message_history=[],
    generation_enabled=False,
)
```

Construct components directly only for explicit diagnostics or controlled ablations; doing so does
not automatically reproduce the production stage order.

## Stage contracts

### Query plan

`build_retrieval_query_plan()` makes no LLM call. It normalizes terminology, extracts grapes,
regions, vintages, classifications, producers, and appellations, detects supported intents, and
builds channel-specific queries. Metadata entities are used as preferences rather than hard filters
because extracted metadata may be incomplete.

### BM25 synchronization

`build_retriever_from_config()` enables hybrid retrieval only when the sidecar manifest matches the
active Chroma collection name, record count, sorted-ID hash, and configured BM25 path. Missing,
empty, invalid, or stale state is logged and produces vector-only retrieval. The analyzer used for
BM25 documents is also used for queries.

### Candidate union

Each channel contributes its complete configured pool. Alternating ranks build a deterministic,
de-duplicated union, while each candidate retains dense/BM25 ranks, scores, channel provenance,
pool sizes, and timings. Fixed 70/30 score blending is not part of production. If reranking is
unavailable, standard unweighted reciprocal-rank fusion is the explicit fallback.

### Reranking and confidence

Metadata matches add a small pre-rerank preference. The local cross-encoder reads the normalized
question with the same contextual search text used during indexing. The active `0.0` threshold
removes negative logits before the best five are selected.

Confidence is a normalized form of the strongest reranker score. Below `0.3` means evidence looks
weak; it does not mean a plausible passage is current or correct. Automatic web fallback remains
off because its focused quality improvement did not justify routine external cost and latency.

### Final context

Semantic deduplication removes near-repeated passages. The context builder presents clean document
bodies and source metadata, not contextual prefixes. TF-IDF compression and small-to-big expansion
remain disabled to preserve full evidence and keep the path understandable.

## Active invariants and failure behavior

- Production, evaluation, and agent tools share `execute_production_rag()`.
- Dense, BM25, and reranking use the shared contextual representation.
- Only a verified BM25 index can participate in hybrid search.
- Candidate channel ranks and scores remain available after union and reranking.
- Metadata can prefer a result but cannot exclude all results.
- Optional web failure preserves the original book result.
- A retrieval error is returned as an inspectable artifact; callers own user-facing behavior.

The complete active values live in `app_config.yml`. The canonical guide provides the readable
snapshot and evaluation decisions without duplicating that configuration in every module README.

## Verification and extension guidance

Focused tests under `tests/retrieval/` cover query plans and analyzers, dense/BM25 behavior,
synchronization validation, balanced union, score provenance, reranking, thresholds, confidence,
deduplication, fallback behavior, and the shared service.

Before changing a stage, create or identify a frozen failure cohort, change one meaningful variable,
and compare common-sample quality, support counts, latency, resource use, and external calls. A small
metric gain does not justify an extra model call or complicated production branch by itself. See
[`src/eval/README.md`](../eval/README.md) for the evaluation workflow.
