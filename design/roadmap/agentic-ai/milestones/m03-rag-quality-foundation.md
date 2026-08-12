# Milestone 3: RAG Quality Foundation — Implementation Specification

> **Project version:** 0.7.3 — last updated 2026-08-12.

**Status**: In implementation — Delivery Gate 0, delivery step 2, and Phase 0 are complete. The
accepted 2026-08-12 post-reindex evidence closes the PDF/EPUB structure and hybrid
candidate-generation correction; Phase 1 is the next delivery step.
**Prerequisite**: Milestone 2 (Eval Harness & Golden Dataset) and its M2R-B API/eval production-RAG
parity are in place, as evidenced by the accepted baseline artifacts in Section 0. Gate 0 extends
that parity to agent RAG tools before any quality behavior changes.
**Effort estimate**: 4 weekends — shared-path and index-synchronization safety first, reranker
confidence second, extraction/indexing quality third, and eval-gated fallback/HyDE last.
**Feature flags**: Runtime-facing sub-features are gated by `app_config.yml`. The extraction
provider and chunking strategy are also config-selected, but removing `unstructured` from the default
path is an intentional implementation change rather than a long-term compatibility mode.

**Design update (2026-07-07)**: Milestone 3 now includes a provider-neutral extraction and chunking
foundation before the original quality phases. The goal was to remove the then-current `unstructured`
dependency path, avoid parser-specific data leaking through the pipeline, and make chunking strategy
selection explicit, testable, and easy to change.

**Historical audit snapshot (2026-06-16)** — retained to show why M3 was prioritized; the status
below predates the implementation recorded later in this document:

| Field | Assessment |
|-------|------------|
| Current status | Not started or not fully implemented; still highly relevant. |
| Blocking prerequisites | M2R-B API/eval parity is satisfied; Delivery Gate 0 must extend parity to agent tools and synchronize BM25 before quality changes. |
| Classification | Core quality milestone. |
| Current recommendation | Do next after M2 remediation; start with reranker threshold/confidence, noise filtering, and contextual enrichment. |
| Remarks | This is the highest-value quality milestone. It addresses actual retrieval failure modes and should come before advanced RAG, planner-executor, multi-agent, or CRAG work. |

---

## 0. Pre-Implementation Baseline and Alignment Audit (2026-07-28)

### 0.1 Baseline artifacts

The pre-M3 baseline uses only the `rag_only` category because all 25 of those samples have curated
`ground_truth_chunk_ids`. The other 35 golden samples do not contribute deterministic retrieval
metrics.

| Artifact | Purpose | Dataset hash | Git state |
|----------|---------|--------------|-----------|
| `eval-results/20260728T084814_retrieval_rag.json` | Deterministic production-RAG retrieval baseline | `3a2be5ba23f772a767035d1659374e09` | `2463daa`, clean |
| `eval-results/20260728T142549_full_rag.json` | Production-RAG generation and judge baseline | `3a2be5ba23f772a767035d1659374e09` | `8e3e5f5`, clean |

Both runs use the shared production RAG target. The full run uses `gemma4:cloud` for answer
generation and `gemma4:31b-cloud` as the Ragas judge, with judge temperature `0.0`,
`max_workers=1`, timeout `120s`, and no judge retries beyond the configured single attempt. Freeze
these model and judge settings for M3 before/after comparisons. Scores produced by the earlier
`gpt-oss:20b-cloud` judge are not comparable with this baseline.

### 0.2 Corpus snapshot

Only the `wine_books` collection is in scope. The separate `wine_test` collection is excluded.

| Field | Baseline |
|-------|----------|
| Collection | `wine_books` |
| Indexed records | 32,798 |
| Embedding dimension | 768 |
| Embedder | `sentence-transformers/all-mpnet-base-v2` |
| Average document length | 736 characters (sampled) |
| Minimum document length | 33 characters (sampled) |
| Maximum document length | 1,024 characters (sampled) |

The current stats command calculates length statistics from at most 100 records. These are sampled
diagnostics, not exact corpus aggregates. The 33-character minimum is evidence that very short
fragments exist, but the exact empty/near-empty rate is not currently measured.

### 0.3 Accepted deterministic retrieval baseline

| Metric | Score | Coverage |
|--------|------:|----------|
| MRR | 0.8533 | 25/25 |
| Precision@3 | 0.5600 | 25/25 |
| Precision@5 | 0.4320 | 25/25 |
| Mean retrieval latency | 932.6 ms | 25/25 |
| Success / error / timeout rate | 1.0 / 0.0 / 0.0 | 25/25 |

This baseline repeated exactly across prior production-RAG retrieval runs and is the primary
deterministic M3 quality gate.

### 0.4 Accepted provisional generation/judge baseline

The user explicitly accepted the following incomplete judge baseline for initial M3 work. It is
directional rather than a strict whole-dataset gate because slow judge jobs timed out. Aggregates
exclude errored metric attempts; missing coverage must never be interpreted as zero.

| Metric | Score | Coverage | Errors |
|--------|------:|----------|-------:|
| Answer relevancy | 0.8924 | 23/25 | 2 timeouts |
| Context precision | 0.7970 | 24/25 | 1 timeout |
| Context recall | 0.6051 | 23/25 | 2 timeouts |
| Faithfulness | 0.9582 | 20/25 | 5 timeouts |
| Mean generation latency | 21,555.7 ms | 25/25 | 0 execution errors |

Faithfulness is particularly vulnerable to upward selection bias because its missing samples include
longer, more complex answers. For any before/after judge comparison:

1. Report metric coverage beside every aggregate.
2. Compare scores over the intersection of sample IDs successfully scored in both runs.
3. Treat a coverage decrease as a regression even if the aggregate score rises.
4. Do not use the provisional judge aggregate as a hard pass/fail gate until coverage is stable.

### 0.5 Alignment decisions

| Status | Misalignment | Resolved decision |
|--------|--------------|-------------------|
| **Resolved for review** | Phases 3D and 3E were specified against `src/agents/tools/rag_tools.py`, but the M3 quality gate executes `src/retrieval/rag_service.py` through the API/eval shared production path. | `execute_production_rag()` is the sole orchestration owner for query normalization, retrieval, metadata boosting, thresholded reranking, confidence, optional web fallback, context construction, and generation. Factories construct optional HyDE and fallback resources. API and eval continue using this path. Agent RAG tools call the same function with `generation_enabled=False`; agent planning/orchestration is unchanged. |
| **Resolved for review** | `make chroma-reindex` rebuilt only ChromaDB and could leave a stale BM25 pickle active. | M3 adds one verified reindex workflow: rebuild Chroma first, rebuild BM25 from the resulting Chroma collection, write BM25 through a temporary file plus atomic replace, and write a synchronization manifest containing collection name, record count, and a hash of sorted chunk IDs. Retrieval validates the manifest and falls back explicitly to vector-only search if validation fails. |
| **Resolved for review** | Phase 0 changes the default extraction dependencies. | The reviewed M3 scope removes `unstructured[all-docs,pdf]` from both dependency declarations and adds `pdfplumber` plus `ebooklib`. PyMuPDF/PyMuPDF4LLM is excluded from M3 entirely; it may be proposed later as a separately reviewed adapter. |
| **Resolved for review** | The detailed spec and `implementation-order.md` disagreed about sequencing. | Delivery order is fixed as: (1) shared-path/BM25 safety, (2) reranker threshold and confidence, (3) extraction/chunking foundation, (4) noise filtering, (5) contextual enrichment, (6) web fallback, (7) HyDE only if its cohort justifies the call, and (8) optional embedding evaluation. Historical labels 3A–3F remain capability names, not execution order. |
| **Planned prerequisite** | The golden dataset has no `semantic_mismatch` or `requires_current_information` cohort, despite Phases 3C and 3E requiring them. The generic `region` tag covers only one RAG sample. | Before its dependent phase, add at least five reviewed `semantic_mismatch` samples, at least five `requires_current_information` samples, and at least five combined region/grape samples. Freeze their IDs and dataset hash in the phase artifact. Do not claim cohort improvements from the 25-sample global average. |
| **Resolved for review** | Exact empty/near-empty rate, chunks per source, web-fallback trigger rate, and actual token usage are not emitted by the current eval/stats reports. | Gate 0 adds exact corpus diagnostics. Phase 5 records fallback use in each result and derives trigger rate. M3 cost acceptance uses recorded model-call estimates and latency; actual token usage is explicitly out of scope unless separate telemetry is approved later. |
| **Comparability risk** | Phase 0 changes chunk boundaries and requires recuration of `ground_truth_chunk_ids`. Pre/post MRR then uses different physical IDs. | Preserve the pre-reindex report and apply the same semantic relevance criteria during recuration. Record the post-reindex dataset hash and treat ID churn as a label migration, not an automatic retrieval regression. |
| **Resolved for review** | Index-time quality controls were incorrectly placed under `chroma.retrieval`, and the proposed query-time metadata filter was incompatible with legacy chunks missing `quality_score`. | Index-time filter and enrichment settings move to `chroma.indexing`. M3 does not add a query-time Chroma `quality_score` predicate: rejected chunks never enter the rebuilt collection or BM25, while retained scores remain metadata for audit and future use. A full reindex is mandatory before enabling the new index. |
| **Resolved for review** | Phase 3E asks for “answer quality,” and milestone acceptance asks for token usage, but full RAG currently exposes relevancy/faithfulness/context metrics and estimated call counts, not answer correctness or token telemetry. | Phase 3E defines answer quality as answer relevancy plus faithfulness over common successfully scored sample IDs, with coverage reported. Its cost signal is fallback trigger count/rate, estimated external calls, and latency. Answer correctness and actual token usage are not M3 gates. |

### 0.6 Approval boundary

Approval of this specification confirms only the future implementation boundaries written here;
implementation still requires a separate explicit request to start:

- the shared production-path adapter and synchronized Chroma/BM25 lifecycle;
- the extraction/chunking structure and the `unstructured` → `pdfplumber`/`ebooklib` dependency
  change;
- the documented config additions/defaults and internal-only RAG result fields;
- extraction of the reusable cached web-search engine from the LangChain tool module into a shared
  service module for Phase 3E;
- the HyDE prompt, but only if the Phase 3 entry gate passes;
- the golden-dataset cohort additions and chunk-ID recuration required by the checkpoints.

It does not authorize public API/schema changes, PyMuPDF/PyMuPDF4LLM, a new retrieval-pipeline
class, production enablement of HyDE or web fallback, an embedding-model switch, or any M4/M12
work. Those remain separate decisions. No implementation begins until the user approves this
specification.

---

## 1. Problem Statement and Current Resolution

The pre-M3 retrieval pipeline had six compounding deficiencies at the indexing and configuration
level, not at the agentic layer. Delivery Gate 0, delivery step 2, and the corrective Phase 0 have
now resolved them as follows:

| # | Pre-M3 problem | Implemented resolution |
|---|----------------|------------------------|
| 1 | Extraction/chunking coupled to `unstructured` objects | Provider-neutral `DocumentElement`/`ChunkCandidate` contracts with configured `pdfplumber` and `ebooklib` adapters |
| 2 | Strategies not cleanly swappable | Extractor/chunker registries and separate ingestion orchestration; reviewed strategies are `section_recursive` and `section_semantic` |
| 3 | Structural noise entered both indexes | Shared structural roles, block-aware rejection, and enforced `ChunkQualityFilter` before embedding |
| 4 | Embeddings lacked document/entry context | One validated contextual search representation shared by dense indexing, BM25, and reranking; stored evidence remains clean |
| 5 | BM25 could contain noise or drift from Chroma | Atomic BM25 rebuild from completed Chroma plus count/ID-hash synchronization manifest and vector-only fallback on mismatch |
| 6 | Production ignored the reranker threshold | Shared production path calls thresholded reranking at `0.0` and emits normalized confidence/low-confidence artifacts |

The observed consequence was structural and off-topic evidence outranking wine prose. The focused
Nebbiolo failure is now the permanent regression diagnostic: the accepted Phase 0 index returns a
direct answer at rank 1, with 9/10 relevant raw candidates and no structural/interleaved/OCR
artifact in that top ten. These were pipeline-foundation defects and were intentionally fixed before
agentic reasoning work in Milestone 12.

---

## 2. Technical Decisions

### 2.1 Libraries

| Concern | Decision | Rationale |
|---------|----------|-----------|
| Extraction abstraction | Define provider-neutral `DocumentExtractor` interface and `DocumentElement` dataclass | Keeps downstream indexing independent of pdfplumber, ebooklib, or any future reviewed provider |
| Default extraction provider | `pdfplumber` for PDF + `ebooklib` for EPUB | Lightweight, local, explicit, and easier to debug than a full document-intelligence framework |
| Optional extraction providers | None in M3 | PyMuPDF/PyMuPDF4LLM is excluded from this milestone and requires a separate licensing/design review |
| Chunking abstraction | Define `DocumentChunker` interface and `ChunkCandidate` dataclass | Allows `section_recursive` and `section_semantic` strategies to share one contract |
| Default chunking strategy | Section-aware recursive chunking | Preserves chapter/section boundaries, stays deterministic, and avoids an extra embedding pass during indexing |
| Semantic chunking | Optional `section_semantic` strategy using the existing local embedder | Useful for long prose sections, but must be eval-gated because it is slower and more sensitive to model/threshold changes |
| Noise detection | Pure Python (re, statistics stdlib) | No ML classifier needed at this corpus size; heuristic rules are transparent and debuggable |
| Contextual enrichment | Rule-based prefix using normalized per-chunk context fields | Zero LLM cost; chapter/section values come from the chunk's own `DocumentElement` lineage, not a file-level scan |
| HyDE expansion | Custom `HyDEExpander` class wrapping existing `src/agents/llm.py` | Full control over blending; avoids LangChain `HypotheticalDocumentEmbedder` black-box |
| Reranker threshold | Existing `DocumentReranker.rerank_with_threshold()` — just enable it | Already implemented, only needs config wiring |
| Query orchestration owner | Extend `execute_production_rag()` in `src/retrieval/rag_service.py` | Keeps API, eval, and agent RAG tools on one measurable production path |
| BM25 synchronization | Rebuild from the completed Chroma collection; atomically replace the pickle and write a chunk-ID manifest | Chroma is the source of truth; stale BM25 is rejected rather than used silently |
| Web fallback | Thin orchestration layer owned by `rag_service.py`, calling a shared cached web-search service extracted from the LangChain tool module | Re-uses cache, TTL, and Tavily logic, avoids a retrieval → agent-tool dependency cycle, and remains visible to production-RAG eval |

### 2.2 No new runtime infrastructure

Most changes live in `src/chroma/` and `src/retrieval/`. Phase 3E moves the reusable web-search
engine/cache code into `src/services/web_search.py`; existing agent tools become thin wrappers.
No new process, Docker service, external provider, database, or migration is required. The existing
web-cache SQLite file is preserved. The default implementation removes
`unstructured[all-docs,pdf]` and replaces it with smaller local parsing libraries (`pdfplumber` and
`ebooklib`). PyMuPDF and PyMuPDF4LLM are not dependencies or implementation targets in this
milestone.

### 2.3 Config-first

All new knobs go into `app_config.yml` under `chroma.extraction`, `chroma.chunking`,
`chroma.indexing`, `chroma.retrieval`, and `web_search`. Index construction controls never live
under retrieval. Feature flags allow incremental rollout and safe rollback.

### 2.4 Reindex requirement

Phase 0, 3A, and 3B are indexing-time changes. They require the verified `make chroma-reindex`
workflow after implementation. That workflow must:

1. Rebuild the configured Chroma collection successfully.
2. Read the completed collection in batches and rebuild BM25 from exactly those records.
3. Write the BM25 pickle to a temporary file and atomically replace the configured index path.
4. Write a sidecar synchronization manifest with collection name, record count, and SHA-256 of
   sorted chunk IDs.
5. Validate the manifest before reporting success.

At query startup, a missing or mismatched manifest disables hybrid retrieval explicitly and logs a
clear vector-only fallback; it must never load a known-stale BM25 index. Because Phase 0 changes
chunk boundaries, `ground_truth_chunk_ids` must be refreshed with the Milestone 2
`chunk_id_lookup.py` utility after reindexing.

---

## 3. Architecture & Data Flow

### 3.1 Current indexing pipeline (corrective Phase 0)

```
ExtractorRegistry.resolve(file_type, provider)
    │
    ▼
DocumentExtractor.extract(path) -> list[DocumentElement]
    │   normalized text plus page/spine, heading, entry, block, column,
    │   reading-order, and structural-role lineage
    │
    ▼
DocumentChunker.chunk(elements) -> list[ChunkCandidate]
    │   default strategy: section_recursive
    │   optional strategy: section_semantic using local embeddings
    │   boundaries: source, entry, chapter, section, role, column, block
    │
    ▼
ChunkQualityFilter.assess(chunk) -> role + quality score + stable reasons
    │
    ├── quality_score < indexing.quality_filter.min_score ──► DISCARD (not indexed, not in BM25)
    │
    ▼
build_contextual_search_text(chunk, metadata) -> search_text: str
    │   heading path: document title > chapter > entry title > section
    │   search_text = "<deduplicated validated heading path>\n<original_text>"
    │   stored document in ChromaDB = <original_text> (unchanged)
    │   quality/structure/layout lineage stored in ChromaDB metadata
    │
    ▼
embedder.embed_documents([search_text, ...])
    │
    ▼
collection.add(embeddings, documents=<original_text>, metadatas=<with quality_score>)
    │
    ▼
BM25Index rebuilt from the completed Chroma snapshot using the same contextual search text
    │
    ▼
atomic pickle + count/sorted-ID-hash synchronization manifest
```

### 3.2 Current query pipeline and later planned extensions

`execute_production_rag()` remains the single orchestration function. The API and eval already call
it. M3 changed agent RAG tools from direct `ChromaRetriever` usage to the same function with
`generation_enabled=False`; the agent still owns final reasoning and generation.

```text
user_query
    -> build_retrieval_query_plan()
       (normalized query, entities, intent, semantic query, sparse query)
    -> dense pool (25) + synchronized BM25 pool (25)
    -> alternating de-duplicated balanced union (maximum 50)
       (unweighted RRF only when the reranker is unavailable)
    -> metadata match boost
    -> thresholded cross-encoder rerank
    -> normalized confidence and low-confidence flag
    -> semantic deduplication
    -> clean formatted context and sources
    -> optional TF-IDF compression
    -> RAGExecutionResult and optional LLM answer
```

HyDE expansion and confidence-triggered cached web fallback remain later, disabled-by-default
capabilities. Their fields already exist in `RAGFeatureUsage`, but Phase 0 does not construct or
invoke those resources.

Construction responsibilities are explicit:

- `build_retriever_from_config()` constructs vector retrieval and activates hybrid retrieval only
  after validating the BM25 synchronization manifest.
- `build_reranker_from_config()` constructs the existing cross-encoder.
- `execute_production_rag()` owns stage ordering and feature-use artifacts.
- API and eval pass generation-enabled resources; agent tools call the same path with generation
  disabled and return the resulting context.
- A future `build_web_fallback_from_config()`/HyDE adapter may be added only in its approved later
  delivery step.

### 3.3 Extraction and chunking contracts

Phase 0 introduces two narrow interfaces. These are intentionally boring: the extraction layer knows
about source formats and provider quirks; the chunking layer knows about text boundaries and context;
the loader knows only about normalized chunk candidates.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DocumentElement:
    text: str
    source_path: str
    file_type: str
    order_index: int
    page_number: int | None = None
    element_type: str = "paragraph"  # title, heading, paragraph, list_item, table, footer, unknown
    heading_level: int | None = None
    document_title: str = ""
    chapter: str = ""
    section: str = ""
    structural_role: str = "unknown"
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


class DocumentExtractor(ABC):
    @abstractmethod
    def extract(self, path: Path) -> list[DocumentElement]:
        """Extract normalized document elements from one source file."""


@dataclass(frozen=True)
class ChunkCandidate:
    text: str
    source_path: str
    file_type: str
    chunk_index: int
    page_number: int | None = None
    start_page: int | None = None
    end_page: int | None = None
    document_title: str = ""
    chapter: str = ""
    section: str = ""
    heading_path: str = ""
    chunking_strategy: str = ""
    extraction_provider: str = ""
    structural_role: str = "unknown"
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


class DocumentChunker(ABC):
    @abstractmethod
    def chunk(self, elements: list[DocumentElement]) -> list[ChunkCandidate]:
        """Build retrieval chunks from normalized document elements."""
```

### 3.4 What is NOT changed

- Public API request/response schemas remain unchanged. Confidence, query-plan, channel-provenance,
  and feature-use values remain internal artifacts and trace attributes in M3.
- LangChain/LangGraph planning and final agent generation are unchanged. Only the agent RAG tool's
  retrieval adapter changes so it no longer bypasses the production path.
- No new `RetrievalPipeline` class is introduced. The existing functional
  `execute_production_rag()` path stays the explicit orchestration boundary.
- FastAPI lifespan continues to use factories. The implemented factory validates BM25
  synchronization; HyDE/fallback construction remains future work.
- Context compression and semantic deduplication retain their existing algorithms. Query analysis,
  hybrid candidate generation, and reranker inputs did change during corrective Phase 0 and are
  documented above.

---

## 4. Configuration (`app_config.yml`)

The checked-in configuration after corrective Phase 0 is shown below. Indexing controls are applied
only by indexing code. HyDE configuration is intentionally absent until its later entry gate passes;
web fallback remains disabled.

```yaml
chroma:
  extraction:
    pdf_provider: pdfplumber            # only reviewed PDF provider in M3
    epub_provider: ebooklib             # only reviewed EPUB provider in M3
    fail_on_unsupported_file: false
    strip_repeated_headers: true
    strip_repeated_footers: true

  chunking:
    strategy: section_recursive         # section_recursive, section_semantic
    chunk_size: 1024
    chunk_overlap: 256
    min_chunk_chars: 200
    preserve_section_boundaries: true
    extract_wine_metadata: true
    semantic:
      enabled: false                    # use local embedding-based chunking inside sections
      breakpoint_threshold_type: percentile
      breakpoint_threshold_amount: 95.0

  indexing:
    quality_filter:
      mode: enforce                     # structural gate accepted during corrective Phase 0
      min_score: 0.4
    bm25:
      rebuild_on_reindex: true
      sync_manifest_path: chroma-data/bm25_index.meta.json

  retrieval:
    n_results: 5
    similarity_threshold: 0.3
    enable_hybrid: true
    semantic_candidate_pool: 25
    bm25_candidate_pool: 25
    reranker_input_limit: 50
    bm25_index_path: chroma-data/bm25_index.pkl
    validate_bm25_sync: true            # stale/missing manifest => explicit vector-only fallback
    enable_reranking: true
    reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
    rerank_top_k: 5
    rerank_threshold: 0.0               # accepted M3 cutoff; filters negative reranker logits
    min_retrieval_confidence: 0.3       # provisional; fallback remains disabled pending failure cohort
    enable_compression: false
    enable_metadata_boost: true
    metadata_boost_factor: 0.1
    use_deduplication: true
    deduplication_threshold: 0.9

  settings:
    # ... existing keys unchanged ...

web_search:
  # ... existing keys unchanged ...
  auto_fallback: false                  # 3E: trigger web search when retrieval confidence is low
```

`min_retrieval_confidence` is the single fallback threshold. A second
`web_search.fallback_threshold` is intentionally not introduced because two independently tunable
thresholds could disagree. When Phase 3/3C is authorized, its approved `hyde` block will be added at
that time rather than documenting a nonexistent runtime switch as current behavior.

---

## 5. DB / Schema Changes

### 5.1 ChromaDB metadata

`ChunkMetadata` and loader assessment now expose extraction/chunking traceability, structural, and
layout fields:

```python
extraction_provider: str = ""      # pdfplumber, ebooklib
chunking_strategy: str = ""        # section_recursive, section_semantic
heading_path: str = ""             # e.g. "France > Burgundy > Cote de Nuits"
structural_role: str = "unknown"
entry_title: str = ""
column_id: int | str = -1
start_block_id: int = -1
end_block_id: int = -1
start_page: int = -1
end_page: int = -1
layout_audit_required: bool = False
reading_order_confidence: float = 1.0
# loader assessment also writes quality_score and comma-separated quality_reasons
```

These fields are stored in ChromaDB metadata for auditability and future ranking experiments. M3
does not add a query-time `quality_score` predicate. The index-time filter is authoritative: rejected
chunks enter neither Chroma nor BM25.

**Migration status**: the required full, verified reindex completed on 2026-08-12. The active corpus
and synchronized BM25 index now share the accepted traceability and quality metadata contract.

### 5.2 Internal RAG result contract

`RAGExecutionResult` gains internal values:

```python
retrieval_confidence: float | None = None
low_confidence: bool = False
rerank_threshold: float | None = None
```

`RAGFeatureUsage` gains internal feature flags:

```python
hyde_expansion: bool = False
rerank_thresholding: bool = False
web_fallback: bool = False
```

They are captured in eval artifacts and traces but do not change the public FastAPI response schema.

### 5.3 SQLite (no changes)

No SQLite schema changes in this milestone.

---

## 6. Implementation Phases

Historical capability labels 3A–3F are retained for roadmap traceability, but delivery follows this
fixed order:

| Delivery step | Capability section | Why now |
|--------------:|--------------------|---------|
| 1 | Delivery Gate 0 — shared path and synchronized reindex | Prevent eval/runtime drift and stale hybrid retrieval before behavior changes |
| 2 | Phase 4 / 3D — reranker threshold and confidence | Highest-value query-time control; no reindex required |
| 3 | Phase 0 — extraction and chunking foundation | Establish deterministic provider-neutral ingestion |
| 4 | Phase 1 / 3A — noise filtering | Remove structural noise from both indices |
| 5 | Phase 2 / 3B — contextual enrichment | Improve embeddings only after chunk boundaries and filtering stabilize |
| 6 | Phase 5 / 3E — web fallback | Depends on the confidence signal; remains disabled by default |
| 7 | Phase 3 / 3C — HyDE | Implement only if the semantic-mismatch cohort justifies an extra LLM call |
| 8 | Phase 6 / 3F — embedding evaluation | Optional final comparison after preceding phases stabilize |

Do not combine checkpoints. Each delivery step receives its own tests and eval artifact before the
next step starts.

---

### Delivery Gate 0 — Shared Production Path and Synchronized Reindex (delivery step 1)

**Goal**: make every later M3 change measurable through one production path and ensure hybrid
retrieval never combines a new Chroma collection with stale BM25 data.

#### Gate 0.1 — Consolidate production-RAG ownership

- Extend `execute_production_rag()` as the sole owner of retrieval stage ordering.
- Keep API and eval on their existing calls to that function.
- Replace the direct `ChromaRetriever` construction in `src/agents/tools/rag_tools.py` with
  `build_retriever_from_config()`, `build_reranker_from_config()`, and
  `execute_production_rag(generation_enabled=False)`.
- Preserve current tool return strings and agent planning behavior.
- Extend `RAGExecutionResult`/`RAGFeatureUsage` with the internal fields in Section 5.2.

**Acceptance tests**:
- Extend `tests/eval/test_rag_parity.py` so API-style, eval-style, and agent-tool retrieval produce
  the same ordered context chunk IDs for the same query and resources.
- Verify the agent tool performs no generation inside the shared RAG call.
- Verify no tool path constructs a bare `ChromaRetriever` independently of the factory.

#### Gate 0.2 — Add verified BM25 rebuilding

Create `src/chroma/bm25_builder.py`:

- Read all records from the completed configured Chroma collection in bounded batches.
- Build `BM25Index` from exactly those IDs, documents, and metadata.
- Write the pickle to a temporary file in the target directory, then use atomic `Path.replace()`.
- Write `chroma-data/bm25_index.meta.json` with collection name, record count, sorted-ID SHA-256,
  build timestamp, and BM25 path.
- Validate Chroma and BM25 record count plus sorted-ID hash after writing.

Update `src/chroma/load_data.py` and `make chroma-reindex` so forced reindexing runs the builder only
after Chroma succeeds. If BM25 construction or validation fails, exit non-zero. At retrieval startup,
`build_retriever_from_config()` validates the manifest; mismatch produces a clear warning and an
explicit vector-only retriever instead of loading stale keyword data.

**Acceptance tests**:
- A successful small-corpus reindex produces matching Chroma/BM25 counts and hashes.
- A deliberately stale manifest prevents hybrid construction and yields vector-only retrieval.
- A failed temporary BM25 build does not overwrite the last valid pickle.
- Legacy missing-manifest behavior is allowed only while `validate_bm25_sync=false`.

#### Gate 0.3 — No-behavior-change checkpoint

Run the 25-sample production-RAG retrieval eval before changing reranker behavior. MRR,
precision@3, precision@5, and ordered per-sample chunk IDs must match the Section 0.3 baseline.
Record the artifact as `eval-results/m3_gate0_shared_path_<date>.json`.

#### Gate 0.4 — Freeze exact corpus diagnostics

Extend `src/chroma/stats.py` with an exact, machine-readable mode that reads the whole configured
collection in bounded batches and records:

- collection name and exact record count;
- average, minimum, and maximum document length;
- empty count, near-empty count below 200 characters, and both rates;
- source-document count plus minimum/average/maximum chunks per source;
- SHA-256 of sorted chunk IDs.

Save the pre-change snapshot as `eval-results/m3_gate0_corpus_<date>.json`. The command must label
sampled and exact modes distinctly so a sampled statistic cannot be used accidentally as an
acceptance gate.

---

### Phase 0 — Extraction & Chunking Foundation (delivery step 3)

**Goal**: replace the `unstructured`-coupled `split_file()` path with provider-neutral extraction and
strategy-based chunking.

#### Step 0.1 — Create extraction contracts

Create `src/chroma/extraction/base.py` with:
- `DocumentElement` dataclass.
- `DocumentExtractor` abstract base class.
- A small `UnsupportedDocumentTypeError` exception for unsupported formats.

Rules:
- `DocumentElement.text` is always stripped, non-empty content.
- `source_path`, `file_type`, and `order_index` are required.
- `page_number` is optional because EPUB content is not page-based.
- Provider-specific details stay inside `metadata`; downstream code must not require provider-native
  objects.

**Acceptance test**: `tests/chroma/test_extraction_base.py`
- `DocumentElement` can represent PDF and EPUB content without provider-specific fields.
- Empty text is rejected or filtered by the provider before returning elements.
- `DocumentExtractor` cannot be instantiated without implementing `extract()`.

#### Step 0.2 — Implement default extraction providers

Update dependencies in the same phase:
- Remove `unstructured[all-docs,pdf]` from `[project].dependencies`.
- Remove it from `[dependency-groups].chroma`.
- Add compatible pinned/minimum versions of `pdfplumber` and `ebooklib` to both places where the
  indexing runtime requires them.
- Regenerate `uv.lock`.
- Do not add PyMuPDF or PyMuPDF4LLM.

Create:
- `src/chroma/extraction/pdfplumber_extractor.py` for PDFs.
- `src/chroma/extraction/ebooklib_extractor.py` for EPUBs.
- `src/chroma/extraction/registry.py` to resolve provider by file type and config.

Provider responsibilities:
- Extract text into ordered `DocumentElement` objects.
- Preserve page numbers for PDFs.
- Infer basic element types (`heading`, `paragraph`, `list_item`, `table`, `footer`, `unknown`) using
  explicit heuristics.
- Track current `document_title`, `chapter`, and `section` while iterating elements.
- Remove repeated headers/footers only when `strip_repeated_headers` / `strip_repeated_footers` are
  enabled and the repeated text appears on enough pages to be structural noise.

`pdfplumber` is the default PDF provider because it exposes enough layout information to detect
lines, words, repeated headers, and page-local order while staying lightweight. `ebooklib` is the
default EPUB provider because EPUB structure is already XHTML and headings can be read from
`h1`/`h2`/`h3` tags.

#### Step 0.3 — Create chunking contracts and default strategy

Create `src/chroma/chunking/base.py` with:
- `ChunkCandidate` dataclass.
- `DocumentChunker` abstract base class.

Create `src/chroma/chunking/section_recursive.py` with `SectionRecursiveChunker`.

Algorithm:
1. Group elements by `document_title` + `chapter` + `section`.
2. Preserve headings as context metadata instead of blindly embedding them as standalone chunks.
3. Pack paragraphs/list items into chunks up to `chunk_size`.
4. If one section exceeds `chunk_size`, split it with `RecursiveCharacterTextSplitter` using
   paragraph, sentence, and whitespace separators in that order.
5. Apply `chunk_overlap` only within the same section; never overlap across unrelated chapters.
6. Drop chunks below `min_chunk_chars` unless they contain high-signal metadata such as grape,
   appellation, classification, or producer names.

This is the default strategy because it is deterministic, cheap, debuggable, and preserves document
structure before adding any semantic complexity.

**Acceptance test**: `tests/chroma/test_section_recursive_chunker.py`
- Chunks never combine text from two different chapters.
- Oversized sections are split below the configured `chunk_size`.
- Heading metadata is preserved on each chunk.
- Small low-signal fragments are dropped, while small high-signal fragments can be retained.
- `chunk_overlap` does not cross section boundaries.

#### Step 0.4 — Add optional section-bounded semantic chunking

Create `src/chroma/chunking/section_semantic.py` with `SectionSemanticChunker`.

Rules:
- Semantic splitting happens inside a section group, not across the whole document.
- Use the existing local embedder from `get_embedder()`.
- Keep the same `ChunkCandidate` output contract as every other chunker.
- Gate with `chroma.chunking.semantic.enabled: false` by default.
- Run eval before switching the default from `section_recursive`.

This option is useful for long, coherent prose sections where fixed packing creates awkward
boundaries. It is not the default because it adds an embedding pass during indexing and makes chunk
boundaries more sensitive to model and threshold changes.

#### Step 0.5 — Refactor `split_file()` into orchestration

Keep a compatibility wrapper if useful, but move responsibilities out of one function:
- `DocumentExtractionPipeline.extract(path) -> list[DocumentElement]`
- `DocumentChunkingPipeline.chunk(elements) -> list[ChunkCandidate]`
- `assemble_chroma_chunks(candidates, extract_metadata=True) -> list[dict]`

The loader should call the pipeline and receive the same dict shape it expects today:
`{"id", "text", "metadata", "importance_score"}`. This keeps Phase 0 isolated from later loader
changes.

#### Step 0.6 — Eval checkpoint

Run the Milestone 2 eval harness after reindexing with `section_recursive`:
- Record total chunk count, average chunk length, empty/near-empty chunk rate, and chunks per source.
- Refresh `ground_truth_chunk_ids` for affected `rag_only` samples.
- Compare `context_precision`, `context_recall`, and `mrr` against the pre-refactor baseline.
- Expected outcome: equivalent or better precision, no recall regression above 2 percentage points,
  and visibly cleaner source snippets in manual inspection.

Record results in `eval-results/m3_phase0_extraction_chunking_<date>.json`.

#### Phase 0 corrective section — Layout-safe extraction and balanced hybrid retrieval

**Approval**: approved by the user on 2026-08-11 after the initial Phase 0 checkpoint failed its
manual-snippet and retrieval-quality gates.

**Why this correction is required**: the 2026-08-11 full reindex completed with 35,676 records from
22 successfully indexed source documents, but the 25-sample retrieval checkpoint regressed from the
accepted baseline to MRR `0.6313`, precision@3 `0.4133`, and precision@5 `0.3200`. For the diagnostic
query "What are the primary flavour characteristics of Nebbiolo?", the first ten raw candidates
contained six interleaved Burgundy tasting-note chunks, three food-pairing worksheets, one
bibliography chunk, and no Nebbiolo metadata match. A coherent direct-answer chunk already existed
in the collection, proving that extraction noise and candidate-generation behavior, rather than
missing source material, caused the failure.

This section is normative where it conflicts with the original Phase 1 and Phase 2 sequencing:

- Keep `pdfplumber`, `ebooklib`, and `lxml`; no new extraction dependency is approved.
- Move the minimum structural-role rejection needed for a clean corpus into Phase 0. Phase 1 keeps
  its audit/enforcement lifecycle but must use the role-aware rules defined here instead of relying
  only on the original five generic deductions.
- Move validated contextual search text into Phase 0 because both embeddings and BM25 need correct
  entity lineage. Phase 2 retains the stored-text-versus-search-text contract and may extend it only
  after this corrective gate passes.
- Keep semantic chunking disabled. It cannot repair malformed extraction order.

##### Step 0.7 — Make PDF extraction layout-aware

Update `PdfPlumberExtractor` without changing providers:

1. Extract positioned words or characters and retain horizontal as well as vertical coordinates.
2. Detect full-width regions, one-column regions, gutters, and multi-column regions per page.
3. Build explicit page-local blocks and emit them in reading order: full-width content first where
   applicable, followed by each detected column from top to bottom and left to right.
4. Never merge text across column or block boundaries. Dehyphenate line wraps only inside one
   validated block.
5. Add provider metadata for `block_id`, `column_id`, and a deterministic reading-order confidence.
6. Reject OCR placeholders such as `(cid:1)` as headings and prevent single-letter form labels,
   numeric scales, page numbers, and rating rows from updating heading context.
7. When reading order is below the accepted confidence, mark the page for audit or exclude its
   uncertain content explicitly; never silently index known interleaving.

**Acceptance tests**:
- A coordinate-based two-column fixture preserves each column independently and never alternates
  lines between them.
- A full-width heading above two columns precedes both columns and supplies their context.
- OCR placeholders and form labels do not become headings.
- The inspected Burgundy layout produces coherent tasting-note blocks.

##### Step 0.8 — Derive EPUB context from structural entry boundaries

Update `EbookLibExtractor` to consume structure already exposed by the approved dependencies:

- Preserve spine item, navigation/TOC anchor, element ID, and CSS class alongside heading level.
- Use navigation anchors and publisher entry boundaries to establish document-entry lineage; raw
  `h1`/`h2`/`h3` nesting alone is insufficient.
- Reset inherited chapter/section context at a verified new entry while permitting explicit
  continuation across spine items.
- Treat a new peer grape/dictionary entry as a context boundary so entries such as `NEGRA MOLE`,
  `NEGRAMOLL`, and `NERELLO` cannot inherit `ENJOYING NEBBIOLO`.
- Record the evidence used to assign or reset structural context for auditability.

**Acceptance tests**:
- EPUB spine order remains deterministic.
- Navigation and heading-class fixtures produce stable entry lineage.
- A `NEBBIOLO` to `NEGRA MOLE` transition resets context.
- Legitimate subsections such as `The taste of Nebbiolo` retain the Nebbiolo parent entry.

##### Step 0.9 — Add structural roles and an authoritative noise gate

Extend the provider-neutral extraction/chunk contract with a structural role independent of the
provider element type. Supported roles are `prose`, `table`, `wine_list`, `toc`, `bibliography`,
`index`, `worksheet`, and `unknown`.

- Deterministically reject confident `toc`, `bibliography`, `index`, `worksheet`, and extraction-
  garbage candidates before either embedding or BM25 construction.
- Keep useful tables and wine lists available, but do not treat list-only material as ordinary
  explanatory prose.
- Use quality scoring only for ambiguous candidates. Signals include citation density, form blanks
  and numeric scales, OCR artifacts, sentence density, repeated short lines, list density, and
  suspected column crossing.
- Store `structural_role`, `quality_score`, and stable rejection reason codes in audit diagnostics.
- Preserve one-source-of-truth behavior: rejected chunks enter neither Chroma nor BM25.

The original Phase 1 word-count rule is insufficient by itself: nine of the ten known bad
Nebbiolo candidates score `0.5` under that rule and would survive the proposed `0.4` cutoff.

##### Step 0.10 — Make recursive chunking block-aware

Revise `SectionRecursiveChunker` so it packs complete compatible blocks instead of flattening all
paragraph, list, table, and unknown elements into one character stream:

- Never cross source, entry, chapter, section, column, block-role, or rejected-content boundaries.
- Prefer complete paragraph and sentence boundaries; split inside a block only when that block alone
  exceeds `chunk_size`.
- Apply overlap using complete trailing sentences or paragraphs, never an arbitrary character
  window that creates a mid-sentence start.
- Keep `chunk_size=1024` for the corrective checkpoint. Parameter tuning is deferred until clean
  extraction isolates chunk-size effects.
- Keep `section_semantic` disabled and make it consume the same validated block groups before any
  future evaluation.

##### Step 0.11 — Build one validated contextual search representation

Keep original clean text as the Chroma document shown to users, but construct a separate local
search representation from validated lineage:

```text
[From: <document title> | Chapter: <chapter> | Section: <section>]
<original clean chunk text>
```

- Embed the contextual search text while storing the original text unchanged.
- Build BM25 tokens from the same contextual search text reconstructed from Chroma metadata and the
  stored document; do not create an independently drifting corpus.
- Extract wine entities from original text plus validated lineage so pronoun-heavy section chunks
  remain discoverable.
- Never include headings marked uncertain, leaked, structural, or OCR-corrupt.
- Preserve the synchronized Chroma/BM25 record count and sorted-ID manifest contract.

##### Step 0.12 — Normalize sparse retrieval and create a deterministic query plan

Replace whitespace-only BM25 tokenization with the same deterministic Unicode-aware analyzer at
index and query time:

- case-fold and normalize punctuation, apostrophes, hyphens, and configured wine terminology;
- make `nebbiolo?`, `Nebbiolo,`, and `nebbiolo` comparable;
- remove low-value question stopwords without removing wine entities, vintages, classifications,
  appellations, or producer names.

Create a local `RetrievalQueryPlan` from the existing query analyzer. It contains the normalized
question, detected entities, an intent-focused semantic query, and sparse terms. For a flavour
question it may deterministically add terms such as `taste`, `aroma`, `tannin`, `acidity`, and
`body`. This step makes no LLM or external API call.

##### Step 0.13 — Replace early weighted fusion with balanced union and reranking

The default production path becomes retrieve, union, then rerank:

1. Retrieve a configurable semantic pool from the contextual embeddings.
2. Retrieve a configurable BM25 pool from the contextual sparse index.
3. Deduplicate the complete pools by chunk ID while preserving dense rank, sparse rank, scores,
   query/entity matches, and channel provenance.
4. Do not truncate either channel before constructing the union. Detected entity matches are recall
   evidence and must remain visible to the reranker; they are not a hard final relevance filter.
5. Rerank the bounded union with the existing local cross-encoder, then apply the reviewed threshold,
   confidence calculation, deduplication, and final `rerank_top_k` selection.

Initial eval settings are `25` candidates per channel and at most `50` unique reranker inputs. They
remain config-driven and must be latency-calibrated before becoming final defaults.

If the reranker is unavailable, fall back to standard unweighted reciprocal-rank fusion. Do not use
the current `0.7/0.3` weighted formula: with `k=60`, vector rank 20 scores `0.00875`, which exceeds
BM25 rank 1 at `0.00492` and can exclude every sparse-only result. If a smaller fallback shortlist
is required, reserve candidates from each channel before filling remaining positions by RRF.

Internal retrieval artifacts must expose channel provenance and ranks. This does not change the
public API response schema.

##### Step 0.14 — Align chunk curation with production retrieval

Update `chunk_id_lookup.py` and `chunk_id_curator.py`:

- Stop silently unwrapping `HybridRetriever` to vector-only retrieval.
- Default candidate discovery to the shared production retrieval behavior with generation disabled.
- Add explicit `vector`, `bm25`, and `hybrid` diagnostic modes.
- Display full query and chunk text plus dense rank, sparse rank, entity matches, fusion/fallback
  score, reranker score, source, structural role, and heading path when available.
- Keep resumable, atomic golden-dataset writes and preserve manual human selection as the authority.

##### Step 0.15 — Corrective Phase 0 validation gate

Validate on a small representative collection before another full reindex. It must include the
affected Burgundy PDF, the Harrington worksheet/reference PDF, a coherent two-column Nebbiolo PDF,
and the `Grapes & Wines` EPUB.

Required focused acceptance for "What are the primary flavour characteristics of Nebbiolo?":

- a direct-answer chunk appears in the top three;
- at least eight of the first ten pre-rerank candidates are relevant to Nebbiolo or its wines;
- no bibliography, worksheet, ToC, index, or interleaved-column chunk appears in the first ten;
- production retrieval returns non-empty context after thresholded reranking.

After the focused gate passes:

1. Run vector-only, BM25-only, and hybrid ablations on the frozen retrieval dataset and report
   candidate recall@10, MRR, precision@3, precision@5, exact-entity hit rate, and latency.
2. Retain BM25 only through the approved balanced-union design and require hybrid retrieval to add
   measurable entity coverage or recall without an unacceptable precision or latency regression.
3. Run the verified full Chroma/BM25 reindex, save exact corpus diagnostics, and inspect
   representative PDF and EPUB snippets.
4. Re-curate affected chunk IDs only after the final index passes structural inspection.
5. Run the Phase 0 production-RAG checkpoint and save
   `eval-results/m3_phase0_extraction_chunking_<date>.json` with configuration, corpus hash, dataset
   hash, metric coverage, ablation results, and the focused Nebbiolo diagnostic.

Phase 1 may begin only after this gate is accepted explicitly.

**Closure accepted 2026-08-12**: the verified reindex produced 37,374 synchronized Chroma/BM25
records from 22 sources with zero empty chunks and chunk-ID SHA-256
`464d855a25cf834c77215ade8c9e28be6c793141fb51b35459aa8773e630a278`. The known malformed
Adam Centamore chunk and its content hash are absent, all 114 curated IDs exist, and the focused
Nebbiolo gate passed with a direct answer at rank 1, 9/10 relevant pre-rerank union candidates, no
structural/interleaved/OCR-artifact candidate in that top ten, and non-empty final context. The
24-scorable-sample production checkpoint achieved MRR `0.8368`, precision@3 `0.6250`, and
precision@5 `0.5833`; hybrid recall@10 was `0.9208` with exact-entity hit rate `1.0`. Final evidence
is recorded in `eval-results/m3_phase0_extraction_chunking_20260812.json`, with reproducible checks
in `docs/m03-phase0-corrective-manual-testing.md`.

---

### Phase 1 — Noise Chunk Filtering (3A, delivery step 4)

**Goal**: exclude structural non-content chunks from the ChromaDB index and BM25 index.

**Current state after corrective Phase 0**: the production mechanics described in Steps 1.1–1.3
are implemented and `quality_filter.mode=enforce` is active. Corrective Phase 0 expanded the
original generic score into shared structural-role assessment and stable rejection reasons. The
remaining Phase 1 delivery work is a separately recorded calibration/eval checkpoint on the final
corpus, not creation of another filter implementation.

#### Step 1.1 — Create `src/chroma/chunk_filter.py`

New module. Contains `ChunkQualityFilter` class with a single public method:

```python
class ChunkQualityFilter:
    def score(self, text: str, metadata: dict) -> float:
        """Return quality score in [0.0, 1.0]. Lower = lower quality."""
```

Internal scoring signals (each deducts from a 1.0 base):

| Signal | Deduction | Implementation |
|--------|-----------|----------------|
| Word count < 80 | −0.5 | `len(text.split()) < 80` |
| ToC pattern: >30% of lines are short (<8 words) AND contain dots or page numbers | −0.6 | regex per line |
| Index/bibliography: >10% of lines match citation patterns (`pp\.`, `ibid\.`, `\[\d+\]`, ISBN) | −0.4 | regex |
| Structure-to-content ratio: heading/list markers > prose sentences | −0.3 | count lines starting with `#`, `-`, `•`, `*`, `\d+\.` vs lines ending with `.`/`?`/`!` |
| Repeated short lines (average line length < 20 chars) | −0.2 | statistics.mean on line lengths |

Scores are clamped to `[0.0, 1.0]`. Return `max(0.0, 1.0 - sum_of_deductions)`.

**Acceptance test**: `tests/chroma/test_chunk_filter.py`
- A ToC-like text (lines of "Chapter 1…………………12") scores < 0.4.
- A normal wine prose paragraph (≥80 words, normal sentence structure) scores ≥ 0.7.
- A bibliography block (lines with `pp.`, `ibid.`) scores < 0.4.
- An empty or near-empty string scores 0.0.
- A single-sentence caption (< 80 words but no structural patterns) scores between 0.4–0.6.

#### Step 1.2 — Wire `ChunkQualityFilter` into `CollectionDataLoader.process_file()`

In `src/chroma/loader.py`:
- Instantiate `ChunkQualityFilter` when `config.chroma.indexing.quality_filter.mode` is `audit` or
  `enforce`; validate and reject any other value.
- In `process_file()`, after `validate_chunks()` and before the duplicate check loop, call
  `filter.score(chunk["text"], chunk["metadata"])` and:
  - Store the score as `chunk["metadata"]["quality_score"]`.
  - In `audit` mode, retain the chunk and increment `chunks_below_quality_threshold` when its score
    is below `config.chroma.indexing.quality_filter.min_score`.
  - In `enforce` mode, skip a below-threshold chunk and increment `chunks_filtered`.
- Add `chunks_filtered: int` to the `stats` dict returned by `process_file()`.
- Add `chunks_below_quality_threshold: int` so audit and enforcement reports are comparable.

**Acceptance test**: extend `tests/chroma/test_loader.py`
- In `enforce` mode with a mix of ToC and prose chunks, verify that chunks below threshold are
  excluded from the `docs` list sent to `collection.add()`.
- Verify that `stats["chunks_filtered"]` is incremented for excluded chunks.
- In `audit` mode, verify all chunks pass through, scores are retained, and
  `chunks_below_quality_threshold` is recorded.
- In `disabled` mode, verify all chunks pass through and no scores are computed.

#### Step 1.3 — Verify BM25 receives the filtered Chroma corpus

The Gate 0 BM25 builder reads the completed Chroma collection, so no second quality-filter
implementation is allowed in BM25 code. After the quality-filtered reindex:

- BM25 record count and sorted-ID hash must match Chroma.
- Known rejected ToC/bibliography chunk IDs must exist in neither index.
- Retained chunks preserve `quality_score` in BM25 document metadata.

This single-source rule prevents Chroma and BM25 from applying subtly different heuristics.

#### Step 1.4 — Add index quality settings to `app_config.yml`

Use `chroma.indexing.quality_filter.mode` and
`chroma.indexing.quality_filter.min_score` as defined in Section 4. Default minimum score is `0.4`.
The historical rollout plan started in `audit`. Corrective Phase 0 completed that inspection and the
checked-in current mode is `enforce`; future tuning must compare against this accepted state.

#### Step 1.5 — Eval checkpoint (before proceeding to Phase 2)

Run the Milestone 2 eval harness:
- Record baseline `context_precision` and `context_recall`.
- Record reduction in total indexed chunk count (`make chroma-stats` before and after reindex).
- Expected outcome: 10–25% fewer indexed chunks; precision should improve or hold steady; recall
  should not degrade by more than 2 percentage points.
- Record results in `eval-results/m3a_noise_filter_<date>.json`.

---

### Phase 2 — Contextual Chunk Enrichment (3B, delivery step 5)

**Goal**: make embeddings document-context-aware by prepending a structured context prefix to the
text that is embedded, while keeping the stored document text clean.

**Current state after corrective Phase 0**: contextual search text is implemented in
`src/chroma/contextual_text.py` and is active without a second configuration flag. It uses validated,
deduplicated `document_title`, `chapter`, `entry_title`, and `section` lineage. Dense embeddings,
BM25 documents, entity extraction, and reranker pairs use the same helper; Chroma still stores the
plain body. Tests live in `tests/chroma/test_contextual_text.py`, loader/BM25 tests, and reranker
tests. The originally proposed `ContextualEnricher` class and
`chroma.indexing.contextual_enrichment.enabled` flag were not created because they would duplicate
this single explicit function and permit search channels to drift.

The remaining delivery-step-5 work is measurement: freeze at least five reviewed region/grape
samples, run a body-only versus contextual-search ablation on the same corpus, and record
`eval-results/m3b_contextual_enrichment_<date>.json`. Keep the active behavior only if cohort
context precision improves and global MRR, precision@3, precision@5, and common-sample context
recall do not regress beyond the reviewed tolerance. Any proposal to make contextual text optional
again is a design/config-default change and requires approval.

---

### Phase 3 — HyDE Query Expansion (3C, conditional delivery step 7)

**Goal**: improve retrieval for queries that have very different surface form from wine book prose,
by generating a hypothetical answer passage and blending its embedding with the raw query embedding.

**Entry gate**: before implementing HyDE, curate at least five `semantic_mismatch` samples and run
them through the post-3E pipeline. Proceed only if at least two samples fail to retrieve a relevant
top-five chunk or cohort `context_recall` is below `0.80`. Otherwise close 3C as measured and not
needed.

#### Step 3.1 — Create `src/retrieval/hyde.py`

New module. Contains `HyDEExpander` class:

```python
class HyDEExpander:
    def __init__(self, llm, embedder, blend_alpha: float = 0.5):
        """
        Args:
            llm: LLM instance from src/agents/llm.py.
            embedder: Embedder from get_embedder() in src/utils/resources.py.
            blend_alpha: Weight for the raw query embedding (0=pure HyDE, 1=pure raw).
        """

    def expand(self, query: str) -> list[float]:
        """
        Generate a blended embedding for the query using HyDE.

        Returns the blended embedding vector as a list of floats.
        Uses a lightweight single LLM call.
        """
```

Prompt for hypothetical passage generation (stored as
`src/agents/prompts/hyde_expansion.md`):

```
You are a wine expert. Write a 2-3 sentence passage from a wine book that directly answers
the following question. Be specific about grape varieties, regions, and wine characteristics.
Do not mention the question — write as if it is a book excerpt.

Question: {query}
```

Implementation details:
- Call `llm.invoke(hyde_prompt)` once.
- Embed the hypothetical passage: `hyde_embedding = embedder.embed_query(hypothetical_passage)`.
- Embed the raw query: `raw_embedding = embedder.embed_query(query)`.
- Return blended: `[blend_alpha * r + (1 - blend_alpha) * h for r, h in zip(raw_embedding, hyde_embedding)]`.
- Normalise to unit vector after blending (cosine similarity requires unit vectors).
- On any LLM error, log a warning and fall back to returning the raw query embedding (fail-safe).

**LLM cost note**: this is one call to the explicit Ollama `hyde.provider` / `hyde.model` configured
in Section 4, only when `hyde.enabled=true`. It must not inherit the production Gemini API model or
whichever eval model happens to be active. The call is sent through the configured Ollama base URL,
including when the selected model is hosted by Ollama Cloud. With `blend_alpha: 0.5`, a
hallucinated hypothetical passage has bounded impact because the raw query embedding contributes
equally.

#### Step 3.2 — Wire `HyDEExpander` into `ChromaRetriever.retrieve()`

In `src/retrieval/vector_retriever.py`:
- Add optional `embedding_override: list[float] | None = None` parameter to `retrieve()`.
- If `embedding_override` is not None, use it instead of calling `embedder.embed_query(query)`.
- The caller (`HybridRetriever`) is responsible for passing the blended HyDE embedding when enabled.

In `src/retrieval/hybrid_retriever.py`:
- In `__init__`, accept an optional `hyde_expander: HyDEExpander | None = None`.
- In `retrieve()`, if `hyde_expander` is set, call `hyde_expander.expand(query)` to get the blended
  embedding, then pass it as `embedding_override` to `vector_retriever.retrieve()`.
- BM25 retrieval always uses the raw `query` text (HyDE is a vector-space concept only).

#### Step 3.3 — Add `hyde` section to `app_config.yml`

As listed in Section 4. `enabled: false` by default.

#### Step 3.4 — Wire through the shared factory

Extend `build_retriever_from_config()` so `hyde.enabled=true` resolves the explicit Ollama HyDE
model and base URL, constructs `HyDEExpander`, and passes it to `HybridRetriever`. API, eval, and
agent tools receive the same configured behavior through the factory. Disabled mode must not load a
HyDE model. This path must never fall back to `model.provider` or the production Gemini API.

**Acceptance tests**: `tests/retrieval/test_hyde.py`
- `expand()` with a mocked LLM that returns a fixed passage → blended embedding has correct shape
  and is unit-normalised.
- `expand()` when LLM raises an exception → falls back to raw query embedding without raising.
- `blend_alpha=1.0` → blended embedding equals raw query embedding.
- `blend_alpha=0.0` → blended embedding equals hypothetical embedding.
- `HybridRetriever` with `hyde_expander=None` → behaves identically to current code (no regression).

#### Step 3.5 — Eval checkpoint

- Measure improvement on "semantic mismatch" query category from the golden dataset.
- Record LLM call count per query (should increase by exactly 1 when HyDE is active).
- Require at least a 5 percentage-point cohort `context_recall` improvement, with no global MRR,
  precision@3, precision@5, or context-precision regression greater than 2 percentage points.
- Record results in `eval-results/m3c_hyde_<date>.json`.

---

### Phase 4 — Reranker Threshold & Confidence Signal (3D, delivery step 2)

**Goal**: stop irrelevant chunks from reaching the LLM context, and produce a machine-readable
confidence signal for the web fallback (Phase 5).

#### Step 4.1 — Add `rerank_threshold` and `min_retrieval_confidence` to config

As listed in Section 4, start with `rerank_threshold: null` so the first deployment preserves the
current no-filter behavior. `min_retrieval_confidence: 0.3` is an initial calibration candidate and
does not trigger external calls while `web_search.auto_fallback=false`.

The accepted 2026-08-04 calibration promoted `rerank_threshold` to `0.0`. The confidence cutoff
remains provisional because the frozen dataset contains no top-five retrieval misses.

#### Step 4.2 — Create `src/retrieval/confidence.py`

New module. Contains `RetrievalConfidenceSignal`:

```python
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class RetrievalResult:
    """Wraps reranked documents with an aggregate confidence signal."""
    documents: List[Dict[str, Any]]
    confidence: float          # normalised max rerank score, 0.0–1.0
    low_confidence: bool       # True if confidence < min_retrieval_confidence
    web_fallback_used: bool = False   # set by Phase 5 if web search was triggered


def compute_confidence(
    documents: List[Dict[str, Any]],
    min_confidence: float,
) -> RetrievalResult:
    """
    Compute normalised confidence from reranked documents.

    Args:
        documents: Reranked documents, each must have a 'rerank_score' key.
        min_confidence: Threshold below which low_confidence = True.

    Returns:
        RetrievalResult with confidence and low_confidence flag.
    """
```

Normalisation: `rerank_score` from `cross-encoder/ms-marco-MiniLM-L-6-v2` is an unbounded logit.
Apply sigmoid to normalise: `confidence = sigmoid(max(rerank_score))` where `sigmoid(x) = 1 / (1 + exp(-x))`.

If `documents` is empty, return `confidence=0.0, low_confidence=True`.

**Acceptance tests**: `tests/retrieval/test_confidence.py`
- List of documents with high rerank scores (> 3.0) → `confidence` close to 1.0, `low_confidence=False`.
- Empty document list → `confidence=0.0`, `low_confidence=True`.
- List with max score = 0.0 (sigmoid output = 0.5) and threshold 0.3 → `low_confidence=False`.
- List with very negative scores → `low_confidence=True`.

#### Step 4.3 — Wire threshold and confidence into the retrieval call site

In `src/retrieval/rag_service.py`:
- When `rerank_threshold` is `null`, retain the current `DocumentReranker.rerank()` call.
- When it is numeric, call `rerank_with_threshold(threshold=cfg.rerank_threshold)`.
- After reranking, call `compute_confidence(reranked_docs, min_confidence=cfg.min_retrieval_confidence)`.
- Copy confidence and threshold-use state into `RAGExecutionResult`/`RAGFeatureUsage`.
- Keep context construction and the public API response unchanged.

**Acceptance tests**: extend `tests/retrieval/test_rag_service.py`
- With `threshold=0.5` and all docs scoring < 0.5, the reranker returns an empty list.
- With `threshold=null`, the existing rank-only behavior is preserved, including negative scores.
- With `threshold=0.0`, negative scores are filtered; do not treat this as baseline parity.
- `low_confidence=True` is set correctly when max score is below threshold.

#### Step 4.4 — Calibrate before selecting the target threshold

Run the 25-sample retrieval eval at thresholds `null`, `0.0`, `0.05`, `0.1`, and `0.2` without
changing any other setting. Select the lowest numeric threshold that improves or preserves
precision@3/precision@5 while keeping MRR and provisional common-sample context recall within
2 percentage points of baseline. If none passes, keep `null`; thresholding is not mandatory merely
because the code exists.

From the same artifact, inspect confidence distributions for relevant-hit and top-five-miss samples.
Keep `0.3` only if it captures retrieval failures while limiting the projected fallback cohort to
20% of the frozen eval set; otherwise record a better cutoff. If no cutoff separates failures
usefully, Phase 5 remains implemented but disabled. Record the selected values and distributions in
`eval-results/m3d_reranker_confidence_<date>.json`.

**Accepted checkpoint (2026-08-04)**:

- Artifact: `eval-results/m3d_reranker_confidence_20260804.json`.
- Thresholds `null`, `0.0`, `0.05`, `0.1`, and `0.2` all retained MRR `0.8533`, precision@3
  `0.5600`, and precision@5 `0.4320` with 25/25 coverage and zero errors/timeouts.
- `0.0` was selected as the lowest numeric candidate. It removed ten negative-score tail chunks
  across four samples without changing deterministic retrieval metrics.
- Common-sample context recall changed from `0.6051` to `0.5942` over 23 shared scored samples,
  a `-0.0109` delta within the allowed `-0.02` gate. Candidate coverage was 25/25.
- All 25 samples were top-five hits and confidence ranged from `0.9101` to `0.9999`; no top-five
  miss cohort exists. `min_retrieval_confidence=0.3` is therefore retained only as an inactive,
  provisional value, and `web_search.auto_fallback=false` remains unchanged.

---

### Phase 5 — Web Search Automatic Fallback (3E, delivery step 6)

**Goal**: transparently supplement retrieval results with web search results when book retrieval
confidence is low — without requiring user rephrasing or explicit agent tool selection.

#### Step 5.1 — Extract the reusable web-search service

Move `WineWebSearchEngine`, `WebSearchCache`, and their non-tool helpers from
`src/agents/tools/web_search_tools.py` to `src/services/web_search.py`. Keep the LangChain
`@tool` functions in their current module as thin wrappers over that service, preserving their
names, arguments, return strings, cache database, TTL behavior, and provider configuration.

This is a code-location change, not a new runtime service or API. It prevents
`src/retrieval/web_fallback.py` from importing the agent-tools package, whose package initializer
also imports RAG tools and would create a circular dependency once those tools use the shared
retrieval path.

**Acceptance tests**:
- Existing web-search tool tests remain unchanged at the public tool boundary.
- Add service-level tests for cache hits, TTL selection, provider failure, and result dictionaries.
- Importing `src.retrieval` must not import or initialize `src.agents.tools`.

#### Step 5.2 — Create `src/retrieval/web_fallback.py`

New module. Contains `WebSearchFallback`:

```python
class WebSearchFallback:
    """
    Wraps web search to be used as an automatic retrieval fallback.

    Converts web search results into the same dict format as ChromaDB retrieved
    documents so they can be passed to context_builder without modifications.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def should_trigger(self, result: RetrievalResult) -> bool:
        """Return True if web fallback should be triggered."""
        return self.enabled and result.low_confidence

    def fetch_and_merge(
        self,
        query: str,
        book_results: RetrievalResult,
    ) -> RetrievalResult:
        """
        If confidence is low, call web search and append results to book_results.

        Web results are appended after book results (book chunks take precedence in context).
        Sets web_fallback_used=True on the returned RetrievalResult.

        Args:
            query: Original user query.
            book_results: RetrievalResult from the book retrieval pipeline.

        Returns:
            Updated RetrievalResult with merged documents.
        """
```

Web result formatting — convert each Tavily result into the standard document dict:

```python
{
    "id": "web_<sha256_of_url>",
    "document": f"{result['title']}\n\n{result['content']}",
    "metadata": {
        "source": "web",
        "url": result["url"],
        "title": result["title"],
        "quality_score": 0.6,   # fixed score for web results
        "filename": "web_search",
    },
    "rerank_score": 0.0,   # web results are not reranked; they appear last in context
}
```

Call site: `WebSearchFallback` receives or constructs the shared `WineWebSearchEngine` from
`src/services/web_search.py` and calls its structured `search()` method. It does not call a
LangChain `@tool` wrapper and does not duplicate the cache or Tavily client.

If the `TAVILY_API_KEY` env var is not set, log a warning, set `web_fallback_used=False`, and
return `book_results` unchanged (fail-safe, consistent with the existing tool-level behaviour).

#### Step 5.3 — Wire `WebSearchFallback` into the retrieval call site

In `src/retrieval/rag_service.py` (same shared call site updated in Phase 4):
- After `compute_confidence()`, if `fallback.should_trigger(result)`, call
  `fallback.fetch_and_merge(query, result)`.
- The merged `RetrievalResult.documents` list is passed to `build_context_from_chunks()` unchanged.
- Map `web_fallback_used` to `RAGFeatureUsage.web_fallback` and Phoenix span attributes.

#### Step 5.4 — Add `auto_fallback` to `web_search` config

As listed in Section 4, `auto_fallback: false` by default. The only trigger threshold is
`chroma.retrieval.min_retrieval_confidence`.

**Acceptance tests**: `tests/retrieval/test_web_fallback.py`
- `should_trigger()` returns `False` when `enabled=False` regardless of confidence.
- `should_trigger()` returns `False` when `low_confidence=False`.
- `should_trigger()` returns `True` when `enabled=True` and `low_confidence=True`.
- `fetch_and_merge()` with mocked Tavily call → returned documents list is `[book_docs..., web_docs...]`.
- `fetch_and_merge()` when Tavily is unavailable → returns original `book_results` unchanged.
- `web_fallback_used=True` is set after a successful web fetch.
- Web result format matches the expected document dict structure (all required keys present).

#### Step 5.5 — Eval checkpoint

- Before implementation, freeze a low-confidence cohort from the post-3D confidence artifacts and
  add/curate at least five reviewed `requires_current_information` samples.
- Define answer quality as Ragas `answer_relevancy` plus `faithfulness`, compared on common scored
  sample IDs with coverage reported.
- Require at least one of those metrics to improve by 2 percentage points on the cohort, neither to
  regress by more than 2 percentage points, and context recall not to regress by more than 2 points.
- Record web fallback trigger rate (how often fallback fires) across the eval set.
- Keep overall fallback trigger rate at or below 20%; otherwise leave the feature disabled and
  recalibrate confidence before rollout.
- Record results in `eval-results/m3e_web_fallback_<date>.json`.

---

### Phase 6 — Embedding Model Evaluation (3F, optional delivery step 8)

Defer until delivery steps 1–7 are complete and eval results are recorded for every delivered step.

If the post-3C pipeline still shows `context_recall` below baseline on specific query categories
despite all prior improvements, run a controlled comparison:

1. Reindex wine books with `BAAI/bge-base-en-v1.5` (same pipeline, only embedder changed).
2. Run the full Milestone 2 eval harness on both indices.
3. Compare `context_precision`, `context_recall`, and latency.
4. Migrate only if the improvement is ≥ 5% on at least two metrics.

No code changes required until the decision is made to switch. The embedder is already
config-driven via `chroma.settings.embedder`.

---

## 7. File Manifest

This manifest separates delivered code from later approved/conditional scope.

| Delivery state | Files | Notes |
|---|---|---|
| **Implemented — Gate 0 / step 2** | `src/chroma/bm25_builder.py`, `src/retrieval/confidence.py`, `src/retrieval/factory.py`, `src/retrieval/rag_service.py`, `src/agents/tools/rag_tools.py`, `src/chroma/stats.py`, `make/chroma.mk` | Shared API/eval/agent path, exact corpus diagnostics, synchronized BM25 lifecycle, threshold and confidence |
| **Implemented — Phase 0 foundation** | `src/chroma/extraction/`, `src/chroma/chunking/`, `src/chroma/ingestion_pipeline.py`, `src/chroma/chunks.py`, `src/chroma/loader.py`, `src/chroma/load_data.py` | Provider-neutral extraction and configurable section-aware chunking |
| **Implemented — corrective Phase 0** | `src/chroma/structural_roles.py`, `src/chroma/chunk_filter.py`, `src/chroma/contextual_text.py`, `src/retrieval/bm25_analyzer.py`, `src/retrieval/hybrid_retriever.py`, `src/retrieval/query_analyzer.py`, `src/retrieval/keyword_search.py`, `src/retrieval/reranker.py` | Layout/entry/block safety, authoritative noise gate, shared contextual search text, wine-aware sparse analysis, deterministic query plan, balanced union |
| **Implemented — curation/eval support** | `src/eval/scripts/chunk_id_curator.py`, `src/eval/scripts/chunk_id_lookup.py`, `src/eval/models.py`, `src/eval/runner.py`, `src/eval/utils.py`, `src/eval/wine_qa_golden.jsonl`, `scripts/rag_quickstart.py` | Production-equivalent discovery, diagnostic modes, full-text display, ablation/provenance artifacts, refreshed curated IDs |
| **Implemented tests** | `tests/chroma/`, `tests/retrieval/`, `tests/eval/test_rag_parity.py`, `tests/eval/test_chunk_id_scripts.py`, `tests/eval/test_utils.py` | Contracts, layout order, entry boundaries, roles/filtering, contextual text, BM25 sync, query plan, union, reranking, curation, and shared-path behavior |
| **Not created — later conditional work** | `src/retrieval/hyde.py`, `src/retrieval/web_fallback.py`, `src/services/web_search.py`, `src/agents/prompts/hyde_expansion.md` | Phase 3/5 entry gates have not authorized implementation |
| **Superseded proposal** | `src/chroma/context_enricher.py`, `tests/chroma/test_context_enricher.py` | Corrective Phase 0 uses the shared functional `contextual_text.py` contract instead |

The dependency change (`pyproject.toml`, `uv.lock`) and original provider-neutral foundation were
delivered in earlier Phase 0 cards. The current corrective work keeps the approved providers and
adds no dependency, database migration, public API schema, or prompt change.

---

## 8. Acceptance Criteria & Test Scenarios

### 8.1 Milestone-level acceptance criteria (all five must be met)

1. **Learning proof**: This document (design note with tradeoffs) — already satisfies the condition.
2. **Safety proof**: API, eval, and agent RAG tools produce ordered-context parity through
   `execute_production_rag()`. A stale/missing BM25 synchronization manifest cannot activate hybrid
   retrieval.
3. **Ingestion proof**: `unstructured` is no longer required for the default PDF/EPUB indexing path;
   provider-specific extraction objects do not appear outside `src/chroma/extraction/`.
4. **Functional proof**: Every delivered phase has its own eval artifact. Deterministic retrieval
   metrics retain complete coverage for samples supported by clean corpus evidence; explicitly
   unsupported samples are reported separately and never treated as zero. Global MRR, precision@3,
   precision@5, and common-sample context recall do not regress by more than 2 percentage points
   unless the user explicitly accepts the tradeoff. Judge comparisons report coverage and use the
   common scored-sample intersection defined in Section 0.4.
5. **Cost proof**: HyDE remains disabled unless its entry gate passes; when enabled it adds exactly
   1 LLM call per query,
   which is within the 2–3 calls per query budget. Web fallback only fires on low-confidence
   queries. Record before/after estimated call counts and latency using the current harness. Actual
   token usage is a separate instrumentation prerequisite if it remains a required acceptance signal.

### 8.2 Unit test matrix

| Test file | Key scenarios |
|-----------|---------------|
| `test_extraction_base.py` | Abstract extractor contract; `DocumentElement` defaults; provider-specific data isolated in `metadata` |
| `test_pdfplumber_extractor.py` | PDF pages return ordered non-empty elements; page numbers preserved; repeated header/footer stripping can be toggled |
| `test_ebooklib_extractor.py` | EPUB headings map to `document_title`/`chapter`/`section`; body text order is stable; unsupported items are skipped |
| `test_section_recursive_chunker.py` | Section boundaries preserved; oversized sections split; heading metadata retained; overlap stays inside section |
| `test_section_semantic_chunker.py` | Mocked embedder creates section-bounded chunks; failures fall back to section-recursive behavior without raising |
| `test_chunk_filter.py` | Structural roles, stable reasons, invalid layout/OCR/form/list signals, prose retention, and enforcement boundaries |
| `test_contextual_text.py` | Validated/deduplicated document/chapter/entry/section lineage, structural heading rejection, clean body preservation |
| `test_bm25_builder.py` | Chroma/BM25 count and ID hash match; atomic replace; failed build preserves valid pickle; stale manifest rejected |
| `test_confidence.py` | High scores → high confidence; empty list → confidence=0.0, low=True; sigmoid boundary (score=0 → confidence=0.5); threshold comparison |
| `test_bm25_analyzer.py` | Unicode/punctuation normalization, longest terminology aliases, question filler removal, persistence-safe exact entity matching |
| `test_hybrid_retriever.py` | Complete channel pools, balanced de-duplicated union, provenance/diagnostics, unweighted RRF fallback |
| `test_query_analyzer.py` | Entity/intent query plan and reviewed flavour semantic/sparse terms |
| `test_reranker.py` | Contextual query/document pairs and clean result bodies |
| `test_rag_service.py` | Shared stage ordering; metadata boost before thresholded reranking; query-plan/provenance/confidence artifacts |
| `test_rag_parity.py` | API, eval, and agent tool return the same ordered context chunk IDs |
| Later planned: `test_hyde.py`, `test_web_fallback.py` | Added only if their conditional delivery steps pass entry gates |

### 8.3 Integration test scenarios

The focused provider/loader/factory tests currently use small local fixtures and mocked Chroma
boundaries. A future `tests/integration/test_rag_quality.py` may consolidate these scenarios when
later M3 capabilities warrant a slower integration suite:

1. **End-to-end extraction and chunking**: index a small PDF and EPUB fixture; verify chunks include
   source filename, page metadata where applicable, extraction provider, chunking strategy, and
   chapter/section metadata.
2. **End-to-end reindex with quality filter**: index a small test corpus containing known ToC and
   prose chunks; verify ChromaDB collection count is lower than without filter and BM25 has the same
   retained IDs.
3. **End-to-end retrieval with contextual embeddings**: index 10 test chunks with document context;
   verify that querying with a context-aware query returns higher-ranked results than a body-only
   ablation (use a pre-seeded test ChromaDB instance).
4. **Reranker threshold in retrieval path**: mock reranker to return scores below and above
   threshold; verify that only above-threshold docs reach the context builder.
5. **Web fallback trigger in retrieval path**: mock `compute_confidence` to return `low_confidence=True`;
   mock Tavily to return two results; verify final context includes both book and web chunks.
6. **Web fallback skip when disabled**: same as above but with `auto_fallback=false`; verify Tavily
   is never called.
7. **Shared-path parity**: run the same fixture query through API-style, eval-style, and agent-tool
   adapters; verify identical ordered context IDs and no agent-tool generation inside the RAG call.

### 8.4 Edge cases

| Edge case | Expected behaviour |
|-----------|--------------------|
| Unsupported file extension | File skipped with structured warning when `fail_on_unsupported_file=false`; error raised when true |
| PDF extractor returns no elements | `chunks_generated=0`, error recorded in file stats, indexing continues for other files |
| EPUB has nested headings but no obvious title | Use filename stem as fallback `document_title`; preserve lower-level headings in `heading_path` |
| One section is larger than `chunk_size` | Split inside that section only; no cross-section overlap |
| Semantic chunking fails or embedder unavailable | Log warning and fall back to `section_recursive` for that section |
| All chunks in a file fail quality filter | `chunks_added=0`, `chunks_filtered=N`, logged as warning |
| BM25 rebuild fails after Chroma succeeds | Reindex command exits non-zero; stale manifest prevents hybrid activation; vector-only retrieval remains available |
| BM25 pickle exists but manifest is missing/mismatched | Log explicit synchronization error and construct vector-only retriever |
| `chapter`, `entry_title`, and `section` empty | Contextual text uses the validated document title only; if every context field is empty it returns the body unchanged |
| HyDE LLM returns empty string | Fallback to raw query embedding; no exception raised |
| `rerank_threshold` set higher than all scores | Reranker returns empty list; confidence=0.0; web fallback triggers if enabled |
| Web fallback enabled but `TAVILY_API_KEY` not set | Log warning, return book results only |
| ChromaDB unreachable during reindex | `process_file()` catches exception, records in `stats["errors"]` — existing behaviour |
| `blend_alpha` outside [0.0, 1.0] | Clamp to [0.0, 1.0] and log a warning |

---

## 9. Risks & Fixed Rollout Decisions

### 9.1 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| New extraction providers produce worse text order than `unstructured` for some PDFs | Medium | High | Keep Phase 0 as its own eval checkpoint. Manually inspect source snippets from representative books before proceeding to 3A. The provider-neutral contract permits a future separately reviewed provider without changing downstream code. |
| Section heading heuristics misclassify decorative lines as headings | Medium | Medium | Keep heuristics explicit and covered by fixtures. Store `element_type` and `heading_path` for auditability. Let quality filtering catch low-value structural chunks. |
| BM25 rebuild fails after Chroma has changed | Low | High | Reindex exits non-zero, atomic replace preserves the last valid pickle, sync validation prevents stale hybrid use, and runtime falls back explicitly to vector-only retrieval. |
| Semantic chunking produces unstable chunk IDs after model or threshold changes | Medium | Medium | Default to deterministic `section_recursive`. Treat `section_semantic` as eval-gated and document that `ground_truth_chunk_ids` must be refreshed after strategy changes. |
| Quality filter over-aggressively removes useful short passages (e.g., wine classification summaries that are short but high-signal) | Medium | Medium | Start with `chroma.indexing.quality_filter.min_score: 0.4` (conservative); tune upward only after eval shows recall improves. Check `chunks_filtered` count on first reindex — if > 30% of chunks are filtered, re-evaluate thresholds before proceeding. |
| Contextual search lineage causes the embedder to over-weight headings over body content | Low | Medium | The shared helper de-duplicates and rejects suspect headings, retains the clean body, and passed the focused/hybrid gate. Phase 2 still measures a body-only ablation on a frozen region/grape cohort. |
| HyDE hypothetical passage hallucinates a confident but wrong wine "fact" as the query anchor | Medium | Low | Controlled by `blend_alpha=0.5` — raw query embedding contributes equally; the hypothetical is a retrieval *direction* hint, not a factual claim. The retrieved content itself is unchanged. |
| `sigmoid(rerank_score)` normalisation is inappropriate for ms-marco CrossEncoder logit distribution | Low | Low | Sigmoid on ms-marco scores is an approximation. If eval shows the confidence signal is poorly calibrated, switch to `min-max` normalisation across the batch instead. |
| Tavily API rate limits triggered by automatic fallback on high-traffic eval runs | Low | Low | Web cache (existing `web_cache.db`) deduplicates identical queries with per-type TTL. Eval runs hit few unique queries. |
| Reindex time on full corpus after enabling filter + enrichment | Low | Low | These are CPU-only operations added to the chunking loop. Expected overhead is < 5% of total reindex time (dominated by embedding generation). |

### 9.2 Fixed rollout defaults and phase-time decisions

There are no unresolved architecture/dependency decisions required to begin Delivery Gate 0 after
the user approves this specification. The following later decisions are deliberately eval-gated:

1. **Semantic chunking**: `section_recursive` remains the M3 default. `section_semantic` stays
   disabled and experimental throughout this milestone; switching the default requires a later
   reviewed decision.
2. **Minimum quality score**: `0.4` with `enforce` is the accepted corrective Phase 0 default. Phase
   1 may recalibrate it only through a reviewed corpus sample and retrieval checkpoint; do not add a
   second noise filter or change the default silently.
3. **HyDE**: remains disabled by default even if implemented. Its delivery step is skipped entirely
   unless the entry gate in Phase 3 passes. Production enablement is a separate reviewed rollout.
4. **Web fallback**: remains disabled by default after implementation. Enablement requires the Phase
   5 quality/cost gate and a separate reviewed rollout because it consumes Tavily quota.
5. **PyMuPDF/PyMuPDF4LLM**: explicitly out of M3 scope. Do not add an adapter, dependency, stub, or
   config value in this milestone.

---

## 10. Cost Analysis

| Feature | Cost per query | Cost at indexing | Notes |
|---------|---------------|-----------------|-------|
| Phase 0 default extraction | 0 | Local PDF/EPUB parsing | Replaces `unstructured`; cost is CPU/disk only |
| Phase 0 `section_recursive` chunking | 0 | Pure Python + LangChain splitter | Deterministic default; no extra model calls |
| Phase 0 `section_semantic` chunking | 0 | Extra local embedding pass during indexing | Optional only; useful for long prose sections if eval proves value |
| 3A Noise filter | 0 | < 1 ms per chunk (pure Python) | One-time reindex cost only |
| 3B Contextual enrichment | 0 | < 0.5 ms per chunk (string concat) | One-time reindex cost only |
| 3C HyDE (when enabled) | +1 configured HyDE-model call | 0 | Explicit `hyde.provider`/`hyde.model`; never inherited implicitly |
| 3D Reranker threshold | 0 | 0 | Uses existing reranker; threshold is just a filter |
| 3E Web fallback (when triggered) | +1 Tavily call | 0 | Only on low-confidence queries; results cached |

**Expected API call budget per query after optional features are enabled**:
- Baseline: 1 configured final-answer generation call
- + HyDE: +1 explicitly configured HyDE-model call
- + Web fallback (only when triggered): +1 Tavily call
- Total maximum: 2 LLM calls + 1 search API call — within the 2–3 call budget (Milestone constraint).

---

## 11. Measurement Plan (Summary)

| Checkpoint | Metric | Expected direction |
|-----------|--------|--------------------|
| After Delivery Gate 0 | MRR, precision@3, precision@5, ordered chunk-ID parity, BM25 sync validation | Exact baseline parity; synchronized hybrid index |
| After 3D (delivery step 2) | MRR, precision@3, precision@5, context precision/recall, confidence distribution | Precision ↑ or flat; no gated metric regression > 2pp |
| After Phase 0 (reindex) | Chunk count, average chunk length, empty chunk rate, context_precision, context_recall, MRR | Cleaner chunks; precision ↑ or flat; recall regression ≤ 2pp |
| After 3A (reindex) | Total indexed chunks, context_precision | Chunk count ↓ 10–25%; precision ↑ or flat |
| After 3B (reindex) | context_precision on region/grape queries | ↑ 3–8% |
| After 3E (delivery step 6) | answer relevancy, faithfulness, context recall, web trigger rate on frozen cohort | one answer metric ↑ ≥ 2pp; no gated regression > 2pp; trigger rate ≤ 20% |
| After 3C if entry gate passes (delivery step 7) | context recall on semantic-mismatch cohort; LLM calls per query | recall ↑ ≥ 5pp; calls +1; no global regression > 2pp |

All results stored in `eval-results/` as timestamped JSON, following the Milestone 2 format.

---

## 12. Review Checklist

Implementation remains blocked until the user reviews and approves this specification. The primary
decisions are:

- [x] Keep `execute_production_rag()` as the sole orchestration owner and route agent RAG tools
  through it with generation disabled.
- [x] Treat Chroma as the BM25 source of truth; rebuild and validate both indices together, with
  explicit vector-only fallback on synchronization failure.
- [x] Replace `unstructured` with `pdfplumber` and `ebooklib`; keep PyMuPDF/PyMuPDF4LLM entirely out
  of M3.
- [x] Use the fixed eight-step delivery order in Section 6, including the safety gate before any
  behavior change and conditional HyDE after web-fallback evaluation.

Also review the scoped supporting decisions: exact corpus diagnostics, nullable calibrated reranker
threshold, shared web-search service extraction, cohort additions, internal artifact fields, config
defaults, and the conditional HyDE prompt. Approval may accept the full specification or request
changes to any item; it does not start implementation automatically.
