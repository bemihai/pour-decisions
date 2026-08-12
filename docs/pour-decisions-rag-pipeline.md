# Pour Decisions RAG Pipeline

> **Project version**: 0.7.3 — last verified 2026-08-12.
> This is the concise implementation overview. See `docs/rag-pipeline-deep-dive.md` for the
> code-level trace and the Milestone 3 Phase 0 evidence.

Pour Decisions answers wine questions from locally indexed PDF and EPUB books. The system keeps
external-call cost at zero for extraction, indexing, embeddings, BM25, query planning, reranking,
and retrieval evaluation. Generation uses the configured application model only after local context
has been assembled.

## Architecture

```text
PDF / EPUB books
    |
    v
layout-/entry-aware extraction
    |
    v
block-aware section chunking
    |
    v
structural role + quality enforcement
    |
    +--> clean original text stored in Chroma
    |
    +--> validated contextual text embedded locally
    |
    +--> forced reindex rebuilds synchronized contextual BM25

user question
    |
    v
deterministic semantic + sparse query plan
    |
    +--> contextual vector pool (25)
    +--> synchronized BM25 pool (25)
              |
              v
      balanced de-duplicated union (<= 50)
              |
              v
      metadata boost -> cross-encoder threshold -> confidence
              |
              v
      semantic deduplication -> formatted context -> LLM/agent
```

The RAG-only API, eval harness, and agent RAG tools share `execute_production_rag()`. Agent tools
disable generation inside that function because the LangGraph agent owns final synthesis.

## 1. Extraction

Extraction is provider-neutral. `DocumentElement` is the stable boundary between providers and
chunking; downstream code does not depend on pdfplumber or ebooklib objects.

### PDF

`PdfPlumberExtractor` uses positioned text to identify full-width regions, columns, gutters, and
blocks. It emits each block in page-local reading order and preserves `block_id`, `column_id`,
layout mode, page number, and reading-order confidence. Invalid/off-canvas geometry is marked
`layout_audit_required` so the quality gate can reject it. Repeated headers and footers are removed
when configured, and OCR placeholders/page numbers/form labels cannot become heading context.

### EPUB

`EbookLibExtractor` reads XHTML in deterministic spine order. Navigation anchors, element IDs,
CSS classes, and verified publisher/peer-entry boundaries establish hierarchy. A new dictionary or
grape entry resets inherited context; real subsections retain their parent entry.

## 2. Chunking and indexing

The default `section_recursive` chunker uses `chunk_size=1024`, `chunk_overlap=256`, and
`min_chunk_chars=200`. It preserves source, entry, chapter, section, role, column, and block
boundaries. Oversized blocks split at paragraph/sentence/whitespace boundaries. Overlap contains
complete trailing units rather than arbitrary character windows.

`section_semantic` is available for controlled evaluation but disabled by default. Legacy
`basic`, `by_title`, and `semantic` strategy names are not supported.

### Structural quality gate

The provider-neutral roles are `prose`, `table`, `wine_list`, `toc`, `bibliography`, `index`,
`worksheet`, and `unknown`. The active `enforce` mode rejects confident ToC, bibliography, index,
worksheet, invalid-layout, extraction-garbage, and below-threshold candidates before embeddings.
Useful tables and wine lists remain searchable. Audit metadata stores the role, quality score, and
reasons.

### Stored text versus search text

Chroma stores the clean original chunk for display. Dense embeddings use a validated local prefix
built from document title, chapter, entry, and section. Wine metadata is extracted from clean text
plus that validated lineage. Uncertain, structural, page-like, score-like, and OCR-corrupt headings
are excluded.

The BM25 builder reads accepted Chroma records and reconstructs the same contextual representation,
so dense and sparse search cannot drift into independently assembled corpora.

### Duplicate handling

Chunk IDs use source stem, chunk index, and clean-text content hash. The loader skips duplicate
content within the current file batch and against existing Chroma records. Incremental file state is
tracked under `chroma-data/manifests/`.

## 3. Chroma/BM25 lifecycle

Use incremental indexing for ordinary additions:

```bash
make chroma-upload
```

An incremental Chroma change can invalidate the existing BM25 manifest. Retrieval detects that
mismatch and uses vector-only search until the verified workflow is run:

```bash
make chroma-reindex
```

Forced reindexing validates the source path before resetting Chroma, fails if any book fails,
rebuilds BM25 from the completed Chroma snapshot, verifies record count and sorted chunk-ID hash,
and atomically replaces:

```text
chroma-data/bm25_index.pkl
chroma-data/bm25_index.meta.json
```

Exact corpus diagnostics are available with:

```bash
make chroma-stats-exact \
  CORPUS_STATS_OUTPUT=eval-results/corpus_$(date +%Y%m%d).json
```

ChromaDB is exposed on host port `8100` (container port `8000`).

## 4. Query planning

`RetrievalQueryPlan` is deterministic and local. It contains the normalized question, detected
entities, intent, semantic query, and sparse query. Supported intents are flavour, aging, pairing,
classification, and region. Intent-specific rewriting occurs only when an entity is detected.

For example:

```text
Question: What are the primary flavour characteristics of Nebbiolo?
Semantic: nebbiolo aroma flavor taste sensory profile tannin acidity body
Sparse:   nebbiolo aroma taste tannin acidity body
```

The sparse analyzer case-folds Unicode, normalizes punctuation/apostrophes/hyphens, applies wine
terminology, and removes low-value question words. Index and query use the same analyzer.

## 5. Hybrid retrieval and reranking

The production retriever obtains complete pools of 25 dense and 25 BM25 candidates. It de-duplicates
them by chunk ID while preserving channel ranks, scores, provenance, and timing. Up to 50 unique
candidates enter the local cross-encoder.

There is no 70/30 weighted fusion. Standard unweighted reciprocal-rank fusion is used only when the
reranker is unavailable. A missing or stale BM25 synchronization manifest produces an explicit
vector-only fallback.

Detected entity matches add a `0.1` metadata boost per matching field. The
`cross-encoder/ms-marco-MiniLM-L-6-v2` model then reranks the union. The active numeric threshold is
`0.0`, so negative logits are removed before the final five chunks. Confidence is the stable sigmoid
normalization of the strongest reranker score; automatic web fallback is still disabled.

Semantic deduplication runs before context formatting. Optional small-to-big expansion and TF-IDF
compression are disabled in the current configuration.

## 6. Retrieval evaluation and curation

Run the production retrieval harness without generation calls:

```bash
PYTHONPATH=. uv run python -m src.eval \
  --mode retrieval \
  --backend rag \
  --categories rag_only
```

Candidate inspection and manual golden-ID curation use the same production path:

```bash
PYTHONPATH=. uv run python -m src.eval.scripts.chunk_id_lookup \
  --question "What are the primary flavour characteristics of Nebbiolo?" \
  --top-k 10 \
  --mode hybrid \
  --full-text \
  --json

PYTHONPATH=. uv run python -m src.eval.scripts.chunk_id_curator \
  --dataset src/eval/wine_qa_golden.jsonl \
  --redo \
  --top-k 10 \
  --mode hybrid
```

Both utilities also provide explicit `vector` and `bm25` diagnostic modes. Full-text review shows
channel ranks/scores, metadata matches, structural role, heading lineage, source, and reranker score.

## 7. Active configuration

```yaml
chroma:
  client:
    host: ${oc.env:CHROMA_HOST, localhost}
    port: ${oc.env:CHROMA_PORT, 8100}

  extraction:
    pdf_provider: pdfplumber
    epub_provider: ebooklib
    fail_on_unsupported_file: false
    strip_repeated_headers: true
    strip_repeated_footers: true

  chunking:
    strategy: section_recursive
    chunk_size: 1024
    chunk_overlap: 256
    min_chunk_chars: 200
    semantic:
      enabled: false

  indexing:
    quality_filter:
      mode: enforce
      min_score: 0.4
    bm25:
      rebuild_on_reindex: true
      sync_manifest_path: chroma-data/bm25_index.meta.json

  retrieval:
    n_results: 5
    similarity_threshold: 0.3
    use_deduplication: true
    deduplication_threshold: 0.9
    enable_hybrid: true
    semantic_candidate_pool: 25
    bm25_candidate_pool: 25
    reranker_input_limit: 50
    bm25_index_path: chroma-data/bm25_index.pkl
    validate_bm25_sync: true
    enable_reranking: true
    reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
    rerank_top_k: 5
    rerank_threshold: 0.0
    min_retrieval_confidence: 0.3
    enable_compression: false
    enable_metadata_boost: true
    metadata_boost_factor: 0.1
```

## 8. Accepted Phase 0 result

The final 2026-08-12 index has 37,374 synchronized Chroma/BM25 records from 22 sources. On the
24 scorable retrieval questions, production achieved MRR `0.8368`, precision@3 `0.6250`, and
precision@5 `0.5833`. Hybrid recall@10 was `0.9208` with exact-entity hit rate `1.0`.

The original Nebbiolo failure now returns a direct answer at rank 1, 9/10 relevant pre-rerank union
candidates, and no structural/interleaved/OCR-artifact candidate in that top ten.

Phase 0 is closed. The next delivery step is Phase 1 — Noise Chunk Filtering, which may extend the
audit/calibration lifecycle around the minimum structural gate already moved into corrective Phase 0.
