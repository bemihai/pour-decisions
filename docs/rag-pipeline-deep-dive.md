# RAG Pipeline Deep Dive

> **Project version**: 0.8.0 — last verified 2026-08-14.
> This is the canonical code-level trace of the completed Milestone 3 pipeline.

Pour Decisions indexes local PDF and EPUB wine books into ChromaDB and a synchronized BM25 index.
The API, eval harness, and agent RAG tools all use `execute_production_rag()` so retrieval behavior
is measured on the same path served to users.

## 1. Implemented Milestone 3 scope

The following behavior is active:

- provider-neutral extraction and chunking contracts;
- layout-aware `pdfplumber` PDF extraction;
- entry-aware `ebooklib` EPUB extraction;
- block-aware `section_recursive` chunking (`1024` characters, `256` overlap);
- structural-role and deterministic quality enforcement before indexing;
- validated contextual text for both dense embeddings and BM25;
- atomic BM25 rebuild plus count/sorted-ID synchronization manifest;
- deterministic dense and sparse query planning;
- balanced dense/BM25 candidate union with channel provenance;
- metadata boosting, local cross-encoder thresholding, confidence, and semantic deduplication;
- a shared production path for API, eval, and agent tools.

`section_semantic` exists but is disabled. Automatic web fallback is implemented behind
`web_search.auto_fallback` and disabled by default. HyDE and the BGE embedding-model switch were
evaluated and rejected; neither has retained runtime code or configuration.

## 2. Indexing pipeline

### 2.1 Entry points and lifecycle

`make chroma-upload` runs incremental indexing. `IndexTracker` skips unchanged files and persists
success after each file, so interrupted work can resume. Incremental changes can make the existing
BM25 manifest stale; retrieval then logs the reason and falls back to vector-only.

`make chroma-reindex` is the authoritative synchronized workflow:

1. Validate that the configured source directory exists and contains supported files.
2. Reset the Chroma collection.
3. Re-extract and reindex all supported books.
4. Fail if any source produced an indexing error.
5. Rebuild BM25 from the completed Chroma collection.
6. Atomically replace the BM25 pickle and synchronization manifest only after validation.

### 2.2 Provider-neutral extraction

`DocumentExtractionPipeline` resolves extractors through `ExtractorRegistry`. Both providers emit
`DocumentElement` values rather than provider-native objects. Required fields are clean text,
source path, file type, and deterministic order index. Optional lineage includes page, element
type, heading level, document title, chapter, section, structural role, and provider audit metadata.

#### PDF extraction

`PdfPlumberExtractor` reads positioned words/characters and constructs page-local lines and blocks.
It detects full-width regions, gutters, columns, invalid geometry, and reading-order confidence.
Content is emitted top-to-bottom inside each block and left-to-right across columns; blocks are not
flattened into alternating column lines. Repeated margin headers/footers can be removed across pages.

Important metadata includes `block_id`, `column_id`, `layout_mode`,
`reading_order_confidence`, and `layout_audit_required`. Invalid or off-canvas geometry is marked
for audit instead of being silently indexed. OCR placeholders, page numbers, rating rows, and
single-letter form labels cannot update heading context.

#### EPUB extraction

`EbookLibExtractor` reads XHTML documents in spine order. It retains spine item, element ID, CSS
class, navigation target, and entry-boundary evidence. Navigation anchors and verified peer entry
headings reset inherited context, preventing one grape/dictionary entry from leaking into the next.
Explicit continuation can preserve legitimate hierarchy across spine items.

### 2.3 Structural roles

`structural_roles.py` classifies provider-neutral content as one of:

- `prose`
- `table`
- `wine_list`
- `toc`
- `bibliography`
- `index`
- `worksheet`
- `unknown`

`toc`, `bibliography`, `index`, and `worksheet` are forbidden index roles. Detection uses headings,
line structure, citation density, form blanks/scales, tasting-form labels, list density, sentence
density, and OCR placeholder density. Useful tables and wine lists remain indexable.

### 2.4 Block-aware chunking

`DocumentChunkingPipeline` resolves `SectionRecursiveChunker` by default. It never crosses source,
entry, chapter, section, structural-role, column, or validated block boundaries. It packs complete
paragraph/list units up to `chunk_size`. A single oversized unit is split at paragraph, sentence,
and whitespace boundaries. Overlap reuses complete trailing units and never creates an arbitrary
mid-sentence window.

Chunks shorter than `min_chunk_chars` are dropped unless wine metadata makes them high signal.
Each `ChunkCandidate` preserves heading path, page range, block/column lineage, provider, strategy,
structural role, reading-order confidence, and audit flags.

### 2.5 Metadata assembly and quality enforcement

`assemble_chroma_chunks()` produces the stable loader dictionary:

```text
{"id", "text", "metadata", "importance_score"}
```

Chunk IDs combine the source stem, chunk index, and the first eight characters of the clean-text
content hash. Wine entities are extracted from clean text plus validated structural lineage so
pronoun-heavy section chunks remain discoverable.

`ChunkQualityFilter` runs in `enforce` mode with a `0.4` minimum. It stores
`structural_role`, `quality_score`, and stable `quality_reasons`. Confident forbidden roles,
invalid layout, dense OCR placeholders, empty/very-short unknown content, and ambiguous candidates
below the threshold are discarded before embeddings. Rejected chunks enter neither Chroma nor BM25.

### 2.6 Contextual embeddings with clean stored text

`build_contextual_search_text()` validates and de-duplicates `document_title`, `chapter`,
`entry_title`, and `section`, then builds:

```text
<document title> > <chapter> > <entry> > <section>

<clean chunk body>
```

Uncertain, structural, OCR-corrupt, score-like, or page-number headings are excluded. The loader
embeds this contextual representation but stores only the clean chunk body as the Chroma document.
Users therefore see clean source text while dense retrieval benefits from local lineage.

### 2.7 Duplicate handling and Chroma writes

The loader validates chunk dictionaries, applies the quality gate, and performs content-hash
deduplication within the file batch and against existing Chroma records. Accepted contextual texts
are embedded in batches; clean documents, metadata, IDs, and embeddings are added to Chroma.

### 2.8 Synchronized BM25

After a successful forced reindex, `rebuild_bm25_from_collection()` reads every accepted Chroma
record in bounded batches. BM25 reconstructs the same contextual representation from the stored
clean document and metadata, then tokenizes it with the shared Unicode-aware analyzer.

The builder writes a temporary pickle and manifest, reloads them, validates record count and the
SHA-256 of sorted chunk IDs, confirms Chroma did not change during the build, and atomically replaces
the live files. Retrieval refuses a missing or mismatched manifest and falls back explicitly to
vector-only search.

## 3. Query and retrieval pipeline

### 3.1 Shared orchestration

`execute_production_rag()` is the single production owner. It is called by:

- the RAG-only API path;
- the eval harness;
- agent RAG tools with `generation_enabled=False`.

Agent planning and final synthesis remain owned by LangGraph, but the tool retrieval itself no
longer bypasses hybrid search, boosting, reranking, confidence, deduplication, or compression.

### 3.2 Deterministic query plan

`build_retrieval_query_plan()` creates:

- the original and normalized question;
- detected grapes, regions, vintages, classifications, producers, and appellations;
- an intent-focused semantic query;
- a sparse query analyzed with the same tokenizer used at BM25 build time.

Supported local intents are flavour, aging, pairing, classification, and region. Intent rewriting
requires a detected wine entity; otherwise the normalized question is preserved. The accepted
flavour semantic terms are `aroma flavor taste sensory profile tannin acidity body`. No LLM or
external API is used.

The BM25 analyzer case-folds Unicode, normalizes apostrophes and hyphens, applies configured wine
terminology, and removes low-value question stopwords without discarding wine entities, vintages,
classifications, appellations, or producer names.

### 3.3 Balanced candidate generation

With current defaults, `HybridRetriever` retrieves 25 dense candidates and 25 BM25 candidates.
The complete pools are de-duplicated by chunk ID and interleaved into a bounded union while
preserving `dense_rank`, `sparse_rank`, dense similarity, BM25 score, channel provenance, and timing
diagnostics. Up to 50 unique candidates reach the reranker.

There is no weighted 70/30 score fusion. If the reranker is unavailable, standard unweighted RRF
orders the complete union. If BM25 synchronization fails, the factory returns the vector retriever
and logs an explicit fallback.

### 3.4 Metadata boosting, reranking, and confidence

Detected entity matches add `metadata_boost_factor` (`0.1`) per matching metadata field before
reranking. Matches are recall/ranking evidence, not hard filters.

The local `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks the bounded union. Because
`rerank_threshold` is numeric (`0.0`), production calls `rerank_with_threshold()` and excludes
negative cross-encoder logits before returning up to `rerank_top_k` chunks (five by default).

`compute_confidence()` applies a stable sigmoid normalization to the strongest reranker score and
compares it with `min_retrieval_confidence`. With the default
`web_search.auto_fallback=false`, the signal is recorded only. When explicitly enabled,
`WebSearchFallback` queries the shared cached Tavily service for low-confidence results and appends
web evidence after book chunks. Provider or credential failure preserves book results unchanged.

The Phase 5 checkpoint found that this trigger catches empty-context failures but can miss stale
book evidence that still scores highly. Keep it disabled unless the external-call budget, added
latency, and this freshness limitation are acceptable.

### 3.5 Context construction

Optional small-to-big expansion is disabled. Semantic deduplication is enabled at `0.9`; it removes
exact content-hash duplicates and near-duplicate context. Source labels and chunk IDs are added only
when the final context string is formatted. TF-IDF compression is available but disabled. The
generation model receives the formatted context and conversation history. Agent tools stop before
generation and return that evidence to the agent.

### 3.6 Retrieval artifacts and curation

`RAGExecutionResult` records the query plan, raw union, final context chunks, channel ranks/scores,
feature usage, sources, threshold, confidence, and retrieval errors without changing the public API
schema. Eval reports preserve these fields for diagnosis.

`chunk_id_lookup.py` and `chunk_id_curator.py` default to the production hybrid path and also expose
explicit `vector`, `bm25`, and `hybrid` modes. Full-text review displays source, structural role,
heading path, channel ranks/scores, metadata matches, and reranker score. Dataset writes are atomic;
manual relevance judgment remains authoritative.

## 4. Active configuration

| Setting | Value |
|---|---:|
| PDF / EPUB provider | `pdfplumber` / `ebooklib` |
| Chunker | `section_recursive` |
| Chunk size / overlap / minimum | `1024` / `256` / `200` |
| Semantic chunking | disabled |
| Quality filter | `enforce`, minimum `0.4` |
| Dense / BM25 pools | `25` / `25` |
| Reranker input limit | `50` |
| Final rerank count | `5` |
| Rerank threshold | `0.0` |
| Metadata boost | enabled, `0.1` per field |
| Semantic deduplication | enabled, `0.9` |
| Context compression | disabled |
| Automatic web fallback | disabled |

## 5. Accepted Phase 0 checkpoint

The 2026-08-12 closing index contains 37,374 synchronized Chroma/BM25 records from 22 sources,
zero empty chunks, and no records missing source. The sorted chunk-ID SHA-256 is
`464d855a25cf834c77215ade8c9e28be6c793141fb51b35459aa8773e630a278`.

On 24 scorable retrieval questions, production hybrid retrieval achieved MRR `0.8368`,
precision@3 `0.6250`, precision@5 `0.5833`, recall@10 `0.9208`, and exact-entity hit rate `1.0`.
For the Nebbiolo failure query, a direct answer ranked first, 9/10 pre-rerank union candidates were
relevant, and no structural, interleaved, or OCR-artifact chunk appeared in that top ten.

The corpus still contains 576 chunks with literal `(cid:...)` OCR tokens and 14.15% of chunks are
shorter than 200 characters. Neither condition failed the Phase 0 gates, but both are explicit
quality observations for Phase 1.

See `docs/m03-phase0-corrective-manual-testing.md` for reproduction commands and
`eval-results/m3_phase0_extraction_chunking_20260812.json` for the local evidence artifact.

## 6. Accepted Phase 2 precision-recall trade-off

The 2026-08-13 Phase 2 checkpoint compared body-only and contextual search representations over
the same 37,412-record corpus. Deterministic metrics covered 24 samples; judge-based context
precision and recall covered the same seven reviewed region/grape samples for every variant.

In the final same-batch judge comparison, contextual search improved global MRR from `0.7431` to
`0.8368`, precision@3 from `0.4583` to `0.5972`, precision@5 from `0.4667` to `0.5583`, and cohort
context recall from `0.4524` to `0.5714`. Cohort context precision decreased from `0.6625` to
`0.5690`.

Two deliberately simple recovery variants were rejected:

- contextual candidate retrieval plus body-only reranking at top five reduced both context
  precision (`0.4143`) and recall (`0.3571`);
- the same mixed representation at top three reduced context precision to `0.3810`, recall to
  `0.1429`, and global precision@5 to `0.3417`.

The reviewed production decision accepts the current contextual representation because its
retrieval coverage gains are substantial and the tested simple alternatives were strictly worse.
This is an explicit exception to the original Phase 2 precision-improvement gate, not evidence that
the precision regression disappeared. Dense indexing, BM25, entity extraction, and reranker pairs
continue to share `build_contextual_search_text()`; clean chunk bodies remain the displayed and
generated evidence. A low-priority backlog item tracks future precision improvement without adding
complexity to the current production path.

Local evidence is recorded in `eval-results/m3b_contextual_enrichment_20260813.json` and
`eval-results/m3b_precision_recovery_20260813.json`.

## 7. Phase 5 web-fallback decision

The frozen five-sample current-information cohort triggered fallback for four empty-context cases.
Answer relevancy improved from `0.0000` to `0.7299`, and the projected combined trigger rate was
`13.3%`, below the `20%` ceiling. Mean cohort latency increased from `3.13 s` to `5.86 s` (`+87%`),
and one stale but plausible classification passage remained falsely high-confidence.

The implementation is retained for explicit opt-in use, with cached Tavily results and fail-safe
preservation of book evidence. The production default remains disabled because confidence is not a
complete freshness detector and every trigger introduces external cost.

## 8. Rejected Phase 3 and Phase 6 experiments

The five-query HyDE experiment changed no retrieval metric, added one model call, and increased mean
latency by about `7.2 s` per query. It was rejected and all experimental runtime code was removed.

The full-corpus embedding comparison changed only the dense model from
`sentence-transformers/all-mpnet-base-v2` to `BAAI/bge-base-en-v1.5`. BGE reduced global MRR from
`0.8368` to `0.8278`, precision@3 from `0.6111` to `0.5694`, and precision@5 from `0.5750` to
`0.4667`; mean retrieval latency increased from `1.56 s` to `2.90 s`. The external context judge
was deliberately skipped because it required sending private book passages to a cloud evaluator
after all local quality signals and latency were already unfavorable. The current embedder remains
unchanged.

## 9. Final Milestone 3 baseline

M3 closes with one explicit quality trade-off: contextual search materially improves retrieval
coverage and recall while reducing judged context precision on the reviewed cohort. The user
accepted that trade-off after simple body-only reranking variants performed worse. All retained
query-time retrieval stages are local and deterministic; the only optional external retrieval call
is disabled by default. Future experiments should start from a frozen failure cohort and be rejected
when added cost or complexity is not justified by meaningful gains.
