# Chroma ingestion module

> **Project version**: 0.7.3 — last verified 2026-08-12.

This module owns the Milestone 3 document-ingestion path: provider-specific PDF/EPUB extraction,
provider-neutral elements, structure-aware chunks, quality filtering, contextual embedding text,
Chroma persistence, and a synchronized BM25 rebuild.

## Active pipeline

```text
PDF / EPUB
  -> configured extractor
  -> DocumentElement stream with layout and heading lineage
  -> section_recursive chunker (section_semantic is eval-only)
  -> ChunkQualityFilter
  -> plain chunk body + contextual search text
  -> Chroma embeddings and stored metadata
  -> verified BM25 index + synchronization manifest
```

The text stored in Chroma's `document` field remains the clean source body shown to the LLM. Dense
embeddings and BM25 use `build_contextual_search_text()`, which prepends a compact, deduplicated
heading path. This adds document context to search without polluting answer evidence.

## Components

| File or package | Responsibility |
|---|---|
| `extraction/` | Configured `pdfplumber` and `ebooklib` providers that emit `DocumentElement` records |
| `structural_roles.py` | Shared classification of prose, contents, index, bibliography, references, and other structural roles |
| `chunking/` | Section-, entry-, block-, page-, and column-aware recursive/semantic chunking |
| `chunk_filter.py` | Quality assessment and enforcement before embedding |
| `contextual_text.py` | Deterministic search-text construction from chunk body and structural metadata |
| `ingestion_pipeline.py` | Extraction/chunking orchestration and the flat Chroma metadata contract |
| `chunks.py` | Compatibility entry point for `split_file()` |
| `loader.py` | Incremental or forced batch ingestion, deduplication, filtering, embedding, and statistics |
| `bm25_builder.py` | Atomic BM25 rebuild and Chroma/BM25 synchronization validation |
| `index_tracker.py` | File-hash manifest for incremental indexing |
| `stats.py` | Sampled operational statistics and exact corpus acceptance artifacts |
| `hierarchical_chunks.py` | Optional small-to-big retrieval support; disabled by default |

## Extraction

The provider registry resolves formats from `chroma.extraction`:

- PDF: `pdfplumber`
- EPUB: `ebooklib`
- unsupported formats: skipped by default, or rejected when `fail_on_unsupported_file` is true

PDF extraction reconstructs reading order from positioned words. It detects supported two-column
layouts, keeps full-width headings in order, assigns page-local block and column lineage, and marks
ambiguous layouts with `layout_audit_required`. Repeated headers and footers are removed when their
normalized text repeats across enough pages.

EPUB extraction follows spine order rather than archive order. XHTML headings maintain document,
chapter, section, and entry context. Explicit navigation fragments, publisher entry classes, and
runs of peer headings establish dictionary-style entry boundaries, preventing one grape entry from
inheriting the title of another.

Both providers classify obvious structural pages/elements (contents, index, bibliography,
references, and similar navigation material) so they can be rejected before chunking or indexing.

## Chunking

### `section_recursive` (production default)

The default chunker groups only compatible elements. A chunk cannot silently cross a document,
chapter, entry, section, structural-role, column, or source-block boundary. Oversized prose is split
at paragraph, sentence, then whitespace boundaries; overlap remains inside the same structural
group. Source page/block ranges and reading-order confidence are preserved as scalar metadata.

```python
from src.chroma.chunks import split_file

chunks = split_file(
    filepath="wine_guide.pdf",
    strategy="section_recursive",
    chunk_size=1024,
    overlap_size=256,
)
```

### `section_semantic` (disabled by default)

This strategy looks for semantic breakpoints independently inside each structural group using the
cached local embedder. A failed semantic split falls back explicitly to recursive splitting for that
content. Enable it only for a controlled evaluation because it adds an embedding pass and produces
less deterministic boundaries.

The removed legacy names `basic`, `by_title`, and `semantic` are not supported.

## Quality gate and contextual search text

`ChunkQualityFilter` runs after chunk assembly and before embedding. In the configured `enforce`
mode it rejects confident structural noise and chunks below `min_score`; it records
`structural_role`, `quality_score`, and `quality_reasons` for assessed candidates. Loader statistics
report both `chunks_filtered` and `chunks_below_quality_threshold`.

`build_contextual_search_text()` uses, in order, `document_title`, `chapter`, `entry_title`, and
`section`, removes repeated headings, rejects structural headings, and appends the plain body. The
same function is used by dense indexing, BM25 construction, and cross-encoder reranking, keeping the
three relevance stages consistent.

## Metadata contract

Each indexed record contains:

- identity/source: `filename`, `file_path`, `file_type`, `chunk_index`, `chunk_id`, `content_hash`
- structure: `document_title`, `chapter`, `entry_title`, `section`, `heading_path`, `structural_role`
- provenance: `extraction_provider`, `chunking_strategy`
- layout lineage: `page_number`, `start_page`, `end_page`, `column_id`, `start_block_id`,
  `end_block_id`, `layout_audit_required`, `reading_order_confidence`
- size/quality: `word_count`, `char_count`, plus quality fields added by the loader
- wine entities: `grapes`, `regions`, `vintages`, `classifications`, `producers`, `appellations`

Chunk-level chapter, section, and entry fields come from the element group that produced the chunk;
they are no longer copied from one file-level "last heading" value.

IDs use `{source_stem}_{chunk_index}_{content_hash_prefix}`. Exact duplicates are skipped by content
hash when configured, while the original document body remains available for citations and context.

## Loading and index lifecycle

Production construction passes the complete extraction, chunking, and indexing configuration to
`CollectionDataLoader`. A simplified direct example is:

```python
from src.chroma import CollectionDataLoader
from src.utils import get_config

cfg = get_config()
loader = CollectionDataLoader(
    collection_name="wine_books",
    collection_metadata={"description": "Professional wine books collection"},
    chroma_host=cfg.chroma.client.host,
    chroma_port=cfg.chroma.client.port,
    embedding_model=cfg.chroma.settings.embedder,
    batch_size=cfg.chroma.settings.batch_size,
    extraction_config=cfg.chroma.extraction,
    chunking_config=cfg.chroma.chunking,
    indexing_config=cfg.chroma.indexing,
)
```

Incremental uploads use the file-hash manifest and retry failed files on the next run. Forced
reindexing resets Chroma, processes the configured corpus, and fails the command if any source
fails. Only after a successful Chroma build does it construct BM25 in temporary files, verify record
count and sorted chunk-ID SHA-256 against Chroma, and atomically replace:

- `chroma-data/bm25_index.pkl`
- `chroma-data/bm25_index.meta.json`

Retrieval validates that manifest before enabling hybrid search. Missing or stale synchronization
state causes an explicit vector-only fallback; it never silently combines mismatched indexes.

```bash
make chroma-up
make chroma-upload       # incremental
make chroma-reindex      # destructive full rebuild + verified BM25 replacement
make chroma-stats        # sampled operational view
make chroma-stats-exact  # exact configured-corpus artifact
```

## Current configuration

The reviewed Phase 0 defaults in `app_config.yml` are:

| Setting | Value |
|---|---:|
| PDF / EPUB providers | `pdfplumber` / `ebooklib` |
| chunking strategy | `section_recursive` |
| chunk size / overlap | 1024 / 256 characters |
| minimum chunk length | 200 characters |
| quality filter | `enforce`, minimum score `0.4` |
| small-to-big | disabled |
| Chroma host port | 8100 |
| batch size | 2500 |

The embedding model is supplied by `EMBEDDING_MODEL`; indexing and querying must use the same model.

## Verification

Focused tests live under `tests/chroma/`. They cover column order, repeated header/footer removal,
EPUB entry boundaries, structural-role filtering, chunk boundary preservation, contextual text,
loader quality statistics, and atomic BM25 synchronization.

The accepted Phase 0 corpus contains 37,374 records from 22 sources with sorted chunk-ID SHA-256
`464d855a25cf834c77215ade8c9e28be6c793141fb51b35459aa8773e630a278`. Generated acceptance
artifacts remain under the ignored `eval-results/` directory; see the Milestone 3 design record and
the manual testing guide for exact commands and results.

## Intentional limitations

- OCR is not part of this phase; image-only or malformed pages can still fail or yield noisy text.
- Ambiguous page layouts are retained with audit metadata rather than guessed aggressively.
- Two source books that could not be extracted reliably were excluded from the accepted 22-source
  corpus and should be revisited with an explicit parser/OCR decision.
- The corpus still contains 576 chunks with literal `(cid:...)` text. Retrieval acceptance passed,
  but this is a tracked cleanup candidate for a later reindex.

See [`docs/rag-pipeline-deep-dive.md`](../../docs/rag-pipeline-deep-dive.md) for the end-to-end query
path and [`docs/m03-phase0-corrective-manual-testing.md`](../../docs/m03-phase0-corrective-manual-testing.md)
for the closure procedure.
