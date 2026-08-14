# Chroma ingestion module

> **Project version**: 0.8.0 — last verified 2026-08-14.

This module turns local PDF and EPUB books into the accepted passages searched by Pour Decisions.
It owns extraction, chunking, quality filtering, contextual search text, Chroma persistence, and the
synchronized BM25 build. It does not own query-time ranking or answer generation.

For the full pipeline in plain English, start with
[`docs/pour-decisions-rag-pipeline.md`](../../docs/pour-decisions-rag-pipeline.md).

## What happens here

```text
PDF / EPUB
  -> provider-specific extractor
  -> provider-neutral DocumentElement records
  -> structure-aware chunks
  -> quality enforcement
  -> clean evidence + contextual search text
  -> Chroma vectors
  -> verified BM25 index and synchronization manifest
```

In plain terms, this module reads books without losing useful page or heading structure, divides
them into passages that still make sense independently, rejects obvious noise, and makes the same
accepted passages available to meaning-based and keyword-based search.

## Component map

| Path | Responsibility |
|---|---|
| `extraction/` | `pdfplumber` and `ebooklib` providers that emit `DocumentElement` records |
| `structural_roles.py` | Shared classification of prose, tables, contents, indexes, bibliographies, and other roles |
| `chunking/` | Section-, entry-, block-, page-, and column-aware chunk construction |
| `chunk_filter.py` | Quality scoring and rejection before embedding |
| `contextual_text.py` | Deterministic search text from clean body and validated headings |
| `ingestion_pipeline.py` | Extraction/chunking orchestration and flat metadata contract |
| `chunks.py` | Compatibility entry point for `split_file()` |
| `loader.py` | Batch ingestion, filtering, embedding, deduplication, and statistics |
| `bm25_builder.py` | Atomic BM25 rebuild and Chroma/BM25 synchronization checks |
| `index_tracker.py` | File-hash manifests for incremental indexing |
| `stats.py` | Sampled and exact collection diagnostics |
| `hierarchical_chunks.py` | Optional small-to-big support; disabled by default |

## Stable boundaries and contracts

### Provider-neutral extraction

All extractors produce `DocumentElement` records. Chunking therefore consumes text, structure,
layout lineage, and audit flags rather than library-specific PDF or EPUB objects. New providers
should implement this boundary instead of leaking their own object model downstream.

PDF extraction reconstructs reading order from positioned words and records pages, blocks,
columns, layout confidence, and suspicious geometry. EPUB extraction follows spine order and uses
headings, anchors, element IDs, and publisher structure to preserve entries and sections.

### Chunk boundaries

The production `section_recursive` chunker does not silently cross document, chapter, entry,
section, structural-role, column, or source-block boundaries. Large units split at paragraphs,
sentences, then whitespace. The optional `section_semantic` strategy is for controlled evaluation
and remains disabled.

### Clean text versus search text

The Chroma `document` field is the clean source body used as answer evidence. Dense embeddings,
BM25, and reranking use `build_contextual_search_text()`, which adds only validated document title,
chapter, entry, and section context. Keeping these representations separate makes passages easier
to find without presenting generated prefixes as source text.

### Metadata

Indexed records carry:

- identity and source: filename, path, type, chunk index/ID, and content hash;
- structure: document title, chapter, entry, section, heading path, and structural role;
- provenance and layout: extractor, chunker, pages, blocks, column, and layout confidence;
- size and quality fields;
- wine entities: grapes, regions, vintages, classifications, producers, and appellations.

Metadata must remain flat and Chroma-compatible. Chunk IDs combine source, position, and a clean
body hash. Exact body duplicates are skipped when configured.

## Index lifecycle

Use incremental indexing for normal additions:

```bash
make chroma-up
make chroma-upload
```

An incremental update can leave the persisted BM25 index stale. Retrieval will then fall back to
vector-only search until the verified workflow is run:

```bash
make chroma-reindex
```

Forced reindex validates the source path, resets and rebuilds Chroma, fails if any book fails,
builds BM25 in temporary files from accepted Chroma records, verifies count and sorted chunk IDs,
and atomically publishes:

```text
chroma-data/bm25_index.pkl
chroma-data/bm25_index.meta.json
```

This ordering prevents retrieval from combining a new vector corpus with an old keyword corpus.

Diagnostics:

```bash
make chroma-stats        # Fast sampled view
make chroma-stats-exact  # Exact configured-corpus report
```

## Active invariants

- Indexing and querying must use the same `EMBEDDING_MODEL`.
- The structural quality gate runs before either search index is published.
- Chroma stores clean answer evidence while all relevance stages share contextual search text.
- BM25 is built from accepted Chroma records, not independently from source files.
- A forced rebuild publishes BM25 only after Chroma and synchronization validation succeed.
- Missing or stale synchronization state is visible and causes vector-only retrieval.

The current defaults live in `app_config.yml`; do not copy its full configuration here. Key active
choices are `pdfplumber`, `ebooklib`, `section_recursive`, enforced quality filtering, and local
embeddings. The canonical guide records the readable configuration snapshot and the reasons behind
these choices.

## Verification and limitations

Focused tests under `tests/chroma/` cover reading order, repeated header/footer removal, EPUB entry
boundaries, chunk boundaries, structural filtering, contextual text, loader statistics, and atomic
BM25 synchronization.

Known constraints include missing OCR support, ambiguous layouts that require audit, and some
literal `(cid:...)` extraction artifacts in the accepted corpus. Address these only through a
measured corpus/retrieval comparison rather than an unverified cleanup heuristic.
