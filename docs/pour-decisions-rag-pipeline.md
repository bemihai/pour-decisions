# Pour Decisions RAG and Retrieval Pipeline

> **Project version**: 0.8.1 — last verified 2026-08-22.
> This is the canonical guide to the project's document indexing and retrieval system.

Pour Decisions answers wine questions using evidence from locally indexed PDF and EPUB books. In
plain English, it turns books into small searchable passages, searches those passages in two
different ways, checks which results best answer the question, and supplies the strongest evidence
to the answering model. This keeps document processing and retrieval local and makes the evidence
visible for evaluation and citations.

## One-minute overview

The system has two paths:

1. **Indexing**, which runs when books are added or rebuilt. It extracts readable text, preserves
   useful document structure, removes obvious noise, divides the text into passages, and writes the
   same accepted corpus to a vector index and a keyword index.
2. **Retrieval**, which runs for every question. It searches by meaning and by exact words, combines
   the candidates, reranks them with a more accurate local model, removes weak or repeated results,
   and formats the remaining passages as context.

```text
INDEXING
PDF / EPUB books
    -> layout-aware extraction
    -> section-aware chunks
    -> structural and quality checks
    -> contextual search text
    -> Chroma vectors + synchronized BM25 index

RETRIEVAL
user question
    -> deterministic query plan
    -> dense search (meaning) + BM25 search (words)
    -> balanced, de-duplicated candidate union
    -> metadata preference + cross-encoder reranking
    -> confidence threshold + semantic deduplication
    -> clean source context
    -> RAG answer or agent synthesis
```

The RAG-only API, evaluation harness, and agent knowledge tools all use
`execute_production_rag()`. This shared entry point prevents evaluation or agent behavior from
quietly drifting away from production retrieval.

## Essential vocabulary

| Term | Plain-English meaning |
|---|---|
| Document element | A provider-neutral piece of extracted content, such as a heading, paragraph, table, or list, with its source and layout information. |
| Chunk | A passage small enough to search and pass to a model, while retaining its chapter, section, page, and source lineage. |
| Contextual search text | A chunk's clean body prefixed with trustworthy title/chapter/section information. It gives search more context without changing the evidence shown to users. |
| Dense or vector search | Search by meaning. A local embedding model converts the question and passages into numeric vectors and finds nearby passages. |
| Sparse or BM25 search | Search by words. It strongly rewards exact and uncommon terms such as grape names, regions, and classifications. |
| Hybrid retrieval | Using dense and BM25 searches together so that semantic matches and exact terminology can recover each other's misses. |
| Reranker | A more precise local model that reads each question-passage pair and reorders the initial search results. |
| Retrieval confidence | A normalized summary of the strongest reranker score. It signals weak evidence but does not prove that evidence is current or factually complete. |
| Context | The final source passages supplied to the answering model. It is not the generated answer. |

## Indexing path

### 1. Extract document structure

`src/chroma/extraction/` converts PDF and EPUB files into `DocumentElement` records. That shared
record is the boundary between file-format code and the rest of the pipeline, so downstream logic
does not depend on `pdfplumber` or `ebooklib` objects.

- PDFs are reconstructed from positioned text. The extractor recognizes full-width text, columns,
  gutters, blocks, page order, and repeated headers or footers. Suspicious geometry is marked for
  audit rather than silently trusted.
- EPUBs are read in spine order. XHTML headings, navigation anchors, element IDs, and publisher
  structure preserve chapter, section, and dictionary-style entry boundaries.

This matters because a visually simple page can have a misleading internal text order. Keeping
layout and heading lineage prevents unrelated columns or adjacent reference entries from being
joined into one passage.

### 2. Build structure-aware chunks

`src/chroma/chunking/` turns compatible elements into chunks. The active
`section_recursive` strategy respects document, chapter, entry, section, structural role, column,
and source-block boundaries. Oversized content splits at paragraphs, then sentences, then
whitespace. Overlap uses complete trailing units instead of arbitrary character slices.

The current limits are 1,024 characters per chunk, 256 characters of overlap, and a 200-character
minimum. The optional `section_semantic` strategy is disabled because it costs an additional
embedding pass and has not earned a production advantage.

### 3. Reject structural noise

`src/chroma/chunk_filter.py` classifies and scores chunks before embedding. Confident tables of
contents, bibliographies, indexes, worksheets, invalid layouts, extraction garbage, and content
below the `0.4` quality score are rejected. Useful prose, tables, and wine lists remain eligible.

In plain terms, the system avoids searching page furniture and reference scaffolding as if those
were explanatory wine passages. It records the role, score, and reasons so filtering can be
audited.

### 4. Create clean evidence and contextual search text

Each accepted record has two related representations:

- The **clean original body** is stored as the document and later shown to the answering model.
- The **contextual search text** prefixes the body with validated document title, chapter, entry,
  and section fields. Dense embeddings, BM25, and reranking use this representation.

`src/chroma/contextual_text.py` rejects uncertain, structural, page-like, score-like, or corrupt
headings and removes duplicates. This improves findability while keeping synthetic prefixes out of
quoted evidence.

### 5. Write synchronized indexes

`src/chroma/loader.py` embeds contextual search text with the configured local embedder and stores
the clean body plus flat metadata in Chroma. Stable chunk IDs combine source, position, and a body
hash; duplicate content is skipped. File hashes under `chroma-data/manifests/` support incremental
updates.

`src/chroma/bm25_builder.py` builds BM25 from the accepted Chroma records and reconstructs the same
contextual representation. A manifest records the collection, count, sorted chunk-ID hash, and
index path. Retrieval enables hybrid search only when that manifest matches the active Chroma
collection.

The normal workflows are:

```bash
make chroma-upload       # Add or update changed books in Chroma
make chroma-reindex      # Rebuild Chroma, then atomically rebuild verified BM25
make chroma-stats        # Sample operational collection statistics
make chroma-stats-exact  # Calculate exact configured-corpus statistics
```

An incremental Chroma update can make BM25 stale. Until a verified full rebuild is run, retrieval
logs the mismatch and continues with vector search only. It never combines mismatched corpora.

## Retrieval path

### 1. Build a deterministic query plan

`src/retrieval/query_analyzer.py` normalizes the question, extracts wine entities, recognizes a
small set of intents, and creates separate semantic and sparse queries. It is deterministic and
makes no LLM call.

For example, a question about Nebbiolo flavour keeps the exact grape name for keyword search and
adds concepts such as aroma, taste, tannin, acidity, and body to the semantic query. Supported
entities include grapes, regions, vintages, classifications, producers, and appellations. Supported
intents are flavour, aging, pairing, classification, and region.

The BM25 analyzer in `src/retrieval/bm25_analyzer.py` applies the same Unicode, punctuation,
apostrophe, hyphen, spelling, and wine-terminology rules to indexed text and queries. It removes
low-value conversational words without stripping meaningful wine names.

### 2. Retrieve two candidate pools

Dense search requests 25 passages that are semantically close to the query. BM25 requests 25
passages with strong keyword matches. `src/retrieval/hybrid_retriever.py` alternates the ranked
lists, removes duplicate chunk IDs, and admits at most 50 unique candidates.

The system deliberately does not blend vector and BM25 scores with fixed weights because the score
scales have different meanings. Each result instead retains its dense rank, BM25 rank, original
scores, channel provenance, and timing. If the reranker is unavailable, unweighted reciprocal-rank
fusion is the explicit fallback.

### 3. Prefer explicit entity matches and rerank

Detected entity matches add `0.1` per matching metadata field before reranking. This is a preference,
not a hard filter, because extracted metadata can be incomplete.

The local `cross-encoder/ms-marco-MiniLM-L-6-v2` model then reads the normalized question together
with every candidate's contextual search text. Unlike initial retrieval, it directly judges each
question-passage pair. Negative logits are removed by the active `0.0` threshold, and the best five
passages remain.

### 4. Calculate confidence and build context

`src/retrieval/confidence.py` converts the strongest reranker score to a stable value between zero
and one. Values below `0.3` are marked low confidence. This is useful for identifying empty or weak
retrieval, but it cannot reliably identify a plausible passage that has become outdated.

`src/retrieval/context_builder.py` then removes semantically repeated passages, formats clean bodies
with their sources, and prepares model context. TF-IDF compression and small-to-big expansion are
available but disabled, keeping the production path simpler and preserving full evidence.

Automatic cached Tavily fallback is also disabled. When explicitly enabled, it runs only for low
confidence and appends web results after book evidence; provider failures preserve the book result.

### 5. Use the shared result

`src/retrieval/rag_service.py` owns the production stage order and returns the query plan, raw
candidates, final context chunks, confidence, sources, feature usage, timings, and errors.

- The `/api/chat` RAG-only mode enables answer generation.
- `src.eval` uses the same path for retrieval and full-pipeline evaluation.
- `src/agents/tools/rag_tools.py` disables generation because the LangGraph agent performs final
  synthesis after tool execution.

Retrieval stops at evidence assembly. For RAG-only chat, `src/agents/llm.py` loads the reviewed
system and user prompt templates from `src/agents/prompts/` and sends the question plus formatted
context to the configured generation model. For intelligent-agent chat, retrieval is a tool result;
the agent decides when to call it and combines that evidence with other tool outputs. This boundary
keeps search behavior independent from the model that writes the final answer.

## Code ownership map

| Path | Responsibility |
|---|---|
| `src/chroma/extraction/` | PDF/EPUB parsing and provider-neutral elements |
| `src/chroma/chunking/` | Structure-aware passage construction |
| `src/chroma/chunk_filter.py` | Structural and quality enforcement |
| `src/chroma/contextual_text.py` | Shared clean/contextual text construction |
| `src/chroma/loader.py` | Chroma ingestion, embedding, metadata, and deduplication |
| `src/chroma/bm25_builder.py` | Atomic BM25 construction and synchronization manifest |
| `src/retrieval/query_analyzer.py` | Entity/intent analysis and channel-specific queries |
| `src/retrieval/vector_retriever.py` | Dense Chroma search |
| `src/retrieval/keyword_search.py` | Persisted BM25 search |
| `src/retrieval/hybrid_retriever.py` | Balanced candidate union and fallback fusion |
| `src/retrieval/reranker.py` | Cross-encoder scoring and thresholding |
| `src/retrieval/confidence.py` | Confidence normalization and low-confidence flag |
| `src/retrieval/context_builder.py` | Deduplication, source display, and context formatting |
| `src/retrieval/factory.py` | Config-driven construction and BM25 validation |
| `src/retrieval/rag_service.py` | Shared production orchestration and result contract |
| `src/api/routes/chat.py` | RAG-only API caller and chat response mapping |
| `src/agents/tools/rag_tools.py` | Agent-facing knowledge-search tools |
| `src/agents/llm.py` and `src/agents/prompts/` | RAG-only prompt loading and final generation |
| `src/eval/` | Frozen datasets, metrics, reports, comparison, and curation tools |

## Active production defaults

The source of truth is `app_config.yml`; this table is a readable snapshot.

| Area | Active choice | Why it matters |
|---|---|---|
| PDF / EPUB extraction | `pdfplumber` / `ebooklib` | Local, format-aware extraction |
| Chunking | `section_recursive`, 1024 size, 256 overlap, 200 minimum | Preserves useful structure with bounded passages |
| Quality filter | Enforced, minimum score `0.4` | Prevents obvious structural noise from entering either index |
| Embedder | `sentence-transformers/all-mpnet-base-v2` via `EMBEDDING_MODEL` | Selected local model; indexing and querying must match |
| Dense / BM25 pools | 25 / 25, union limit 50 | Gives both channels room before precise reranking |
| Dense similarity threshold | `0.3` | Removes weak vector candidates before the union |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local pairwise relevance scoring |
| Final results / threshold | 5 / `0.0` | Keeps the strongest non-negative reranker results |
| Confidence boundary | `0.3` | Provisional weak-retrieval signal |
| Metadata boost | Enabled, `0.1` per matching field | Prefers explicit wine entities without hard filtering |
| Semantic deduplication | Enabled, `0.9` similarity boundary | Avoids spending context on near-duplicates |
| Compression / small-to-big | Disabled / disabled | Avoids complexity without proven production gain |
| Automatic web fallback | Disabled | Avoids routine external cost and latency |

Chroma runs on host port `8100` by default (container port `8000`). The BM25 index and manifest are
`chroma-data/bm25_index.pkl` and `chroma-data/bm25_index.meta.json`.

## Failure behavior and operations

| Situation | Behavior |
|---|---|
| Unsupported source file | Skip by default; configuration can make it an error. |
| One or more files fail during forced reindex | Fail the rebuild; do not publish a partial BM25 replacement. |
| BM25 index or manifest is missing, invalid, or stale | Log the reason and use vector-only retrieval. |
| Reranker is unavailable | Use unweighted reciprocal-rank fusion rather than fixed score blending. |
| No passage survives the threshold | Return empty/low-confidence book context; automatic web fallback remains off unless enabled. |
| Optional web provider fails | Keep the original book result. |
| Retrieval fails inside the shared service | Return an inspectable error and empty retrieval artifacts; the caller decides how to present failure. |

Forced reindex is intentionally strict: Chroma must complete first, then BM25 is built in temporary
files, verified against record count and sorted chunk IDs, and atomically published. This avoids a
half-updated search system.

## Evaluation: what the metrics mean

Retrieval evaluation normally runs without answer generation, so it measures search behavior
without spending model calls:

```bash
PYTHONPATH=. uv run python -m src.eval \
  --mode retrieval \
  --backend rag \
  --categories rag_only
```

- **MRR** asks how early the first known-relevant passage appears. A score near 1 means the system
  usually puts a useful answer near the top.
- **Precision@3 / Precision@5** ask what fraction of the first three or five passages are known to
  be relevant. Higher precision means less distracting context reaches the model.
- **Recall@10** asks how much of the known-relevant evidence appears anywhere in the first ten
  candidates. Higher recall means the system misses fewer useful passages.
- **Context precision / recall** use an answer-aware judge. Precision asks how much supplied context
  is useful; recall asks how much evidence needed for the answer was supplied.
- **Latency** is the time spent retrieving. A small quality gain can still be rejected if it adds a
  model call, external cost, or a large delay.

Precision and recall often pull in opposite directions. Returning more broadly related passages can
find evidence that a narrow search missed, improving recall, while also adding passages that are not
directly useful, reducing precision. The objective is not to maximize one number in isolation; it is
to give the answering model enough correct evidence without flooding it with noise.

For commands, datasets, support accounting, reports, and result schemas, see
[`src/eval/README.md`](../src/eval/README.md).

## Milestone 3 evidence and decisions

These figures come from different named checkpoints and must not be compared unless the cohort is
stated. Generated result files stay in the ignored local `eval-results/` directory.

### Accepted retrieval baseline

The 2026-08-12 Phase 0 production run evaluated 25 RAG-only questions. One question had no supported
golden chunk IDs, so ranking metrics scored 24 questions: MRR `0.8368`, precision@3 `0.6250`, and
precision@5 `0.5833`. A focused hybrid ablation reached recall@10 `0.9208` and exact-entity hit rate
`1.0`. In ordinary terms, useful evidence generally appeared early, hybrid search recovered most
known evidence in the wider candidate set, and explicit wine entities were consistently found.

### Accepted contextual-search trade-off

The fair Phase 2 comparison used the same 37,412-record corpus for every variant. Contextual search
improved MRR from `0.7431` to `0.8368`, precision@3 from `0.4583` to `0.5972`, precision@5 from
`0.4667` to `0.5583`, and context recall from `0.4524` to `0.5714`. Context precision fell from
`0.6625` to `0.5690` on the same seven answer-judged samples.

The contextual representation was retained because it made relevant material substantially easier
to find. Simple recovery variants that retrieved contextually but reranked only the body made
precision worse; reducing the result count also damaged recall. The lower answer-aware context
precision is therefore an accepted, documented trade-off and a low-priority improvement area, not a
claim that every quality gate passed.

### Rejected or disabled additions

- **Automatic web fallback** improved current-information answers in a focused five-question cohort,
  but increased mean latency from 3.13 to 5.86 seconds, can add an external call, and missed one
  plausible stale passage. The implementation remains opt-in and disabled by default.
- **HyDE query expansion** added one model call and roughly 7.2 seconds per question without a
  retrieval improvement. The experiment was rejected and its production code was discarded.
- **BGE embeddings** were tested on the same 37,412-record corpus with the rest of retrieval fixed.
  MRR fell from `0.8368` to `0.8278`, precision@3 from `0.6111` to `0.5694`, precision@5 from
  `0.5750` to `0.4667`, and mean latency rose from 1.56 to 2.90 seconds. The project retained
  `all-mpnet-base-v2`.

The conclusion is intentionally conservative: the retained local pipeline produced meaningful
quality gains, while additions with no gain or poor cost/latency trade-offs were not promoted.

## Known limitations

- OCR is not implemented; image-only or badly encoded pages may be missing or noisy.
- Ambiguous layouts are marked for audit instead of being aggressively guessed.
- Contextual search increases coverage but the answer-aware precision regression remains open.
- Retrieval confidence detects weak evidence, not factual freshness.
- Metadata extraction can be incomplete, which is why metadata matches boost rather than filter.
- The corpus still has some literal `(cid:...)` extraction artifacts; accepted retrieval quality did
  not justify adding cleanup complexity during Milestone 3.

## Rules for future retrieval changes

1. Change one meaningful variable at a time against a frozen dataset and corpus.
2. Report metric support counts and compare only common samples.
3. Evaluate quality together with latency, local resource use, external calls, and maintenance cost.
4. Prefer the simpler option when a small gain requires a new model call or complicated branching.
5. Keep `app_config.yml`, indexing, production retrieval, evaluation, and this guide aligned.
6. Use `execute_production_rag()` for production-equivalent behavior; direct component assembly is
   for explicit diagnostics or ablations.

## Related documentation

- [`src/chroma/README.md`](../src/chroma/README.md) — ingestion module boundaries and contracts
- [`src/retrieval/README.md`](../src/retrieval/README.md) — retrieval module boundaries and usage
- [`src/eval/README.md`](../src/eval/README.md) — evaluation handbook
- [`quick-reference.md`](quick-reference.md) — common project commands and configuration
- [`rag-pipeline.md`](rag-pipeline.md) — generic, project-independent RAG tutorial
