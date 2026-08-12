# Milestone 3 Phase 0 Corrective Manual Testing

> **Project version**: 0.7.3 — last verified 2026-08-12.
> **Status**: Phase 0 closed against the 2026-08-12 index.

This guide reproduces the six closure checks for the layout-safe extraction, block-aware chunking,
quality filtering, contextual indexing, synchronized BM25, and balanced hybrid retrieval correction.
It distinguishes commands that only inspect the current index from the destructive full-reindex
command.

## Accepted checkpoint

| Item | Accepted value |
|---|---:|
| Chroma/BM25 records | 37,374 |
| source books | 22 |
| empty chunks | 0 |
| sorted chunk-ID SHA-256 | `464d855a25cf834c77215ade8c9e28be6c793141fb51b35459aa8773e630a278` |
| golden retrieval samples | 25 total / 24 scorable / 1 unsupported |
| curated IDs | 114 unique / 0 missing |
| MRR | 0.8368 |
| Precision@3 | 0.6250 |
| Precision@5 | 0.5833 |
| mean production retrieval latency | 1,351 ms |

Local generated evidence:

- `eval-results/m3_phase0_corpus_20260812_final.json`
- `eval-results/20260812T133822_retrieval_rag.json`
- `eval-results/m3_phase0_extraction_chunking_20260812.json`

`eval-results/` is intentionally ignored by Git. Preserve or copy these files before cleaning a
worktree if they are needed for an audit.

## Prerequisites

```bash
cd /path/to/pour-decisions
make chroma-up
```

The environment must provide `EMBEDDING_MODEL` and `WINE_BOOKS_PATH`. The active Chroma collection
must have been indexed with the same embedding model configured for retrieval.

## Step 1 — Verify implementation and focused tests

Run the areas that directly enforce the correction:

```bash
PYTHONPATH=. uv run pytest -q \
  tests/chroma \
  tests/retrieval \
  tests/eval/test_rag_parity.py \
  tests/eval/test_chunk_id_scripts.py \
  tests/eval/test_utils.py
```

Then run the complete Python suite:

```bash
PYTHONPATH=. uv run pytest -q --tb=short
```

Accepted result: `867 passed, 11 skipped, 0 failed`. A later suite may contain more tests; zero
failures is authoritative, while the historical count is only a checkpoint.

## Step 2 — Verify the synchronized corpus

This is read-only against Chroma:

```bash
make chroma-stats-exact \
  CORPUS_STATS_OUTPUT=eval-results/m3_phase0_corpus_$(date +%Y%m%d)_verify.json
```

Confirm the report shows:

- collection `wine_books`;
- 37,374 records and 22 sources for the accepted snapshot;
- zero empty documents and zero records missing source;
- sorted chunk-ID hash equal to the value at the top of this guide.

The factory validates the live BM25 sidecar whenever it constructs hybrid retrieval. A successful
hybrid lookup in Step 4 confirms that the BM25 record count and ID hash still match Chroma. A stale
or missing manifest must log a warning and use vector-only retrieval explicitly.

### Full rebuild, only when intentionally requested

`make chroma-reindex` resets the configured collection. Do not use it for ordinary verification.
When a new source/configuration requires a rebuild, run:

```bash
make chroma-reindex
```

Success requires every configured source to index and the atomic BM25 rebuild/manifest validation
to complete. Any source error exits non-zero and prevents Phase 0 acceptance of that rebuild.

## Step 3 — Validate the golden dataset and curated IDs

```bash
make eval-validate

PYTHONPATH=. uv run python -m src.eval.scripts.chunk_id_curator \
  --dataset src/eval/wine_qa_golden.jsonl \
  --top-k 10 \
  --mode hybrid
```

Without `--redo`, the curator skips already curated samples and reports completion. Use `--redo`
only after chunk IDs change. Review complete text manually before accepting IDs; the script writes
the JSONL atomically, but human relevance judgment remains authoritative.

The accepted dataset contains 114 unique curated chunk IDs and no missing IDs. `rag_only_021` is
intentionally marked unsupported because the current corpus has no clean chunk supporting its full
Dealu Mare ground truth; unsupported is not scored as zero.

## Step 4 — Re-run the focused Nebbiolo diagnostic

```bash
PYTHONPATH=. uv run python -m src.eval.scripts.chunk_id_lookup \
  --question "What are the primary flavour characteristics of Nebbiolo?" \
  --top-k 10 \
  --mode hybrid \
  --full-text
```

Review the query plan and the first ten raw candidates. Acceptance requires:

- a direct flavour answer in the top three (accepted result: rank 1);
- at least 8/10 candidates relevant to Nebbiolo or its wines (accepted result: 9/10);
- no contents, index, bibliography, worksheet, interleaved-column, or obvious OCR-artifact chunk;
- non-empty final context after thresholded reranking;
- the semantic query includes `nebbiolo aroma flavor taste sensory profile tannin acidity body`.

Use the single channels to diagnose a regression without changing production configuration:

```bash
PYTHONPATH=. uv run python -m src.eval.scripts.chunk_id_lookup \
  --question "What are the primary flavour characteristics of Nebbiolo?" \
  --top-k 10 --mode vector --full-text

PYTHONPATH=. uv run python -m src.eval.scripts.chunk_id_lookup \
  --question "What are the primary flavour characteristics of Nebbiolo?" \
  --top-k 10 --mode bm25 --full-text
```

## Step 5 — Run the production retrieval checkpoint

```bash
PYTHONPATH=. uv run python -m src.eval \
  --mode retrieval \
  --backend rag \
  --categories rag_only
```

This makes no generation or judge LLM calls. Verify coverage as well as aggregates: 24 samples must
be scored, `rag_only_021` must be unsupported, and errors/timeouts must be zero.

Accepted metrics are recorded at the top of this guide. Compared over the 24 common scorable
samples with Gate 0, Phase 0 changed MRR from `0.8472` to `0.8368` (-1.04 percentage points),
Precision@3 from `0.5556` to `0.6250` (+6.94 points), and Precision@5 from `0.4333` to `0.5833`
(+15.00 points). Mean latency increased from about `884 ms` to `1,359 ms` on the common cohort.

The accepted three-channel top-10 ablation is stored in the Phase 0 evidence artifact:

| Mode | Recall@10 | MRR | P@3 | P@5 | Entity hit | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| vector | 0.5854 | 0.5825 | 0.4306 | 0.3583 | 0.8125 | 198 ms |
| BM25 | 0.3781 | 0.5667 | 0.3194 | 0.2333 | 0.9375 | 105 ms |
| hybrid | 0.9208 | 0.8368 | 0.6250 | 0.5833 | 1.0000 | 1,304 ms |

The public eval CLI currently runs the production mode only; vector/BM25 corpus-wide ablation was a
closure analysis recorded in the evidence artifact. The lookup utility provides stable per-query
single-channel diagnostics. If repeated corpus-wide ablations become a regular gate, add a reviewed
reusable eval command rather than relying on an ad-hoc shell script.

## Step 6 — Close or reject the checkpoint

Phase 0 can be closed only when all of the following are true:

- full Chroma/BM25 rebuild, when performed, completed with no source failures;
- exact corpus and BM25 synchronization checks passed;
- the known malformed/interleaved evidence is absent and no malformed replacement was introduced;
- all supported golden chunk IDs exist;
- the Nebbiolo focused gate passed;
- production retrieval has complete supported coverage and no execution error;
- the final evidence artifact records configuration, corpus/dataset hashes, coverage, metrics,
  ablation, focused diagnostic, test results, and known observations.

The 2026-08-12 checkpoint passed all six conditions. Two books excluded because of extraction
failures remain outside the accepted 22-source corpus. Also track the 576 retained chunks containing
literal `(cid:...)` tokens and the 5,289 chunks below 200 characters; neither appeared in the focused
failure or violated the accepted Phase 0 gate, but both are valid Phase 1 calibration inputs.
