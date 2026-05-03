# Eval Harness — `src/eval/`

This package provides a repeatable, automated evaluation pipeline for the Pour Decisions
RAG pipeline and agentic AI layer. It measures retrieval quality, answer faithfulness, and
relevance on a curated golden dataset so that every code change can be objectively assessed.

---

## What does it mean to evaluate an AI agent?

Traditional software has deterministic outputs — the same input always produces the same
result, and correctness is provable by inspection. LLM-powered agents do not have this
property. An answer about Barolo aging might be factually correct, fluent, and well-cited
one day, and subtly wrong or hallucinated on the next model version, after a config change,
or after a retrieval pipeline update.

Evaluation for AI agents therefore means:

- **Measuring, not verifying.** We cannot know if an answer is correct from the code path
  alone. We need empirical measurement against known-good reference data.
- **Tracking over time.** A single score is a snapshot. The value is in the trajectory —
  did a code change make things better or worse?
- **Separating failure modes.** A bad answer could be a retrieval failure (wrong chunks),
  a reasoning failure (correct chunks, wrong synthesis), or a faithfulness failure
  (answer not grounded in retrieved context). Each requires a different fix.
- **Distinguishing true regressions from data drift.** If a cellar-dependent question
  stops scoring well because the wine was consumed, that is not a system regression — it
  is a stale dataset. The harness explicitly handles this.

Our evaluation harness is built around two philosophies:

1. **Local-first and cost-minimized.** Retrieval metrics (MRR, precision@k) are pure
   Python with zero API calls. Full LLM-as-judge scoring uses Gemini Flash and costs
   approximately $0.03 per full 60-question run.
2. **Longitudinal.** Every run is time-stamped and saved to disk. The `make eval-report`
   command compares the two most recent runs to surface regressions immediately.

---

## How we evaluate

### Two modes

| Mode | Flag | LLM calls | When to use |
|------|------|-----------|-------------|
| `retrieval` | `--mode retrieval` | 0 (free) | Every commit, as a fast sanity check |
| `full` | `--mode full` | ~420 for 60 samples (~$0.03) | Before/after meaningful pipeline changes |

Default is `retrieval` — safe to run without API cost at any time.

### Two backends

| Backend | What it tests | LLM calls per sample |
|---------|--------------|----------------------|
| `rag` | RAG pipeline in isolation: hybrid retrieval + single LLM generation | 1 |
| `agent` | Full intelligent agent with tool planning and multi-step reasoning | 2–3 |

Default is `rag` — faster and cheaper; use `agent` to validate tool-calling behavior.

### Execution pipeline

```
wine_qa_golden.jsonl
        |
        v
GoldenDataset.load() + filter()
        |
        v
EvalRunner.run()   (async, bounded concurrency)
   |                |
   v                v
_run_rag_sync    _run_agent_sync
 (retriever       (WineAgent.invoke)
  + invoke_llm)
        |
        v
per-sample: answer, contexts, chunk_ids, tool_calls, latency_ms, error
        |
        +---> RetrievalMetrics (mrr, precision@k)  [no LLM]
        |          (only for samples with ground_truth_chunk_ids)
        |
        +---> RagasScorer.score()  [4 LLM-as-judge metrics]
        |          (mode=full only; skips errors + empty contexts)
        |
        v
EvalReporter.build()  -> aggregate + per-category means
EvalReporter.save()   -> tests/eval/results/{timestamp}_{mode}_{backend}.json
EvalReporter.print_summary()
```

---

## The golden dataset

Location: `tests/eval/wine_qa_golden.jsonl`

The golden dataset is a version-controlled JSONL file of 60 hand-authored wine Q&A pairs.
It is the **ground truth for the pipeline** — every evaluated metric is relative to it.

### Category breakdown

| Category | Count | Description |
|----------|-------|-------------|
| `rag_only` | 25 | Wine knowledge questions answerable from indexed wine books |
| `cellar` | 15 | Queries against the user's live cellar database via tool calls |
| `pairing` | 10 | Food and wine pairing questions |
| `multi_hop` | 10 | Questions requiring synthesis from both RAG and cellar sources |

### Difficulty breakdown

| Level | Count | Characteristics |
|-------|-------|-----------------|
| `easy` | 20 | Single-source, factual, unambiguous |
| `medium` | 25 | Require terminology awareness or moderate context |
| `hard` | 15 | Multi-hop, classification edge cases, ambiguous terminology |

### Sample schema

Each line in the JSONL file is one JSON object:

```json
{
  "id": "rag_only_001",
  "question": "What is the minimum aging requirement for Barolo DOCG?",
  "category": "rag_only",
  "difficulty": "easy",
  "expected_facts": [
    "38 months minimum from harvest",
    "at least 18 months must be in oak"
  ],
  "expected_tool_calls": [],
  "ground_truth": "Barolo DOCG requires a minimum of 38 months aging from harvest, with at least 18 months in oak, extended to 62 months for Riserva.",
  "ground_truth_chunk_ids": ["abc123", "def456"],
  "tags": ["barolo", "aging", "docg"],
  "notes": null
}
```

**Key fields:**

- `ground_truth` — A complete, factual sentence (not a vague description). This is the
  reference answer for Ragas `context_recall` scoring and LLM-as-judge prompts.
- `ground_truth_chunk_ids` — ChromaDB chunk IDs that are known to contain the answer.
  Used for MRR and precision@k. Can be empty; if so, retrieval metrics are skipped for
  that sample without failing the run.
- `expected_tool_calls` — Tools expected to fire for agent-backend runs. Agent evaluation
  quality is partly assessed by whether the right tools were used.
- `notes` — Special skip conditions. Samples with `"skip if DB is empty"` in `notes` are
  automatically skipped when the cellar database has no inventory.

### Cellar questions and ground truth authoring

Cellar questions are grounded in the live SQLite database, so their correct answer changes
over time as bottles are consumed or added. Ground truths for cellar samples therefore use
**structural assertions** rather than exact current values:

- **Do not write:** "The cellar contains 4 bottles of Barolo Vigna Rionda."
- **Write instead:** "The answer must state whether a Barolo wine is present and provide
  the producer name and earliest drinking year."

This ensures Ragas `context_recall` can evaluate whether the right *shape* of answer was
produced without hardcoding values that will become stale.

---

## Metrics

### Retrieval metrics (local, zero API cost)

These are computed purely from `ground_truth_chunk_ids` in the golden dataset. They require
no model or API calls and execute in milliseconds. They are skipped for any sample where
`ground_truth_chunk_ids` is empty.

**Reciprocal Rank (RR)**

For a single query: `1 / rank` of the first relevant chunk in the retrieved list, where
rank is 1-based. If no relevant chunk is retrieved, RR = 0.

```
retrieved = [C, A, B, D]   relevant = [A, B]
RR = 1/2 = 0.50   (A appears at position 2)
```

**Mean Reciprocal Rank (MRR)**

The average RR across all samples that have `ground_truth_chunk_ids`. The primary summary
metric for retrieval quality.

- MRR = 1.0 — the first retrieved chunk is always relevant
- MRR > 0.7 — strong retrieval; relevant content is near the top
- MRR 0.4–0.7 — moderate; relevant content is present but buried
- MRR < 0.4 — weak retrieval; relevant content is often missing or deeply ranked

**Precision@k**

Fraction of the top-k retrieved chunks that are relevant. The denominator is always k
(fixed-cutoff), so returning fewer than k results is penalized.

```
retrieved = [A, C, B, D, E]   relevant = [A, B]   k=3
precision@3 = 2/3 = 0.67
```

We compute `precision_at_3` and `precision_at_5` by default (configurable via
`eval.retrieval_metrics.k_values` in `app_config.yml`).

- precision@3 > 0.67 — at least 2 of the 3 most visible chunks are relevant
- precision@5 > 0.60 — relevant content dominates the visible context window
- Values below 0.33 suggest the retriever is polluting context with irrelevant noise

---

### Ragas metrics (LLM-as-judge, full mode only)

These metrics use Gemini Flash as an evaluator LLM and are only computed in `--mode full`.
They operate on the triad of `(question, retrieved_contexts, answer)` and optionally
`ground_truth`.

**Faithfulness**

Measures whether every factual claim in the answer is supported by the retrieved contexts.
Ragas decomposes the answer into atomic claims and uses NLI classification to check each
one against the context.

- Score near 1.0 — answer contains no hallucinations; all facts are backed by evidence
- Score below 0.7 — the model is adding knowledge not present in retrieved context
- Low faithfulness is a hallucination signal, not a relevance signal

**Answer Relevancy**

Measures whether the answer actually addresses the question. Ragas generates synthetic
questions from the answer and measures semantic similarity to the original question.

- Score near 1.0 — the answer is tightly focused on the question asked
- Low score — the answer is drifting off-topic, over-explaining, or incomplete
- Does not assess factual correctness; a wrong but on-topic answer can score high

**Context Precision**

Measures the signal-to-noise ratio in the retrieved context: are the retrieved chunks
actually relevant to the question? Ragas scores each chunk individually using LLM judgment.

- Score near 1.0 — retrieved context is clean; every chunk contributes to the answer
- Low score — retriever is pulling in off-topic material that may confuse generation
- Context precision and faithfulness often move together: noisy context → hallucination

**Context Recall**

Measures whether the retrieved context contains all the information needed to produce the
ground truth answer. Requires `ground_truth` to be set on the sample.

- Score near 1.0 — retrieval is comprehensive; no required information was missed
- Low score — relevant knowledge exists in the index but was not retrieved
- The metric most sensitive to retrieval strategy changes (chunk size, hybrid weights)

**Reading scores together:**

| Pattern | Diagnosis |
|---------|-----------|
| Low faithfulness, high context precision | Model is hallucinating despite good retrieval |
| High faithfulness, low context recall | Good at using what it finds, but finding too little |
| Low context precision + low faithfulness | Retriever is noisy; fix retrieval before fixing generation |
| All metrics high in `rag_only`, low in `multi_hop` | Multi-hop synthesis is the bottleneck |

---

## Practical handbook

### Prerequisites

1. `GOOGLE_API_KEY` set in `.env` (required for `--mode full` only)
2. ChromaDB running (`make chroma-up`) (required for RAG queries)
3. `cellar-data/wine_cellar.db` present with populated inventory (for cellar samples)

### Day-to-day commands

```bash
# Free, fast retrieval-only check (no API key needed)
make eval

# Full LLM scoring — use before/after pipeline changes
make eval-full

# Compare latest two result files
make eval-report

# Validate the dataset is not stale against the current cellar
make eval-validate
```

### Running subsets

Use the CLI directly for targeted runs:

```bash
# Only wine knowledge questions, easy difficulty
python -m src.eval --mode retrieval --categories rag_only --difficulties easy

# Pairing and multi-hop questions with full Ragas scoring
python -m src.eval --mode full --categories pairing,multi_hop

# Agent backend instead of RAG pipeline
python -m src.eval --mode retrieval --backend agent

# Custom dataset and output location
python -m src.eval --dataset path/to/custom.jsonl --output-dir path/to/results
```

Full CLI reference:

```
usage: python -m src.eval
  [--mode {retrieval,full}]
  [--backend {rag,agent}]
  [--categories CATEGORIES]     e.g. "rag_only,pairing"
  [--difficulties DIFFICULTIES] e.g. "easy,medium"
  [--tags TAGS]                 e.g. "barolo,aging"
  [--dataset PATH]              default: tests/eval/wine_qa_golden.jsonl
  [--output-dir PATH]           default: tests/eval/results/
  [--max-concurrency N]         default: 3
```

### Comparing results

```bash
# Compare latest 2 runs
python -m src.eval.compare_results

# Compare latest 3 runs
python -m src.eval.compare_results --latest 3

# Custom results directory
python -m src.eval.compare_results --results-dir path/to/results
```

The comparison tool prints a delta table with green/red coloring in terminal:

```
Metric                Previous   Latest     Delta
------------------------------------------------
answer_relevancy        0.8912     0.9134   +0.0222
context_precision       0.7401     0.7109   -0.0292
faithfulness            0.8200     0.8450   +0.0250
mrr                     0.6100     0.6350   +0.0250
```

### Validating dataset freshness

Cellar questions reference specific wine types that must still be present in the database.
Run this validation before trusting eval results, especially after syncing cellar data:

```bash
make eval-validate

# Or directly:
python -m src.eval.dataset_validator

# Machine-readable output for CI integration:
python -m src.eval.dataset_validator --json
```

Exit code 0 = all cellar questions still valid. Exit code 1 = stale questions detected.

If stale samples are reported:

1. Open `tests/eval/wine_qa_golden.jsonl`
2. Locate the sample IDs listed in the report
3. Update the question to reference a wine that is currently in the cellar, or reclassify
   the question to `rag_only` if it can be answered from books alone
4. Re-run `make eval-validate` to confirm the fix

### Updating `ground_truth_chunk_ids`

Chunk IDs are content-hash-based ChromaDB identifiers. They become stale whenever the
index is rebuilt (e.g., after a chunking strategy change). After any full reindex:

```bash
# Look up candidates for one question (ChromaDB must be running):
python -m src.eval.chunk_id_lookup \
    --question "What is the minimum aging for Barolo DOCG?" \
    --top-k 10

# JSON output for scripting:
python -m src.eval.chunk_id_lookup \
    --question "What are the primary grape varieties in Châteauneuf-du-Pape?" \
    --json
```

The tool returns ranked candidates with chunk IDs, similarity scores, source files, and
text previews. Inspect the top results and copy relevant IDs into `ground_truth_chunk_ids`
in the JSONL for that sample.

If `ground_truth_chunk_ids` is empty for a sample (the default for cellar/pairing
questions), MRR and precision@k are simply not computed for that sample — the run does
not fail.

---

## Result files

Each run writes one JSON file to `tests/eval/results/` (gitignored):

```
{YYYYMMDDTHHMMSS}_{mode}_{backend}.json
e.g.  20260503T143022_retrieval_rag.json
```

Top-level fields:

| Field | Description |
|-------|-------------|
| `run_id` | ISO timestamp string used as a unique ID |
| `timestamp` | Full ISO 8601 UTC timestamp |
| `mode` | `retrieval` or `full` |
| `backend` | `rag` or `agent` |
| `git_sha` | Short commit hash for reproducibility |
| `config_snapshot` | Model name, embedder, n_results, feature flags |
| `aggregate_metrics` | Mean score per metric across all evaluated samples |
| `metrics_by_category` | Per-category metric breakdown |
| `per_sample` | Full per-sample results including answer, contexts, latency |
| `summary` | `evaluated`, `skipped`, `errors`, `total_llm_calls`, `total_latency_ms` |

The `summary.skipped` count captures samples that were intentionally skipped (e.g., cellar
samples when the DB is empty). `summary.errors` captures unexpected failures. Neither
abort the run — eval runs to completion even when individual samples fail.

---

## Configuration

All defaults live in the `eval:` section of `app_config.yml`:

```yaml
eval:
  dataset_path: tests/eval/wine_qa_golden.jsonl
  results_dir: tests/eval/results
  default_mode: retrieval
  default_backend: rag
  max_concurrency: 3
  ragas:
    evaluator_model: gemini-2.5-flash
    metrics:
      - faithfulness
      - answer_relevancy
      - context_precision
      - context_recall
  retrieval_metrics:
    k_values: [3, 5]
  skip_cellar_samples_if_empty: true
```

The `max_concurrency` setting controls how many samples run in parallel during eval.
Increasing it speeds up a run but also increases the rate of LLM API calls — stay within
your API quota when running `--mode full`.

---

## Cost reference

| Run type | LLM calls | Approx. tokens | Approx. cost |
|----------|-----------|----------------|--------------|
| `make eval` (retrieval, 60 samples) | 60 (generation) | ~60k | < $0.005 |
| `make eval-full` (full Ragas, 60 samples) | ~420 | ~360k | ~$0.03 |
| Monthly (1 full run/week) | ~1680 | ~1.4M | ~$0.11/month |

All estimates assume Gemini Flash April 2026 pricing. The retrieval-only mode is free in
the sense that it costs only the same LLM calls as a normal chat turn.

---

## Module reference

| File | Responsibility |
|------|---------------|
| `models.py` | Pydantic models: `GoldenSample`, `SampleResult`, `EvalRunResult` |
| `dataset.py` | `GoldenDataset`: load and filter the JSONL golden file |
| `dataset_validator.py` | Detect stale cellar-dependent samples against the live DB |
| `runner.py` | `EvalRunner`: async execution against RAG or agent backend |
| `metrics.py` | Pure local functions: `reciprocal_rank`, `precision_at_k`, means |
| `ragas_scorer.py` | `RagasScorer`: wrap Ragas `evaluate()` for full-mode scoring |
| `reporter.py` | `EvalReporter`: aggregate results, save JSON, print summary |
| `compare_results.py` | CLI: compare latest N result files with delta table |
| `chunk_id_lookup.py` | Dev utility: find ChromaDB chunk IDs for dataset authoring |
| `__main__.py` | CLI entry point: orchestrates the full eval pipeline |
| `__init__.py` | Package exports |

---

## Adding new eval samples

1. Open `tests/eval/wine_qa_golden.jsonl`.
2. Add a new line following the schema above. IDs must be unique and follow
   `{category}_{NNN}` format.
3. For `rag_only` samples: run `chunk_id_lookup.py` to populate `ground_truth_chunk_ids`.
4. For `cellar` / `multi_hop` samples: write `ground_truth` as a structural assertion, not
   a value. Add a `notes` field with `"skip if DB is empty"` if the question depends on
   specific inventory.
5. Run `make eval-validate` to confirm no new stale entries.
6. Run `pytest tests/eval/test_dataset.py -v` to verify schema validity and distribution.

---

## Tests

The test suite for this module lives in `tests/eval/`. Run the full eval test suite:

```bash
python -m pytest tests/eval/ -v -m "not eval"
```

Ragas scorer tests require a live `GOOGLE_API_KEY` and are gated by `@pytest.mark.eval`:

```bash
python -m pytest tests/eval/test_ragas_scorer.py -m eval -v
```

