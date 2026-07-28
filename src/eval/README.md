# Eval Harness

- **Doc version**: 0.8.1
- **Last update**: 2026-07-28

---

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

1. **Cost-minimized.** Retrieval metrics (MRR, precision@k) are pure Python with zero
   API calls. Full LLM-as-judge scoring uses the separately configured Ollama evaluator,
   which may be local or cloud-hosted.
2. **Longitudinal.** Every run is time-stamped and saved to disk. The `make eval-report`
   command compares the two most recent runs to surface regressions immediately.

---

## How we evaluate

### Two modes

| Mode | Flag | LLM-as-judge calls | When to use |
|------|------|--------------------|-------------|
| `retrieval` | `--mode retrieval` | 0 (free) | Every commit, as a fast sanity check |
| `full` | `--mode full` | up to ~780 estimated (model-dependent) | Before/after meaningful pipeline changes |

Default is `retrieval` — safe to run without API cost at any time.

### Three backends

| Backend | What it tests | LLM calls per sample |
|---------|--------------|----------------------|
| `rag` | Shared production RAG pipeline, with generation only in `--mode full` | 0 in `retrieval`, 1 in `full` |
| `retriever` | Raw vector/hybrid retriever benchmark; retrieval mode only | 0 |
| `agent` | Full intelligent agent with tool planning and multi-step reasoning | 2–3 |

Default is `rag` — faster and cheaper; use `agent` to validate tool-calling behavior.

### Execution pipeline

```
wine_qa_golden.jsonl
        |
        v
load_golden_dataset() + filter_golden_samples()
        |
        v
EvalRunner.run()   (async, bounded concurrency)
   |                    |                    |
   v                    v                    v
shared production RAG   raw retriever        intelligent agent
(generation optional)   benchmark            (WineAgent.invoke)
        |
        v
per-sample: answer, raw/final chunks, exact context, sources, feature flags, latency, status
        |
        +---> pure metric functions (mrr, precision@k)  [no LLM]
        |          (only for samples with ground_truth_chunk_ids)
        |
        +---> RagasScorer.score()  [4 LLM-as-judge metrics]
        |          (mode=full only; skips errors + empty contexts)
        |
        v
EvalReporter.build()  -> grouped aggregates + per-category means + metric coverage
EvalReporter.save()   -> eval-results/{timestamp}_{mode}_{backend}.json
EvalReporter.print_summary()
```

---

## The golden dataset

Location: `src/eval/wine_qa_golden.jsonl`

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
| `easy` | 18 | Single-source, factual, unambiguous |
| `medium` | 27 | Require terminology awareness or moderate context |
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

These metrics use the configured Ollama evaluator LLM and are only
computed in `--mode full`. They operate on the triad of
`(question, retrieved_contexts, answer)` and optionally `ground_truth`.

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

### Agent metrics

Agent runs preserve each tool call and classify its output as `rag_context`,
`cellar_result`, `pairing_result`, `web_result`, `taste_profile_result`, or
`other_result`. Deterministic trajectory metrics compare the observed call sequence with
`expected_tool_calls`:

- `tool_recall` — fraction of required calls that occurred.
- `tool_precision` — fraction of observed calls that were expected.
- `tool_exact_match` — whether the complete observed and expected sequences match.
- `tool_ordered_match` — whether required calls occurred in the expected order, allowing
  additional calls between them.

In full agent mode, `answer_correctness` judges the final answer against `ground_truth` and
`expected_facts`. Ragas context metrics only receive outputs classified as `rag_context`;
cellar, pairing, web, and other tool results are not misrepresented as retrieved book evidence.

---

## Practical handbook

### Prerequisites

1. Local Ollama running (`make ollama-up`) for `--mode full` scoring
2. ChromaDB running (`make chroma-up`) (required for RAG queries)
3. `cellar-data/wine_cellar.db` present with populated inventory (for cellar samples)

### Day-to-day commands

```bash
# Free, fast retrieval check
make eval

# Full LLM scoring — use before/after pipeline changes
make eval-full

# Compare latest two result files
make eval-report

# Validate the dataset is not stale against the current cellar
make eval-validate
```

The Make targets execute commands through `uv run`, so they work without manually activating
`.venv`. Retrieval-only, reporting, validation, and curation targets use the base environment
and do not start Ollama. Full and Phoenix targets request the `eval` extra; full-mode targets
also start the local Ollama relay used to reach local or cloud-hosted Ollama models.

### Main CLI: `uv run python -m src.eval`

This is the primary entrypoint for the eval harness. It always performs the same
high-level steps:

1. Load config from `app_config.yml`.
2. Parse CLI filters and dataset/output settings.
3. Load the selected golden dataset.
4. Validate requested categories, difficulties, and tags.
5. Run environment preflight checks for the selected mode/backend.
6. Filter the dataset to the requested sample subset.
7. Execute the selected backend with `EvalRunner`.
8. Attach local retrieval metrics where `ground_truth_chunk_ids` exist.
9. Optionally run Ragas scoring when `--mode full`, using independent judge
   timeout, retry, reasoning, and output-budget controls.
10. Save a result JSON file and print a terminal summary.
11. Optionally push the finished report to Phoenix.

#### CLI reference

```text
usage: uv run python -m src.eval
  [--mode {retrieval,full}]
  [--backend {rag,retriever,agent}]
  [--categories CATEGORIES]     e.g. "rag_only,pairing"
  [--difficulties DIFFICULTIES] e.g. "easy,medium"
  [--tags TAGS]                 e.g. "barolo,aging"
  [--sample-id SAMPLE_ID]       e.g. "multi_hop_001"
  [--dataset PATH]              default: src/eval/wine_qa_golden.jsonl
  [--output-dir PATH]           default: eval-results/
  [--max-concurrency N]         default: 1
  [--push-to-phoenix]
  [--phoenix-url URL]
```

#### Flags and semantics

- `--mode retrieval`
  Scope: execute the selected backend without Ragas judge scoring. For `--backend rag`,
  this also computes local retrieval metrics such as `mrr` and `precision_at_k` where
  chunk IDs are available. For `--backend agent`, it computes deterministic tool trajectory
  metrics without LLM-as-judge scoring.
  Does not do: Ragas or any LLM-as-judge scoring. For `--backend rag`, all enabled
  production retrieval and context-building stages run, but final answer generation is disabled.

- `--mode full`
  Scope: run the same backend execution as retrieval mode, then add Ragas
  LLM-as-judge scoring on top of the captured outputs.
  Does not do: change the backend behavior itself; it only changes the post-processing step.

- `--backend rag`
  Scope: evaluate the shared production RAG pipeline used by API `rag_only`. In
  `--mode retrieval`, it runs through final context construction without generation.
  In `--mode full`, it also runs a single answer-generation call.
  Use this when you want to assess retrieval quality and grounded answer generation
  without agent planning noise.

- `--backend retriever`
  Scope: evaluate the raw configured retriever before production boosting, reranking,
  deduplication, compression, or generation. It only supports `--mode retrieval`.
  Use this for low-level retriever diagnostics, not M3 production quality gates.

- `--backend agent`
  Scope: evaluate the full intelligent agent invocation path, including tool selection
  and multi-step reasoning.
  Use this when you want to know how the production agent behaves end-to-end, not just
  whether the retriever is healthy.

- `--categories`, `--difficulties`, `--tags`
  Scope: subset the dataset before execution.
  Behavior: invalid categories and difficulties always fail fast. Invalid tags fail fast
  unless `eval.validate_tag_filters=false` in config.

- `--sample-id`
  Scope: subset the dataset to one or more exact sample ids.
  Behavior: accepts a comma-separated list and fails fast if any requested id is not present
  in the selected dataset file.

- `--dataset`
  Scope: replace the default golden dataset with another JSONL file using the same schema.

- `--output-dir`
  Scope: write the result JSON somewhere other than `eval-results/`.

- `--max-concurrency`
  Scope: bound the number of in-flight sample executions.
  Default is `1` to keep agent evaluation conservative until thread-safety is proven.

- `--push-to-phoenix`
  Scope: after the local result file is built, send the finished report to Phoenix.
  Behavior: this does not change scoring; it only adds reporting.

- `--phoenix-url`
  Scope: override the Phoenix base URL used by `--push-to-phoenix`.

#### The five execution paths

There are five supported `mode` + `backend` combinations. Full mode is intentionally
rejected for the low-level `retriever` backend.

**Path 1: Retrieval mode + RAG backend**

```bash
uv run python -m src.eval --mode retrieval --backend rag
```

Scope:
- Fastest and cheapest main CLI path.
- Executes the shared production RAG path through final context construction, without answer generation.
- Computes local retrieval metrics for samples with `ground_truth_chunk_ids`.
- Produces raw/final chunk artifacts, exact context text, source metadata, feature flags,
  retrieved chunk IDs, status, and latency.
- Leaves `answer` empty by design because this path is measuring retrieval rather than generation.

Use it when:
- You changed retrieval logic, chunking, ranking, filtering, or prompt wiring.
- You want a daily or per-commit regression check.

Does not cover:
- Agent planning quality.
- Generation quality.
- Ragas faithfulness/relevancy scoring.

**Path 2: Full mode + RAG backend**

```bash
uv run --extra eval python -m src.eval --mode full --backend rag
```

Scope:
- Runs the same RAG retrieval path as Path 1, then adds a single answer-generation call.
- Adds Ragas scoring after the run completes.
- Produces both retrieval metrics and LLM-as-judge metrics.

Use it when:
- You need to know whether retrieved context actually supports the generated answer.
- You are validating a retrieval or generation change more deeply than a smoke test.

Does not cover:
- Agent tool selection behavior.
- Multi-step planner execution quality.

**Path 3: Retrieval mode + retriever backend**

```bash
uv run python -m src.eval --mode retrieval --backend retriever
```

Scope:
- Runs only the configured vector/hybrid retriever.
- Bypasses production boosting, reranking, context construction, compression, and generation.
- Writes `summary.evaluation_target=retriever_benchmark`.

Use it when:
- You need to isolate raw retriever behavior from production post-processing.
- You are diagnosing vector/BM25 changes.

Does not cover:
- Production RAG parity or answer quality.
- Full mode or Ragas scoring.

**Path 4: Retrieval mode + agent backend**

```bash
uv run python -m src.eval --mode retrieval --backend agent
```

Scope:
- Executes the full intelligent agent for each sample.
- Computes required-tool recall, precision, exact match, and ordered match where
  `expected_tool_calls` are defined.
- Captures final answers, typed tool outputs, RAG-only evidence, tool calls, status, and latency.
- Still skips Ragas scoring because mode is `retrieval`.

Use it when:
- You want a cheaper end-to-end agent check.
- You are debugging tool usage, routing, or latency without paying the extra evaluation cost.

Does not cover:
- Faithfulness/relevancy scoring from Ragas.
- Retrieval metrics such as `mrr` or `precision_at_k`.
- Final-answer correctness judging.

Debugging note:
- For focused debugging, run one exact sample with `--sample-id`.
- Example:

```bash
uv run python -m src.eval --mode retrieval --backend agent --sample-id multi_hop_001
```

**Path 5: Full mode + agent backend**

```bash
uv run --extra eval python -m src.eval --mode full --backend agent
```

Scope:
- This is the broadest and most expensive main CLI path.
- Executes the full intelligent agent.
- Runs Ragas context metrics only on RAG-classified evidence.
- Adds `answer_correctness` against `ground_truth` and `expected_facts`.
- Gives you the closest thing to an end-to-end quality readout from the current harness.

Use it when:
- You changed agent architecture, tool prompts, routing, or orchestration.
- You want the most complete available eval signal before a major change is accepted.

Does not cover:
- A full trace-based audit of every intermediate agent decision.
- Hard pass/fail CI gates; those are intentionally deferred until stable baselines exist.

#### Common targeted invocations

```bash
# Only wine knowledge questions, easy difficulty
uv run python -m src.eval --mode retrieval --backend rag --categories rag_only --difficulties easy

# Pairing and multi-hop questions with full Ragas scoring
uv run --extra eval python -m src.eval --mode full --backend rag --categories pairing,multi_hop

# End-to-end agent run without judge scoring
uv run python -m src.eval --mode retrieval --backend agent

# Full agent run with judge scoring
uv run --extra eval python -m src.eval --mode full --backend agent

# Restrict to tagged questions
uv run python -m src.eval --mode retrieval --tags barolo,aging

# Custom dataset and output location
uv run python -m src.eval --dataset path/to/custom.jsonl --output-dir path/to/results

# Push a completed run to Phoenix
uv run --extra eval python -m src.eval --mode retrieval --push-to-phoenix --phoenix-url http://localhost:6006
```

#### Failure and exit behavior

- Invalid categories, difficulties, or strict tag filters fail before execution starts.
- Preflight failures fail before sample execution starts.
- If filters select zero samples, the CLI exits cleanly with no run file.
- Individual sample failures do not abort the run; they are recorded in `per_sample`.
- Sample timeouts are recorded as timeout results when `eval.sample_timeout_seconds` is set.
- Judge metric timeouts use `eval.ragas.timeout_seconds` and are recorded in
  `metric_errors` as `ragas_timeout`; they do not change the overall sample status.
- `--push-to-phoenix` happens after the local result file is written, so a Phoenix push
  failure does not erase the local run artifact.

### Comparing results

```bash
# Compare latest 2 runs
uv run python -m src.eval.scripts.compare_results

# Compare latest 3 runs
uv run python -m src.eval.scripts.compare_results --latest 3

# Custom results directory
uv run python -m src.eval.scripts.compare_results --results-dir path/to/results
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
uv run python -m src.eval.scripts.dataset_validator

# Machine-readable output for CI integration:
uv run python -m src.eval.scripts.dataset_validator --json
```

Exit code 0 = all cellar questions still valid. Exit code 1 = stale questions detected.

If stale samples are reported:

1. Open `src/eval/wine_qa_golden.jsonl`
2. Locate the sample IDs listed in the report
3. Update the question to reference a wine that is currently in the cellar, or reclassify
   the question to `rag_only` if it can be answered from books alone
4. Re-run `make eval-validate` to confirm the fix

### Updating `ground_truth_chunk_ids`

Chunk IDs are content-hash-based ChromaDB identifiers. They become stale whenever the
index is rebuilt (e.g., after a chunking strategy change). After any full reindex:

```bash
# Look up candidates for one question (ChromaDB must be running):
uv run python -m src.eval.scripts.chunk_id_lookup \
    --question "What is the minimum aging for Barolo DOCG?" \
    --top-k 10

# JSON output for scripting:
uv run python -m src.eval.scripts.chunk_id_lookup \
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

Each run writes one JSON file to `eval-results/` (gitignored):

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
| `backend` | `rag`, `retriever`, or `agent` |
| `git_sha` | Short commit hash for reproducibility |
| `config_snapshot` | Model name, embedder, n_results, feature flags |
| `aggregate_metrics` | Mean score per metric across all evaluated samples |
| `metrics_by_category` | Per-category metric breakdown |
| `metric_groups` | Aggregates separated into retrieval, RAG judge, agent tool, agent answer, and operational families |
| `metric_coverage` | Scored, unsupported, skipped, and errored counts overall and by category |
| `per_sample` | Full per-sample results, artifacts, scores, `metric_errors`, and a status/reason for every active metric |
| `schema_version` | Version of the eval result JSON schema |
| `summary` | `evaluated`, `skipped`, `errors`, `timeouts`, `estimated_llm_calls`, `total_latency_ms`, plus structured dataset/filter/execution metadata |

The `summary.skipped` count captures samples that were intentionally skipped (e.g., cellar
samples when the DB is empty). `summary.errors` captures unexpected failures. Neither
abort the run — eval runs to completion even when individual samples fail.

The current writer uses result schema version 6. Comparison tooling reads versions 1–6 so
historical baselines remain usable. Missing metrics are rendered as `n/a`, never as zero,
and comparisons include scored support counts to prevent a change in sample coverage from
looking like a quality change.

Latest verified retrieval run (2026-07-27): 60/60 samples passed with zero errors,
timeouts, generation calls, or judge calls. MRR was `0.8533`, precision@3 `0.5600`, and
precision@5 `0.4320`; each metric scored the 25 RAG-grounded samples and explicitly marked
the remaining 35 samples unsupported.

---

## Configuration

All defaults live in the `eval:` section of `app_config.yml`:

```yaml
eval:
  dataset_path: src/eval/wine_qa_golden.jsonl
  results_dir: eval-results
  default_mode: retrieval
  default_backend: rag
  execution_provider: ollama
  execution_model: gemma4:cloud
  ollama:
    base_url: http://localhost:11434
  max_concurrency: 1
  sample_timeout_seconds: 300
  ragas:
    evaluator_provider: ollama
    evaluator_model: gpt-oss:20b-cloud
    temperature: 0.0
    reasoning: false
    num_predict: 2048
    timeout_seconds: 120
    max_retries: 1
    max_workers: 1
    metrics:
      - faithfulness
      - answer_relevancy
      - context_precision
      - context_recall
  retrieval_metrics:
    k_values: [3, 5]
  skip_cellar_samples_if_empty: true
```

The `max_concurrency` setting controls how many samples run in parallel during backend
execution. Judge scoring has separate Ragas worker behavior and per-metric limits.
Increase execution concurrency only after validating the selected Ollama plan and backend.

Eval is intentionally Ollama-only. The CLI preflight accepts Ollama-hosted local and
cloud models, but rejects other providers.

---

## Cost reference

| Run type | LLM calls | Approx. tokens | Approx. cost |
|----------|-----------|----------------|--------------|
| `make eval` (retrieval, 60 samples) | 0 | 0 generated tokens | free |
| `make eval-full` (full Ragas, 60 samples) | up to ~780 estimated | model-dependent | Ollama cloud usage for `:cloud` models |
| Monthly (1 full run/week) | up to ~3120 estimated | model-dependent | Ollama cloud usage for `:cloud` models |

These are stable estimates for longitudinal comparison, not provider telemetry. Actual Ragas
calls depend on metric internals and sample support. Execution and judging remain
separately configured:

- sample execution uses `eval.execution_provider` / `eval.execution_model`
- Ragas judge scoring uses `eval.ragas.evaluator_provider` / `eval.ragas.evaluator_model`

Judge sampling and runtime are configured independently under `eval.ragas`. Reasoning is
disabled and output is capped for predictable structured scoring. The Ragas timeout is
shorter than the agent sample timeout because it applies to one metric, not the complete
agent execution. Retries default to one to avoid repeating paid cloud calls.

If Ragas returns NaN or a non-numeric metric value, the sample records a metric-level
error and excludes that value from aggregates rather than treating it as a score of zero.
Executor failures such as `TimeoutError` are preserved as stable reasons such as
`ragas_timeout`.

---

## Module reference

| File | Responsibility |
|------|---------------|
| `models.py` | Pydantic models: `GoldenSample`, `SampleResult`, `EvalRunResult` |
| `dataset.py` | `load_golden_dataset()` and `filter_golden_samples()` for the JSONL golden file |
| `scripts/dataset_validator.py` | Detect stale cellar-dependent samples against the live DB |
| `runner.py` | `EvalRunner`: async backend execution for raw eval sample outputs |
| `metrics.py` | Pure local functions: `reciprocal_rank`, `precision_at_k`, means |
| `ragas_scorer.py` | `RagasScorer`: wrap Ragas `evaluate()` for full-mode scoring |
| `reporter.py` | `EvalReporter`: aggregate results, save JSON, print summary |
| `scripts/compare_results.py` | CLI: compare latest N result files with delta table |
| `scripts/chunk_id_lookup.py` | Dev utility: find ChromaDB chunk IDs for dataset authoring |
| `phoenix_reporter.py` | `PhoenixReporter`: push results to Phoenix as experiments |
| `__main__.py` | CLI entry point: orchestrates the full eval pipeline |
| `__init__.py` | Package exports |

---

## Adding new eval samples

1. Open `src/eval/wine_qa_golden.jsonl`.
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
uv run pytest tests/eval/ -v -m "not eval"
```

Ragas scorer integration tests require a live Ollama server and are gated by `@pytest.mark.eval`:

```bash
uv run --extra eval pytest tests/eval/test_ragas_scorer.py -m eval -v
```

---

## Phoenix experiment integration

When a Phoenix server is running (`make phoenix`), eval results can be pushed as named
experiments for visual comparison in the Phoenix UI.

Phoenix remains REST-based for now; the project does not depend on the Phoenix Python SDK.

### Preconditions

- Phoenix is running at `http://localhost:6006` (or configured in
  `observability.phoenix.endpoint` in `app_config.yml`)
- `uv` can install the eval extra declared in `pyproject.toml`

### Usage

```bash
# Retrieval-only run + push to Phoenix
make eval-phoenix

# Full Ragas run + push to Phoenix
make eval-phoenix-full

# Or manually with a custom Phoenix URL
uv run --extra eval python -m src.eval --mode retrieval --push-to-phoenix --phoenix-url http://myserver:6006
```

### What Phoenix shows

Each eval run creates one **experiment** in Phoenix, named
`eval_{mode}_{backend}_{run_id}` (e.g. `eval_retrieval_rag_20260503T143022`).

The experiment contains:
- **Dataset**: `eval_golden_dataset` — the full golden Q&A set, uploaded as a versioned
  snapshot so each experiment is tied to the exact questions used.
- **Runs**: one row per evaluated sample, showing the generated answer and latency.
- **Evaluations**: one annotation per metric per sample, with a numeric score and a
  quality label (`excellent` / `good` / `fair` / `poor`).
  - Retrieval metrics (MRR, precision@k) use `annotator_kind = CODE`
  - Ragas metrics use `annotator_kind = LLM`

This enables side-by-side comparison of multiple runs in the Phoenix UI with per-category
filtering and trend charts.

### Implementation

`src/eval/phoenix_reporter.py` — `PhoenixReporter` class. Uses the Phoenix REST API
directly (no SDK dependency). Fail-open: any error logs a warning and returns `None`
without aborting the eval run.

Method sequence: `push()` -> `_upload_dataset()` -> `_list_example_ids()` ->
`_create_experiment()` -> `_push_runs()` -> `_push_evaluations()`
