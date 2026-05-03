# Eval Dataset — `tests/eval/`

This directory contains the golden Q&A evaluation dataset and test results for the
Pour Decisions eval harness (Milestone 2).

---

## Directory layout

```
tests/eval/
├── README.md                   # This file
├── wine_qa_golden.jsonl        # 60-question golden dataset (version-controlled)
├── results/                    # Timestamped JSON result files (gitignored)
│   └── .gitkeep
├── test_models.py              # Phase 1: model + dataset loader unit tests
├── test_dataset.py             # Phase 2: golden dataset integrity tests
├── test_metrics.py             # Phase 3: retrieval metric unit tests
├── test_runner.py              # Phase 4: eval runner unit tests (mocked)
├── test_ragas_scorer.py        # Phase 5: Ragas scorer tests (@pytest.mark.eval)
└── test_reporter.py            # Phase 6: reporter unit tests
```

---

## `wine_qa_golden.jsonl` — format specification

Each line is a JSON object with the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `str` | Yes | Unique identifier in the format `{category}_{NNN}` |
| `question` | `str` | Yes | The exact question passed to the system under test |
| `category` | `str` | Yes | One of: `rag_only`, `cellar`, `pairing`, `multi_hop` |
| `difficulty` | `str` | Yes | One of: `easy`, `medium`, `hard` |
| `expected_facts` | `list[str]` | Yes | Key facts the answer must contain (used in LLM-as-judge prompts) |
| `expected_tool_calls` | `list[str]` | No | Tool names expected to be invoked (agent backend only) |
| `ground_truth` | `str` | Yes | Reference answer (full, factual sentence) for Ragas `context_recall` |
| `ground_truth_chunk_ids` | `list[str]` | No | ChromaDB chunk IDs known to contain the answer |
| `tags` | `list[str]` | Yes | Multi-labels for result slicing |
| `notes` | `str` | No | Special conditions or skip criteria |

### Category distribution (target: 60 questions)

| Category | Target count | Description |
|----------|-------------|-------------|
| `rag_only` | 25 | Wine knowledge from indexed books |
| `cellar` | 15 | Queries against the user's cellar DB |
| `pairing` | 10 | Food and wine pairing |
| `multi_hop` | 10 | Synthesis across RAG + cellar |

### Difficulty breakdown

| Level | Target count | Characteristics |
|-------|-------------|-----------------|
| `easy` | 20 | Single-source, factual, unambiguous |
| `medium` | 25 | Require wine terminology or moderate context |
| `hard` | 15 | Multi-hop, classification edge cases, ambiguous terminology |

---

## Ground truth authoring guidelines

`ground_truth` must be a **complete, factual sentence** — not a description of what the
answer should contain.

- **Bad:** "Something about Barolo's aging requirement."
- **Good:** "Barolo DOCG requires a minimum of 38 months aging from harvest, with at
  least 18 months in oak, extended to 62 months for Riserva."

For `cellar` category questions, ground truths use **structural assertions** rather than
exact values, because cellar contents change over time:

- **Bad:** "The cellar contains 16 bottles of Barolo."
- **Good:** "The answer must state whether a Barolo wine is present in the cellar and
  provide its name, producer, and earliest drinking year."

---

## `ground_truth_chunk_ids` maintenance

Chunk IDs are content-hash-derived ChromaDB identifiers. They **become stale** whenever
the index is rebuilt (e.g., after a chunking strategy change in Milestone 3).

After any reindex, re-run the lookup utility to refresh IDs:

```bash
python -m src.eval.chunk_id_lookup --question "What is the minimum aging for Barolo?"
```

If `ground_truth_chunk_ids` is empty for a sample, MRR and precision@k are simply
skipped for that sample — the run does not fail.

---

## Running evaluations

```bash
# Retrieval-only mode (free, ~0 API calls)
make eval

# Full Ragas scoring (uses Gemini Flash API — ~$0.03 per run)
make eval-full

# Compare the two most recent runs
make eval-report
```

Or directly:

```bash
python -m src.eval --mode retrieval --backend rag
python -m src.eval --mode full --backend rag --categories rag_only pairing
```

---

## Results

Result files are gitignored. Each file is named:

```
{YYYYMMDDTHHMMSS}_{mode}_{backend}.json
```

Example: `20260501T143022_retrieval_rag.json`

The `.gitkeep` ensures the directory is tracked by git even when empty.

---

## Local Retrieval Metrics (Phase 3)

Retrieval-only evaluation uses pure local metrics from `src/eval/metrics.py`:

- `reciprocal_rank`: `1 / rank` of first relevant chunk, else `0.0`
- `mean_reciprocal_rank` (MRR): average reciprocal rank across samples
- `precision_at_k`: relevant chunks in top-k divided by `k`
- `mean_precision_at_k`: average precision@k across samples

These metrics are deterministic, require no API keys, and make zero LLM calls.

Run the metric unit tests:

```bash
python -m pytest tests/eval/test_metrics.py -v
```
