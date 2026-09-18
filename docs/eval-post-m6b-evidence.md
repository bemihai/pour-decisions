# Post-M6B full-evaluation evidence

- **Project version:** 0.8.7
- **Last verified:** 2026-09-18
- **Status:** diagnostic cohort frozen; no agent or retrieval redesign approved

## Baseline and provenance

The local schema-v7 artifacts are `eval-results/20260918T065333_full_rag.json` and
`eval-results/20260918T071727_full_agent.json`. Both contain all 60 samples from golden
dataset hash `0c000ebde4ad64ef6f68f0f23bc5bb2e`, at Git SHA `633d2ff` on
`m06b-closeout`. Both report `git.is_dirty=true`. The effective eval-only working-tree
diff was the Ragas dependency workaround in `pyproject.toml` and `uv.lock`, plus the
import-error preflight and its test in `src/eval/preflight.py` and
`tests/eval/test_main.py`. The lockfile pins an upstream Ragas fix at
`28618898a677515c04d31e5b322da4d8129a769c`; this changes full-eval import
compatibility, not the agent, retrieval, prompts, or dataset. Those four files should
be reviewed and integrated with this evidence work. The historical artifacts cannot
truthfully be relabeled clean after that integration.

Both runs used Ollama `gemma4:cloud` for generation, `gemma4:31b-cloud` as judge,
temperature 0 for the judge, `num_predict=2048`, 120-second judge timeout, and one
retry. Each ran with sample concurrency 1 and a 300-second sample timeout. These
results are adequate for a diagnostic cohort and milestone choice because the
effective config and dataset match. They do **not** satisfy a literal clean-commit
baseline gate. A new full run is needed if that gate is enforced for an implementation
comparison; repeating both 60-sample runs now merely to clear the dirty flag would
cost roughly 1,518 *estimated* model calls (699 RAG plus 819 agent), without resolving
the failures identified below. These estimates are not provider billing counts.

| Run | Execution | Quality and tool signals | Judge coverage limits |
| --- | --- | --- | --- |
| RAG | 60/60 executed, 0 timeouts, 0 execution errors, mean 2,707 ms | RAG-only: relevancy .863, faithfulness .949, context precision .592, recall .443, MRR .837 | Context/faithfulness 51/60; retrieval metrics 24/60. Cellar questions are outside this backend's inventory scope. |
| Agent | 60/60 executed, 0 timeouts, mean 5,315 ms | Relevancy .895, correctness .561, tool exact .543, ordered .743, precision .661, recall .771. Multi-hop exact .20, recall .50. | Correctness 52/60, relevancy 53/60, faithfulness 26/60, tool scores 35/60. Seven blanks were incorrectly reported passed; four judge metrics errored. |

## Frozen agent failure cohort

The historical agent run has empty final answers for `cellar_001`, `cellar_004`,
`cellar_015`, `multi_hop_001`, `multi_hop_003`, `multi_hop_006`, and `multi_hop_009`.
All seven had status `passed` because the runner formerly checked only whether an
exception occurred. Under the corrected runner they would be explicit
`failed/empty_agent_final_answer` samples: execution success would be 53/60 rather
than 60/60, an 11.7% user-visible empty-answer rate. Their historical judge scores
were absent, so quality means do not measure these failed answers.

The old artifact lacks raw final-message finish metadata, so it cannot establish the
cause of each historical blank. A bounded replay with the configured Ollama model
reproduced blanks on `cellar_001`, `multi_hop_003`, `multi_hop_006`, and
`multi_hop_009`; another replay reproduced three of these. Final AI messages had
empty visible content, no pending tool calls, `done_reason=stop`, and 25–28 output
tokens on the inspected examples. This supports a model-visible-output failure,
not text discarded by the eval adapter. `cellar_004`, `cellar_015`, and
`multi_hop_001` returned nonempty answers on replay, indicating run variance.
The narrow fix is eval status classification; an agent fallback or model contract
change needs separate review.

The ten multi-hop traces were reviewed against expected names, actual tool order,
outputs, and final-answer presence. The exact-name metric is intentionally strict,
but a mismatch is not always a bad plan:

| ID | Trace assessment |
| --- | --- |
| `multi_hop_001` | No tools; empty answer. Missing decomposition and terminal output. |
| `multi_hop_002` | Cellar plus `search_wine_region_info` instead of expected generic knowledge search. Specialized region search is a plausible substitute; faithfulness scored 0, so answer quality remains a concern. |
| `multi_hop_003` | No tools; empty answer. Missing decomposition and terminal output. |
| `multi_hop_004` | Cellar and knowledge search each called twice. Repeated names merit argument-level trace review; low faithfulness (.235) and recall (0) show a real evidence problem. |
| `multi_hop_005` | Cellar plus six `get_pairing_for_wine` calls instead of `get_food_pairing_wines`. These may be distinct per-wine lookups; repeated names alone do not prove redundant calls. Correctness .470. |
| `multi_hop_006` | Term-definition search only; empty answer. Required cellar and knowledge work missing. |
| `multi_hop_007` | Extra grape search before both expected tools. This may help the answer; do not penalize it as waste without checking arguments. Correctness .468. |
| `multi_hop_008` | Both expected tools in order; correctness .495 and context precision 0 show that selection alone did not ensure a good answer. |
| `multi_hop_009` | Region search only; empty answer. Useful specialized call, but required cellar step missing. |
| `multi_hop_010` | Both expected tools in order; correctness .425. |

Seven extra calls repeat a **tool name** within their sample (two in `_004`, five
in `_005`); the artifact does not persist arguments needed to label them redundant.
Do not alter golden expected names or scoring until specialized-tool equivalence and
argument-level calls are reviewed. The confirmed planning cohort is `_001`, `_003`,
`_006`, `_009`; the potential scoring-contract cohort is `_002`, `_005`, `_007`.

## RAG evidence cohort

Eight RAG-only samples had judged context recall 0. This is a mixed cohort, not
evidence of one retrieval defect:

| ID | Evidence assessment |
| --- | --- |
| `rag_only_003` | Relevant Barolo/Barbaresco chunks retrieved (MRR .5); exact aging-rule reference details need source-level confirmation. |
| `rag_only_008` | Sancerre/Sauvignon Blanc material retrieved (MRR .5); mixed red/rosé context and missing reference flavor/soil detail. Partial support. |
| `rag_only_012` | Top-ranked ground-truth chunk and judged precision 1; reference additionally names IGT and specific examples. Reference/context mismatch needs adjudication. |
| `rag_only_020` | Only one context chunk, despite MRR 1; answer covers the grape and country, while reference requires Romanian subregions, aromas, and aging. Retrieved support is incomplete. |
| `rag_only_021` | Two chunks, judged precision 1, but no ground-truth chunk IDs for retrieval scoring; answer lacks reference climate, variety, and aging claims. Partial source support. |
| `rag_only_022` | MRR 1 but judged precision and recall 0; retrieved Rhône material does not clearly support the full regulatory reference. Ground-truth ID alone is insufficient evidence. |
| `rag_only_023` | MRR and precision 0; answer explicitly says the context lacks the 1855 ranking. Verified missing-context retrieval failure and the strongest candidate for corrective retrieval work. |
| `rag_only_025` | MRR and judged precision 1; one source says 65–67°F for Bordeaux while the reference says 61–64°F plus decanting. Adjudicate reference/source conflict before changing retrieval. |

Only `_023` is a cleanly verified missing-context entry case from this artifact.
`_020` and `_021` show incomplete evidence, but may reflect corpus limits or
over-specified references. This cohort does not yet justify broad retrieval changes.

## Judge parsing and production-model check

Agent judge errors affected faithfulness on `rag_only_007`, `_012`, and `_022`,
and correctness on `cellar_014`. Each stored error is an Ragas
`OutputParserException` for structured `NLIStatementOutput` or
`ClassificationWithReason`. The artifact stores only a truncated error message,
not the complete raw judge response. It cannot distinguish malformed structure
from model output truncation or a Ragas parser defect. The configured 2,048-token
output cap is a plausible contributor, not a proven cause. The scorer correctly
records these as metric errors and excludes them from aggregates; a regression test
now confirms this handling for parser errors. Do not substitute zero scores or add
paid retries without a bounded reproduced judge-output trace.

Four frozen agent cases were replayed through the production Google
`gemini-2.5-flash` agent with a five-call-per-question limit (20 maximum; eight
generation calls observed). This was a direct agent check, not a comparable full
Ragas evaluation. `cellar_001` answered after `get_cellar_wines` (2 calls,
2,754 ms); `multi_hop_003` answered after cellar plus region search (3 calls,
16,983 ms); `multi_hop_009` answered after region plus cellar search (2 calls,
7,558 ms). `multi_hop_005` answered without any tools (1 call, 3,216 ms), asserting
an inventory conclusion without consulting the cellar. Thus the sampled Ollama
blank-output failure did not reproduce on Gemini, while a missing-tool planning
failure did. No private cellar answer text is included in this report.

The next milestone choice should prioritize the confirmed agent planning and
answer-completion cohort, with model-specific validation before broad changes.
Corrective retrieval should begin with `_023` and additional verified missing
context cases, not the aggregate recall number. This is evidence for a future
design decision; it does not change an approved design or agent contract.
