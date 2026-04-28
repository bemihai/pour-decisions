---
mode: agent
description: "Spike - Quick proof-of-concept to validate a risky assumption"
---

You are writing a quick throwaway proof-of-concept to validate an assumption before committing to a design.

**What to validate:** ${input:assumption:e.g. Can we run cross-encoder reranking locally within 200ms per query?}
**Constraint context:** ${input:constraints:e.g. Must run without GPU, must be free tier}

Read `#file:AGENTS.md` for the project's technology stack and constraints.

## Rules

- Write the minimal code necessary to prove or disprove the assumption. Brevity over elegance.
- Include a clear conclusion at the top of the script: what was proven or disproven.
- Do NOT write production-quality code. No docstrings, no full error handling required.
- Output a summary memo at the end stating whether the approach is viable, any observed limitations, and recommended next steps for Phase 2 planning.

