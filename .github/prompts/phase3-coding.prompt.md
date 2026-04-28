---
mode: agent
description: "Phase 3 - Implement one phase/step from a design document"
---

You are an expert coding agent implementing a specific step from a design specification.

**Design document:** ${input:designFile:e.g. design/agentic/llm-observability-design.md}
**Step to implement:** ${input:step:e.g. Phase 1, Step 2: Add DB migration for sync_logs table}

Read `#file:AGENTS.md`, `#file:.github/copilot-instructions.md`, and the design document above before writing any code.

## Rules

- Implement ONLY the step specified above. Do not implement subsequent steps.
- Follow all project conventions: type hints on every function, Google-style docstrings, `from src.utils import logger` (never `print()`), Black 120-char formatting, isort imports.
- Write the accompanying unit tests defined in the Acceptance Criteria for this step.
- If you discover an architectural flaw or need to deviate from the design, STOP and ask before proceeding. Then update the design document to reflect reality before writing code.
- Do not produce placeholder or TODO code — every function must be fully implemented.

