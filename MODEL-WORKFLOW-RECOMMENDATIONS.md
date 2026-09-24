# Model Recommendations for Pour Decisions Workflows

- **Project version**: 0.9.0
- **Last updated**: 2026-09-24

Use **GPT-6 Sol** as the main implementation model, **GPT-6 Astra** for difficult design audits and correctness work, and **GPT-6 Luna** for clear, bounded maintenance and mechanical preparation.

These recommendations are based on the repository instructions and current official OpenAI model documentation. They are advisory, not measured benchmark results or changes to project policy.

## Review scope

- Root `AGENTS.md` and `frontend/AGENTS.md`
- Existing workflow assignments for design refresh, phase delivery, closeout, and project tracking
- Current official Codex model-selection and pricing guidance

The recommendations concern the development assistant. They do not propose replacing the application's Gemini/local model configuration.

## What changes with GPT-6

The previous recommendation used GPT-5.6 Sol for implementation, Terra for maintenance, and Luna only for mechanical work. Replace that default split with GPT-6 Sol and Luna: use Sol when the task requires investigation or judgment, and Luna when the behavior, scope, and verification are already clear. Keep Astra for the hardest work. This is a project recommendation inferred from the documented roles, not a measured equivalence between model generations. [Official model guidance](https://learn.chatgpt.com/docs/models#choosing-astra-sol-and-luna).

GPT-5.6 models remain available during rollout, but are fallback choices when a GPT-6 model is unavailable or a representative task demonstrates a regression. Availability depends on client, sign-in method, rollout, and workspace settings. [Codex models](https://learn.chatgpt.com/docs/models#recommended-models).

## Workflow recommendations

| Workflow | Recommended model / reasoning effort | Rationale |
|---|---|---|
| Milestone design audit: `refresh-milestone-design`, audit mode | **GPT-6 Astra · high** | Reconstruct delivered behavior, challenge milestone value, and identify unnecessary complexity across subsystems. |
| Apply an approved design refresh | **GPT-6 Sol · medium** | Encode settled decisions and check consistency between scope, interfaces, phases, and acceptance criteria. Use Astra when approved changes substantially reshape the milestone. |
| Normal milestone phase delivery: `deliver-milestone-phase` | **GPT-6 Sol · medium; high for demanding phases** | Sustain implementation, sequential card execution, tests, Git discipline, and accurate handoffs within reviewed scope. |
| Complex milestone phase involving orchestration, concurrency, or retrieval consistency | **GPT-6 Astra · high** | Reason through interacting failure paths involving shared deadlines, SQLite retries, immutable tool snapshots, or Chroma/BM25 consistency. |
| Milestone closeout | **GPT-6 Sol · high** | Verify merged phases, prove acceptance, run regressions, and update documentation and versions. Escalate to Astra for ambiguous cross-system failures. |
| Architecture proposals and milestone prioritization | **GPT-6 Astra · high** | Evaluate consequential tradeoffs involving reuse, infrastructure, operating cost, maintainability, and measurable value. Decisions remain subject to user approval. |
| Small backend fixes and straightforward tests | **GPT-6 Luna · high** | Use when the cause and expected behavior are already known and targeted tests can verify the result. Use Sol when diagnosis or scope is uncertain. |
| Difficult debugging and regression investigation | **GPT-6 Astra · high** | Investigate symptoms spanning retrieval, agent execution, database access, and API lifecycle boundaries. |
| Prompt/tool-contract design and evaluation analysis | **GPT-6 Astra · high** | Analyze changes that can affect tool selection and failure behavior across requests before seeking required approval. |
| Approved frontend feature implementation | **GPT-6 Sol · medium; high for complex state changes** | Coordinate Next.js behavior, typed API calls, server/client state, and usability within approved direction. |
| Small frontend fixes | **GPT-6 Luna · high** | Make localized changes following existing components and the required installed Next.js documentation. Use Sol for state, accessibility, or interaction changes requiring investigation. |
| Planka task decomposition and reconciliation | **GPT-6 Sol · medium** | Define independently verifiable work and reconcile actual delivery state. |
| Routine Planka updates from clear evidence | **GPT-6 Luna · high** | Apply explicit templates and status rules after checking evidence and existing cards. |
| Maintained documentation and PR descriptions | **GPT-6 Luna · high** | Summarize verified changes. Use Sol when establishing the documented behavior requires tracing multiple modules. |
| Mechanical summaries and formatting drafts | **GPT-6 Luna · high; low for simple formatting** | Transform supplied facts into a specified format. Do not use as the primary judge of design readiness or release acceptance. |

Reasoning settings are suggested starting points, not experimentally established optima. OpenAI recommends starting with **Sol medium**, **Luna high**, and **Astra Light (`low`)**. The Astra high assignments above deliberately spend more reasoning on difficult project decisions; use Light or medium for narrower investigations. Effort levels are not directly comparable across generations. [Reasoning guidance](https://learn.chatgpt.com/docs/models#pick-a-reasoning-effort).

Use Extra High only when high leaves a concrete reasoning problem unresolved. Max spends more time on a single task; Ultra uses subagents for parallel work. Neither is a routine default, and Luna supports Max but not Ultra. Use Ultra only for explicitly authorized delegation with meaningful independent subtasks. [Max and Ultra](https://learn.chatgpt.com/docs/models#know-when-to-use-max-or-ultra).

## Operating guidance

- Keep one model responsible for a delivery phase. The delivery skill requires sequential card execution; introducing a model or agent handoff for every card adds coordination work. Prefer natural boundaries such as audit, approved refresh, and implementation.
- Preserve approval gates. Model selection does not authorize architecture, prompt, contract, default, dependency, migration, reviewed-document, or frontend-direction changes. State which decisions have already been approved to avoid repeated clarification.
- Scale verification to risk. Higher reasoning effort should not automatically trigger broader or repeated testing. Follow the repository's targeted-test and phase-gate requirements.
- Distinguish development-assistant spending from application operating cost. Evaluate the former by cost per successfully completed task, including retries and review effort.
- Escalate Luna to Sol when a task needs diagnosis, crosses module boundaries, or requires repeated corrections. Escalate Sol to Astra for unresolved interactions or consequential tradeoffs. Fix missing context or unclear acceptance criteria before increasing effort.
- Choose the model and effort in the IDE picker; in the CLI use `/model` or launch with `codex --model gpt-6-sol`. These are selection instructions, not changes to saved configuration. [Model selection controls](https://learn.chatgpt.com/docs/models).

## Cost and usage

For Standard speed, the published Codex credit rates are:

| Model | Input credits / 1M tokens | Cached input credits / 1M tokens | Output credits / 1M tokens |
|---|---:|---:|---:|
| GPT-6 Astra | 250 | 25 | 1,250 |
| GPT-6 Sol | 50 | 5 | 250 |
| GPT-6 Luna | 2.5 | 0.25 | 12.5 |

These are credit rates, not dollar prices or fixed subscription message limits. API-key sessions follow separate API billing. Included usage depends on the task, context, tools, reasoning, and caching; check the usage dashboard rather than estimating task counts from these ratios. Prefer Standard speed for routine work and assess total cost per accepted result, including retries and review. [Codex pricing](https://learn.chatgpt.com/docs/pricing#token-rates).

## Evidence and limitations

The official model guidance describes Astra as the strongest option for difficult workflows, Sol for everyday and complex coding, and Luna for focused, repeatable work, including focused coding. Those roles support the workflow assignments above, but do not establish that Luna can replace Sol for arbitrary small changes. [Codex model roles](https://learn.chatgpt.com/docs/models#choosing-astra-sol-and-luna).

No comparative model evaluations were run. Actual latency, account availability, usage, and cost per completed task were not measured. Validate these starting points on representative maintenance, phase-delivery, and audit tasks, recording correctness, review corrections, elapsed time, and usage. Revisit the assignments when that evidence or the available models change.
