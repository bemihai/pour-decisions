# Model Recommendations for Pour Decisions Workflows

Project version: 0.8.4  
Last verified: 2026-09-05

Use GPT-5.6 Sol as the main implementation model, GPT-6 Astra for design audits and difficult correctness work, GPT-5.6 Terra for bounded maintenance, and GPT-5.6 Luna for mechanical preparation.

These recommendations are based on the repository instructions and current official OpenAI model documentation. They are advisory, not measured benchmark results or changes to project policy.

## Scope reviewed

- Root `AGENTS.md` and `frontend/AGENTS.md`
- `refresh-milestone-design` skill, including audit and approved-refresh workflows
- `deliver-milestone-phase` skill, including phase execution and milestone closeout
- `planka.md`

The recommendations concern the development assistant. They do not propose replacing the application's Gemini/local model configuration.

## Workflow recommendations

| Workflow | Recommended model / reasoning effort | Rationale |
|---|---|---|
| Milestone design audit: `refresh-milestone-design`, audit mode | **GPT-6 Astra · high** | Reconstruct delivered behavior, challenge milestone value, and identify unnecessary complexity across subsystems. |
| Apply an approved design refresh | **GPT-5.6 Sol · high** | Encode settled decisions and check consistency between scope, interfaces, phases, and acceptance criteria. Use Astra when approved changes substantially reshape the milestone. |
| Normal milestone phase delivery: `deliver-milestone-phase` | **GPT-5.6 Sol · high** | Sustain implementation, sequential card execution, tests, Git discipline, and accurate handoffs within reviewed scope. |
| Complex milestone phase involving orchestration, concurrency, or retrieval consistency | **GPT-6 Astra · high** | Reason through interacting failure paths involving shared deadlines, SQLite retries, immutable tool snapshots, or Chroma/BM25 consistency. |
| Milestone closeout | **GPT-5.6 Sol · high** | Verify merged phases, prove acceptance, run regressions, and update documentation and versions. Escalate to Astra for ambiguous cross-system failures. |
| Architecture proposals and milestone prioritization | **GPT-6 Astra · high** | Evaluate consequential tradeoffs involving reuse, infrastructure, operating cost, maintainability, and measurable value. Decisions remain subject to user approval. |
| Small backend fixes and straightforward tests | **GPT-5.6 Terra · medium** | Follow existing patterns within clear scope. Escalate to Sol when the cause or affected behavior is uncertain. |
| Difficult debugging and regression investigation | **GPT-6 Astra · high** | Investigate symptoms spanning retrieval, agent execution, database access, and API lifecycle boundaries. |
| Prompt/tool-contract design and evaluation analysis | **GPT-6 Astra · high** | Analyze changes that can affect tool selection and failure behavior across requests before seeking required approval. |
| Approved frontend feature implementation | **GPT-5.6 Sol · high** | Coordinate Next.js behavior, typed API calls, server/client state, and usability within approved direction. |
| Small frontend fixes | **GPT-5.6 Terra · medium** | Make localized changes following existing components and the required installed Next.js documentation. |
| Planka task decomposition and reconciliation | **GPT-5.6 Sol · medium** | Define independently verifiable work and reconcile actual delivery state. |
| Routine Planka updates from clear evidence | **GPT-5.6 Terra · low or medium** | Apply explicit templates and status rules after checking evidence and existing cards. |
| Maintained documentation and PR descriptions | **GPT-5.6 Terra · medium** | Summarize verified changes. Use Sol when establishing the documented behavior requires tracing multiple modules. |
| Mechanical summaries and formatting drafts | **GPT-5.6 Luna · low** | Transform supplied facts into a specified format. Do not use as the primary judge of design readiness or release acceptance. |

Reasoning settings are suggested starting points, not experimentally established optima.

## Operating guidance

- Keep one model responsible for a delivery phase. The delivery skill requires sequential card execution; introducing a model or agent handoff for every card adds coordination work. Prefer natural boundaries such as audit, approved refresh, and implementation.
- Preserve approval gates. Model selection does not authorize architecture, prompt, contract, default, dependency, migration, reviewed-document, or frontend-direction changes. State which decisions have already been approved to avoid repeated clarification.
- Scale verification to risk. Higher reasoning effort should not automatically trigger broader or repeated testing. Follow the repository's targeted-test and phase-gate requirements.
- Distinguish development-assistant spending from application operating cost. Evaluate the former by cost per successfully completed task, including retries and review effort.
- Start with Astra high for design audits, Sol high for phase delivery, and Terra medium for everyday small tasks. Reserve Luna for mechanical work.

## Evidence and limitations

Official OpenAI documentation positions the models as follows:

- [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra): most capable model for difficult end-to-end work.
- [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol): flagship model for complex professional work.
- [GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra): balances intelligence and cost, roughly corresponding to the earlier mini tier.
- [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna): optimized for cost-sensitive, high-volume work, roughly corresponding to the earlier nano tier.

The workflow assignments are an assessment based on those roles and the repository's instructions, not an official OpenAI ranking of these project tasks. No comparative model evaluations were run. Actual latency, usage limits, account availability, and cost per completed task were not measured. Revisit these recommendations when model capabilities or project workflows change.
