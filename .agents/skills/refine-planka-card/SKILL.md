---
name: refine-planka-card
description: Refine one manually added Planka Inbox bug, routine task, small feature, or spike into a clear, bounded, implementation-ready Pour Decisions Backlog item. Route roadmap-sized or strategy-, design-, or experiment-dependent proposals before they move to Ready.
---

# Refine a Planka Card

Turn one rough Inbox card into an executable work item through repository-grounded analysis and an
iterative conversation with the user. Refinement ends only when another engineer or agent could
implement and verify the card without guessing about product behavior, scope, or material technical
decisions.

## Required input

The user must provide the Planka card's current name, for example:

```text
$refine-planka-card card="fix inventory sorting"
```

Accept equivalent natural-language invocations when the card name is unambiguous. If the name is
missing, ask for it before accessing Planka.

## Boundaries and authorization

Invocation with a card name authorizes reading Planka and, after the confirmation gate below,
updating only the uniquely identified existing card. Use the project's dedicated Planka MCP server;
do not use browser automation or another integration as a fallback.

Do not implement the card, edit repository files, change reviewed design documents, or mutate other
cards. Do not create, delete, archive, or split cards, create labels, or change assignees, dates, or
attachments unless the user separately and explicitly approves that action. Preserve useful facts
already recorded on the card.

## Establish context and evidence

1. Read `AGENTS.md` and `planka.md`.
2. Find the card on the `Backlog` board, normally in `Inbox`. Prefer an exact title match. If there
   are zero matches or multiple plausible matches, show concise identifying details and ask the user
   to identify the intended card; do not guess.
3. Read the complete card, including its description, labels, comments, and available attachments.
4. Perform a lightweight related-work scan using the card's distinctive terms, affected subsystem,
   existing labels, and apparent initiative. Inspect plausible duplicates, blockers, superseding
   work, or roadmap links, but do not mutate them. Report whether the target is standalone, related,
   overlapping, or potentially duplicate.
5. Investigate the current repository behavior relevant to the card. Use SigMap first for unfamiliar
   code, then read the identified implementation, tests, and authoritative documentation. Treat code
   and tests as evidence, not as permission to resolve product or design choices silently.
6. Keep these categories distinct throughout refinement:
   - **Recorded fact**: supplied by the card, attachment, comment, or user.
   - **Repository evidence**: directly confirmed in current code, tests, or authoritative docs.
   - **Hypothesis**: a plausible explanation or implementation direction that remains unverified.
   - **Approved decision**: behavior, scope, or design explicitly chosen by the user.

If the card is not on `Inbox`, report its current list and ask whether the user still wants it
refined before making any change.

For formatting or UI bugs, preserve an exact reproducible example in a fenced block or attachment
when ordinary Markdown could alter the evidence.

## Classify and route the work

Classify the target before asking detailed questions. Use the classification to choose evidence and
readiness requirements; do not infer the final labels until repository and Planka context are known.

### Bug

Existing behavior violates an established expectation or previously working behavior. Establish:

- reproduction conditions and the smallest representative example
- observed behavior versus expected behavior
- user impact, affected modes or environments, and relevant boundaries
- whether it is a confirmed regression, longstanding defect, or unknown
- regression coverage and any essential manual verification

A root-cause hypothesis is useful context, but it is not an acceptance criterion unless confirmed.

### Routine task

A bounded maintenance, refactor, cleanup, documentation, test, or operational change with no new
product behavior. Establish:

- concrete outcome and affected workflow or code path
- constraints, dependencies, and compatibility requirements
- observable completion evidence and proportionate verification

### Small feature

A bounded new capability with an already-understood user outcome that can ship independently.
Establish:

- user problem, desired outcome, and representative workflow
- approved UX or product behavior, meaningful alternatives, and why the chosen option fits
- included behavior, non-goals, failure behavior, and compatibility or rollout constraints
- every applicable architecture, API, schema, prompt, configuration, dependency, or design approval

Do not turn an unresolved product or design proposal into a Ready card merely by writing a detailed
implementation specification.

### Research or technical spike

Work whose deliverable is evidence and a decision rather than production behavior. Establish:

- explicit question or risky hypothesis
- bounded, reproducible investigation or experiment
- evidence to collect and success, failure, or comparison thresholds
- expected decision output and what remains out of scope

### Roadmap feature

Work that spans multiple independently movable items, requires strategy or design, or needs milestone
visibility is not one executable Backlog card. Explain the mismatch and recommend the relevant
workflow without invoking it automatically:

- unclear value, user outcome, or strategic alternatives: `research-feature-strategy`
- roadmap-sized work needing a new reviewed design: `create-milestone-design`
- an existing milestone whose design may be stale: `refresh-milestone-design`
- a risky technical assumption that must be tested first: `run-technical-spike`

Keep the target out of `Ready` until the prerequisite decision or artifact exists and its executable
scope is clear. Propose decomposition when appropriate and ask the user to choose the structure; do
not create or split cards under this skill's default authorization.

## Clarify until implementation-ready

Identify every material gap that would force an implementer to guess. Apply the type-specific
requirements above and consider only relevant cross-cutting dimensions:

- included scope, non-goals, and affected interfaces or workflows
- edge and failure behavior, concurrency, security, performance, and cost
- dependencies, rollout or migration constraints, and related or roadmap work
- applicable repository approval gates
- observable acceptance criteria and the automated or manual evidence that verifies them

Inventory gated changes early. State which approvals are already recorded, which choices remain
open, and whether confirming the final specification will approve a clearly identified gated change.

Ask focused questions in coherent batches of no more than three, prioritizing answers that determine
later scope or design. Ask as many follow-up rounds as needed. When an answer exposes another
ambiguity, investigate it and ask again. Offer concrete options and a recommendation when repository
evidence supports them, including tradeoffs, but never select a material decision on the user's
behalf.

While questions remain open, stop before updating Planka. Do not treat silence, a partial answer,
plausible defaults, or inferred intent as resolution. Avoid low-value questions whose answers are
already established by the card, repository, project instructions, or prior user responses.

## Specify outcomes without over-prescribing implementation

Make acceptance criteria describe observable behavior, externally meaningful contracts, and required
constraints. Prescribe an internal implementation only when the user approved it, a repository
invariant requires it, or leaving it open would materially affect architecture, cost, reliability,
security, or compatibility.

Record unverified internals as hypotheses and optional approaches in `Notes`, not as facts or
acceptance criteria. Distinguish required tests from implementation suggestions. Prefer automated
verification; require manual checks only where automation cannot provide sufficient evidence.

## Readiness gate

Every card is implementation-ready only when:

- its classification, outcome, and motivation are understandable without hidden conversation context
- recorded facts, repository evidence, hypotheses, and approved decisions are not conflated
- scope and meaningful non-goals are explicit
- all behavior- or design-affecting decisions and applicable approval gates are resolved
- acceptance criteria are observable and unambiguous
- verification is proportionate and covers the changed behavior
- dependencies, related work, initiative linkage, and deferred follow-ups are recorded
- the work is one bounded executable unit and no material question remains

Also apply the relevant type-specific gate:

- **Bug**: reproduction, observed versus expected behavior, impact, boundaries, and regression
  coverage are explicit.
- **Routine task**: outcome, affected path, constraints, dependencies, and completion evidence are
  explicit.
- **Small feature**: user outcome, approved behavior, alternatives, non-goals, failure behavior,
  rollout or compatibility, and gated decisions are resolved.
- **Research or spike**: hypothesis, method, evidence, thresholds, and decision output are explicit.
- **Roadmap feature**: do not move it to `Ready` under this workflow; route or decompose it first.

Repository approval gates still apply. If implementation requires a decision or artifact that has
not been approved, leave the card out of `Ready`. Never claim readiness merely because the
description follows a template.

## Draft and confirm the update

Once the readiness gate passes, prepare a concise proposed update following the Backlog card template
in `planka.md`. Use the natural sections for the card type and omit empty boilerplate:

- an action-oriented title using the project's naming convention, prefixed with the exact next
  workflow skill as `$skill-name: `
- `Context`, `Scope`, `Non-goals`, `Acceptance`, `Verification`, `Dependencies and Related Work`,
  `References`, and useful `Notes`
- `Evidence / Reproduction` for bugs and evidence-driven work when it materially helps execution
- one existing primary work-type label and relevant area labels
- exactly one existing initiative label when the work belongs to a roadmap item
- destination list `Ready`

Choose the prefix for the workflow that should execute the Ready card:

- `$deliver-planka-card:` for an independently deliverable bug, routine task, or small feature
- `$run-technical-spike:` for a bounded experiment with a falsifiable question and explicit evidence
  thresholds
- `$research-feature-strategy:` for evidence-backed strategic comparison or recommendation work
- `$deliver-milestone-phase:` for an approved, bounded phase of a reviewed milestone

Do not use `$refine-planka-card:` on a card moved to `Ready`; refinement is complete at that point.
Replace an existing work-type or workflow prefix rather than stacking prefixes. Preserve initiative
identifiers such as `m10:` immediately after the workflow prefix. Do not use
`$create-milestone-design:` or `$refresh-milestone-design:` on a Ready Backlog card under this
workflow; cards needing those skills remain outside `Ready` until the required design work is
complete.

Put exact examples in fenced blocks when Markdown interpretation could change them. References must
point to stable, useful artifacts. Do not cite local design documents in committed repository
artifacts; Planka references may identify them when relevant to the private planning workflow.

Show the user the complete proposed title, description, label additions and removals, gated changes
approved by confirmation, and list move. Ask for explicit confirmation or corrections. If the user
changes scope or behavior, reassess the readiness gate and continue clarification as needed.

## Apply and verify

After explicit confirmation:

1. Re-read the card and verify it is still the same card and has not materially changed. If it has,
   show the conflict and ask how to proceed.
2. Update that existing card with the confirmed title, description, and labels, then move it to
   `Ready`. Do not create a replacement card.
3. Read the card back from Planka and verify the persisted title, description, labels, board, and
   list. Correct only discrepancies from the confirmed update.
4. Report the final card name and location, summarize the classification and decisions, identify
   related work and explicitly deferred follow-ups, and state any approved gated changes. Do not
   begin implementation.

If Planka is unavailable or a required board, list, or label is missing, stop and report the exact
blocker. Do not substitute another tool or invent project structure.
