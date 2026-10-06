---
name: refine-planka-card
description: Refine one manually added Planka Inbox card into a clear, bounded, implementation-ready Pour Decisions Backlog item. Use when the user supplies the card's name and wants missing context, decisions, examples, edge cases, acceptance criteria, or tests resolved before the card moves to Ready.
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

## Establish context

1. Read `AGENTS.md` and `planka.md`.
2. Find the card on the `Backlog` board, normally in `Inbox`. Prefer an exact title match. If there
   are zero matches or multiple plausible matches, show concise identifying details and ask the user
   to identify the intended card; do not guess.
3. Read the complete card, including its description, labels, comments, and available attachments.
   Inspect related Planka cards when needed for linkage or duplicate detection, but do not mutate
   them.
4. Investigate the current repository behavior relevant to the card. Use SigMap first for unfamiliar
   code, then read the identified implementation, tests, and authoritative documentation. Treat
   code and tests as evidence, not as permission to resolve product or design choices silently.
5. Distinguish recorded facts, repository evidence, and proposed decisions. Call out conflicts or
   stale assumptions instead of smoothing them over.

If the card is not on `Inbox`, report its current list and ask whether the user still wants it
refined before making any change.

## Clarify until implementation-ready

Identify every material gap that would force an implementer to guess. Consider only dimensions
relevant to this card, including:

- the problem, user impact, and desired outcome
- current behavior, expected behavior, and reproducible examples or data
- included scope, non-goals, and affected interfaces or workflows
- product, UX, architecture, API, schema, prompt, configuration, or compatibility decisions
- failure behavior and relevant boundary, error, concurrency, security, performance, or cost cases
- dependencies, rollout or migration constraints, and links to roadmap work
- objectively verifiable acceptance criteria and the tests or manual checks that provide evidence

Ask focused questions in coherent batches, prioritizing decisions that affect later questions. Ask
as many follow-up rounds as needed. When an answer exposes another ambiguity, investigate it and ask
again. Offer concrete options and a recommendation when repository evidence supports them, including
tradeoffs, but never select a material decision on the user's behalf.

While questions remain open, stop before updating Planka. Do not treat silence, a partial answer,
plausible defaults, or inferred intent as resolution. Avoid low-value questions whose answers are
already established by the card, repository, project instructions, or prior user responses.

If the work is roadmap-sized or contains independently movable tasks, explain why it is not a
single executable Backlog item and propose a decomposition. Ask the user to choose the structure;
do not create or split cards under this skill's default authorization.

## Readiness gate

The card is implementation-ready only when all of the following are true:

- its outcome and motivation are understandable without hidden conversation context
- scope and meaningful non-goals are explicit
- all behavior- or design-affecting decisions are resolved with user approval
- relevant examples and edge or failure behavior are specified
- acceptance criteria are observable and unambiguous
- verification covers the changed behavior at an appropriate level
- dependencies, initiative linkage, and applicable constraints are recorded
- the work is one bounded unit and no material open question remains

Repository approval gates still apply. If implementation would require a decision or artifact that
must be approved first, obtain that decision during refinement or leave the card out of `Ready`.
Never claim readiness merely because the description follows a template.

## Draft and confirm the update

Once the readiness gate passes, prepare a concise proposed update following the Backlog card template
in `planka.md`:

- an action-oriented title using the project's naming convention
- `Context`, `Scope`, `Acceptance`, `References`, and useful `Notes`
- at least one existing work-type label
- exactly one existing initiative label when the work belongs to a roadmap item
- destination list `Ready`

Put test expectations, important examples, constraints, and edge cases in the most natural section;
do not add empty boilerplate. References must point to stable, useful artifacts. Do not cite local
design documents in committed repository artifacts; Planka references may identify them when they
are relevant to the private planning workflow.

Show the user the complete proposed title, description, label changes, and list move. Ask for explicit
confirmation or corrections. If the user changes scope or behavior, reassess the readiness gate and
continue clarification as needed.

## Apply and verify

After explicit confirmation:

1. Re-read the card and verify it is still the same card and has not materially changed. If it has,
   show the conflict and ask how to proceed.
2. Update that existing card with the confirmed title, description, and labels, then move it to
   `Ready`. Do not create a replacement card.
3. Read the card back from Planka and verify the persisted title, description, labels, board, and
   list. Correct only discrepancies from the confirmed update.
4. Report the final card name and location, summarize what was clarified, and identify any explicitly
   deferred follow-up. Do not begin implementation.

If Planka is unavailable or a required board, list, or label is missing, stop and report the exact
blocker. Do not substitute another tool or invent project structure.
