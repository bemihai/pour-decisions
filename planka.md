# Planka Guide

## Purpose

Use Planka to track delivery work at two levels:

- `Roadmap` tracks large roadmap items and their lifecycle.
- `Backlog` tracks executable work items such as bugs, routine tasks, small features, research
  spikes, refactors, and maintenance.

This guide defines the required board structure, list semantics, label rules, naming conventions, and markdown templates for cards.

## Core Model

- `1 roadmap card = 1 roadmap item`
- `1 backlog card = 1 executable work item`
- Roadmap-item status is represented by the card's list on the `Roadmap` board.
- Task workflow status is represented by the card's list on the `Backlog` board.
- Labels are for classification and linkage, not for workflow phase.

Do not create duplicate cards with the same scope on both boards.

## Boards

### `Roadmap`

Use this board only for large features and roadmap-sized efforts.

Lists:

- `Idea` - worth tracking, but not properly shaped yet
- `Design` - scope, tradeoffs, or design decisions are still being worked through
- `Ready for Implementation` - sufficiently defined and can be decomposed into backlog work
- `In Implementation` - active execution is in progress
- `Validation` - implementation is mostly complete and is being tested or polished
- `Blocked` - blocked at the roadmap level
- `Done` - finished and no longer needs active tracking

### `Backlog`

Use this board for all actual execution work.

Lists:

- `Inbox` - unprocessed notes, ideas, bugs, or candidate tasks
- `Triaged` - understood and classified, but not selected for immediate work
- `Ready` - clear, bounded, and actionable now
- `In Progress` - actively being worked
- `Blocked` - blocked at the task level
- `Done` - completed

## Labels

Use a small, predictable label set.

### Primary Work-Type Labels

Primary work-type labels describe why the work exists. Use a short fixed set such as `bug`,
`feature`, `tech-debt`, `maintenance`, `research`, `docs`, or `test`.

### Area Labels

Area labels describe the affected subsystem rather than the reason for the work. Use existing labels
such as `backend`, `frontend`, `retrieval`, `agents`, `database`, or `infra` when they help
routing and ownership. Area labels do not replace an applicable primary work-type label.

### Initiative Labels

Create one short label per roadmap initiative, epic, milestone, or feature group, for example `m01`,
`search`, or `frontend-v2`. Apply an initiative label only when a `Backlog` card belongs to a
specific roadmap item.

## Label Rules

- Every `Backlog` card must have one primary work-type label.
- Add only the area labels that materially improve classification; do not label every subsystem a
  task may incidentally touch.
- A `Backlog` card tied to a roadmap item must also have exactly one initiative label.
- `Roadmap` cards usually do not need primary work-type or area labels.
- Do not use labels for status such as `design`, `blocked`, `ready`, or `done`.
- Do not create ad hoc labels when an existing label already fits.

## Naming Conventions

### Roadmap Cards

Format:

```text
m03 - Retrieval Quality Foundation
```

Rules:

- Start with the initiative ID or short initiative key when one exists.
- Use title case after the ID.
- Keep the title stable over time unless the roadmap item scope changes materially.

### Backlog Cards

Format for roadmap-linked work:

```text
m03: implement hybrid retrieval in chat flow
```

Format for standalone work:

```text
bug: fix inventory sort regression
```

Rules:

- Use a short, action-oriented title.
- Prefer a verb-first phrase for implementation tasks.
- Keep one card scoped to one independently movable unit of work.

## When To Create Which Card

Create a `Roadmap` card when:

- the work is a large feature, epic, initiative, or milestone
- it spans multiple tasks or sessions
- it needs roadmap-level visibility
- it has lifecycle phases such as design, implementation, and validation

Create a `Backlog` card when:

- it is directly executable
- it can move through the task workflow independently
- it is a bug, routine task, small feature, research spike, refactor, cleanup, or test task

If unsure:

- if it needs decomposition, create or update a roadmap card
- if it can be picked up and done, create a backlog card

## Roadmap Card Template

Use this template for roadmap cards on the `Roadmap` board.

```md
## Goal
Short statement of the feature outcome.

## Why
Why this milestone exists and what problem it solves.

## Scope
- Included item 1
- Included item 2

## Non-goals
- Explicitly out of scope item 1
- Explicitly out of scope item 2

## Success Criteria
- Criterion 1
- Criterion 2
- Criterion 3

## References
- Related design doc, spec, issue, or user instruction

## Linked Backlog Work
- Add linked task card titles here as they are created

## Execution Status
Current summary in 3-6 lines.

## Next Gate
What must happen before this card moves to the next list.
```

## Backlog Card Template

Use this template for executable work on the `Backlog` board. Keep the sections that materially
help execution and omit empty boilerplate. Bugs and evidence-driven work should normally include
`Evidence / Reproduction`; routine tasks may omit it.

```md
## Context
What this task is, why it exists, and the confirmed current behavior.

## Evidence / Reproduction
- Exact reproduction, representative example, or investigation evidence
- Observed behavior versus expected behavior

## Scope
- Concrete change 1
- Concrete change 2

## Non-goals
- Explicitly excluded behavior or follow-up

## Acceptance
- Verifiable outcome 1
- Verifiable outcome 2

## Verification
- Automated test or check
- Essential manual check only when automation is insufficient

## Dependencies and Related Work
- Required approval, prerequisite, related card, or roadmap link

## References
- Related design doc, spec, issue, ticket, or user instruction
- Stable code path or authoritative documentation

## Notes
Optional constraints, hypotheses, implementation recommendations, or deferred observations.
```

## Agent Operating Rules

- Do not create a roadmap card for a small bug or maintenance task.
- Do not create a backlog card so large that it hides multiple independent tasks.
- Distinguish confirmed facts and repository evidence from hypotheses and proposed decisions.
- Keep acceptance criteria behavior-focused; put optional implementation approaches in `Notes`.
- Do not encode the same status in both the list and labels.
- Do not invent arbitrary initiative naming schemes if the user already has one.
- Do not rename roadmap cards away from their source-aligned names without explicit reason.
- Prefer updating an existing relevant card over creating a near-duplicate.
- Keep descriptions concise but sufficient for another engineer or agent to pick up the work.
- If the required board, list, or labels do not exist and the agent cannot create them directly, pause and ask the user to create them or approve an explicit exception.

## Minimum Required Data

For a new roadmap card:

- correct board
- correct list
- milestone-aligned title
- description using the roadmap template

For a new backlog card:

- correct board
- correct list
- action-oriented title
- one primary work-type label
- relevant area labels when useful
- one milestone label if linked to a milestone
- description using the backlog template

## Anti-Patterns

- one board per roadmap item
- one list per milestone
- labels used as workflow phase
- giant backlog cards that should be split
- small chores tracked on the roadmap board
- roadmap cards copied verbatim into backlog cards with the same scope
