---
name: create-milestone-design
description: Draft a new Pour Decisions milestone specification from an agreed objective or roadmap item, resolving design decisions and defining delivery phases and acceptance tests.
---

# Create a Milestone Design

Translate an agreed objective into a specification that a coding agent can implement one phase at a
time. For auditing or refreshing an existing milestone specification, use `refresh-milestone-design`.

## Inputs

```text
$create-milestone-design milestone=m15 objective="agreed feature outcome"
```

Accept a named feature instead of a milestone ID and a supplied roadmap item instead of an objective.
Do not invent initiative identifiers. Resolve missing or ambiguous scope with the user. Reuse decisions
already approved in the conversation; do not ask for them again.

## Authority and baseline

Read `AGENTS.md`, applicable directory instructions, and the selected roadmap material. Repository
policy remains authoritative. Inspect current implementation and tests using SigMap-assisted discovery;
do not rely on a roadmap's description of delivered behavior. Check prerequisites and existing
specifications to avoid duplicate designs. If a matching specification exists, report it and resolve
whether the user intends a refresh before writing anything.

This workflow does not authorize production code, dependency installation, migrations, tracking
updates, branches, or PRs. External research requires the current task's authorization. Design files
remain local-only; do not add references to them in committed artifacts.

## Resolve design choices

Identify choices affecting scope, architecture, interfaces, schemas, prompts, defaults, dependencies,
and frontend behavior. Present concrete options, project-specific tradeoffs, and a recommendation
using the priority order in `AGENTS.md`. Distinguish existing constraints from new proposals.

Draft enough detail in the conversation for the user to review each material choice. Ask focused
questions and continue across turns until the necessary decisions are resolved. Do not silently
select libraries, schema changes, or architecture. If a risky assumption needs empirical validation,
propose a bounded technical spike and keep dependent design choices open until evidence is available.

## Specification content

Use the repository's established document structure where appropriate. Include the information
needed for this milestone without filling irrelevant sections:

- Objective, motivation, scope, non-goals, and the verified current baseline.
- Approved decisions, alternatives rejected with reasons, and applicable cost constraints.
- Component responsibilities, explicit data flow, affected contracts, and failure behavior.
- Prerequisites and ordered implementation phases. Give each phase a bounded outcome, affected
  files or symbols, test expectations, and an observable completion gate.
- Acceptance criteria tied to user-visible or operational outcomes. Map them to specific unit,
  integration, evaluation, or manual scenarios, including relevant boundary and failure cases.
- Validation commands grounded in repository configuration. Distinguish existing tests from
  tests to create and identify external prerequisites or costs; do not claim checks were executed.
- Relevant rollout, compatibility, migration, rollback, performance, and operational considerations.
- Risks and unresolved questions, each with its impact and required next decision or evidence.

Prefer phases that can be verified independently. Make unavoidable ordering and intermediate
limitations explicit. Avoid acceptance criteria that merely require adding a file or class.
Use the exact project version from `pyproject.toml` and a separate date for versioned documents.

## Review, save, and handoff

Check consistency across scope, data flow, contracts, phases, file map, acceptance criteria, and
tests. Verify referenced repository paths and commands. Every acceptance criterion should have a
credible verification method; every prerequisite should be satisfied or explicitly gated.

Show the proposed specification and material decisions before requesting any still-missing approval.
Saving requires an explicit request or approval to create the design at a resolved location; a
request to create a new specification covers that new file once the location is unambiguous, but
does not approve unresolved architecture or overwriting an existing reviewed design.

After saving, read the document back and report its location and readiness. If material decisions,
dependencies, or unverified assumptions remain, label it a draft and state the next gate. Do not
start implementation or silently hand off to a delivery skill.
