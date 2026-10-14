---
name: deliver-planka-card
description: Deliver one refined Pour Decisions Backlog card from Ready through implementation, verification, and a single pull request. Use for an isolated task, not milestone phases or cards that still need refinement or design.
---

# Deliver a Planka Card

Implement one bounded, already-refined Backlog card as an ordinary pull request into `master`.
This workflow is for independently deliverable bugs, routine tasks, small features, and technical
spikes. It does not create a milestone stack or coordinate multiple cards.

## Required input

The user must identify the Planka card by its current name, for example:

```text
$deliver-planka-card card="bug: fix inventory sort regression"
```

Accept an equivalent natural-language request when the card name is unambiguous. If the name is
missing, ask for it before changing Planka, Git, GitHub, or repository files.

## Establish authority and eligibility

Before acting:

1. Read the repository `AGENTS.md`, `planka.md`, and any more specific instructions governing the
   files in scope.
2. Find exactly one matching card on the Planka `Backlog` board using the project's dedicated
   Planka MCP server. Read its full description, labels, comments, attachments, dependencies, and
   related work. Do not use browser automation or another integration as a fallback.
3. Confirm the card is in `Ready`, is one bounded executable unit, has observable acceptance and
   verification criteria, and records any required gated decisions or approvals.
4. Inspect the working tree, current and remote branches, existing pull requests, and the current
   `master` baseline. Reuse an existing branch or PR only after verifying that it represents the
   same card and scope.
5. Confirm recorded prerequisites are complete and that the card does not depend on unmerged work
   outside its branch.

If the card is not uniquely identifiable or is not in `Ready`, stop without mutating it and report
what must be resolved. Do not invoke refinement automatically. If it is roadmap-sized, requires
multiple independently movable tasks, or depends on a milestone branch or another unmerged PR,
route it to the appropriate refinement or milestone workflow rather than forcing it through this
skill.

Treat the refined card as the implementation specification. The invocation authorizes its bounded
scope, including gated changes explicitly identified and approved on the card. It does not authorize
unrecorded architecture, design, prompt, API, schema, configuration-default, dependency, migration,
or design-document changes. Stop for a material ambiguity, a missing approval, or a required scope
change; never edit a reviewed design document without explicit permission.

Invocation authorizes reading and updating only the selected card, creating or reusing one task
branch, implementing and committing the card, and—after successful verification—pushing that branch
and creating or updating one ready-for-review pull request into `master`. Do not mutate linked
Roadmap cards or other Backlog cards. Do not merge or close a PR, delete branches, force-push,
rewrite shared history, change unrelated external state, or perform destructive recovery without
separate explicit authorization.

## Execute the card

Read and follow [references/card-execution.md](references/card-execution.md).

## Delivery rules

- Preserve unrelated user changes. Stop if a dirty working tree makes safe branch setup or scoped
  commits ambiguous.
- Use the current card title, description, and acceptance criteria without silently expanding or
  redesigning them. Record newly discovered follow-up work in the handoff; do not create another
  card unless the user asks.
- Move the card based on observed state: `Ready` before work, `In Progress` while delivery is active,
  `Done` only after implementation, required verification, PR publication, and read-back checks all
  succeed, and `Blocked` only when an actual unresolved blocker prevents safe continuation.
- Run the card's specified checks plus tests proportionate to the touched behavior and shared-code
  risk. Do not claim manual, external-service, or live-environment checks passed without direct
  evidence.
- Keep the branch and commit history limited to this card. Do not include unrelated working-tree
  changes or opportunistic cleanup.
- Put the same plain-English outcome and verification evidence in the card, PR, and final handoff.
  Do not create a standalone summary document.
- A card delivered with an open PR is implementation-complete but unintegrated and awaiting review.
  Never represent the PR as merged or the behavior as released.
