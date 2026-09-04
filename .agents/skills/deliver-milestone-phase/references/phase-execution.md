# Numbered Phase Execution

Use this workflow for a numbered implementation phase.

## 1. Prepare milestone tracking

Apply the board, list, label, naming, and description rules in `planka.md`.

- Find or create one Roadmap card for the milestone. Move it to `In Implementation` when phase work
  begins, unless its current state correctly represents a blocker or later lifecycle stage.
- Extract the selected phase's tasks from the specification and turn them into bounded Backlog
  cards before coding. Prefer one card per independently verifiable work item; combine adjacent
  specification steps only when they cannot be delivered or verified independently.
- Put new actionable cards in `Ready`. Give each exactly one milestone initiative label and at least
  one fitting work-type label.
- Link the Backlog card titles from the Roadmap card and set its next gate to the selected phase
  gate.

If required Planka structures or labels are missing and cannot be created safely, follow the stop
rule in `planka.md`.

## 2. Prepare Git branches

Resolve the repository's default branch and inspect remote branches before creating anything.

- The milestone branch is the long-lived integration branch. Prefer the established branch for the
  milestone; otherwise use the normalized milestone identifier, such as `m05`, based on the current
  default branch.
- Create the phase branch from the current milestone branch. Follow an established naming pattern
  for that milestone when one exists; otherwise use
  `<milestone>-phase-<number>-<short-phase-slug>`.
- If a matching branch or PR exists, verify that it represents the same phase and continue it
  instead of silently replacing it.

Do not start from another feature branch merely because it is currently checked out. Push new
branches only when needed for the requested PR workflow.

## 3. Implement cards sequentially

For each selected-phase Backlog card, in specification order:

1. Move the card to `In Progress`.
2. Re-read the card scope and the corresponding specification task.
3. Implement only that bounded task and its tests. Do not pull work forward from a later phase.
4. Run the narrowest meaningful validation for the task and fix in-scope failures.
5. Record useful verification evidence on the card, then move it to `Done` only when its acceptance
   conditions pass.

If a task is blocked, move it to `Blocked`, update the Roadmap execution status and next gate, and
stop. Do not mark downstream cards complete.

## 4. Verify the phase gate

After all selected-phase cards pass:

- Run the exact phase-gate checks and targeted tests named by the specification.
- Run additional relevant tests required by repository policy or shared-code risk.
- Compare the result with the selected phase scope and file map. Flag out-of-scope changes and
  unresolved deviations.
- Confirm that all phase cards have evidence matching their acceptance conditions.

Do not update the milestone specification to record completion unless the user explicitly permits
editing that reviewed artifact.

## 5. Raise the phase PR

Push the phase branch and open or update a PR whose base is the milestone branch, never the default
branch. Use a title such as:

```text
M05 Phase 1: Prompt manifest and immutable registry
```

Write the PR body in plain English with these sections:

- `Summary`: what behavior or capability the phase adds
- `Completed work`: the implemented Planka cards
- `Verification`: commands and meaningful results, including the phase gate
- `Not verified`: manual or unavailable checks
- `Deviations and decisions`: approved divergences or `None`
- `Next gate`: review and merge into the milestone branch before the next phase starts

Update the Roadmap card's execution status with the same concise outcome and PR link. Leave the
Roadmap card in `In Implementation` while more phases remain. Hand off the PR as awaiting review;
do not merge it.

