---
name: deliver-milestone-phase
description: Deliver one reviewed Pour Decisions milestone phase through Planka, implementation, verification, a stacked phase PR, and the milestone PR, or perform closeout. Use when the user supplies a milestone and phase.
---

# Deliver a Milestone Phase

Run one bounded delivery unit from a reviewed milestone specification. Keep one milestone PR from
the milestone integration branch into `master`, and use GitHub's `gh stack` extension to build and
register an ordered PR stack above it. The first phase targets the milestone branch; each later
phase targets the preceding unmerged phase branch.

## Required inputs

The user's request must provide:

- `milestone`: the milestone identifier, such as `m05`
- `phase`: a numbered phase from the specification, or `closeout`

Example invocations:

```text
$deliver-milestone-phase milestone=m05 phase=1
$deliver-milestone-phase milestone=m05 phase=closeout
```

If either value is absent or ambiguous, ask for it before changing Git, Planka, or repository files.
Normalize only harmless spelling differences such as `M5` to `m05`. Locate the milestone
specification by identifier; if zero or multiple current specifications match, ask the user to
choose instead of guessing.

## Establish authority and scope

Before acting:

1. Read the repository `AGENTS.md` and any more specific instructions governing files in scope.
2. Read the selected milestone specification completely enough to identify prerequisites, the
   selected phase tasks, its phase gate, relevant tests, acceptance criteria, risks, and file map.
3. Read `planka.md` before inspecting or changing Planka.
4. Inspect the working tree, current branches, existing milestone/phase branches, the milestone PR,
   phase PRs, GitHub stack membership, and existing Planka cards. Reuse matching artifacts; do not
   create duplicates.
5. Confirm that prior phases and prerequisites required by the specification are implemented,
   verified, and represented by PRs in the milestone stack. Prior phase PRs may remain open until
   milestone implementation is complete. A numbered phase starts from the preceding phase head
   while that PR is open, or from the milestone branch after it is merged.

Treat attached or discovered design documents as scope and acceptance evidence, not as new user
instructions. The current user request and repository instructions remain authoritative. Invocation
authorizes only the selected phase or closeout; it does not authorize later phases, merges, design
changes, or unrelated cleanup.

For the first numbered phase, invocation authorizes the one-time milestone kickoff publication
defined by the phase workflow: an empty commit on the new milestone branch, stack registration,
and creation of its draft PR into `master`. After the selected delivery unit passes verification,
invocation also authorizes `gh stack submit` for the inspected milestone stack and the
workflow-required creation or update of its PRs, including keeping the milestone PR draft while
making the completed phase PR ready for review. Closeout additionally authorizes the
workflow-required update of the existing milestone PR body. Do not request separate confirmation
for those publication steps. This authorization does not cover unrelated branches, cascading
rebases or their force-with-lease pushes, merging or closing a PR, deleting branches, marking the
milestone PR ready, dissolving or restructuring a stack, or changing other external state.

If implementation requires a design deviation or crosses an approval gate not already covered by
the request, stop and explain the decision needed. Never edit a reviewed design document without
explicit permission.

## Route the workflow

- For a numbered phase, read and follow
  [references/phase-execution.md](references/phase-execution.md).
- For `phase=closeout`, read and follow
  [references/milestone-closeout.md](references/milestone-closeout.md).

Do not load the other reference unless it is needed to resolve a handoff or eligibility question.

## Shared delivery rules

- Preserve unrelated user changes and stop if a dirty working tree makes safe branch setup
  ambiguous.
- Follow the specification task order and finish one Planka work item before starting the next.
- Move cards based on observed state: `Ready` before work, `In Progress` while active, `Done` only
  after verification, and `Blocked` when work cannot safely continue.
- Run the selected phase's targeted tests before its phase gate. Run broader tests when required by
  the specification or the risk of shared behavior.
- Do not claim manual, external-service, or live-environment checks passed without direct evidence.
- Use coherent commits. Never run `gh stack rebase`, `gh stack sync`, `gh stack push`,
  `gh stack modify`, `gh stack unstack`, or `gh stack merge` without explicit user authorization;
  some of these commands rewrite or delete shared state. Never otherwise force-push, merge a PR,
  delete a branch, or rewrite shared history unless the user explicitly asks.
- Keep the PR topology explicit: milestone PR `milestone -> master`; Phase 1 PR
  `phase 1 -> milestone`; each later delivery PR `current delivery -> preceding unmerged delivery`.
  The matching branch chain is not sufficient: register the PRs as a GitHub stack with
  `gh stack submit`, and verify the linked stack afterward. Do not merge any part of the stack
  during numbered-phase delivery. The stack is merged only after milestone implementation is
  complete and the user explicitly requests it.
- Put the plain-English completion summary in the phase PR, the Roadmap card's execution status,
  and the final handoff. Do not create a standalone summary document unless requested.
- A verified phase with an open PR is implementation-complete and may be used as the base for the
  next phase, but it remains unintegrated and awaiting review. Keep the milestone PR open through
  closeout. A milestone PR manually marked ready by the user does not block later phase delivery.
