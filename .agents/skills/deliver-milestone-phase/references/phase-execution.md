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

Verify that the repository trunk is `master`, then inspect remote branches and PRs before creating
anything. Use GitHub's official `gh stack` extension for branch tracking and remote stack
registration. See the
[stacked PR CLI command reference](https://docs.github.com/en/pull-requests/reference/stacked-prs-cli-commands).

- Check `gh stack --help` before mutating Git state. If the extension is unavailable, ask for
  permission to install it with `gh extension install github/gh-stack`; do not silently fall back
  to ordinary `git switch -c`, `git push`, or `gh pr create` as the stacking mechanism.
- Inspect registered local or remote stack state with `gh stack view --json` or
  `gh stack checkout <PR-number-or-branch>`. Also inspect PR head/base pairs independently; a valid
  branch chain may exist without being linked as a GitHub stack.

- The milestone branch is the long-lived integration branch. Prefer the established branch for the
  milestone; otherwise use the normalized milestone identifier, such as `m05`, based on current
  `master`.
- Every milestone has one long-lived draft PR whose head is the milestone branch and whose base is
  `master`. Reuse it when it exists. If another open PR already uses that head/base pair, treat it as
  the milestone PR after verifying its scope rather than creating a duplicate.
- If no local stack exists, initialize one with `gh stack init --base master <milestone-branch>`.
  When the complete branch chain already exists, adopt it from bottom to top with
  `gh stack init --base master <milestone-branch> <phase-1-branch> ...`; do not recreate branches.
- When the milestone branch is new and still identical to `master`, create one explicit empty
  kickoff commit on the milestone branch, such as `chore(m08): open milestone delivery`. Do not add
  a marker file or mix implementation into this commit. This one-time bootstrap provides the
  distinct commit history GitHub needs to compare the milestone branch with `master`.
- Title the milestone PR from the specification, such as `M08: Session Memory and Thread Control`.
  Its initial body identifies the milestone objective, planned phases and gates, current delivery
  status, and the phase-PR stacking convention. State clearly that it is an integration view and is
  not ready to merge.
- Ensure the topmost current stack branch is checked out, then create the requested phase branch
  with `gh stack add <branch-name>`. For Phase 1, the current top is the milestone branch. For a
  later phase, it is the immediately preceding unmerged phase branch; if prior phases have already
  merged, first use `gh stack sync` only with explicit authorization because it may cascade a
  rebase and force-with-lease push. Do not require prior phase PRs to merge before continuing the
  stack. Follow an established naming pattern for that milestone when one exists; otherwise use
  `<milestone>-phase-<number>-<short-phase-slug>`.
- If a matching branch or PR exists, verify that it represents the same phase and continue it
  instead of silently replacing it.

The resulting PR topology for multiple open phases is:

```text
master
└── milestone branch      milestone PR -> master
    └── phase 1 branch    Phase 1 PR -> milestone branch
        └── phase 2       Phase 2 PR -> Phase 1 branch
            └── ...
```

This is both the required branch topology and the required registered GitHub stack. Keep the
milestone and delivery branches in the same repository; GitHub stacks do not support cross-fork
branches. Do not merge or bulk-land the stack while delivering a numbered phase.

Do not start from another feature branch merely because it is currently checked out. Apart from the
one milestone kickoff commit, publish new branches only through the requested stack workflow.

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

Confirm that the milestone PR is still open with base `master`, or that this first submission will
create it. From any branch in the locally tracked stack, run `gh stack submit`. This pushes the
tracked branches, creates or updates their PRs with the correct bases, and links the PRs as one
GitHub stack. In its editor, keep the milestone PR draft and make the completed phase PR ready for
review. In a non-interactive environment, use `gh stack submit --auto`, then set the required PR
titles, bodies, and draft/readiness states explicitly. Never use `gh pr create` alone as a
substitute for stack submission. Use a phase title such as:

```text
M05 Phase 1: Prompt manifest and immutable registry
```

Write the PR body in plain English with these sections:

- `Summary`: what behavior or capability the phase adds
- `Completed work`: the implemented Planka cards
- `Verification`: commands and meaningful results, including the phase gate
- `Not verified`: manual or unavailable checks
- `Deviations and decisions`: approved divergences or `None`
- `Stack`: milestone PR link, preceding phase PR links, and the full branch chain to `master`
- `Next gate`: review this phase; the next phase may stack on its head without merging it

After submission, run `gh stack view --json` and independently inspect the PRs. Verify that GitHub
reports one linked stack in bottom-to-top order, every PR base matches the preceding branch, the
milestone PR targets `master` and remains draft, and the completed phase PR is ready for review. If
an older eligible PR chain was adopted but submission did not link it, use
`gh stack link --base master <milestone-PR-or-branch> <phase-1-PR-or-branch> ...` in bottom-to-top
order, then verify again. `gh stack link` is an adoption fallback, not the normal phase workflow.

Update the Roadmap card's execution status with the same concise outcome and the milestone/phase PR
links. Leave the Roadmap card in `In Implementation` while more phases remain. Hand off the phase
PR as implementation-complete, unintegrated, and awaiting review; do not merge any PR.
