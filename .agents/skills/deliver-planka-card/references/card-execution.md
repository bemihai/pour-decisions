# Single-Card Execution

Use this workflow for one eligible Backlog card in `Ready`.

## 1. Prepare the branch

Verify that the repository trunk is `master` and inspect remote freshness, existing branches, and
open pull requests before creating anything.

- Start from the current remote `master`, not from whichever feature branch happens to be checked
  out. If updating the baseline would overwrite, mix with, or strand user changes, stop and report
  the conflict.
- Reuse a matching branch or PR only when its head, base, commits, and stated scope represent this
  exact card. Do not create duplicates or silently replace existing work.
- When no matching branch exists, create one from the verified `master` baseline. Follow an
  established standalone-task naming pattern when one exists; otherwise use
  `<work-type>/<short-card-slug>`, such as `bug/fix-inventory-sort-regression`.
- Keep the branch outside any milestone stack. If the card needs an unmerged milestone or feature
  branch as its base, it is not eligible for this workflow.

## 2. Implement the card

After safe branch setup, move the card from `Ready` to `In Progress`, re-read its specification, and
implement only its recorded scope.

1. Use repository-grounded discovery before editing unfamiliar code. Follow the repository's
   SigMap instructions when applicable, then read the identified implementation and relevant tests.
2. Implement the smallest complete change that satisfies the acceptance criteria. Preserve existing
   architecture and patterns unless the card explicitly contains the required approved decision.
3. Add or update tests required by the card and by the risk of the changed behavior.
4. Run the narrowest meaningful checks during iteration and fix failures caused by the change.
5. Review the working-tree diff for scope, sensitive-file safety, accidental generated files, and
   unrelated edits before committing.

If new evidence invalidates the card's approved behavior or requires a gated change not recorded on
the card, stop before making that change. Update the card to `Blocked` only when the issue is an
actual implementation blocker; otherwise leave it `In Progress` and explain the decision required.

## 3. Verify and commit

Run every check listed in the card's `Verification` section, then add broader targeted checks when
repository policy or shared-code risk requires them. Compare the completed behavior directly with
each acceptance criterion and record:

- commands or checks run and their meaningful results
- acceptance criteria satisfied
- manual, external, or environment-dependent checks not performed
- approved deviations, or `None`

Do not weaken tests, change acceptance criteria, or reinterpret failures merely to complete the
card. Report unrelated failures without broadening scope.

Create coherent commit or commits containing only the card's work. Use the repository's established
commit style and verify the committed diff against `master` before publication.

## 4. Publish one pull request

Push the task branch and create or update one non-draft pull request whose base is `master`. Use the
GitHub CLI rather than browser automation. If a matching PR exists, preserve useful discussion and
update it instead of creating another.

Use the refined card title as the PR title unless a minor adjustment is needed to match repository
style. Write the body in plain English with:

- `Summary`: the user-visible or engineering outcome
- `Completed work`: the bounded changes made for the card
- `Acceptance`: how the implementation satisfies each criterion
- `Verification`: commands and meaningful results
- `Not verified`: manual or unavailable checks, or `None`
- `Deviations and decisions`: approved divergences, or `None`
- `Planka`: the selected card's title and link or stable identifier

After publication, inspect the remote PR and verify its head branch, `master` base, ready-for-review
state, title, body, and checks visible at that time. Do not merge it.

## 5. Complete tracking and handoff

Update only the selected Planka card with a concise outcome, verification evidence, commit and PR
links, unverified checks, and any deferred observations. Preserve the refined specification rather
than replacing it with a completion note. Move the card to `Done`, then read it back to verify its
board, list, title, description, labels, and completion evidence.

If PR publication or final Planka verification fails, do not mark the card `Done`. Leave it `In
Progress` unless an unresolved blocker justifies `Blocked`, record the exact state when possible,
and hand off the remaining action without retrying destructive or duplicate-producing operations.

Report the card, branch, commit or commits, PR, verification performed, unverified checks, and any
deferred follow-up. State explicitly that the result is implementation-complete, unintegrated, and
awaiting review.
