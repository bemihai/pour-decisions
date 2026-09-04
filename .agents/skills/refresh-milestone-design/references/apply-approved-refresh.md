# Apply an Approved Milestone Refresh

Use this stage only after the user explicitly authorizes design-document editing and resolves every
blocking decision that the approved refresh depends on.

## 1. Freeze the approved change set

Restate the approved factual refreshes and decision IDs before editing. Exclude recommendations the
user rejected, deferred, or did not address.

Recheck the project version, working tree, target document, and relevant repository artifacts. If
the baseline changed since the audit, determine whether the approved conclusions still hold. Stop
for another decision when new evidence materially changes scope, architecture, cost, contracts,
dependencies, defaults, or acceptance.

## 2. Refresh the document coherently

Edit the existing milestone specification in place. Preserve useful rationale, but make every
section describe one consistent current design.

As applicable, update:

- the exact current project version from `pyproject.toml`, plus a separate current verification date
- milestone status and implementation readiness
- current baseline and prerequisite release state
- goal, motivation, scope, and non-goals
- technical decisions and rejected alternatives
- interfaces, data flow, configuration, and failure behavior
- phased implementation order and phase gates
- test plan, runnable validation commands, and measurable acceptance criteria
- new/modified file map, risks, rollout, cost, and delivery checkpoints

Remove superseded steps and historical assertions that no longer help implementation. Do not leave
both old and new approaches as if they were simultaneously authoritative. Preserve explicit
non-goals and rejected alternatives when they prevent likely scope creep.

This skill refreshes the design against the current version; it does not increment the version.

## 3. Make the implementation plan executable

Confirm that:

- each phase is bounded, ordered by real dependencies, and independently verifiable
- each phase gate states observable evidence required before the next phase
- file and symbol references exist or are clearly marked as planned additions
- commands use current repository entry points and valid test paths
- acceptance criteria prove the objective rather than merely proving files were added
- cost and reliability claims include how they will be measured
- optional or evidence-gated work is not presented as mandatory foundation
- approval-gated changes are called out explicitly rather than buried in implementation steps

Do not add implementation details unsupported by current evidence solely to make the document look
complete. Record a bounded open question or decision gate instead.

## 4. Verify the refreshed design

Perform read-only checks proportional to the document:

- search every referenced existing path, symbol, config key, command, and dependency
- compare project-version declarations and status language for internal consistency
- cross-check the roadmap order and prerequisites
- check that scope, phases, tests, acceptance criteria, risks, and file map agree
- verify current external claims against the primary sources used in the approved audit
- inspect the diff for accidental implementation, unrelated documentation edits, or unapproved
  decisions

Do not run the milestone's implementation test suite merely to validate a design edit unless a
specific current-state claim requires it.

## 5. Hand off readiness

Report:

- sections changed and stale content removed
- approved decisions encoded, with their IDs
- factual refreshes applied
- checks performed and claims not independently verified
- remaining open questions or deferred work
- final verdict: `ready for implementation` or `not ready`, with concrete blockers

Do not create implementation branches, Planka cards, or phase PRs. Those belong to the delivery
workflow after the refreshed design receives implementation approval.

