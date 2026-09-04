# Milestone Closeout

Use this workflow only for `phase=closeout`. Closeout is a delivery unit, not permission to repair
unrelated defects or redesign the milestone.

## 1. Verify closeout eligibility

Read the entire milestone specification and confirm:

- every numbered implementation phase and phase gate is complete
- all phase PRs are merged into the milestone branch
- no milestone Backlog card remains `Ready`, `In Progress`, or `Blocked`
- the milestone branch contains the expected implementation and has no unresolved design deviation

If any condition fails, report the exact missing phase, PR, card, or decision and stop closeout.
Move the Roadmap card to `Validation` only when implementation is genuinely complete.

## 2. Create closeout work and branch

Create or reuse bounded Backlog cards for the closeout work described by the specification. Keep
acceptance verification, documentation, and release/version work separate when they can be reviewed
or retried independently.

Create a closeout branch from the current milestone branch. Follow an existing milestone naming
pattern or default to `<milestone>-closeout`. The closeout PR targets the milestone branch.

## 3. Prove acceptance

Build an acceptance matrix from every milestone acceptance criterion:

```text
criterion | status | evidence | remaining action
```

- Inspect the final implementation, not only phase summaries.
- Run the regression commands required by the specification and repository policy.
- Perform required live or manual checks only when the environment and authorization are available.
- Mark unavailable checks as unverified. Never replace required evidence with an assumption.
- Fix only defects clearly within the approved milestone design. Stop for approval if a fix changes
  architecture, contracts, prompts, defaults, dependencies, migrations, frontend direction, or the
  reviewed design.

Do not proceed to release work while a required acceptance criterion is failing.

## 4. Update documentation

Update maintained repository documentation affected by the delivered behavior and keep technical
claims aligned with verified implementation. Do not add committed links or citations to local-only
milestone documents.

The milestone specification is a reviewed artifact. Update its completion evidence, status,
project version, and verification date only after the user explicitly permits editing it. If that
permission is absent, report the exact pending design-document update instead of silently changing
it.

## 5. Choose and apply the version bump

Read the current version from `pyproject.toml`; it is the source of truth.

- Use a patch bump for internal stabilization, reliability, observability, documentation, or other
  backward-compatible corrections without a new public capability.
- Use a minor bump for a backward-compatible user-facing capability or intentional public contract
  expansion.
- If the milestone does not fit clearly, propose patch versus minor with the tradeoff and obtain the
  user's decision before editing versions.

Update maintained version declarations consistently and keep `last updated` or `last verified`
dates separate from the semantic version. Regenerate existing lock metadata with repository tooling
when required, without changing dependencies.

## 6. Raise closeout and integration PRs

Open or update the closeout PR into the milestone branch. Its plain-English body must include:

- milestone outcome
- acceptance matrix summary
- automated and manual verification
- documentation updated and intentionally deferred
- old and new project versions with the bump rationale
- unresolved risks or `None`

Open or refresh a draft milestone integration PR from the milestone branch into the repository's
default branch. The draft will update after the closeout PR is merged. Do not merge either PR.

Update the Roadmap execution status and set the next gate to merging closeout, then completing
review of the milestone integration PR. Keep the Roadmap card in `Validation`; move it to `Done`
only after the milestone PR is merged and all required evidence is present.

