# Milestone Design Audit

Perform this stage without editing repository files or external project-tracking systems.

## 1. Reconstruct what changed

Start from the current released tree and work backward only as needed.

- Map each prerequisite milestone to its release, delivered interfaces, behavior, and known
  limitations.
- Compare the candidate milestone's baseline assumptions with current code, tests, configuration,
  prompts, schemas, dependencies, tracing, eval artifacts, and maintained documentation.
- Identify work already delivered, superseded, made unnecessary, or made more complicated by prior
  milestones.
- Distinguish a stale specification from a still-valid decision whose wording is merely old.

Do not treat a prior milestone document as proof that its implementation matches the current tree.

## 2. Challenge the milestone on three axes

### Relevance

Determine whether the original objective still addresses a real project gap.

- Is the problem still present and important in current behavior?
- Does the milestone duplicate capability now delivered elsewhere?
- Is it still in the approved implementation order, and are its prerequisites satisfied?
- Have current tools or techniques made the approach obsolete or offered a materially simpler path?
- Would implementing it improve a measured outcome, or only add architectural novelty?

Be willing to recommend `retain`, `narrow`, `redesign`, `defer`, `merge into another milestone`, or
`retire`.

### Freshness

Verify the document at the level needed for implementation:

- project version, verification date, status, prerequisites, and release references
- architecture and data-flow descriptions
- file paths, modules, classes, functions, models, schemas, configuration keys, and defaults
- library/API/model assumptions and supported behavior
- scope, non-goals, technical decisions, implementation phases, tests, commands, and file map
- acceptance criteria, risks, rollout assumptions, and delivery checkpoints

Flag vague future tense, historical commentary presented as current fact, obsolete alternatives,
and phase steps that no longer follow from the current architecture.

### Usefulness versus resources

Challenge the design proportionally to its expected value:

- implementation and review effort
- new dependencies, services, infrastructure, migrations, schemas, configuration, or UI work
- added LLM, embedding, retrieval, storage, network, or observability cost
- ongoing maintenance, debugging, operational, privacy, and failure-recovery burden
- test complexity and ability to prove the promised outcome
- opportunity cost relative to the roadmap's current evidence gates

For each complicated mechanism, ask what current component can be reused and what the smallest
useful version would be. Reject complexity justified only by hypothetical future needs.

## 3. Classify findings

Give every finding a stable ID and classify it as one of:

- `FACT`: demonstrably stale or incorrect repository fact
- `ASSUMPTION`: important claim lacking evidence
- `DECISION`: user-owned choice that changes design direction or scope
- `SIMPLIFY`: avoidable complexity or duplicate mechanism
- `GAP`: missing implementation detail, test, acceptance proof, or failure handling
- `REMOVE`: obsolete or no-longer-useful content/work

Rate impact as `blocking`, `important`, or `minor`. A stylistic preference is not a design finding.

## 4. Produce the audit report

Lead with one readiness verdict:

- `ready as written`
- `ready after factual refresh`
- `approval required for targeted changes`
- `material redesign required`
- `defer or retire`

Then report:

1. **Baseline checked** — project version, branch/release evidence, prerequisite releases, and the
   main current artifacts inspected.
2. **What remains valuable** — objective and decisions worth preserving.
3. **Findings** — ID, category, impact, current evidence, consequence, and proposed resolution.
4. **Decision requests** — options, tradeoffs, risks, and a clear recommendation for every key
   decision. Include the option to keep the current design when credible.
5. **Proposed edit map** — document sections that would change, be removed, or be added.
6. **Readiness blockers** — unresolved decisions, prerequisites, evidence, or external checks.
7. **Approval request** — distinguish permission to apply factual refreshes from approval of named
   decision outcomes.

Do not rewrite the document in the report. Give enough detail for the user to make decisions without
hiding the consequences, then stop for approval.

