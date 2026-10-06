---
name: research-feature-strategy
description: Investigate a proposed Pour Decisions feature and compare strategic options before detailed design. Use for new feature research or a high-level roadmap proposal.
---

# Research a Feature Strategy

Develop an evidence-backed recommendation about whether and how to pursue a feature. Keep the
outcome at strategy level: motivation, options, tradeoffs, risks, and a proposed sequence of outcomes.

## Inputs

```text
$research-feature-strategy feature="cellar recommendations" scope="reduce unnecessary model calls"
```

Require a feature or problem statement; scope is optional. Ask about missing goals or constraints
when they materially change the investigation. Use answers already provided in the conversation.

## Authority and scope

Read `AGENTS.md` and relevant directory instructions. They remain authoritative over this skill.
Research recommendations are proposals, not approved architecture or permission to implement.
Do not edit code, publish tracking items, or change reviewed documents as part of research.

Use repository evidence first. External browsing, integrations, APIs, and paid experiments require
the current task's explicit authorization under `AGENTS.md`. When authorized, verify consequential
claims about current techniques or services using primary documentation. Otherwise state which
claims remain unverified and seek approval only when external evidence is necessary to proceed.

## Investigation

1. Establish the user problem, intended users, success measures, and meaningful non-goals.
2. Use SigMap when available to locate current capabilities; inspect the relevant implementation,
   tests, configuration, and task-relevant planning artifacts. Separate implemented behavior from
   proposals. Look for reuse and overlap before proposing new components.
3. Compare a small set of credible approaches, including retaining or extending existing behavior
   when viable. Evaluate cost, maintainability, reliability, learning value, and modernity in the
   order defined by `AGENTS.md`. Account for inference calls, caching, local retrieval, storage,
   operational effort, and service costs where relevant. Respect the current inference policy.
4. Explain effects on affected subsystems and interfaces, without prescribing implementation code.
   Identify risky assumptions and the smallest experiments that could settle them; do not run
   experiments automatically as part of research.
5. Propose a high-level sequence with outcomes, dependencies, and decision gates. Distinguish
   user decisions from facts that can be established through investigation.

## Deliverable and stopping point

Present the problem, baseline evidence, options and tradeoffs, recommendation, success measures,
proposed sequence, and unresolved decisions. Cite actual evidence and label estimates or assumptions.
Do not present an unresolved option as an approved decision.

Provide the result in the conversation unless the user requests a saved roadmap. Resolve the target
path before saving, preserve existing documents, and follow the design approval and local-only
artifact rules in `AGENTS.md`. A request to research does not authorize creating a design file.
If research changes the scope or reveals a material decision, stop at that decision with concrete
options. Detailed specification and implementation are separate tasks.
