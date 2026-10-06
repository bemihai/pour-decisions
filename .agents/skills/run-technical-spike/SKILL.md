---
name: run-technical-spike
description: Run a bounded experiment to validate a risky Pour Decisions technical assumption before design or implementation, with reproducible evidence and an explicit conclusion.
---

# Run a Technical Spike

Answer one falsifiable technical question using the smallest useful experiment. A successful spike
can disprove an approach or show that evidence is inconclusive; it need not produce a viable feature.

## Inputs and experiment contract

```text
$run-technical-spike assumption="reranking meets our latency budget" constraints="CPU only"
```

Read `AGENTS.md` and relevant directory instructions. Establish the assumption, representative input,
environment, measurable success threshold, and relevant time, call, or cost limits before execution.
Use values supplied by the user or established project requirements. When a material threshold or
resource limit is missing, propose a concrete experiment and ask the user to settle it. Do not choose
an arbitrary threshold and then claim the assumption is proven.

Inspect the current implementation, dependencies, configuration, and tests first, using SigMap when
available. Existing evidence may answer the question without new code. State that conclusion and
its limits if so.

## Scope and permissions

`AGENTS.md` remains authoritative, including style, logging, sensitive-file, and approval rules.
Prototype code is small, but still uses the required type hints, docstrings, and explicit control flow.
Implement only the experiment, not production integration or design changes.

Use installed dependencies, disposable fixtures, and an isolated temporary directory by default.
Do not change project dependency files, environment files, persistent application data, or running
services as an incidental setup step. Dependency changes and external services require the approval
specified in `AGENTS.md`; invocation does not grant it. Resolve resource access before execution.

## Execute and evaluate

1. Describe the experiment briefly: what is varied, what is measured, what remains fixed, and what
   result would support or reject the assumption. State the stopping limit.
2. Build only the harness needed to measure it. Reuse existing project interfaces where practical.
   Keep fixtures synthetic or user-approved and avoid copying sensitive data into artifacts.
3. Run the experiment within the agreed limits. For performance claims, distinguish setup or cold
   start from steady state and collect enough observations to expose meaningful variability.
   Record commands, environment details, input characteristics, and results needed for reproduction.
4. Investigate surprising results within scope. If additional work would exceed the experiment's
   budget, require a new service, or test a different assumption, stop and explain the next decision.
5. Compare observed results against the agreed threshold. Separate measurements from extrapolation;
   success on a fixture does not establish production performance or reliability.

## Handoff

Report a conclusion of supported, disproved, or inconclusive, with the evidence, limitations,
reproduction steps, and recommended next decision. Identify artifacts retained and any side effects.
Provide the report in the conversation unless the user requested a saved memo.

Keep useful temporary artifacts available for inspection and report their location. Remove them only
when cleanup is requested or was explicitly included in the experiment contract, using exact paths.
Do not promote the prototype to production, add packages permanently, or update a reviewed design
without the corresponding approval. A follow-up design may use the evidence but still requires its
own decisions.
