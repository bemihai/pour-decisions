# AGENTS.md

> **Project version**: 0.12.0 — last updated 2026-10-06.
> Authoritative repository policy; consult task-specific references for architecture details.

## Project Overview

Pour Decisions is a RAG-powered wine chatbot with cellar management. **Cost minimization is the #1 architectural constraint** - prefer local retrieval models, free-tier services, caching, and batching where they meet the need. Generative inference uses direct Ollama Cloud only; local embeddings and reranking remain supported.

## Collaboration Policy

- The user retains **100% ownership** over architecture, structure, and design decisions.
- This repository is a **learning lab**. Favor explicit, understandable code and clear control flow over clever abstractions, hidden behavior, or convenience magic.
- **Do not** make architecture, structure, or design changes without explicit approval.
- **Do not use external integrations, third-party services, browser automation, or external APIs
  without explicit user approval for that integration in the current task.** Prefer the project's
  dedicated MCP server when it supports the requested workflow; browser automation is not an
  implicit substitute.
- For **medium or large changes**, present a short plan before implementation even when the direction appears obvious.
- For **small, localized tasks**, implement directly while remaining within established patterns and approved boundaries.
- Surface assumptions, identify tradeoffs, and do not silently broaden scope.
- Repository skills and tool-specific prompts must follow `AGENTS.md`. If a proposed rule or skill
  conflicts with it, explain the conflict and obtain explicit user approval before changing the policy.
- When behavior depends on unfamiliar wine-domain facts, verify them against repository evidence or
  an approved source before encoding them. External research still requires the approval described above.

## Approval Gates

The following changes require explicit approval before implementation:

- architecture, structure, or design changes
- prompt changes
- API or schema contract changes
- config default changes
- dependency additions or removals
- database migrations or migration edits
- design doc updates
- major frontend refactors

Approval of a bounded plan or specification authorizes the changes explicitly described in it,
including identified gated changes. Do not ask for the same approval again. Request approval when
new evidence requires a material deviation, additional access, or expanded scope. Sensitive-file
operations still require approval for the exact operation.

If a task crosses a gate not already covered by approval, stop, explain why, and request approval
before proceeding.

## Sensitive Local Files

- Treat local secret and credential files as user-owned, sensitive state. This includes `.env`,
  `.env.local`, `.env.*.local`, private-key files, credential exports, and equivalent untracked
  configuration. `.env.example` is a public template and is not covered by this sensitive-file
  rule.
- **Do not create, edit, overwrite, format, rename, copy, move, or delete a sensitive local file
  unless the user explicitly approves that exact file operation in the current task.** General
  permission to implement a feature, change configuration, run tests, or invoke a skill does not
  authorize changes to sensitive local files.
- Do not directly inspect, print, log, or expose sensitive-file contents without explicit user
  approval. Normal project commands may load these files through the application's established
  environment-loading path, but that does not authorize displaying their values or modifying the
  files.
- Prefer checked-in configuration, command-scoped environment overrides, test fixtures, or changes
  to `.env.example` when documentation or test setup needs an environment value.
- Before running a formatter, generator, setup command, or script that may write configuration,
  confirm that it cannot modify sensitive local files. If a tool unexpectedly changes one, stop,
  report the exact file, and ask the user how to proceed; do not silently keep or revert the change.

## Decision Priorities

When evaluating alternatives, use this priority order:

1. **Cost-consciousness** - default to low-cost solutions, but paid options are acceptable when they are clearly the best choice.
2. **Maintainability** - prefer code that is easy to read, extend, and debug.
3. **Reliability** - favor predictable behavior and explicit failure handling.
4. **Learning value** - prefer implementations that help the user understand the system and the tradeoffs.
5. **Modernity** - prefer current, well-supported approaches over legacy or outdated patterns.

If these priorities conflict, explain the tradeoff explicitly. Do not resolve the conflict through an unstated assumption.

## Agent Checklist

Before starting work:

1. Read this file and any task-relevant design docs.
2. Identify whether the request is a small localized task or a medium/large change.
3. Check whether the task crosses any approval gate.
4. Check whether existing approval covers the proposed changes; ask only for uncovered decisions.

### Supplemental Instructions

- Load `planka.md` only when the user asks for Planka work, card creation, board organization, backlog management, or closely related project-tracking tasks. Do not load it by default for ordinary coding work.

### SigMap-Assisted Code Discovery

- Use SigMap when available to reduce unnecessary source-file reads.
- Before answering questions about unfamiliar code, run `sigmap ask "<question>"`.
- Before editing shared code, run `sigmap --impact <file>` and, when relevant,
  `sigmap --callers <symbol>`.
- Use `sigmap --query "<topic>"` to locate relevant files and symbols.
- Treat SigMap as a discovery index. Read the identified implementation and relevant tests before
  editing or making definitive behavioral claims.
- Run `sigmap validate` after changing source directories or SigMap configuration.

While implementing:

1. Stay within the approved scope.
2. Prefer explicit code paths, clear naming, and understandable control flow.
3. Avoid implicit fallbacks, hidden side effects, and unnecessary abstraction.
4. Keep cost, maintainability, and reliability visible in design choices.

For implementation tasks, continue through implementation, relevant verification, and diff review.
Fix failures caused by the requested change within the approved scope. Stop when acceptance criteria
are met, a material decision is required, or an external prerequisite blocks progress. Report
unrelated failures without expanding scope.

Before handoff:

1. Run the appropriate tests for the change size.
2. Report what was changed, what was verified, and what was not verified.
3. Call out assumptions, tradeoffs, and any follow-up decisions the user should review.

Do not create standalone change summaries or handoff documents unless requested. Put the summary in
the final response or an existing artifact required by the approved workflow. Update relevant existing
documentation within scope; design-document edits still require explicit approval.

## LLM Development Workflow

We use a strict **Strategy → Design → Implementation** workflow for LLM-assisted feature development.
Implement step-by-step from approved phased design documents. When implementation requires a design
deviation, explain it and obtain approval before proceeding. Update the affected design document
only when that edit is explicitly authorized.

### Design Authority

- Existing design documents are reviewed artifacts. **Do not update them without explicit permission.**
- If implementation reveals a needed design change, explain the divergence clearly and request approval before editing design docs.
- “Small doc cleanup” is **not** exempt from this rule.
- Milestone design documents and other design documents are local-only working artifacts. They are
  intentionally not committed to Git and may change without notice.
- Do not add references, links, or citations to local design documents in committed artifacts,
  including source code, repository documentation, docstrings, comments, tests, configuration, or
  generated public-facing content.

## Repository Map and Task References

- `src/chroma/` indexes documents; `src/retrieval/` executes retrieval.
- `src/agents/` contains the intelligent agent, prompts, tools, and description generation.
- `src/database/` contains SQLite models, repositories, and migrations; `src/etl/` imports cellar data.
- `src/api/` contains FastAPI routes, schemas, and lifespan-owned resources.
- `frontend/` contains the Next.js application; `tests/` mirrors backend source areas.

Read only the references relevant to the task, then verify affected implementation and tests:

- System architecture and subsystem overview: [README.md](README.md#architecture).
- Indexing and retrieval behavior: [pipeline guide](docs/pour-decisions-rag-pipeline.md).
- Commands, environment, runtime controls, and implementation entry points:
  [quick reference](docs/quick-reference.md).
- Frontend conventions and UI direction: [frontend/AGENTS.md](frontend/AGENTS.md).

## Essential Coding Invariants

- Load settings from `app_config.yml` through `get_config()`.
- Use `from src.utils import logger, get_config, get_embedder`; never use `print()`.
- Use `get_embedder()` to reuse cached embeddings models configured by the project.
- Use `with get_db_connection() as conn:` for SQLite access with foreign keys enforced.
  Preserve the raw-SQL repository pattern and Pydantic validation.
- Keep TypeScript interfaces in `frontend/src/lib/types.ts` synchronized with affected
  Pydantic API schemas in `src/api/schemas/`.
- Preserve lifespan-owned async resources on API retrieval paths. Post-start streaming failures
  and disconnects are uncertain outcomes and must never trigger automatic replay.

## Development and Verification

```bash
make install          # Install Python dependency groups and eval extra with uv
make dev-full         # Start ChromaDB, FastAPI, and Next.js for development
make test-fast        # No coverage; excludes slow, integration, and eval tests
make test-fast TEST_PATH=tests/utils/test_config.py  # Targeted verification
make test-unit        # Unit suite, including slow tests; 80% coverage gate
make test             # All Python tests with coverage, then frontend tests
make test-watch       # Python watch mode
make format-check     # Check Black/isort on src and tests without changing files
make format PYTHON_PATHS=src/path.py  # Apply formatting only to selected Python paths
make frontend-test    # Vitest, one pass
make frontend-build   # Next.js production build
```

Python test targets use `uv run --group test python` by default and accept `TEST_PATH`.
Use targeted fast tests during iteration; keep the broader coverage gate for suite-level validation.
The full `test` target includes integration/eval tests and may need external prerequisites and approval.
Formatting targets accept `PYTHON_PATHS` (default `src tests`); avoid unrelated formatting changes.
For other commands and environment setup, consult the quick reference.

Shared backend fixtures are in `tests/conftest.py`. Markers are `slow`, `integration`, and `eval`.
Frontend-specific test commands and conventions are in `frontend/AGENTS.md`.

### Testing Policy

- Unit tests must stay fast.
- For **small changes**, run targeted tests for the touched area.
- For **large changes**, run the full relevant suite before handoff.
- If full validation is skipped because of scope, time, environment limits, or missing prerequisites, state that explicitly.

## Style

- Type hints required on all functions. Google-style docstrings.
- Black formatting at 120 chars. isort for imports.
- No emojis in code comments, docstrings, or logs.
- Standard library, then third-party, then local imports - grouped and separated.

### Design and Explicitness

- Avoid hidden magic, implicit assumptions, and surprising fallbacks.
- Prefer explicit data flow, explicit configuration, and clear interfaces.
- Code involving agents or orchestration must be easy to follow step-by-step.
- Keep logging balanced: log what helps debugging or analysis, avoid noisy or decorative logs.
- Challenge unclear, weak, or inconsistent technical assumptions instead of blindly implementing them.
- Do not prefer OOP over functional programming. Use classes when appropriate and mix with functional programming following Python's best practices. 
- Versioned documents use the exact project version from `pyproject.toml`; documents do not maintain
  independent semantic versions.
- Keep the document's `last updated` or `last verified` date separate from the project version and
  update that date when the document is edited or re-verified.

### Working Style

- For short, focused tasks: be execution-focused and concise.
- For larger tasks, design work, research, or brainstorming: be collaborative, explicit about tradeoffs, and open about alternatives.
- Compliance is not passivity: follow instructions carefully, but raise concerns when a request is technically weak, risky, or inconsistent with repo goals.
