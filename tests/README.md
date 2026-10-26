# Test Suite

> **Project version**: 0.13.0 — last verified 2026-10-26.

Backend tests mirror the source areas under `tests/`; frontend tests live beside frontend code.
Shared backend fixtures are in `tests/conftest.py` and pytest settings are in `pyproject.toml`.

## Supported commands

```bash
make test-fast        # No coverage; excludes slow, integration, and eval; stops on first failure
make test-fast TEST_PATH=tests/api/test_app.py  # Target a file or directory
make test-unit        # Excludes integration/eval and enforces 80% backend coverage
make test             # All Python markers with coverage, then frontend tests
make test-watch       # Python watch mode
make frontend-test    # Vitest, one pass
make format-check     # Black/isort check for src and tests
```

Python test targets use `uv run --group test python`. `TEST_PATH` defaults to `tests/`; prefer a
focused `test-fast` run while iterating and the broader relevant suite before handoff. `make test`
includes integration and eval markers, so it may require services, data, credentials, and explicit
approval for any external call.

Equivalent direct invocations should retain the project environment and marker policy:

```bash
uv run --group test python -m pytest tests/chroma/test_loader.py -v
uv run --group test python -m pytest tests/api -v -m "not slow and not integration and not eval"
uv run --group test python -m pytest tests -k "deduplication" -v
uv run --group test python -m pytest tests -v --durations=10
```

## Organization

The main areas are:

| Path | Coverage |
|---|---|
| `tests/agents/` | Agent graph, prompts, tools, guardrails, memory, and provenance |
| `tests/api/` | Routes, schemas, lifespan resources, streaming, tracing, and errors |
| `tests/chroma/` | Extraction, chunking, filtering, loading, indexes, and diagnostics |
| `tests/database/` | Models, repositories, connection behavior, and migrations |
| `tests/etl/` | CellarTracker and Vivino import behavior |
| `tests/eval/` | Dataset, metrics, runners, reports, and bounded evaluation workflows |
| `tests/retrieval/` | Query planning, dense/BM25 retrieval, reranking, and context |
| `tests/services/` | Shared service boundaries such as cached web search |
| `tests/utils/` | Configuration, resources, terms, logging, and tracing helpers |
| `tests/test_data/` | Checked-in deterministic fixtures |

Frontend tests use Vitest and React Testing Library under `frontend/src/**/__tests__/` or beside
the relevant library/store code. See [frontend/README.md](../frontend/README.md#testing).

## Markers and prerequisites

- `slow`: long-running local tests, excluded by `make test-fast`.
- `integration`: tests requiring live components or integration setup, excluded by `test-fast` and
  `test-unit`.
- `eval`: evaluation-specific tests, excluded by `test-fast` and `test-unit`.

Unit tests must stay fast and isolate model, embedding, network, and external-service boundaries.
Do not treat mocked tests as evidence that a live service or model behaved correctly. For manual
API checks, follow [tests/api/TEST_PLAN.md](api/TEST_PLAN.md).

## Writing tests

- Name files `test_*.py`, classes `Test*`, and functions `test_*`.
- Add type hints and concise behavior-focused docstrings.
- Reuse fixtures from `tests/conftest.py`; keep area-specific fixtures near their tests.
- Use `pytest-mock` or `unittest.mock` at external and expensive boundaries.
- Assert observable contracts, explicit failure behavior, and preservation of valid prior state.
- Mark slow, integration, and eval behavior accurately; do not weaken markers to make a target pass.
- Keep temporary databases and files isolated through pytest fixtures.

Coverage HTML from `make test` or `make test-unit` is written to `htmlcov/`. The `80%` threshold is
enforced by `make test-unit`; focused runs are for iteration and do not replace that suite-level
signal when the touched risk warrants it.
