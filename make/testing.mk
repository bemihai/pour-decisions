# ============================================================================
# Testing Commands
# ============================================================================

PROJECT_PYTHON ?= uv run --group test python
EVAL_PYTHON ?= uv run --extra eval python
TEST_PATH ?= tests/
PYTHON_PATHS ?= src tests

.PHONY: test
test:
	@echo "Running Python tests with coverage..."
	@PYTHONPATH=$(shell pwd) $(PROJECT_PYTHON) -m pytest $(TEST_PATH) -v --cov=src --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"
	@echo ""
	@echo "Running frontend tests..."
	@cd frontend && npm test --if-present

.PHONY: test-unit
test-unit:
	@echo "Running unit tests with coverage threshold (80%)..."
	@PYTHONPATH=$(shell pwd) $(PROJECT_PYTHON) -m pytest $(TEST_PATH) -v -m "not integration and not eval" --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=80

.PHONY: test-fast
test-fast:
	@echo "Running tests quickly (no coverage, stop at first failure)..."
	@PYTHONPATH=$(shell pwd) $(PROJECT_PYTHON) -m pytest $(TEST_PATH) -v -x -m "not slow and not integration and not eval"

.PHONY: test-watch
test-watch:
	@echo "Running tests in watch mode..."
	@PYTHONPATH=$(shell pwd) $(PROJECT_PYTHON) -m pytest_watch $(TEST_PATH) -- -v --cov=src --cov-report=term-missing

# Restrict defaults to Python source and tests; never format local environment files.
.PHONY: format-check
format-check:
	@uv run --group dev python -m black --check $(PYTHON_PATHS)
	@uv run --group dev python -m isort --check-only $(PYTHON_PATHS)

.PHONY: format
format:
	@uv run --group dev python -m isort $(PYTHON_PATHS)
	@uv run --group dev python -m black $(PYTHON_PATHS)

.PHONY: test-coverage
test-coverage:
	@echo "Opening coverage report in browser..."
	@open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null || echo "Coverage report: htmlcov/index.html"

.PHONY: eval
eval:
	@echo "Running eval harness (retrieval-only mode, free)..."
	@PYTHONPATH=$(shell pwd) $(PROJECT_PYTHON) -m src.eval --mode retrieval --backend rag

.PHONY: eval-full
eval-full:
	@echo "Running full eval harness (Ollama Cloud scoring)..."
	@PYTHONPATH=$(shell pwd) $(EVAL_PYTHON) -m src.eval --mode full --backend rag

.PHONY: eval-report
eval-report:
	@echo "Comparing eval runs..."
	@PYTHONPATH=$(shell pwd) $(PROJECT_PYTHON) -m src.eval.scripts.compare_results --latest 2 $(ARGS)

.PHONY: eval-validate
eval-validate:
	@echo "Checking golden dataset for stale cellar-dependent questions..."
	@PYTHONPATH=$(shell pwd) $(PROJECT_PYTHON) -m src.eval.scripts.dataset_validator

.PHONY: eval-curate
eval-curate:
	@echo "Interactive chunk ID curation for golden dataset..."
	@PYTHONPATH=$(shell pwd) $(PROJECT_PYTHON) -m src.eval.scripts.chunk_id_curator

.PHONY: eval-contextual-ablation
eval-contextual-ablation:
	@echo "Running M3 Phase 2 body-only versus contextual-search ablation..."
	@PYTHONPATH=$(shell pwd) $(PROJECT_PYTHON) -m src.eval.scripts.contextual_enrichment_ablation

.PHONY: eval-phoenix
eval-phoenix:
	@echo "Running eval harness and pushing results to Phoenix..."
	@PYTHONPATH=$(shell pwd) $(EVAL_PYTHON) -m src.eval --mode retrieval --backend rag --push-to-phoenix

.PHONY: eval-phoenix-full
eval-phoenix-full:
	@echo "Running full eval harness and pushing results to Phoenix (Ollama Cloud)..."
	@PYTHONPATH=$(shell pwd) $(EVAL_PYTHON) -m src.eval --mode full --backend rag --push-to-phoenix
