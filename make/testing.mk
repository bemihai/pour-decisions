# ============================================================================
# Testing Commands
# ============================================================================

.PHONY: test
test:
	@echo "Running Python tests with coverage..."
	@PYTHONPATH=$(shell pwd) pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"
	@echo ""
	@echo "Running frontend tests..."
	@cd frontend && npm test --if-present

.PHONY: test-unit
test-unit:
	@echo "Running unit tests with coverage threshold (80%)..."
	@PYTHONPATH=$(shell pwd) pytest tests/ -v -m "not integration and not eval" --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=80

.PHONY: test-fast
test-fast:
	@echo "Running tests quickly (no coverage, stop at first failure)..."
	@PYTHONPATH=$(shell pwd) pytest tests/ -v -x -m "not integration and not eval"

.PHONY: test-watch
test-watch:
	@echo "Running tests in watch mode..."
	@PYTHONPATH=$(shell pwd) ptw tests/ -- -v --cov=src --cov-report=term-missing

.PHONY: test-coverage
test-coverage:
	@echo "Opening coverage report in browser..."
	@open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null || echo "Coverage report: htmlcov/index.html"

.PHONY: eval
eval:
	@echo "Running eval harness (retrieval-only mode, free)..."
	@PYTHONPATH=$(shell pwd) python -m src.eval --mode retrieval --backend rag

.PHONY: eval-full
eval-full:
	@echo "Running full eval harness (LLM scoring, local Ollama by default)..."
	@PYTHONPATH=$(shell pwd) python -m src.eval --mode full --backend rag

.PHONY: eval-report
eval-report:
	@echo "Comparing last 2 eval runs..."
	@PYTHONPATH=$(shell pwd) python -m src.eval.compare_results --latest 2

.PHONY: eval-validate
eval-validate:
	@echo "Checking golden dataset for stale cellar-dependent questions..."
	@PYTHONPATH=$(shell pwd) python -m src.eval.dataset_validator

.PHONY: eval-curate
eval-curate:
	@echo "Interactive chunk ID curation for golden dataset..."
	@PYTHONPATH=$(shell pwd) python -m src.eval.chunk_id_curator

.PHONY: eval-phoenix
eval-phoenix:
	@echo "Running eval harness and pushing results to Phoenix..."
	@PYTHONPATH=$(shell pwd) python -m src.eval --mode retrieval --backend rag --push-to-phoenix

.PHONY: eval-phoenix-full
eval-phoenix-full:
	@echo "Running full eval harness and pushing results to Phoenix (local Ollama by default)..."
	@PYTHONPATH=$(shell pwd) python -m src.eval --mode full --backend rag --push-to-phoenix

