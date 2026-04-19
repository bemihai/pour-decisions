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
	@PYTHONPATH=$(shell pwd) pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=80

.PHONY: test-fast
test-fast:
	@echo "Running tests quickly (no coverage, stop at first failure)..."
	@PYTHONPATH=$(shell pwd) pytest tests/ -v -x

.PHONY: test-watch
test-watch:
	@echo "Running tests in watch mode..."
	@PYTHONPATH=$(shell pwd) ptw tests/ -- -v --cov=src --cov-report=term-missing

.PHONY: test-coverage
test-coverage:
	@echo "Opening coverage report in browser..."
	@open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null || echo "Coverage report: htmlcov/index.html"

