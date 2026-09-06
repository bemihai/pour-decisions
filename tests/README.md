# Test Suite

> **Project version**: 0.8.5 — last verified 2026-09-06.
> Test structure mirrors `src/`. New test files will be added for each milestone feature.
> Coverage threshold is 80% on `make test-unit`.

This directory contains unit and integration tests for the Pour Decisions project.

## Running Tests

### Quick Commands

```bash
make test          # Run all tests with coverage report
make test-unit     # Run tests with 80% coverage threshold (fails if below)
make test-fast     # Quick run without coverage, stops at first failure
make test-watch    # Watch mode for continuous testing during development
make test-coverage # Open HTML coverage report in browser
```

### Direct pytest Usage

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# Run specific test file
pytest tests/chroma/test_deduplication.py -v

# Run specific test class
pytest tests/chroma/test_deduplication.py::TestDeduplicateByContentHash -v

# Run specific test
pytest tests/chroma/test_deduplication.py::TestDeduplicateByContentHash::test_no_duplicates -v

# Run tests matching pattern
pytest tests/ -k "deduplication" -v

# Stop at first failure
pytest tests/ -x

# Show slow tests (>1s)
pytest tests/ --durations=10
```

## Test Structure

```
tests/
├── conftest.py           # Shared fixtures and configuration
├── test_data/            # Test data files (sample PDFs, CSVs, etc.)
│   ├── knowledge/        # Test wine PDFs
│   ├── ct/               # CellarTracker test data
│   └── vivino/           # Vivino test data
├── chroma/               # Tests for ChromaDB and indexing components
│   ├── test_chunks.py
│   ├── test_deduplication.py
│   ├── test_hierarchical_chunks.py
│   ├── test_index_tracker.py
│   ├── test_loader.py
│   ├── test_metadata_extractor.py
│   └── test_utils.py
└── agents/               # Tests for agent tools
    └── test_web_search_tools.py
```

## Coverage Reports

After running tests with coverage:
- **Terminal report**: Shows coverage percentage and missing lines
- **HTML report**: Open `htmlcov/index.html` in a browser for detailed coverage

Target coverage: **80%** minimum for the codebase.

## Test Data

Use `tests/test_data/` for any test files needed:
- Sample wine PDFs
- Mock CSV files
- Test database fixtures

## Writing Tests

Follow these guidelines:

1. **Naming**: Test files must start with `test_`, test classes with `Test`, test functions with `test_`
2. **Docstrings**: Add clear docstrings to test classes and functions
3. **Mocking**: Use `pytest-mock` for mocking external dependencies (LLM calls, embeddings, etc.)
4. **Fixtures**: Define reusable fixtures in `conftest.py`
5. **Markers**: Use markers for test categories:
   - `@pytest.mark.slow` - For tests taking >1s
   - `@pytest.mark.integration` - For integration tests

Example:
```python
import pytest
from unittest.mock import Mock, patch

class TestMyFeature:
    """Test my feature functionality."""

    def test_basic_case(self):
        """Test the basic happy path."""
        assert True

    @pytest.mark.slow
    def test_slow_operation(self):
        """Test that takes longer to run."""
        assert True

    @patch("src.module.function")
    def test_with_mock(self, mock_func):
        """Test using mocked dependencies."""
        mock_func.return_value = "mocked"
        assert True
```

## Continuous Integration

Tests are automatically run on:
- Pull requests
- Pushes to main branch
- Pre-commit hooks (if configured)

Coverage reports are generated and can be viewed in CI artifacts.
