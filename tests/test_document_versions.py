"""Tests for project-version metadata in Markdown documents."""

import re
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "node_modules",
}
LEGACY_DOC_VERSION_PATTERN = re.compile(r"\*\*Doc version\*\*\s*:")
PROJECT_VERSION_PATTERN = re.compile(r"\*\*Project version\*\*\s*:\s*(\d+\.\d+\.\d+)")


def _iter_markdown_files() -> list[Path]:
    """Return repository Markdown files outside generated and dependency directories."""
    return sorted(
        path
        for path in PROJECT_ROOT.rglob("*.md")
        if not any(part in EXCLUDED_DIRECTORY_NAMES for part in path.relative_to(PROJECT_ROOT).parts)
    )


def _project_version() -> str:
    """Read the authoritative project version from pyproject.toml."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    return str(pyproject["project"]["version"])


def test_markdown_documents_use_project_version() -> None:
    """Reject independent document versions and project-version mismatches."""
    expected_version = _project_version()
    legacy_markers: list[str] = []
    mismatched_markers: list[str] = []

    for markdown_file in _iter_markdown_files():
        content = markdown_file.read_text(encoding="utf-8")
        relative_path = markdown_file.relative_to(PROJECT_ROOT)

        if LEGACY_DOC_VERSION_PATTERN.search(content):
            legacy_markers.append(str(relative_path))

        for match in PROJECT_VERSION_PATTERN.finditer(content):
            actual_version = match.group(1)
            if actual_version != expected_version:
                mismatched_markers.append(f"{relative_path}: {actual_version}")

    assert not legacy_markers, f"Legacy 'Doc version' markers found: {legacy_markers}"
    assert not mismatched_markers, (
        f"Markdown project versions must match pyproject.toml ({expected_version}): {mismatched_markers}"
    )
