"""Validate local links in the maintained RAG documentation set."""

import re
from pathlib import Path
from urllib.parse import unquote

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAG_DOCUMENTS = (
    Path("README.md"),
    Path("AGENTS.md"),
    Path("docs/pour-decisions-rag-pipeline.md"),
    Path("docs/rag-pipeline.md"),
    Path("docs/quick-reference.md"),
    Path("src/chroma/README.md"),
    Path("src/retrieval/README.md"),
    Path("src/eval/README.md"),
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:")


def _local_link_targets(document: Path) -> list[Path]:
    """Return resolved filesystem targets for local Markdown links in one document."""
    content = document.read_text(encoding="utf-8")
    targets: list[Path] = []

    for raw_target in MARKDOWN_LINK.findall(content):
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
        if not target or target.startswith("#") or target.startswith(EXTERNAL_PREFIXES):
            continue

        path_without_anchor = unquote(target.split("#", maxsplit=1)[0].split("?", maxsplit=1)[0])
        targets.append((document.parent / path_without_anchor).resolve())

    return targets


@pytest.mark.parametrize("relative_document", RAG_DOCUMENTS, ids=str)
def test_rag_document_local_links_exist(relative_document: Path) -> None:
    """Ensure maintained RAG docs do not point to deleted or local-only files."""
    document = REPOSITORY_ROOT / relative_document
    missing_targets = [target for target in _local_link_targets(document) if not target.exists()]

    assert not missing_targets, f"{relative_document} has missing local links: {missing_targets}"
