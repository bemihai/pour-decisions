"""Tests for validated contextual search representations."""

from src.chroma.contextual_text import build_contextual_search_text, validated_context_parts


def test_builds_deduplicated_context_without_changing_body() -> None:
    """Trusted lineage should prefix the clean evidence exactly once."""
    body = "It is notably tannic, acidic, and scented with roses and tar."
    metadata = {
        "document_title": "Grapes & Wines",
        "chapter": "NEBBIOLO",
        "entry_title": "NEBBIOLO",
        "section": "taste",
        "structural_role": "prose",
    }

    contextual = build_contextual_search_text(body, metadata)

    assert contextual == "Grapes & Wines > NEBBIOLO > taste\n\n" + body
    assert contextual.endswith(body)


def test_excludes_structural_and_corrupt_context_values() -> None:
    """Noise-like headings must never become embedding or BM25 enrichment."""
    metadata = {
        "document_title": "Wine Guide",
        "chapter": "CONTENTS",
        "entry_title": "(cid:127) (cid:42)",
        "section": "2010 TO 2020 17.5",
        "structural_role": "prose",
    }

    assert validated_context_parts(metadata) == ["Wine Guide"]
    assert build_contextual_search_text("Clean body.", metadata) == "Wine Guide\n\nClean body."


def test_rejected_structural_role_disables_all_enrichment() -> None:
    """Rejected corpus roles should remain isolated from inherited headings."""
    metadata = {
        "document_title": "Wine Guide",
        "chapter": "Nebbiolo",
        "section": "Worksheet",
        "structural_role": "worksheet",
    }

    assert build_contextual_search_text("Name: ______", metadata) == "Name: ______"
