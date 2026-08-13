"""Tests for deterministic wine-aware BM25 analysis and retrieval."""

from pathlib import Path

import pytest

from src.retrieval.bm25_analyzer import analyze_bm25_text
from src.retrieval.keyword_search import BM25Index


@pytest.mark.parametrize("text", ["Nebbiolo", "NEBBIOLO?", "‘Nebbiolo,’", "nebiolo!"])
def test_nebbiolo_variants_share_the_same_token(text: str) -> None:
    """Case, punctuation, quotes, and configured misspellings should converge."""
    assert analyze_bm25_text(text) == ["nebbiolo"]


def test_unicode_apostrophes_hyphens_and_diacritics_are_stable() -> None:
    """Wine and producer names should remain searchable across typography variants."""
    assert analyze_bm25_text("Grüner Veltliner; Côte-Rôtie; Nero d’Avola") == [
        "gruner",
        "veltliner",
        "cote",
        "rotie",
        "nero",
        "d'avola",
    ]
    assert analyze_bm25_text("Château d’Yquem") == ["chateau", "d'yquem"]


def test_question_filler_is_removed_but_wine_entities_are_preserved() -> None:
    """Sparse queries should emphasize exact entities and discriminating values."""
    analyzed = analyze_bm25_text(
        "What are the primary flavour characteristics of Barolo DOCG Nebbiolo 2016?"
    )

    assert analyzed == ["barolo", "docg", "nebbiolo", "2016"]


def test_configured_synonyms_canonicalize_identically() -> None:
    """Configured grape aliases should produce their canonical multi-token form."""
    assert analyze_bm25_text("Shiraz") == analyze_bm25_text("Syrah") == ["syrah"]
    assert analyze_bm25_text("Pinot Gris") == analyze_bm25_text("Pinot Grigio") == [
        "pinot",
        "grigio",
    ]


def test_exact_entity_retrieval_survives_persistence(tmp_path: Path) -> None:
    """A persisted index should retrieve contextual Nebbiolo evidence with punctuation."""
    documents = [
        {
            "id": "nebbiolo",
            "document": "It has high tannin, acidity, and aromas of roses and tar.",
            "search_text": "Grapes & Wines > NEBBIOLO > taste\n\nIt has high tannin and acidity.",
            "metadata": {"chapter": "NEBBIOLO"},
        },
        {
            "id": "pinot",
            "document": "It has red cherry and subtle earthy aromas.",
            "search_text": "Grapes & Wines > PINOT NOIR > taste\n\nIt has red cherry aromas.",
            "metadata": {"chapter": "PINOT NOIR"},
        },
        {
            "id": "riesling",
            "document": "It has lime, floral, and slate aromas.",
            "search_text": "Grapes & Wines > RIESLING > taste\n\nIt has lime and slate aromas.",
            "metadata": {"chapter": "RIESLING"},
        },
    ]
    index_path = tmp_path / "bm25.pkl"
    bm25 = BM25Index()
    bm25.build_index(documents)
    bm25.save(index_path)

    loaded = BM25Index(index_path=index_path)
    results = loaded.search("What are the primary flavour characteristics of Nebbiolo?", top_k=3)

    assert [result["id"] for result in results] == ["nebbiolo"]
    assert results[0]["document"].startswith("It has high tannin")
