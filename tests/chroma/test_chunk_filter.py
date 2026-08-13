"""Tests for structural role classification and authoritative chunk filtering."""

import pytest

from src.chroma.chunk_filter import ChunkQualityFilter
from src.chroma.structural_roles import classify_structural_role


NORMAL_PROSE = (
    "Nebbiolo is a late-ripening red grape associated with Barolo and Barbaresco. "
    "Its wines combine high acidity with firm tannins and can develop for decades. "
    "Classic aromas include roses, violets, dried cherries, herbs, leather, tar, and spice. "
    "The best examples retain a core of fruit while their tannins become more supple with age. "
    "Site, vintage, extraction, and maturation choices influence the final balance, but the grape's "
    "structure and perfume remain recognizable in well-made examples from Piedmont."
)

WORKSHEET = """0 5 10
Food Flavor Type: Fruity Nutty Smoky Cheesy Herbal Umami Earthy Other
Wine Flavor Type: Fruity Nutty Smoky Buttery Herbal Floral Earthy Other
Food Observations: ___________________________________________
Wine Observations: ___________________________________________
Level of Food & Wine Match: Match based on: Complementary Contrast
Comments: ______________________________________"""

BIBLIOGRAPHY = """1. S. E. Ebeler, Sensory Analysis and Analytical Flavor Chemistry (2004), pp. 41-50.
2. G. Reineccius, Source Book of Flavors, 2nd ed. (1994), pp. 20-31.
3. Ibid.
4. B. P. Halpern et al., Retronasal Olfaction (2001), pp. 51-63."""


def test_normal_wine_prose_is_retained() -> None:
    """Substantive explanatory prose should remain safely above the quality threshold."""
    assessment = ChunkQualityFilter(mode="enforce", min_score=0.4).assess(
        NORMAL_PROSE,
        {"section": "The taste of Nebbiolo"},
    )

    assert assessment.structural_role == "prose"
    assert assessment.quality_score >= 0.7
    assert assessment.should_reject is False


def test_worksheet_is_rejected_even_when_it_contains_wine_vocabulary() -> None:
    """Form labels and scales should override superficially relevant flavour terms."""
    assessment = ChunkQualityFilter(mode="enforce", min_score=0.4).assess(WORKSHEET, {})

    assert assessment.structural_role == "worksheet"
    assert assessment.should_reject is True
    assert "worksheet_form_signals" in assessment.rejection_reasons


def test_generic_tasting_form_overrides_declared_prose_role() -> None:
    """A provisional prose label must not mask a generic tasting worksheet."""
    text = """Wine________________ Producer________________ Year________
Visual Clarity: Cloudy--------Clear--------Brilliant
Taste Sweetness: Bone Dry-------Dry-------Sweet
Tactile Sensations: Tannin-------Alcohol-------Body"""

    assessment = ChunkQualityFilter(mode="enforce", min_score=0.4).assess(
        text,
        {"structural_role": "prose"},
    )

    assert assessment.structural_role == "worksheet"
    assert assessment.should_reject is True
    assert "worksheet_form_signals" in assessment.rejection_reasons


def test_very_short_unknown_fragment_is_rejected_at_boundary_score() -> None:
    """Bare unknown labels must not survive by landing exactly on the cutoff."""
    assessment = ChunkQualityFilter(mode="enforce", min_score=0.4).assess("THE BEST WINES", {})

    assert assessment.structural_role == "unknown"
    assert assessment.quality_score == 0.0
    assert assessment.below_threshold is True
    assert assessment.should_reject is True
    assert "very_short_unknown" in assessment.rejection_reasons


def test_bibliography_heading_is_authoritative() -> None:
    """A NOTES section should be rejected even when its citation block is long."""
    assessment = ChunkQualityFilter(mode="enforce", min_score=0.4).assess(
        BIBLIOGRAPHY,
        {"heading_path": "Food and Wine Pairing > Chapter 2 > NOTES"},
    )

    assert assessment.structural_role == "bibliography"
    assert assessment.should_reject is True
    assert "bibliography_heading" in assessment.rejection_reasons


def test_numbered_wine_summary_is_not_misclassified_as_bibliography() -> None:
    """Generic numbered teaching material needs an actual citation signal."""
    text = """1. Cabernet Sauvignon: high acid, high tannin, blackcurrant, cedar, and mint.
2. Merlot: medium acidity, plum fruit, chocolate, and a softer tannin profile.
3. Syrah: full body, blackberry, pepper, clove, leather, game, and tar.
4. Grenache: red fruit, spice, high alcohol, and moderate tannin."""

    assessment = ChunkQualityFilter(mode="enforce", min_score=0.4).assess(text, {})

    assert assessment.structural_role != "bibliography"
    assert assessment.should_reject is False


def test_toc_and_index_roles_are_rejected() -> None:
    """Known structural headings should reject their content deterministically."""
    toc = ChunkQualityFilter(mode="enforce").assess(
        "Chapter One ........ 12\nChapter Two ........ 34\nChapter Three ........ 58",
        {"section": "Contents"},
    )
    index = ChunkQualityFilter(mode="enforce").assess(
        "Barolo, 12, 18\nBarbaresco, 22\nNebbiolo, 11, 40",
        {"chapter": "Index"},
    )

    assert toc.structural_role == "toc" and toc.should_reject
    assert index.structural_role == "index" and index.should_reject


def test_wine_list_is_classified_but_not_hard_rejected() -> None:
    """Useful lists remain eligible for specialized or later retrieval behavior."""
    text = "\n".join(
        [
            "Altare Brunate",
            "Brovia Villero",
            "Giacomo Conterno Monfortino",
            "Bartolo Mascarello Barolo",
            "Roagna La Pira",
            "G D Vajra Albe",
        ]
    )
    role = classify_structural_role(text, {})
    assessment = ChunkQualityFilter(mode="enforce", min_score=0.4).assess(text, {})

    assert role.role == "wine_list"
    assert assessment.should_reject is False


def test_layout_audit_requirement_rejects_uncertain_extraction() -> None:
    """Known uncertain page order must not silently enter the index."""
    assessment = ChunkQualityFilter(mode="enforce").assess(
        NORMAL_PROSE,
        {"layout_audit_required": True},
    )

    assert assessment.structural_role == "unknown"
    assert assessment.should_reject is True
    assert "layout_audit_required" in assessment.rejection_reasons


@pytest.mark.parametrize("mode", ["disabled", "audit", "enforce"])
def test_supported_modes_are_explicit(mode: str) -> None:
    """Only the reviewed quality-filter modes should be accepted."""
    assert ChunkQualityFilter(mode=mode).mode == mode


def test_invalid_filter_configuration_is_rejected() -> None:
    """Configuration errors should fail before indexing begins."""
    with pytest.raises(ValueError, match="mode"):
        ChunkQualityFilter(mode="sometimes")
    with pytest.raises(ValueError, match="between"):
        ChunkQualityFilter(mode="enforce", min_score=1.1)
