"""Tests for CellarTracker tasting-note reconciliation."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from src.database import Tasting, Wine, initialize_database
from src.etl.cellartracker_importer import CellarTrackerImporter
from src.etl.utils import normalize_cellartracker_wine_type, normalize_wine_type, parse_bool


@pytest.fixture
def importer(temp_dir: Path) -> CellarTrackerImporter:
    """Create an importer backed by an isolated database."""
    db_path = temp_dir / "cellar.db"
    assert initialize_database(db_path)

    with patch("src.etl.cellartracker_importer.cellartracker.CellarTracker"):
        return CellarTrackerImporter("user", "password", str(db_path))


def _create_tasting(
    importer: CellarTrackerImporter,
    *,
    external_id: str,
    is_defective: bool,
    personal_rating: int | None = None,
    tasting_notes: str | None = None,
    do_like: bool | None = None,
    community_rating: float | None = None,
    like_votes: int | None = 0,
    like_percentage: float | None = None,
    last_tasted_date: date | None = None,
) -> int:
    """Create one CellarTracker wine and its tasting row."""
    wine_id = importer.wine_repo.create(
        Wine(
            source="cellar_tracker",
            external_id=external_id,
            wine_name=f"Test wine {external_id}",
        )
    )
    importer.tasting_repo.create(
        Tasting(
            wine_id=wine_id,
            is_defective=is_defective,
            personal_rating=personal_rating,
            tasting_notes=tasting_notes,
            do_like=do_like,
            community_rating=community_rating,
            like_votes=like_votes,
            like_percentage=like_percentage,
            last_tasted_date=last_tasted_date,
        )
    )
    return wine_id


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("False", False),
        ("false", False),
        ("0", False),
        ("True", True),
        ("true", True),
        ("1", True),
    ],
)
def test_parse_bool_handles_cellartracker_defective_values(raw_value: str, expected: bool) -> None:
    """CellarTracker boolean strings are parsed without string-truthiness."""
    assert parse_bool(raw_value) is expected


@pytest.mark.parametrize(
    ("source_type", "expected"),
    [
        ("Red", "Red"),
        ("White", "White"),
        ("Rosé", "Rosé"),
        ("Dessert", "Dessert"),
        ("Fortified", "Fortified"),
        ("Sparkling", "Sparkling"),
        ("White - Sparkling", "Sparkling"),
        ("Rosé - Sparkling", "Sparkling"),
        ("White - offdry", "White"),
        ("  wHiTe - SpArKlInG  ", "Sparkling"),
        ("  rOsÉ - offdry  ", "Rosé"),
    ],
)
def test_normalize_cellartracker_wine_type(source_type: str, expected: str) -> None:
    """Exact and compound CellarTracker types map to a canonical family."""
    assert normalize_cellartracker_wine_type(source_type) == expected


@pytest.mark.parametrize("source_type", [None, "", "   ", "Orange - Skin Contact"])
def test_normalize_cellartracker_wine_type_warns_for_unknown_source(
    source_type: str | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unsupported source values remain visible instead of silently becoming red."""
    with caplog.at_level("WARNING"):
        assert normalize_cellartracker_wine_type(source_type) == "Unknown"

    assert caplog.records[-1].getMessage() == f"Unrecognized CellarTracker wine type: {source_type!r}"


def test_vivino_wine_type_fallback_is_unchanged() -> None:
    """The source-specific fix does not alter the shared Vivino normalization path."""
    assert normalize_wine_type("Red Wine") == "Red"
    assert normalize_wine_type("unsupported") == "Red"


@pytest.mark.parametrize(
    ("wine_name", "source_type", "expected"),
    [
        ("CastellRoig Corpinnat Sabaté I Coca Reserva Brut 2023", "White - Sparkling", "Sparkling"),
        ("Schloss Johannisberg Riesling Gelblack feinherb 2022", "White - offdry", "White"),
        ("Billecart-Salmon Champagne Le Rosé", "Rosé - Sparkling", "Sparkling"),
    ],
)
def test_reported_cellartracker_wines_use_source_type_family(
    importer: CellarTrackerImporter,
    wine_name: str,
    source_type: str,
    expected: str,
) -> None:
    """Reported compound types are normalized from the CellarTracker Type field."""
    wine = importer._get_wine_object_from_inventory_record(
        {
            "iWine": wine_name,
            "Wine": wine_name,
            "Type": source_type,
            "Producer": "Regression Producer",
            "Country": "France",
            "Region": "Regression Region",
        }
    )

    assert wine.wine_type == expected


def test_next_inventory_sync_corrects_type_without_creating_duplicates(
    importer: CellarTrackerImporter,
) -> None:
    """A normal sync updates the existing CT wine and bottle in place."""
    source_record = {
        "iWine": "compound-type-1",
        "Barcode": "bottle-1",
        "Wine": "Billecart-Salmon Champagne Le Rosé",
        "Type": "Red",
        "Vintage": "2022",
        "Producer": "Billecart-Salmon",
        "Country": "France",
        "Region": "Champagne",
    }
    importer._process_inventory([source_record])

    existing_wine = importer.wine_repo.get_by_external_id("compound-type-1")
    assert existing_wine is not None
    existing_bottles = importer.bottle_repo.get_by_wine(existing_wine.id)
    assert len(existing_bottles) == 1

    source_record["Type"] = "Rosé - Sparkling"
    importer._process_inventory([source_record])

    updated_wine = importer.wine_repo.get_by_external_id("compound-type-1")
    assert updated_wine is not None
    updated_bottles = importer.bottle_repo.get_by_wine(updated_wine.id)
    assert updated_wine.id == existing_wine.id
    assert updated_wine.wine_type == "Sparkling"
    assert [bottle.id for bottle in updated_bottles] == [bottle.id for bottle in existing_bottles]


def test_source_false_clears_reported_wine_without_changing_other_tasting_data(
    importer: CellarTrackerImporter,
) -> None:
    """A positive rating does not prevent an authoritative false correction."""
    external_id = "5240460"
    tasting_date = date(2025, 5, 11)
    note = "Great Merlot from Dealu Mare."
    stored_note = f"[{tasting_date.isoformat()}] {note}"
    wine_id = _create_tasting(
        importer,
        external_id=external_id,
        is_defective=True,
        personal_rating=88,
        tasting_notes=stored_note,
        do_like=True,
        community_rating=87.25,
        like_votes=4,
        like_percentage=0.8,
        last_tasted_date=tasting_date,
    )
    source_record = {
        "iNote": "11784648",
        "iWine": external_id,
        "Defective": "False",
        "Rating": "88",
        "TastingDate": "5/11/2025",
        "TastingNotes": note,
        "fLikeIt": "True",
        "CScore": "87.25",
        "LikeVotes": "4",
        "LikePercent": "0.8",
    }

    with patch.object(
        importer.tasting_repo,
        "update_defective_flag",
        wraps=importer.tasting_repo.update_defective_flag,
    ) as update_defective_flag:
        importer._process_tasting_notes([source_record])
        importer._process_tasting_notes([source_record])

    tasting = importer.tasting_repo.get_latest_by_wine(wine_id)
    assert tasting is not None
    assert tasting.is_defective is False
    assert tasting.personal_rating == 88
    assert tasting.tasting_notes == stored_note
    assert tasting.do_like is True
    assert tasting.community_rating == 87.25
    assert tasting.like_votes == 4
    assert tasting.like_percentage == 0.8
    assert tasting.last_tasted_date == tasting_date
    assert update_defective_flag.call_count == 1


def test_explicit_source_false_clears_stale_flag_without_other_personal_data(
    importer: CellarTrackerImporter,
) -> None:
    """An explicit false value is personal data even when all other fields are empty."""
    wine_id = _create_tasting(importer, external_id="false-only", is_defective=True)

    importer._process_tasting_notes([{"iWine": "false-only", "Defective": "False"}])

    tasting = importer.tasting_repo.get_latest_by_wine(wine_id)
    assert tasting is not None
    assert tasting.is_defective is False


def test_source_true_sets_and_idempotently_preserves_defective_flag(
    importer: CellarTrackerImporter,
) -> None:
    """An authoritative true value is set once and preserved on repeated imports."""
    wine_id = _create_tasting(importer, external_id="true-flag", is_defective=False)
    source_record = {"iWine": "true-flag", "Defective": "True"}

    with patch.object(
        importer.tasting_repo,
        "update_defective_flag",
        wraps=importer.tasting_repo.update_defective_flag,
    ) as update_defective_flag:
        importer._process_tasting_notes([source_record])
        importer._process_tasting_notes([source_record])

    tasting = importer.tasting_repo.get_latest_by_wine(wine_id)
    assert tasting is not None
    assert tasting.is_defective is True
    assert update_defective_flag.call_count == 1
