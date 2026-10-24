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
    ("window_source", "raw_available", "expected"),
    [
        ("manual", "-0.25", -0.25),
        ("cellar_tracker", "1.75", 1.75),
        ("llm", "0", 0.0),
        ("heuristic", 0.375, 0.375),
    ],
)
def test_availability_refreshes_raw_value_independently_of_window_source(
    importer: CellarTrackerImporter,
    window_source: str,
    raw_available: str | float,
    expected: float,
) -> None:
    """Raw availability accepts the full numeric range for every window source."""
    external_id = f"availability-{window_source}"
    wine_id = importer.wine_repo.create(
        Wine(
            source="cellar_tracker",
            external_id=external_id,
            wine_name="Availability test wine",
            drink_from_year=2025,
            drink_to_year=2027,
            drink_index=88.0,
            cellar_tracker_available=0.5,
            drink_window_source=window_source,
        )
    )

    importer._process_availability([{"iWine": external_id, "Available": raw_available}])

    updated = importer.wine_repo.get_by_id(wine_id)
    assert updated is not None
    assert updated.cellar_tracker_available == expected
    assert updated.drink_index == 88.0
    assert updated.drink_window_source == window_source


@pytest.mark.parametrize("raw_available", ["", "not-a-number", "nan", "inf", "-inf"])
def test_invalid_availability_is_logged_and_does_not_modify_stored_value(
    importer: CellarTrackerImporter,
    raw_available: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Invalid availability is non-fatal and leaves the last valid value intact."""
    wine_id = importer.wine_repo.create(
        Wine(
            source="cellar_tracker",
            external_id="invalid-availability",
            wine_name="Invalid availability test wine",
            cellar_tracker_available=0.625,
        )
    )

    with caplog.at_level("WARNING"):
        importer._process_availability(
            [{"iWine": "invalid-availability", "Available": raw_available}]
        )

    updated = importer.wine_repo.get_by_id(wine_id)
    assert updated is not None
    assert updated.cellar_tracker_available == 0.625
    assert f"Could not parse available score '{raw_available}'" in caplog.text
    assert importer.stats["errors"] == []


def test_absent_availability_does_not_modify_stored_value(importer: CellarTrackerImporter) -> None:
    """A missing availability value leaves the last synchronized value untouched."""
    wine_id = importer.wine_repo.create(
        Wine(
            source="cellar_tracker",
            external_id="missing-availability",
            wine_name="Missing availability test wine",
            cellar_tracker_available=0.625,
        )
    )

    importer._process_availability([{"iWine": "missing-availability"}])

    updated = importer.wine_repo.get_by_id(wine_id)
    assert updated is not None
    assert updated.cellar_tracker_available == 0.625


def _inventory_record(external_id: str, begin: str, end: str) -> dict:
    """Build a valid CellarTracker inventory record with a drinking window."""
    return {
        "iWine": external_id,
        "Barcode": f"bottle-{external_id}",
        "Wine": "Window precedence wine",
        "Type": "Red",
        "Producer": "Window Producer",
        "Country": "France",
        "Region": "Bordeaux",
        "BeginConsume": begin,
        "EndConsume": end,
    }


def test_manual_window_survives_inventory_sync(importer: CellarTrackerImporter) -> None:
    """Generic inventory updates cannot bypass manual window precedence."""
    wine_id = importer.wine_repo.create(
        Wine(
            source="cellar_tracker",
            external_id="manual-window",
            wine_name="Manual window wine",
            drink_from_year=2030,
            drink_to_year=2040,
            drink_index=42.0,
            drink_window_source="manual",
        )
    )

    importer._process_inventory([_inventory_record("manual-window", "2025", "2027")])

    updated = importer.wine_repo.get_by_id(wine_id)
    assert updated is not None
    assert (updated.drink_from_year, updated.drink_to_year) == (2030, 2040)
    assert updated.drink_index == 42.0
    assert updated.drink_window_source == "manual"


@pytest.mark.parametrize("existing_source", ["llm", "heuristic"])
def test_cellartracker_window_replaces_lower_priority_window(
    importer: CellarTrackerImporter,
    existing_source: str,
) -> None:
    """A synchronized CellarTracker window replaces LLM and heuristic estimates."""
    external_id = f"replace-{existing_source}"
    wine_id = importer.wine_repo.create(
        Wine(
            source="cellar_tracker",
            external_id=external_id,
            wine_name="Estimated window wine",
            drink_from_year=2030,
            drink_to_year=2040,
            drink_window_source=existing_source,
        )
    )

    importer._process_inventory([_inventory_record(external_id, "2025", "2027")])

    updated = importer.wine_repo.get_by_id(wine_id)
    assert updated is not None
    assert (updated.drink_from_year, updated.drink_to_year) == (2025, 2027)
    assert updated.drink_window_source == "cellar_tracker"


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


def test_repeated_inventory_import_links_explicit_grapes(importer: CellarTrackerImporter) -> None:
    """Repeated imports share canonical records without duplicate or inferred links."""
    records = [
        {"iWine": "901", "Wine": "First", "Type": "Red", "Varietal": "Cabernet Franc / Shiraz", "Barcode": "9011"},
        {"iWine": "902", "Wine": "Second", "Type": "Red", "Varietal": "Syrah", "Barcode": "9021"},
        {"iWine": "903", "Wine": "Blend", "Type": "Red", "Varietal": "Bordeaux Blend", "Barcode": "9031"},
    ]
    for record in records:
        record.update({"Producer": "Test Estate", "Country": "France", "Region": "Test Region"})
    importer._process_inventory(records)
    importer._process_inventory(records)
    assert not importer.stats["errors"]
    first = importer.wine_repo.get_by_external_id("901")
    second = importer.wine_repo.get_by_external_id("902")
    blend = importer.wine_repo.get_by_external_id("903")
    assert first and second and blend
    grapes = importer.grape_repo.get_by_wine(first.id)
    assert [grape.name for grape in grapes] == ["cabernet franc", "syrah"]
    assert importer.grape_repo.get_by_wine(second.id)[0].id == grapes[1].id
    assert importer.grape_repo.get_by_wine(blend.id) == []
    assert blend.varietal == "Bordeaux Blend"


def test_bottle_import_links_new_and_existing_wines(importer: CellarTrackerImporter) -> None:
    """The historical-bottle path also links explicit grapes idempotently."""
    records = [{"iWine": "904", "Wine": "White", "Type": "White", "Varietal": "Pinot Gris", "Barcode": "9041"}]
    records[0].update({"Producer": "Test Estate", "Country": "France", "Region": "Test Region"})
    importer._process_bottles(records)
    importer._process_bottles(records)
    assert not importer.stats["errors"]
    wine = importer.wine_repo.get_by_external_id("904")
    assert wine
    assert [grape.name for grape in importer.grape_repo.get_by_wine(wine.id)] == ["pinot grigio"]
