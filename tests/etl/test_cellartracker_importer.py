"""Tests for CellarTracker tasting-note reconciliation."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from src.database import Tasting, Wine, initialize_database
from src.etl.cellartracker_importer import CellarTrackerImporter
from src.etl.utils import parse_bool


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
