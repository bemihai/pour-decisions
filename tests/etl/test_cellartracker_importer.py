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
