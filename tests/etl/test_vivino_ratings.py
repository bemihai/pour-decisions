"""Tests for Vivino personal-rating calibration and recalculation."""

import csv
import shutil
import sqlite3
from pathlib import Path

import pytest

from src.database import get_db_connection, initialize_database
from src.etl.recalibrate_vivino_ratings import recalibrate_vivino_ratings
from src.etl.utils import (
    VIVINO_PERSONAL_RATING_ANCHORS,
    convert_vivino_community_rating,
    convert_vivino_personal_rating,
    generate_external_id,
)
from src.etl.vivino_importer import VivinoImporter

CSV_FIELDS = [
    "Winery",
    "Wine name",
    "Vintage",
    "Region",
    "Country",
    "Average rating",
    "Scan date",
    "Your rating",
    "Your review",
    "Personal Note",
    "Wine type",
    "Drinking Window",
]


def write_vivino_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a minimal full-wine-list fixture."""
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def vivino_row(
    winery: str,
    wine_name: str,
    rating: str,
    *,
    vintage: str = "2020",
    community_rating: str = "4.2",
) -> dict[str, str]:
    """Build one representative Vivino export row."""
    return {
        "Winery": winery,
        "Wine name": wine_name,
        "Vintage": vintage,
        "Region": "Dealu Mare",
        "Country": "Romania",
        "Average rating": community_rating,
        "Scan date": "2026-01-02",
        "Your rating": rating,
        "Your review": "",
        "Personal Note": "",
        "Wine type": "Red Wine",
        "Drinking Window": "",
    }


@pytest.mark.parametrize("raw_rating, expected", VIVINO_PERSONAL_RATING_ANCHORS)
def test_personal_conversion_preserves_approved_anchors(raw_rating: float, expected: int) -> None:
    """Every approved anchor maps to its exact score."""
    assert convert_vivino_personal_rating(raw_rating) == expected


@pytest.mark.parametrize(
    ("raw_rating", "expected"),
    [(2.6, 81), (2.7, 82), (2.8, 83), (3.1, 86), (3.7, 91), (4.2, 95)],
)
def test_personal_conversion_interpolates_and_rounds(raw_rating: float, expected: int) -> None:
    """Representative values use the approved piecewise interpolation."""
    assert convert_vivino_personal_rating(raw_rating) == expected


@pytest.mark.parametrize("raw_rating", [0, 0.5, 1.0])
def test_personal_conversion_maps_low_valid_values_to_70(raw_rating: float) -> None:
    """The complete valid range through one star maps to Avoid's upper bound."""
    assert convert_vivino_personal_rating(raw_rating) == 70


@pytest.mark.parametrize("raw_rating", [-0.1, 5.1, "bad", "", None, float("nan"), float("inf")])
def test_personal_conversion_rejects_invalid_values(raw_rating: object) -> None:
    """Malformed, missing, non-finite, and out-of-range values do not score."""
    assert convert_vivino_personal_rating(raw_rating) is None  # type: ignore[arg-type]


def test_community_conversion_preserves_existing_mapping() -> None:
    """Community averages retain their prior conversion independently."""
    assert convert_vivino_community_rating(1.0) == 73
    assert convert_vivino_community_rating(3.5) == 85
    assert convert_vivino_community_rating(4.2) == 92
    assert convert_vivino_community_rating(5.0) == 100


def test_importer_uses_highest_raw_rating_and_allows_downward_replacement(tmp_path: Path) -> None:
    """Normal imports use the same personal conversion and authoritative replacement rules."""
    database_path = tmp_path / "cellar.db"
    csv_path = tmp_path / "full_wine_list.csv"
    assert initialize_database(database_path)

    write_vivino_csv(
        csv_path,
        [
            vivino_row("Example Winery", "Example Wine", "3.0"),
            vivino_row("Example Winery", "Example Wine", "3.5"),
        ],
    )
    VivinoImporter(str(database_path)).import_full_wine_list_csv(csv_path)

    with get_db_connection(database_path) as connection:
        first = connection.execute("SELECT personal_rating, do_like, community_rating FROM tastings").fetchone()
    assert dict(first) == {"personal_rating": 88, "do_like": 1, "community_rating": 92}

    write_vivino_csv(csv_path, [vivino_row("Example Winery", "Example Wine", "1.0")])
    VivinoImporter(str(database_path)).import_full_wine_list_csv(csv_path)

    with get_db_connection(database_path) as connection:
        second = connection.execute("SELECT personal_rating, do_like, community_rating FROM tastings").fetchone()
    assert dict(second) == {"personal_rating": 70, "do_like": 0, "community_rating": 92}

    with get_db_connection(database_path) as connection:
        connection.execute("UPDATE tastings SET personal_rating = 76")
        connection.commit()
    write_vivino_csv(csv_path, [vivino_row("Example Winery", "Example Wine", "2.0")])
    VivinoImporter(str(database_path)).import_full_wine_list_csv(csv_path)
    with get_db_connection(database_path) as connection:
        two_star = connection.execute("SELECT personal_rating, do_like FROM tastings").fetchone()
    assert tuple(two_star) == (75, 0)

    write_vivino_csv(csv_path, [vivino_row("Example Winery", "Example Wine", "3.0")])
    VivinoImporter(str(database_path)).import_full_wine_list_csv(csv_path)
    with get_db_connection(database_path) as connection:
        three_star = connection.execute("SELECT personal_rating, do_like FROM tastings").fetchone()
    assert tuple(three_star) == (85, 0)


def insert_vivino_tasting(
    database_path: Path,
    winery: str,
    wine_name: str,
    rating: int,
    do_like: bool,
    *,
    source: str = "vivino",
    vintage: int = 2020,
    community_rating: int = 91,
) -> None:
    """Insert a wine and tasting for recalibration tests."""
    external_id = generate_external_id(winery, wine_name, vintage)
    with get_db_connection(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO wines (source, external_id, wine_name, wine_type, vintage)
            VALUES (?, ?, ?, 'Red', ?)
            """,
            (source, external_id, wine_name, vintage),
        )
        connection.execute(
            """
            INSERT INTO tastings (wine_id, personal_rating, do_like, community_rating)
            VALUES (?, ?, ?, ?)
            """,
            (cursor.lastrowid, rating, do_like, community_rating),
        )
        connection.commit()


def test_recalibration_preview_apply_isolation_and_idempotency(tmp_path: Path) -> None:
    """Preview is read-only; apply is scoped, atomic, community-safe, and idempotent."""
    database_path = tmp_path / "cellar.db"
    backup_path = tmp_path / "cellar-backup.db"
    csv_path = tmp_path / "full_wine_list.csv"
    assert initialize_database(database_path)
    insert_vivino_tasting(database_path, "Changed", "Wine", 73, True, community_rating=92)
    insert_vivino_tasting(database_path, "Unchanged", "Wine", 88, True, community_rating=90)
    insert_vivino_tasting(database_path, "Manual", "Wine", 99, True, source="manual", community_rating=99)
    shutil.copy2(database_path, backup_path)
    write_vivino_csv(
        csv_path,
        [
            vivino_row("Changed", "Wine", "1.0"),
            vivino_row("Unchanged", "Wine", "3.5"),
            vivino_row("Missing", "Wine", "4.0"),
            vivino_row("No Rating", "Wine", ""),
            vivino_row("Invalid", "Wine", "6.0"),
        ],
    )

    preview = recalibrate_vivino_ratings(csv_path, database_path)
    assert preview.to_dict() == {
        "matched": 2,
        "changed": 1,
        "unchanged": 1,
        "unmatched": 1,
        "unrated": 1,
        "invalid": 1,
        "old_distribution": {"73": 1, "88": 1},
        "new_distribution": {"70": 1, "88": 1},
        "applied": False,
    }
    with get_db_connection(database_path) as connection:
        before_apply = connection.execute("""
            SELECT t.personal_rating, t.do_like, t.community_rating
            FROM tastings t JOIN wines w ON w.id = t.wine_id
            WHERE w.wine_name = 'Wine' AND w.source = 'vivino'
            ORDER BY w.external_id
            """).fetchall()
    assert {tuple(row) for row in before_apply} == {(73, 1, 92), (88, 1, 90)}

    applied = recalibrate_vivino_ratings(
        csv_path,
        database_path,
        apply=True,
        backup_path=backup_path,
    )
    assert applied.applied is True
    assert applied.changed == 1

    with get_db_connection(database_path) as connection:
        changed_row = connection.execute(
            """
            SELECT t.personal_rating, t.do_like, t.community_rating
            FROM tastings t JOIN wines w ON w.id = t.wine_id
            WHERE w.source = 'vivino' AND w.external_id = ?
            """,
            (generate_external_id("Changed", "Wine", 2020),),
        ).fetchone()
        manual_row = connection.execute("""
            SELECT t.personal_rating, t.do_like, t.community_rating
            FROM tastings t JOIN wines w ON w.id = t.wine_id
            WHERE w.source = 'manual'
            """).fetchone()
    assert tuple(changed_row) == (70, 0, 92)
    assert tuple(manual_row) == (99, 1, 99)

    second_apply = recalibrate_vivino_ratings(
        csv_path,
        database_path,
        apply=True,
        backup_path=backup_path,
    )
    assert second_apply.changed == 0
    assert second_apply.unchanged == 2


def test_recalibration_requires_verified_backup_for_apply(tmp_path: Path) -> None:
    """Apply mode refuses to write without a separate healthy backup."""
    database_path = tmp_path / "cellar.db"
    csv_path = tmp_path / "full_wine_list.csv"
    assert initialize_database(database_path)
    write_vivino_csv(csv_path, [vivino_row("Example", "Wine", "3.0")])

    with pytest.raises(ValueError, match="requires --backup-path"):
        recalibrate_vivino_ratings(csv_path, database_path, apply=True)


def test_recalibration_rolls_back_all_updates_on_database_failure(tmp_path: Path) -> None:
    """A failed update leaves all prior personal and community values intact."""
    database_path = tmp_path / "cellar.db"
    backup_path = tmp_path / "cellar-backup.db"
    csv_path = tmp_path / "full_wine_list.csv"
    assert initialize_database(database_path)
    insert_vivino_tasting(database_path, "Rollback", "Wine", 73, True, community_rating=92)
    with get_db_connection(database_path) as connection:
        connection.execute("""
            CREATE TRIGGER fail_recalibration
            BEFORE UPDATE OF personal_rating ON tastings
            BEGIN
                SELECT RAISE(ABORT, 'forced recalibration failure');
            END
            """)
        connection.commit()
    shutil.copy2(database_path, backup_path)
    write_vivino_csv(csv_path, [vivino_row("Rollback", "Wine", "1.0")])

    with pytest.raises(sqlite3.IntegrityError, match="forced recalibration failure"):
        recalibrate_vivino_ratings(csv_path, database_path, apply=True, backup_path=backup_path)

    with get_db_connection(database_path) as connection:
        row = connection.execute("SELECT personal_rating, do_like, community_rating FROM tastings").fetchone()
    assert tuple(row) == (73, 1, 92)
