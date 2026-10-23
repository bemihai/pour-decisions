"""Regression coverage for favorite rankings and best-wine metadata."""

from pathlib import Path

import pytest

from src.database import get_db_connection, initialize_database
from src.database.repository import StatsRepository


@pytest.fixture()
def favorites_db(tmp_path: Path) -> Path:
    """Create groups with distinct counts, ratings, ties, and unrated wines."""
    db_path = tmp_path / "favorites.db"
    assert initialize_database(db_path)
    with get_db_connection(db_path) as conn:
        for group_id, name in enumerate(["Most tasted", "Higher rated", "Unrated"], 1):
            conn.execute("INSERT INTO producers (id, name, country) VALUES (?, ?, ?)", (group_id, name, name))
            conn.execute("INSERT INTO regions (id, primary_name, country) VALUES (?, ?, ?)", (group_id, name, name))
        # The same rating and date choose the higher wine ID, not the first row.
        for wine_id, group_id, rating, count in [
            (1, 1, 80, 2),
            (2, 1, 90, 1),
            (3, 1, 90, 1),
            (4, 2, 98, 2),
            (5, 3, None, 2),
        ]:
            conn.execute(
                """INSERT INTO wines (id, source, wine_name, producer_id, region_id, vintage, appellation, wine_type)
                   VALUES (?, 'manual', ?, ?, ?, ?, ?, 'Red')""",
                (wine_id, f"Wine {wine_id}", group_id, group_id, 2017 + group_id, f"Appellation {group_id}"),
            )
            conn.execute("INSERT INTO tastings (wine_id, personal_rating) VALUES (?, ?)", (wine_id, rating))
            for _ in range(count):
                conn.execute(
                    """INSERT INTO bottles (wine_id, source, status, consumed_date)
                       VALUES (?, 'manual', 'consumed', '2026-01-01')""",
                    (wine_id,),
                )
        conn.commit()
    return db_path


METHODS = [
    "get_producer_preferences",
    "get_region_preferences",
    "get_country_stats",
    "get_vintage_stats",
    "get_appellation_stats",
]


@pytest.mark.parametrize("method", METHODS)
def test_metadata_preserves_ranking_and_limit(favorites_db: Path, method: str) -> None:
    """Metadata follows the existing winner, count ordering, and vintage ordering."""
    query = getattr(StatsRepository(favorites_db), method)
    rows = query(limit=5)
    expected_ids = [4, 3, None] if method == "get_vintage_stats" else [3, 4, None]
    assert [row["best_wine_id"] for row in rows] == expected_ids
    assert query(limit=1) == rows[:1]
    for row in rows:
        wine_id = row["best_wine_id"]
        if wine_id is None:
            assert row["best_wine_name"] is None
            assert row["best_wine_vintage"] is None
            assert row["best_wine_producer_name"] is None
        else:
            assert row["best_wine_name"] == f"Wine {wine_id}"
            assert row["best_wine_vintage"] == (2018 if wine_id == 3 else 2019)
            assert row["best_wine_producer_name"] == ("Most tasted" if wine_id == 3 else "Higher rated")
    assert next(row for row in rows if row["best_wine_id"] == 3)["avg_rating"] == 85


@pytest.mark.parametrize("method", METHODS)
def test_best_wine_uses_latest_consumption_before_id(favorites_db: Path, method: str) -> None:
    """A later consumption wins a rating tie even with a lower wine ID."""
    with get_db_connection(favorites_db) as conn:
        conn.execute("UPDATE bottles SET consumed_date = '2026-02-01' WHERE wine_id = 2")
        conn.commit()
    rows = getattr(StatsRepository(favorites_db), method)()
    best = next(row for row in rows if row["highest_rating"] == 90)
    assert best["best_wine_id"] == 2
    assert best["best_wine_name"] == "Wine 2"


def test_vintages_require_two_consumed_bottles(favorites_db: Path) -> None:
    """Single-consumption vintages remain excluded."""
    with get_db_connection(favorites_db) as conn:
        conn.execute("DELETE FROM bottles WHERE wine_id = 4 AND id = (SELECT MAX(id) FROM bottles WHERE wine_id = 4)")
        conn.commit()
    assert [row["vintage"] for row in StatsRepository(favorites_db).get_vintage_stats()] == [2018, 2020]


def test_best_wine_missing_optional_metadata(favorites_db: Path) -> None:
    """A known best wine can have no vintage or producer without losing its ID."""
    with get_db_connection(favorites_db) as conn:
        conn.execute("UPDATE wines SET vintage = NULL, producer_id = NULL WHERE id = 3")
        conn.commit()
    row = StatsRepository(favorites_db).get_country_stats()[0]
    assert row["best_wine_id"] == 3
    assert row["best_wine_name"] == "Wine 3"
    assert row["best_wine_vintage"] is None
    assert row["best_wine_producer_name"] is None
