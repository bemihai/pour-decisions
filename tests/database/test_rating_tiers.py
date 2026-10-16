"""Tests for approved backend personal-rating tier boundaries."""

from pathlib import Path

import pytest

from src.database import get_db_connection, initialize_database
from src.database.repository import StatsRepository
from src.etl.utils import get_rating_description


@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        (0, "Avoid"),
        (70, "Avoid"),
        (71, "Below Average"),
        (79, "Below Average"),
        (80, "Good"),
        (85, "Good"),
        (86, "Very Good"),
        (89, "Very Good"),
        (90, "Excellent"),
        (93, "Excellent"),
        (94, "Outstanding"),
        (97, "Outstanding"),
        (98, "Extraordinary"),
        (100, "Extraordinary"),
    ],
)
def test_rating_descriptions_use_approved_boundaries(rating: int, expected: str) -> None:
    """Backend descriptions return the approved label at every boundary."""
    assert get_rating_description(rating) == expected


def test_cellar_rating_distribution_uses_all_approved_tiers(tmp_path: Path) -> None:
    """Cellar statistics use the same seven labels and boundaries."""
    database_path = tmp_path / "cellar.db"
    assert initialize_database(database_path)
    ratings = [70, 71, 80, 86, 90, 94, 98]
    with get_db_connection(database_path) as connection:
        for index, rating in enumerate(ratings):
            wine = connection.execute(
                """
                INSERT INTO wines (source, external_id, wine_name, wine_type)
                VALUES ('manual', ?, ?, 'Red')
                """,
                (f"wine-{index}", f"Wine {index}"),
            )
            connection.execute(
                "INSERT INTO tastings (wine_id, personal_rating) VALUES (?, ?)",
                (wine.lastrowid, rating),
            )
            connection.execute(
                "INSERT INTO bottles (wine_id, source, quantity, status) VALUES (?, 'manual', 1, 'in_cellar')",
                (wine.lastrowid,),
            )
        connection.commit()

    assert StatsRepository(str(database_path)).get_cellar_rating_distribution() == [
        {"tier": "Extraordinary (98-100)", "wines": 1},
        {"tier": "Outstanding (94-97)", "wines": 1},
        {"tier": "Excellent (90-93)", "wines": 1},
        {"tier": "Very Good (86-89)", "wines": 1},
        {"tier": "Good (80-85)", "wines": 1},
        {"tier": "Below Average (71-79)", "wines": 1},
        {"tier": "Avoid (0-70)", "wines": 1},
    ]
