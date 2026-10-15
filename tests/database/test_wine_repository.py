"""Focused tests for structured wine-region filtering."""

from pathlib import Path

import pytest

from src.database import get_db_connection, initialize_database
from src.database.repository import WineRepository


@pytest.fixture()
def regional_wines_db(temp_dir: Path) -> Path:
    """Create wines spanning primary, secondary, aliased, and unrelated regions."""
    db_path = temp_dir / "regional-wines.db"
    assert initialize_database(db_path)

    with get_db_connection(db_path) as conn:
        avincis = conn.execute(
            "INSERT INTO producers (name, country) VALUES (?, ?)",
            ("Avincis", "Romania"),
        ).lastrowid
        vinarte = conn.execute(
            "INSERT INTO producers (name, country) VALUES (?, ?)",
            ("Vinarte", "Romania"),
        ).lastrowid
        bauer = conn.execute(
            "INSERT INTO producers (name, country) VALUES (?, ?)",
            ("Crama Bauer", "Romania"),
        ).lastrowid
        domaine = conn.execute(
            "INSERT INTO producers (name, country) VALUES (?, ?)",
            ("Domaine Test", "France"),
        ).lastrowid

        dragasani = conn.execute(
            "INSERT INTO regions (primary_name, secondary_name, country) VALUES (?, ?, ?)",
            ("Oltenia", "Drăgăşani", "Romania"),
        ).lastrowid
        samburesti = conn.execute(
            "INSERT INTO regions (primary_name, country) VALUES (?, ?)",
            ("Sâmbureşti", "Romania"),
        ).lastrowid
        dealurile_olteniei = conn.execute(
            "INSERT INTO regions (primary_name, country) VALUES (?, ?)",
            ("Dealurile Olteniei", "Romania"),
        ).lastrowid
        burgundy = conn.execute(
            "INSERT INTO regions (primary_name, country) VALUES (?, ?)",
            ("Burgundy", "France"),
        ).lastrowid

        wines = (
            (1, "Avincis Olt de Vie", avincis, 2019, "Red", "Cabernet Sauvignon", dragasani),
            (2, "Soare", vinarte, 2018, "Red", "Cabernet Sauvignon", samburesti),
            (3, "Bauer Merlot", bauer, 2020, "Red", "Merlot", dealurile_olteniei),
            (4, "Burgundy Pinot Noir", domaine, 2021, "Red", "Pinot Noir", burgundy),
        )
        for wine in wines:
            conn.execute(
                """
                INSERT INTO wines (
                    id, source, wine_name, producer_id, vintage, wine_type, varietal, region_id
                ) VALUES (?, 'manual', ?, ?, ?, ?, ?, ?)
                """,
                wine,
            )
        conn.commit()

    return db_path


@pytest.mark.parametrize("region", ["Dragasani", "Drăgășani", "Drăgăşani", "  DRAGASANI  "])
def test_region_filter_normalizes_secondary_region_spellings(regional_wines_db: Path, region: str) -> None:
    """Accentless, comma-below, cedilla, case, and whitespace variants match."""
    wines = WineRepository(regional_wines_db).get_all(region_name=region)

    assert [wine.id for wine in wines] == [1]


@pytest.mark.parametrize("region", ["Samburesti", "Sâmburești", "Sâmbureşti"])
def test_region_filter_normalizes_primary_region_spellings(regional_wines_db: Path, region: str) -> None:
    """Primary structured region labels use the same Unicode normalization."""
    wines = WineRepository(regional_wines_db).get_all(region_name=region)

    assert [wine.id for wine in wines] == [2]


def test_region_filter_expands_aliases_and_preserves_partial_matching(regional_wines_db: Path) -> None:
    """Configured aliases and normalized partial searches remain supported."""
    repository = WineRepository(regional_wines_db)

    assert [wine.id for wine in repository.get_all(region_name="Bourgogne")] == [4]
    assert [wine.id for wine in repository.get_all(region_name="Burgundy")] == [4]
    assert [wine.id for wine in repository.get_all(region_name="drag")] == [1]


def test_region_filter_unknown_and_combined_filters_remain_bounded(regional_wines_db: Path) -> None:
    """Unknown input stays empty and region matching composes with other filters."""
    repository = WineRepository(regional_wines_db)

    assert repository.get_all(region_name="not a wine region") == []
    assert repository.get_all(region_name="   ") == []

    wines = repository.get_all(
        region_name="Drăgășani",
        producer_name="Avincis",
        wine_type="Red",
        vintage=2019,
        country="Romania",
        varietal="Cabernet",
    )
    assert [wine.id for wine in wines] == [1]

    assert repository.get_all(region_name="Drăgășani", producer_name="Crama Bauer") == []
