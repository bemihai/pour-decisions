"""Focused tests for cellar wine lookup behavior."""

from pathlib import Path

import pytest

from src.agents.tools import cellar_tools
from src.database import get_db_connection, initialize_database


@pytest.fixture()
def regional_cellar_db(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create representative in-cellar wines for normalized region lookups."""
    db_path = temp_dir / "regional-cellar.db"
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

        wines = (
            (1, "Avincis Olt de Vie", avincis, 2019, "Red", "Cabernet Sauvignon", dragasani),
            (2, "Soare", vinarte, 2018, "Red", "Cabernet Sauvignon", samburesti),
            (3, "Bauer Merlot", bauer, 2020, "Red", "Merlot", dealurile_olteniei),
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
            conn.execute(
                """
                INSERT INTO bottles (wine_id, source, quantity, status, location)
                VALUES (?, 'manual', 1, 'in_cellar', 'Test Rack')
                """,
                (wine[0],),
            )
        conn.commit()

    monkeypatch.setattr(cellar_tools, "get_default_db_path", lambda: db_path)
    return db_path


@pytest.mark.parametrize("region", ["Dragasani", "Drăgășani", "Drăgăşani"])
def test_get_cellar_wines_matches_secondary_region_variants(regional_cellar_db: Path, region: str) -> None:
    """The cellar tool returns only the structured Drăgăşani match."""
    wines = cellar_tools.get_cellar_wines.invoke({"region": region})

    assert [wine["wine_id"] for wine in wines] == [1]
    assert wines[0]["producer"] == "Avincis"
    assert all(wine["producer"] != "Crama Bauer" for wine in wines)


@pytest.mark.parametrize("region", ["Samburesti", "Sâmburești", "Sâmbureşti"])
def test_get_cellar_wines_returns_vinarte_soare_cabernet_sauvignon(
    regional_cellar_db: Path,
    region: str,
) -> None:
    """Every supported Sâmbureşti spelling returns the reported stored wine."""
    wines = cellar_tools.get_cellar_wines.invoke({"region": region})

    assert [(wine["name"], wine["producer"], wine["varietal"]) for wine in wines] == [
        ("Soare", "Vinarte", "Cabernet Sauvignon")
    ]


def test_get_cellar_wines_region_composes_with_other_tool_filters(regional_cellar_db: Path) -> None:
    """Region normalization preserves the tool's remaining filters and inventory boundary."""
    wines = cellar_tools.get_cellar_wines.invoke(
        {
            "region": "  dragasani ",
            "country": "Romania",
            "producer": "Avincis",
            "wine_type": "Red",
            "varietal": "Cabernet",
            "vintage": 2019,
        }
    )

    assert [wine["wine_id"] for wine in wines] == [1]
    assert cellar_tools.get_cellar_wines.invoke({"region": "unknown region"}) == []
