"""Grape schema, migration, explicit matching and shared persistence tests."""

import sqlite3
from pathlib import Path

import pytest

from src.database.db import get_db_connection, initialize_database, verify_grapes_schema
from src.database.migrations.create_grapes import run_migration
from src.database.repository.grape import GrapeRepository, extract_explicit_grapes


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Cabernet Franc", ["cabernet franc"]),
        ("Pinot Grigio", ["pinot grigio"]),
        ("Pinot Gris / Chardonnay", ["chardonnay", "pinot grigio"]),
        ("Shiraz, Syrah, cab sauv", ["cabernet sauvignon", "syrah"]),
        ("Cabernet Franc / Cabernet Sauvignon", ["cabernet franc", "cabernet sauvignon"]),
        ("Bordeaux Blend", []),
        ("Red Blend", []),
        ("Unknown variety", []),
        (None, []),
        ("", []),
    ],
)
def test_explicit_matching(text: str | None, expected: list[str]) -> None:
    """Only explicitly matched names survive, with full names preferred."""
    assert extract_explicit_grapes(text) == expected


def seed_wines(path: Path) -> None:
    """Create a small legacy cellar with retained data and multiple varieties."""
    assert initialize_database(path)
    with get_db_connection(path) as conn:
        conn.executemany(
            "INSERT INTO wines(id, source, wine_name, wine_type, varietal, description) "
            "VALUES (?, 'manual', ?, 'Red', ?, 'existing wine description')",
            [(1, "First", "Syrah / Cabernet Franc"), (2, "Second", "Shiraz"), (3, "Blend", "Bordeaux Blend")],
        )
        conn.execute("INSERT INTO bottles(wine_id, source) VALUES (1, 'manual')")
        conn.commit()


def test_legacy_migration_parity_idempotency_and_shared_cache(tmp_path: Path) -> None:
    """Migration preserves cellar data and shares descriptions across wines."""
    path = tmp_path / "cellar.db"
    seed_wines(path)
    with get_db_connection(path) as conn:
        before_wines = [tuple(row) for row in conn.execute("SELECT * FROM wines")]
        before_bottles = [tuple(row) for row in conn.execute("SELECT * FROM bottles")]
        conn.execute("DROP TABLE wine_grapes")
        conn.execute("DROP TABLE grapes")
        conn.commit()
    run_migration(path)
    repository = GrapeRepository(path)
    first = repository.get_by_wine(1)
    assert [grape.name for grape in first] == ["cabernet franc", "syrah"]
    assert repository.get_by_wine(3) == []
    syrah = first[1]
    assert repository.update_description(syrah.id, "A cached grape description")
    run_migration(path)
    assert repository.get_by_wine(2)[0].id == syrah.id
    assert repository.get_by_wine(2)[0].description == "A cached grape description"
    assert repository.get_by_id(999) is None
    assert not repository.update_description(999, "Missing")
    with get_db_connection(path) as conn:
        verify_grapes_schema(conn.cursor())
        assert [tuple(row) for row in conn.execute("SELECT * FROM wines")] == before_wines
        assert [tuple(row) for row in conn.execute("SELECT * FROM bottles")] == before_bottles
        assert conn.execute("SELECT count(*) FROM grapes").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM wine_grapes").fetchone()[0] == 3
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_fresh_schema_constraints_and_link_reconciliation(tmp_path: Path) -> None:
    """Fresh databases enforce foreign keys and uniqueness, and imports can relink."""
    path = tmp_path / "fresh.db"
    seed_wines(path)
    repository = GrapeRepository(path)
    repository.sync_wine(1, "Syrah / Shiraz")
    repository.sync_wine(1, "Syrah / Shiraz")
    grape = repository.get_by_wine(1)[0]
    with get_db_connection(path) as conn:
        verify_grapes_schema(conn.cursor())
        for query, values in [
            ("INSERT INTO grapes(name) VALUES (?)", ("syrah",)),
            ("INSERT INTO wine_grapes VALUES (?, ?)", (1, grape.id)),
            ("INSERT INTO wine_grapes VALUES (?, ?)", (999, grape.id)),
            ("INSERT INTO wine_grapes VALUES (?, ?)", (1, 999)),
        ]:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(query, values)
            conn.rollback()
    repository.sync_wine(1, "Chardonnay")
    assert [item.name for item in repository.get_by_wine(1)] == ["chardonnay"]
    repository.sync_wine(1, "Bordeaux Blend")
    assert repository.get_by_wine(1) == []
    assert repository.get_by_id(grape.id) is not None


@pytest.mark.parametrize("partial", ["grapes", "wine_grapes"])
def test_incompatible_partial_schema_rolls_back(tmp_path: Path, partial: str) -> None:
    """An incompatible existing table cannot leave half of the migration applied."""
    path = tmp_path / "partial.db"
    seed_wines(path)
    with get_db_connection(path) as conn:
        conn.execute("DROP TABLE wine_grapes")
        conn.execute("DROP TABLE grapes")
        if partial == "grapes":
            conn.execute("CREATE TABLE grapes(id INTEGER PRIMARY KEY, name TEXT)")
        else:
            conn.execute("CREATE TABLE wine_grapes(wine_id INTEGER, grape_id INTEGER)")
        conn.commit()
        before = [tuple(row) for row in conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY name")]
    with pytest.raises(ValueError, match="incompatible"):
        run_migration(path)
    with get_db_connection(path) as conn:
        assert [tuple(row) for row in conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY name")] == before
        assert conn.execute("SELECT count(*) FROM wines").fetchone()[0] == 3


def test_migration_does_not_create_missing_database(tmp_path: Path) -> None:
    """A mistaken migration path cannot silently create an empty cellar."""
    path = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        run_migration(path)
    assert not path.exists()
