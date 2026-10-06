"""Tests for declared-preference schema, normalization, and repository behavior."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import threading

import pytest

from src.database import get_db_connection, initialize_database
from src.database.models import PreferenceSubjectKind
from src.database.repository import (
    DeclaredPreferenceRepository,
    PreferenceCombinationError,
    PreferenceCountChangedError,
    PreferenceDuplicateError,
    PreferenceLimitReachedError,
    PreferenceNotFoundError,
    PreferenceStoreBusyError,
    PreferenceValueUnresolvedError,
    PreferenceVersionConflictError,
    normalize_preference_text,
)


@pytest.fixture()
def preference_db(temp_dir: Path) -> Path:
    """Create a temporary initialized cellar with deterministic reference rows."""
    db_path = temp_dir / "preferences.db"
    assert initialize_database(db_path)
    with get_db_connection(db_path) as conn:
        producer_id = conn.execute(
            "INSERT INTO producers (name, country) VALUES (?, ?)",
            ("Fixture Producer", "USA"),
        ).lastrowid
        region_id = conn.execute(
            "INSERT INTO regions (primary_name, country) VALUES (?, ?)",
            ("Napa Valley", "USA"),
        ).lastrowid
        conn.execute(
            """
            INSERT INTO wines (source, wine_name, producer_id, wine_type, varietal, region_id)
            VALUES ('manual', 'Fixture Nebbiolo', ?, 'Red', 'Nebbiolo', ?)
            """,
            (producer_id, region_id),
        )
        conn.commit()
    return db_path


def test_new_database_contains_strict_preference_schema(preference_db: Path) -> None:
    """Initialization creates the approved table, columns, and indexes."""
    with get_db_connection(preference_db) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(declared_preferences)")]
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(declared_preferences)")}

        assert columns == [
            "id",
            "subject_kind",
            "stance",
            "normalized_value",
            "display_value",
            "price_minor_units",
            "currency",
            "provenance",
            "version",
            "created_at",
            "updated_at",
        ]
        assert indexes == {
            "uq_declared_preferences_identity",
            "idx_declared_preferences_subject_stance",
        }

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO declared_preferences (
                    subject_kind, stance, normalized_value, display_value
                ) VALUES ('price_ceiling', 'like', 'price_ceiling', 'invalid')
                """
            )


def test_normalize_preference_text_preserves_accents_and_rejects_controls() -> None:
    """Display and identity normalization follow the approved Unicode rules."""
    display, identity = normalize_preference_text("  Côtes\u00a0du   Rhône  ")
    assert display == "Côtes du Rhône"
    assert identity == "côtes du rhône"

    with pytest.raises(PreferenceCombinationError):
        normalize_preference_text("Napa\tValley")
    with pytest.raises(PreferenceCombinationError):
        normalize_preference_text("   ")


def test_create_resolves_every_supported_subject(preference_db: Path) -> None:
    """Repository resolves domain labels and creates all five approved kinds."""
    repository = DeclaredPreferenceRepository(preference_db)

    grape = repository.create("grape", stance="like", value="nebb")
    region = repository.create("region", stance="dislike", value="Napa Valley")
    producer = repository.create("producer", stance="avoid", value="fixture producer")
    style = repository.create("wine_style", stance="like", value="rose")
    price = repository.create("price_ceiling", price_minor_units=4_000, currency="EUR")

    assert (grape.normalized_value, grape.display_value) == ("nebbiolo", "Nebbiolo")
    assert (region.normalized_value, region.display_value) == ("napa valley", "Napa Valley")
    assert (producer.normalized_value, producer.display_value) == ("fixture producer", "Fixture Producer")
    assert (style.normalized_value, style.display_value) == ("rose", "Rosé")
    assert price.price_minor_units == 4_000
    assert price.currency.value == "EUR"
    assert price.stance is None
    assert price.provenance == "explicit_user"
    assert [item.subject_kind for item in repository.list_all()] == sorted(
        [
            PreferenceSubjectKind.GRAPE,
            PreferenceSubjectKind.REGION,
            PreferenceSubjectKind.PRODUCER,
            PreferenceSubjectKind.WINE_STYLE,
            PreferenceSubjectKind.PRICE_CEILING,
        ],
        key=lambda item: item.value,
    )


def test_duplicate_identity_ignores_stance(preference_db: Path) -> None:
    """Changing stance requires update rather than a duplicate create."""
    repository = DeclaredPreferenceRepository(preference_db)
    repository.create("grape", stance="like", value="Nebbiolo")

    with pytest.raises(PreferenceDuplicateError):
        repository.create("grape", stance="avoid", value="nebb")

    assert len(repository.list_all()) == 1


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"subject_kind": "grape", "stance": "like", "value": "Unknown Grape"}, PreferenceValueUnresolvedError),
        ({"subject_kind": "producer", "stance": "like", "value": "Unknown Producer"}, PreferenceValueUnresolvedError),
        ({"subject_kind": "wine_style", "stance": "like", "value": "still"}, PreferenceCombinationError),
        (
            {"subject_kind": "price_ceiling", "stance": "like", "price_minor_units": 100, "currency": "EUR"},
            PreferenceCombinationError,
        ),
        ({"subject_kind": "price_ceiling", "price_minor_units": 0, "currency": "EUR"}, PreferenceCombinationError),
    ],
)
def test_invalid_create_is_atomic(preference_db: Path, kwargs: dict, error_type: type[Exception]) -> None:
    """Resolution and combination failures leave no partial rows."""
    repository = DeclaredPreferenceRepository(preference_db)

    with pytest.raises(error_type):
        repository.create(**kwargs)

    assert repository.list_all() == []


def test_update_delete_and_reset_use_current_versions(preference_db: Path) -> None:
    """Mutable operations enforce versions and reset-count confirmation."""
    repository = DeclaredPreferenceRepository(preference_db)
    grape = repository.create("grape", stance="like", value="Nebbiolo")
    price = repository.create("price_ceiling", price_minor_units=4_000, currency="EUR")

    updated_grape = repository.update_mutable(grape.id, grape.version, stance="avoid")
    updated_price = repository.update_mutable(
        price.id,
        price.version,
        price_minor_units=5_000,
        currency="RON",
    )
    assert updated_grape.stance.value == "avoid"
    assert updated_grape.version == 2
    assert updated_price.price_minor_units == 5_000
    assert updated_price.currency.value == "RON"

    with pytest.raises(PreferenceVersionConflictError):
        repository.update_mutable(grape.id, grape.version, stance="dislike")
    with pytest.raises(PreferenceCombinationError):
        repository.update_mutable(updated_grape.id, updated_grape.version, price_minor_units=100, currency="EUR")
    with pytest.raises(PreferenceCountChangedError):
        repository.reset(expected_count=1)

    assert repository.delete(updated_grape.id, updated_grape.version) == (updated_grape.id, 2)
    assert repository.get_by_id(updated_grape.id) is None
    assert repository.reset(expected_count=1) == 1
    assert repository.list_all() == []


def test_missing_and_stale_delete_are_distinct(preference_db: Path) -> None:
    """Missing IDs and stale versions produce different repository outcomes."""
    repository = DeclaredPreferenceRepository(preference_db)
    preference = repository.create("wine_style", stance="like", value="sparkling")

    with pytest.raises(PreferenceVersionConflictError):
        repository.delete(preference.id, preference.version + 1)
    with pytest.raises(PreferenceNotFoundError):
        repository.delete(999_999, 1)


def test_capacity_check_and_insert_are_atomic(preference_db: Path) -> None:
    """The 100-row limit rejects the next insert without changing state."""
    with get_db_connection(preference_db) as conn:
        conn.executemany(
            """
            INSERT INTO declared_preferences (
                subject_kind, stance, normalized_value, display_value
            ) VALUES ('grape', 'like', ?, ?)
            """,
            [(f"fixture-{index}", f"Fixture {index}") for index in range(100)],
        )
        conn.commit()

    repository = DeclaredPreferenceRepository(preference_db)
    with pytest.raises(PreferenceLimitReachedError):
        repository.create("producer", stance="like", value="Fixture Producer")
    assert len(repository.list_all()) == 100


def test_concurrent_duplicate_create_has_exactly_one_winner(preference_db: Path) -> None:
    """Serialized creates produce one row and one stable duplicate failure."""
    barrier = threading.Barrier(2)

    def create_style() -> str:
        repository = DeclaredPreferenceRepository(preference_db)
        barrier.wait()
        try:
            repository.create("wine_style", stance="like", value="sparkling")
            return "created"
        except PreferenceDuplicateError:
            return "duplicate"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: create_style(), range(2)))

    assert sorted(outcomes) == ["created", "duplicate"]
    assert len(DeclaredPreferenceRepository(preference_db).list_all()) == 1


def test_write_lock_exhaustion_is_bounded(preference_db: Path) -> None:
    """A held writer lock becomes the explicit busy outcome after the bound."""
    lock_connection = sqlite3.connect(preference_db, isolation_level=None)
    lock_connection.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(PreferenceStoreBusyError):
            DeclaredPreferenceRepository(preference_db).create(
                "wine_style",
                stance="like",
                value="sparkling",
            )
    finally:
        lock_connection.rollback()
        lock_connection.close()
