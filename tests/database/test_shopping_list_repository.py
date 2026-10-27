"""Tests for deterministic Shopping List repository behavior."""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.database import get_db_connection, initialize_database
from src.database.models import ShoppingListStatus
from src.database.repository.shopping_list import (
    ShoppingListDuplicateError,
    ShoppingListRepository,
    ShoppingListStoreUnavailableError,
    ShoppingListValidationError,
    ShoppingListVersionConflictError,
)


@pytest.fixture
def shopping_db(temp_dir: Path) -> Path:
    """Create a disposable cellar with two catalogue wines and inventory."""
    path = temp_dir / "shopping.db"
    assert initialize_database(path)
    with get_db_connection(path) as connection:
        producer_id = connection.execute("INSERT INTO producers (name) VALUES ('Example Producer')").lastrowid
        connection.execute(
            "INSERT INTO wines (id, source, external_id, wine_name, producer_id, vintage, wine_type, bottle_size) "
            "VALUES (10, 'manual', 'wine-10', 'Catalogue Wine', ?, 2020, 'Red', '1.5L')",
            (producer_id,),
        )
        connection.execute(
            "INSERT INTO wines (id, source, external_id, wine_name, vintage, wine_type) "
            "VALUES (11, 'manual', 'wine-11', 'Literal % Wine', 2021, 'White')"
        )
        connection.execute(
            "INSERT INTO bottles (wine_id, source, quantity, status) VALUES (10, 'manual', 2, 'in_cellar')"
        )
        connection.commit()
    return path


def test_manual_and_catalogue_items_survive_reopen(shopping_db: Path) -> None:
    """Both identity modes persist their authoritative fields across connections."""
    repository = ShoppingListRepository(shopping_db)
    manual = repository.create_manual(
        "  Château   Test  ",
        vintage=2019,
        quantity=3,
        priority="high",
        notes=" Find it ",
        retailer_url="https://shop.test/wine",
    )
    catalogue = repository.create_catalogue(10)

    reopened = ShoppingListRepository(shopping_db)
    assert reopened.get_by_id(manual.id).model_dump() == manual.model_dump()
    assert reopened.get_by_id(catalogue.id).model_dump() == catalogue.model_dump()
    assert catalogue.name == "Catalogue Wine"
    assert catalogue.vintage == 2020
    assert catalogue.link_available is True


def test_active_duplicate_identity_is_exact_and_reports_existing_item(shopping_db: Path) -> None:
    """Whitespace/case normalize, while accents, punctuation, and vintages remain distinct."""
    repository = ShoppingListRepository(shopping_db)
    original = repository.create_manual("Cuvée Test", vintage=None)

    with pytest.raises(ShoppingListDuplicateError) as duplicate:
        repository.create_manual("  CUVÉE   TEST ")
    assert duplicate.value.existing_item_id == original.id

    assert repository.create_manual("Cuvee Test").id != original.id
    assert repository.create_manual("Cuvée-Test").id != original.id
    assert repository.create_manual("Cuvée Test", vintage=2022).id != original.id

    catalogue = repository.create_catalogue(10)
    with pytest.raises(ShoppingListDuplicateError) as catalogue_duplicate:
        repository.create_catalogue(10)
    assert catalogue_duplicate.value.existing_item_id == catalogue.id


def test_concurrent_equivalent_creates_leave_one_wanted_item(shopping_db: Path) -> None:
    """The unique index resolves concurrent create races authoritatively."""
    start = threading.Event()

    def create() -> tuple[str, int]:
        start.wait()
        try:
            item = ShoppingListRepository(shopping_db).create_manual("Concurrent Wine", vintage=2024)
            return "created", item.id
        except ShoppingListDuplicateError as error:
            return "duplicate", error.existing_item_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(create) for _ in range(2)]
        start.set()
        results = [future.result() for future in futures]

    assert sorted(result[0] for result in results) == ["created", "duplicate"]
    assert results[0][1] == results[1][1]


def test_purchase_restore_preserves_inventory_and_restore_conflict_is_atomic(shopping_db: Path) -> None:
    """Lifecycle changes never alter inventory and a conflicting restore keeps history unchanged."""
    repository = ShoppingListRepository(shopping_db)
    item = repository.create_catalogue(10, quantity=4)
    with get_db_connection(shopping_db) as connection:
        before = tuple(connection.execute("SELECT q_quantity FROM wines WHERE id = 10").fetchone()) + tuple(
            connection.execute("SELECT quantity, status FROM bottles WHERE wine_id = 10").fetchone()
        )

    purchased = repository.update(item.id, item.version, {"status": "purchased"})
    assert purchased.status == ShoppingListStatus.PURCHASED
    assert purchased.purchased_at is not None
    wanted = repository.create_catalogue(10)

    with pytest.raises(ShoppingListDuplicateError) as conflict:
        repository.update(purchased.id, purchased.version, {"status": "wanted"})
    assert conflict.value.existing_item_id == wanted.id
    unchanged = repository.get_by_id(purchased.id)
    assert unchanged.status == ShoppingListStatus.PURCHASED
    assert unchanged.version == purchased.version

    with get_db_connection(shopping_db) as connection:
        after = tuple(connection.execute("SELECT q_quantity FROM wines WHERE id = 10").fetchone()) + tuple(
            connection.execute("SELECT quantity, status FROM bottles WHERE wine_id = 10").fetchone()
        )
    assert after == before


def test_stale_updates_and_deletes_do_not_overwrite_newer_state(shopping_db: Path) -> None:
    """Optimistic versions protect update and delete operations."""
    repository = ShoppingListRepository(shopping_db)
    item = repository.create_manual("Versioned Wine")
    updated = repository.update(item.id, item.version, {"quantity": 2})

    with pytest.raises(ShoppingListVersionConflictError):
        repository.update(item.id, item.version, {"quantity": 3})
    with pytest.raises(ShoppingListVersionConflictError):
        repository.delete(item.id, item.version)
    assert repository.get_by_id(item.id).quantity == 2
    assert repository.delete(item.id, updated.version) == (item.id, updated.version)


def test_catalogue_deletion_preserves_snapshot_and_original_identity(shopping_db: Path) -> None:
    """A missing catalogue link remains readable and cannot silently relink."""
    repository = ShoppingListRepository(shopping_db)
    item = repository.create_catalogue(10)
    with get_db_connection(shopping_db) as connection:
        connection.execute("DELETE FROM wines WHERE id = 10")
        replacement_id = connection.execute(
            "INSERT INTO wines (source, external_id, wine_name, vintage, wine_type) "
            "VALUES ('manual', 'replacement', 'Catalogue Wine', 2020, 'Red')"
        ).lastrowid
        connection.commit()

    retained = repository.get_by_id(item.id)
    assert retained.wine_id is None
    assert retained.catalogue_identity_id == 10
    assert retained.name == "Catalogue Wine"
    assert retained.link_available is False
    assert replacement_id != retained.catalogue_identity_id
    assert repository.update(item.id, item.version, {"notes": "Still wanted"}).notes == "Still wanted"


def test_manual_identity_edits_revalidate_duplicates(shopping_db: Path) -> None:
    """Manual edits recheck active uniqueness while catalogue identity stays immutable."""
    repository = ShoppingListRepository(shopping_db)
    first = repository.create_manual("First", vintage=2020)
    second = repository.create_manual("Second", vintage=2020)
    with pytest.raises(ShoppingListDuplicateError) as duplicate:
        repository.update(second.id, second.version, {"name": " FIRST "})
    assert duplicate.value.existing_item_id == first.id
    assert repository.get_by_id(second.id).name == "Second"

    catalogue = repository.create_catalogue(10)
    with pytest.raises(ShoppingListValidationError, match="immutable"):
        repository.update(catalogue.id, catalogue.version, {"name": "Override"})


def test_search_paging_and_catalogue_options_are_literal_and_bounded(shopping_db: Path) -> None:
    """Literal wildcards, totals, priority order, and catalogue pages are deterministic."""
    repository = ShoppingListRepository(shopping_db)
    normal = repository.create_manual("Literal % Match")
    high = repository.create_manual("Other", priority="high", notes="literal % note")
    repository.create_manual("Ordinary")

    items, total = repository.list_items(search="%", limit=1)
    assert total == 2
    assert [item.id for item in items] == [high.id]
    second_page, _ = repository.list_items(search="%", limit=1, offset=1)
    assert [item.id for item in second_page] == [normal.id]

    options, has_more = repository.list_catalogue_options("%", limit=1)
    assert [option.id for option in options] == [11]
    assert has_more is False
    assert options[0].bottle_size == "750ml"


def test_missing_schema_and_invalid_url_fail_explicitly(temp_dir: Path, shopping_db: Path) -> None:
    """Unavailable schema and unsafe retailer URLs are never converted to empty success."""
    missing_schema = temp_dir / "legacy.db"
    sqlite3.connect(missing_schema).close()
    with pytest.raises(ShoppingListStoreUnavailableError):
        ShoppingListRepository(missing_schema).list_items()
    with pytest.raises(ShoppingListValidationError, match="credentials"):
        ShoppingListRepository(shopping_db).create_manual("Unsafe URL", retailer_url="https://user:pass@shop.test")
