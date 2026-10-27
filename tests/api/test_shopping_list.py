"""API contract tests for the local Shopping List."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.database import get_db_connection, initialize_database
from src.database.repository import (
    ShoppingListDuplicateError,
    ShoppingListNotFoundError,
    ShoppingListRepository,
    ShoppingListStoreBusyError,
    ShoppingListStoreUnavailableError,
    ShoppingListValidationError,
    ShoppingListVersionConflictError,
)


@pytest.fixture()
def shopping_api_db(temp_dir: Path) -> Path:
    """Create a temporary cellar with catalogue choices and inventory."""
    path = temp_dir / "shopping-api.db"
    assert initialize_database(path)
    with get_db_connection(path) as connection:
        producer_id = connection.execute("INSERT INTO producers (name) VALUES ('API Producer')").lastrowid
        connection.execute(
            "INSERT INTO wines (id, source, external_id, wine_name, producer_id, vintage, wine_type, bottle_size) "
            "VALUES (21, 'manual', 'api-21', 'API Catalogue Wine', ?, 2021, 'Red', '1.5L')",
            (producer_id,),
        )
        connection.execute(
            "INSERT INTO bottles (wine_id, source, quantity, status) VALUES (21, 'manual', 2, 'in_cellar')"
        )
        connection.commit()
    return path


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, shopping_api_db: Path) -> TestClient:
    """Bind Shopping List routes to the disposable database."""
    from src.api.main import app

    monkeypatch.setattr(
        "src.api.routes.shopping_list.ShoppingListRepository",
        lambda: ShoppingListRepository(shopping_api_db),
    )
    return TestClient(app)


def test_empty_list_and_catalogue_search_have_bounded_shapes(client: TestClient) -> None:
    """List defaults and catalogue metadata are explicit and bounded."""
    empty = client.get("/api/shopping-list/items")
    options = client.get("/api/shopping-list/catalogue-options?search=API")

    assert empty.status_code == 200
    assert empty.json() == {"items": [], "total": 0, "limit": 50, "offset": 0}
    assert options.status_code == 200
    assert options.json() == {
        "items": [
            {
                "id": 21,
                "name": "API Catalogue Wine",
                "vintage": 2021,
                "producer": "API Producer",
                "bottle_size": "1.5L",
            }
        ],
        "has_more": False,
        "limit": 20,
        "offset": 0,
    }


def test_full_manual_lifecycle_is_authoritative_and_inventory_safe(client: TestClient, shopping_api_db: Path) -> None:
    """Create, edit, purchase, restore, and delete preserve inventory state."""
    created = client.post(
        "/api/shopping-list/items",
        json={
            "identity_kind": "manual",
            "name": " Manual Wine ",
            "vintage": 2020,
            "quantity": 2,
            "priority": "high",
            "notes": "Find locally",
            "retailer_url": "https://shop.test/wine",
        },
    )
    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "Manual Wine"
    assert body["status"] == "wanted"
    assert body["version"] == 1

    edited = client.patch(
        f"/api/shopping-list/items/{body['id']}",
        json={"expected_version": 1, "quantity": 3, "notes": None},
    )
    assert edited.status_code == 200
    assert edited.json()["quantity"] == 3
    assert edited.json()["notes"] is None
    assert edited.json()["version"] == 2

    purchased = client.patch(
        f"/api/shopping-list/items/{body['id']}",
        json={"expected_version": 2, "status": "purchased"},
    )
    assert purchased.status_code == 200
    assert purchased.json()["purchased_at"].endswith("Z")
    restored = client.patch(
        f"/api/shopping-list/items/{body['id']}",
        json={"expected_version": 3, "status": "wanted"},
    )
    assert restored.status_code == 200
    assert restored.json()["purchased_at"] is None

    deleted = client.delete(f"/api/shopping-list/items/{body['id']}?expected_version=4")
    assert deleted.status_code == 200
    assert deleted.json() == {"id": body["id"], "deleted_version": 4}
    with get_db_connection(shopping_api_db) as connection:
        assert tuple(connection.execute("SELECT quantity, status FROM bottles WHERE wine_id = 21").fetchone()) == (
            2,
            "in_cellar",
        )


def test_catalogue_creation_uses_server_snapshot_and_blocks_duplicate(client: TestClient) -> None:
    """Catalogue display fields are server-owned and duplicate errors identify the existing item."""
    created = client.post(
        "/api/shopping-list/items",
        json={"identity_kind": "catalogue", "wine_id": 21, "quantity": 2},
    )
    override = client.post(
        "/api/shopping-list/items",
        json={"identity_kind": "catalogue", "wine_id": 21, "name": "Override"},
    )
    duplicate = client.post(
        "/api/shopping-list/items",
        json={"identity_kind": "catalogue", "wine_id": 21},
    )

    assert created.status_code == 201
    assert created.json()["name"] == "API Catalogue Wine"
    assert created.json()["vintage"] == 2021
    assert created.json()["link_available"] is True
    assert override.status_code == 422
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == {
        "code": "shopping_list_duplicate",
        "message": "A Wanted item already uses this identity.",
        "existing_item_id": created.json()["id"],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"identity_kind": "manual", "name": ""},
        {"identity_kind": "manual", "name": "Wine", "quantity": 0},
        {"identity_kind": "manual", "name": "Wine", "vintage": 999},
        {"identity_kind": "manual", "name": "Wine", "unknown": True},
        {"identity_kind": "catalogue", "wine_id": 21, "vintage": 2021},
    ],
)
def test_create_rejects_invalid_or_unknown_fields(client: TestClient, payload: dict[str, object]) -> None:
    """Strict request variants reject invalid values without writes."""
    response = client.post("/api/shopping-list/items", json=payload)
    assert response.status_code == 422
    assert client.get("/api/shopping-list/items").json()["total"] == 0


def test_stale_patch_and_delete_return_conflicts(client: TestClient) -> None:
    """HTTP mutations expose optimistic-concurrency failures without overwriting state."""
    item = client.post("/api/shopping-list/items", json={"identity_kind": "manual", "name": "Versioned"}).json()
    assert (
        client.patch(f"/api/shopping-list/items/{item['id']}", json={"expected_version": 1, "quantity": 2}).status_code
        == 200
    )
    stale_patch = client.patch(f"/api/shopping-list/items/{item['id']}", json={"expected_version": 1, "quantity": 3})
    stale_delete = client.delete(f"/api/shopping-list/items/{item['id']}?expected_version=1")

    assert stale_patch.status_code == 409
    assert stale_patch.json()["detail"]["code"] == "shopping_list_version_conflict"
    assert stale_delete.status_code == 409
    assert client.get("/api/shopping-list/items").json()["items"][0]["quantity"] == 2


def test_status_search_and_paging_are_deterministic(client: TestClient) -> None:
    """Status filters, literal wildcards, priority, totals, and offsets remain stable."""
    normal = client.post("/api/shopping-list/items", json={"identity_kind": "manual", "name": "Literal % Wine"}).json()
    high = client.post(
        "/api/shopping-list/items",
        json={"identity_kind": "manual", "name": "Other", "notes": "literal %", "priority": "high"},
    ).json()
    purchased = client.post("/api/shopping-list/items", json={"identity_kind": "manual", "name": "Purchased"}).json()
    client.patch(
        f"/api/shopping-list/items/{purchased['id']}",
        json={"expected_version": 1, "status": "purchased"},
    )

    first = client.get("/api/shopping-list/items?search=%25&limit=1")
    second = client.get("/api/shopping-list/items?search=%25&limit=1&offset=1")
    history = client.get("/api/shopping-list/items?status=purchased")
    assert first.json()["total"] == 2
    assert first.json()["items"][0]["id"] == high["id"]
    assert second.json()["items"][0]["id"] == normal["id"]
    assert [item["id"] for item in history.json()["items"]] == [purchased["id"]]


@pytest.mark.parametrize(
    ("error", "status_code", "code"),
    [
        (ShoppingListNotFoundError("raw"), 404, "shopping_list_not_found"),
        (ShoppingListDuplicateError(42), 409, "shopping_list_duplicate"),
        (ShoppingListVersionConflictError("raw"), 409, "shopping_list_version_conflict"),
        (ShoppingListValidationError("raw"), 422, "shopping_list_mutation_invalid"),
        (ShoppingListStoreBusyError("raw"), 503, "shopping_list_store_busy"),
        (ShoppingListStoreUnavailableError("raw"), 503, "shopping_list_schema_unavailable"),
    ],
)
def test_repository_errors_have_stable_sanitized_contracts(
    monkeypatch: pytest.MonkeyPatch, error: Exception, status_code: int, code: str
) -> None:
    """Expected storage failures map to bounded public errors."""
    from src.api.main import app

    repository = MagicMock()
    repository.list_items.side_effect = error
    monkeypatch.setattr("src.api.routes.shopping_list.ShoppingListRepository", lambda: repository)
    response = TestClient(app).get("/api/shopping-list/items")
    assert response.status_code == status_code
    assert response.json()["detail"]["code"] == code
    assert "raw" not in response.text
    if code == "shopping_list_store_busy":
        assert response.headers["Retry-After"] == "1"


def test_unexpected_failure_is_500_and_ordinary_crud_makes_no_external_calls(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Unexpected errors are sanitized and local CRUD never invokes URL or model clients."""
    from src.api.main import app

    with (
        patch("urllib.request.urlopen", side_effect=AssertionError("external URL call")),
        patch("src.agents.llm.load_base_model", side_effect=AssertionError("model call")),
    ):
        assert (
            client.post(
                "/api/shopping-list/items",
                json={"identity_kind": "manual", "name": "Local Only", "retailer_url": "https://shop.test"},
            ).status_code
            == 201
        )

    repository = MagicMock()
    repository.list_items.side_effect = RuntimeError("SELECT secret FROM private_table")
    monkeypatch.setattr("src.api.routes.shopping_list.ShoppingListRepository", lambda: repository)
    response = TestClient(app).get("/api/shopping-list/items")
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "shopping_list_store_unavailable"
    assert "secret" not in response.text
