"""API contract tests for declared taste preferences."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.database import get_db_connection, initialize_database
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
)


@pytest.fixture()
def preference_db(temp_dir: Path) -> Path:
    """Create an initialized temporary cellar with resolvable domain values."""
    db_path = temp_dir / "taste-preferences-api.db"
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


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, preference_db: Path) -> TestClient:
    """Bind preference routes to a real repository over the temporary cellar."""
    from src.api.main import app

    monkeypatch.setattr(
        "src.api.routes.taste_profile.DeclaredPreferenceRepository",
        lambda: DeclaredPreferenceRepository(preference_db),
    )
    return TestClient(app)


def test_list_and_options_are_typed_sorted_and_empty(client: TestClient) -> None:
    """An empty profile and canonical creation options have stable wire shapes."""
    list_response = client.get("/api/taste-profile/preferences")
    options_response = client.get("/api/taste-profile/preferences/options")

    assert list_response.status_code == 200
    assert list_response.json() == {"items": [], "total": 0, "max_items": 100}

    assert options_response.status_code == 200
    options = options_response.json()
    assert options["subject_kinds"] == ["grape", "region", "producer", "wine_style", "price_ceiling"]
    assert options["stances"] == ["like", "dislike", "avoid"]
    assert options["wine_styles"] == ["red", "white", "rose", "sparkling", "dessert", "fortified"]
    assert options["currencies"] == ["EUR", "RON", "USD", "GBP", "CHF"]
    assert "Nebbiolo" in options["grapes"]
    assert "Napa Valley" in options["regions"]
    assert options["producers"] == ["Fixture Producer"]
    assert options["grapes"] == sorted(set(options["grapes"]), key=str.casefold)
    assert options["regions"] == sorted(set(options["regions"]), key=str.casefold)
    assert options["max_items"] == 100


@pytest.mark.parametrize(
    "payload",
    [
        {"subject_kind": "grape", "stance": "like", "value": "nebb"},
        {"subject_kind": "region", "stance": "dislike", "value": "Napa Valley"},
        {"subject_kind": "producer", "stance": "avoid", "value": "fixture producer"},
        {"subject_kind": "wine_style", "stance": "like", "value": "rose"},
        {"subject_kind": "price_ceiling", "price_amount": "40", "currency": "EUR"},
    ],
)
def test_create_supports_every_subject_variant(client: TestClient, payload: dict[str, object]) -> None:
    """Each discriminated request variant returns its persisted representation."""
    response = client.post("/api/taste-profile/preferences", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["subject_kind"] == payload["subject_kind"]
    assert body["provenance"] == "explicit_user"
    assert body["version"] == 1
    assert body["created_at"].endswith("Z")
    assert body["updated_at"].endswith("Z")
    if payload["subject_kind"] == "price_ceiling":
        assert body["price_amount"] == "40.00"
        assert body["stance"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"subject_kind": "grape", "stance": "like", "value": "Nebbiolo", "currency": "EUR"},
        {"subject_kind": "price_ceiling", "price_amount": 40, "currency": "EUR"},
        {"subject_kind": "price_ceiling", "price_amount": "40.123", "currency": "EUR"},
        {"subject_kind": "wine_style", "stance": "like", "value": "still"},
        {"subject_kind": "grape", "stance": "like", "value": "x" * 121},
    ],
)
def test_create_rejects_cross_variant_and_invalid_fields_atomically(
    client: TestClient,
    payload: dict[str, object],
) -> None:
    """Request-shape failures return standard validation errors without writes."""
    response = client.post("/api/taste-profile/preferences", json=payload)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert client.get("/api/taste-profile/preferences").json()["total"] == 0


def test_create_duplicate_and_unresolved_values_return_bounded_errors(client: TestClient) -> None:
    """Identity and domain failures never echo user-entered values."""
    first = client.post(
        "/api/taste-profile/preferences",
        json={"subject_kind": "grape", "stance": "like", "value": "Nebbiolo"},
    )
    duplicate = client.post(
        "/api/taste-profile/preferences",
        json={"subject_kind": "grape", "stance": "avoid", "value": "nebb"},
    )
    unresolved = client.post(
        "/api/taste-profile/preferences",
        json={"subject_kind": "producer", "stance": "like", "value": "Sensitive Unknown Producer"},
    )

    assert first.status_code == 201
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "preference_duplicate"
    assert unresolved.status_code == 422
    assert unresolved.json()["detail"]["code"] == "preference_value_unresolved"
    assert "Sensitive Unknown Producer" not in unresolved.text


def test_patch_dispatches_by_fields_and_enforces_target_kind(client: TestClient) -> None:
    """Patch variants update only the mutable fields valid for the target row."""
    grape = client.post(
        "/api/taste-profile/preferences",
        json={"subject_kind": "grape", "stance": "like", "value": "Nebbiolo"},
    ).json()
    price = client.post(
        "/api/taste-profile/preferences",
        json={"subject_kind": "price_ceiling", "price_amount": "40", "currency": "EUR"},
    ).json()

    grape_update = client.patch(
        f"/api/taste-profile/preferences/{grape['id']}",
        json={"expected_version": 1, "stance": "avoid"},
    )
    price_update = client.patch(
        f"/api/taste-profile/preferences/{price['id']}",
        json={"expected_version": 1, "price_amount": "55.25", "currency": "USD"},
    )
    mismatched = client.patch(
        f"/api/taste-profile/preferences/{grape['id']}",
        json={"expected_version": 2, "price_amount": "20", "currency": "EUR"},
    )

    assert grape_update.status_code == 200
    assert grape_update.json()["stance"] == "avoid"
    assert grape_update.json()["version"] == 2
    assert price_update.status_code == 200
    assert price_update.json()["price_amount"] == "55.25"
    assert price_update.json()["currency"] == "USD"
    assert price_update.json()["version"] == 2
    assert mismatched.status_code == 422
    assert mismatched.json()["detail"]["code"] == "preference_combination_invalid"


def test_delete_is_immediate_and_distinguishes_stale_from_missing(client: TestClient) -> None:
    """Delete applies immediately and preserves optimistic-conflict semantics."""
    created = client.post(
        "/api/taste-profile/preferences",
        json={"subject_kind": "wine_style", "stance": "like", "value": "red"},
    ).json()

    stale = client.delete(f"/api/taste-profile/preferences/{created['id']}?expected_version=2")
    deleted = client.delete(f"/api/taste-profile/preferences/{created['id']}?expected_version=1")
    missing = client.delete(f"/api/taste-profile/preferences/{created['id']}?expected_version=1")

    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "preference_version_conflict"
    assert deleted.status_code == 200
    assert deleted.json() == {"id": created["id"], "deleted_version": 1}
    assert client.get("/api/taste-profile/preferences").json()["total"] == 0
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "preference_not_found"


def test_reset_requires_confirmation_and_current_count(client: TestClient) -> None:
    """Reset is explicit, count-checked, atomic, and immediately visible."""
    client.post(
        "/api/taste-profile/preferences",
        json={"subject_kind": "grape", "stance": "like", "value": "Nebbiolo"},
    )
    missing_confirmation = client.post(
        "/api/taste-profile/preferences/reset",
        json={"expected_count": 1},
    )
    stale = client.post(
        "/api/taste-profile/preferences/reset",
        json={"confirm": True, "expected_count": 0},
    )

    assert missing_confirmation.status_code == 422
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "preference_count_changed"
    assert client.get("/api/taste-profile/preferences").json()["total"] == 1

    reset = client.post(
        "/api/taste-profile/preferences/reset",
        json={"confirm": True, "expected_count": 1},
    )

    assert reset.status_code == 200
    assert reset.json() == {"deleted_count": 1}
    assert client.get("/api/taste-profile/preferences").json()["total"] == 0


@pytest.mark.parametrize(
    ("error", "status_code", "code"),
    [
        (PreferenceNotFoundError("raw"), 404, "preference_not_found"),
        (PreferenceDuplicateError("raw"), 409, "preference_duplicate"),
        (PreferenceVersionConflictError("raw"), 409, "preference_version_conflict"),
        (PreferenceCountChangedError("raw"), 409, "preference_count_changed"),
        (PreferenceLimitReachedError("raw"), 409, "preference_limit_reached"),
        (PreferenceValueUnresolvedError("raw"), 422, "preference_value_unresolved"),
        (PreferenceCombinationError("raw"), 422, "preference_combination_invalid"),
        (PreferenceStoreBusyError("raw"), 503, "preference_store_busy"),
    ],
)
def test_repository_errors_have_bounded_public_contracts(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status_code: int,
    code: str,
) -> None:
    """Every repository failure maps to its approved status and bounded code."""
    from src.api.main import app

    repository = MagicMock()
    repository.list_all.side_effect = error
    monkeypatch.setattr("src.api.routes.taste_profile.DeclaredPreferenceRepository", lambda: repository)

    response = TestClient(app).get("/api/taste-profile/preferences")

    assert response.status_code == status_code
    assert response.json()["detail"]["code"] == code
    assert "raw" not in response.text
    if status_code == 503:
        assert response.headers["Retry-After"] == "1"


def test_unexpected_repository_error_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected storage failures do not expose exception or SQL details."""
    from src.api.main import app

    repository = MagicMock()
    repository.list_all.side_effect = RuntimeError("SELECT secret FROM private_table")
    monkeypatch.setattr("src.api.routes.taste_profile.DeclaredPreferenceRepository", lambda: repository)

    response = TestClient(app).get("/api/taste-profile/preferences")

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "preference_store_unavailable",
        "message": "Preference store is unavailable.",
    }
    assert "SELECT" not in response.text
