"""Deterministic integration tests for declared preferences in taste tools."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.agents.tools import taste_profile_tools
from src.database.models import DeclaredPreference
from src.database.repository import PreferenceStoreBusyError


def _preference(
    preference_id: int,
    subject_kind: str,
    *,
    stance: str | None = None,
    normalized_value: str,
    display_value: str | None = None,
    price_minor_units: int | None = None,
    currency: str | None = None,
) -> DeclaredPreference:
    """Build one valid persisted preference fixture."""
    timestamp = datetime(2026, 10, 5, tzinfo=timezone.utc)
    return DeclaredPreference(
        id=preference_id,
        subject_kind=subject_kind,
        stance=stance,
        normalized_value=normalized_value,
        display_value=display_value,
        price_minor_units=price_minor_units,
        currency=currency,
        provenance="explicit_user",
        version=1,
        created_at=timestamp,
        updated_at=timestamp,
    )


def _patch_profile_repositories(
    monkeypatch: pytest.MonkeyPatch,
    *,
    preferences: list[DeclaredPreference],
    tastings: list[dict[str, object]],
) -> tuple[MagicMock, MagicMock]:
    """Bind the profile tool to controlled repository results."""
    preference_repository = MagicMock()
    preference_repository.list_all.return_value = preferences
    tasting_repository = MagicMock()
    tasting_repository.get_all_with_wine_info.return_value = tastings
    monkeypatch.setattr(
        taste_profile_tools,
        "DeclaredPreferenceRepository",
        lambda _db_path: preference_repository,
    )
    monkeypatch.setattr(
        taste_profile_tools,
        "TastingRepository",
        lambda _db_path: tasting_repository,
    )
    return preference_repository, tasting_repository


def test_profile_exposes_bounded_declared_items_without_observed_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declared facts remain visible with empty history and hide storage internals."""
    preferences = [
        _preference(
            1,
            "grape",
            stance="like",
            normalized_value="nebbiolo",
            display_value="Nebbiolo",
        ),
        _preference(
            2,
            "price_ceiling",
            normalized_value="price_ceiling",
            price_minor_units=4_025,
            currency="EUR",
        ),
    ]
    preference_repository, tasting_repository = _patch_profile_repositories(
        monkeypatch,
        preferences=preferences,
        tastings=[],
    )

    result = taste_profile_tools.get_user_taste_profile.invoke({})

    assert result["total_wines_rated"] == 0
    assert result["declared_preferences"] == {
        "items": [
            {
                "subject_kind": "grape",
                "stance": "like",
                "value": "Nebbiolo",
                "provenance": "explicit_user",
            },
            {
                "subject_kind": "price_ceiling",
                "stance": None,
                "price_amount": "40.25",
                "currency": "EUR",
                "provenance": "explicit_user",
            },
        ],
        "total": 2,
    }
    assert result["preference_conflicts"] == []
    assert "normalized_value" not in str(result)
    assert "created_at" not in str(result)
    assert "updated_at" not in str(result)
    preference_repository.list_all.assert_called_once_with()
    tasting_repository.get_all_with_wine_info.assert_called_once_with(has_rating=True)


def test_profile_reports_only_approved_observed_rating_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact canonical identities disclose high/low conflicts and ignore neutral evidence."""
    preferences = [
        _preference(
            1,
            "grape",
            stance="like",
            normalized_value="nebbiolo",
            display_value="Nebbiolo",
        ),
        _preference(
            2,
            "region",
            stance="avoid",
            normalized_value="napa valley",
            display_value="Napa Valley",
        ),
        _preference(
            3,
            "wine_style",
            stance="dislike",
            normalized_value="white",
            display_value="White",
        ),
    ]
    tastings = [
        {
            "personal_rating": 75,
            "varietal": "Nebbiolo",
            "region_name": "Piedmont",
            "producer_name": "Producer A",
            "wine_type": "Red",
        },
        {
            "personal_rating": 95,
            "varietal": "Cabernet Sauvignon",
            "region_name": "Napa Valley",
            "producer_name": "Producer B",
            "wine_type": "Red",
        },
        {
            "personal_rating": 85,
            "varietal": "Chardonnay",
            "region_name": "Burgundy",
            "producer_name": "Producer C",
            "wine_type": "White",
        },
    ]
    _patch_profile_repositories(monkeypatch, preferences=preferences, tastings=tastings)

    result = taste_profile_tools.get_user_taste_profile.invoke({})

    assert result["preference_conflicts"] == [
        {
            "subject_kind": "grape",
            "value": "Nebbiolo",
            "declared_stance": "like",
            "observed_signal": "low_rating",
            "average_rating": 75.0,
            "count": 1,
        },
        {
            "subject_kind": "region",
            "value": "Napa Valley",
            "declared_stance": "avoid",
            "observed_signal": "high_rating",
            "average_rating": 95.0,
            "count": 1,
        },
    ]
    assert {item["varietal"] for item in result["favorite_varietals"]} == {
        "Cabernet Sauvignon",
        "Chardonnay",
        "Nebbiolo",
    }
    assert result["declared_preferences"]["total"] == 3


def test_profile_propagates_preference_store_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preference read failures never masquerade as an empty declared profile."""
    preference_repository = MagicMock()
    preference_repository.list_all.side_effect = PreferenceStoreBusyError("synthetic busy")
    monkeypatch.setattr(
        taste_profile_tools,
        "DeclaredPreferenceRepository",
        lambda _db_path: preference_repository,
    )

    with pytest.raises(PreferenceStoreBusyError):
        taste_profile_tools.get_user_taste_profile.invoke({})

    preference_repository.list_all.assert_called_once_with()


def test_canonical_identity_reuses_persistence_terminology() -> None:
    """Tool matching canonicalizes aliases with the repository identity contract."""
    grape_signals = taste_profile_tools._observed_preference_signals(
        [{"personal_rating": 91, "varietal": "nebb"}]
    )

    assert grape_signals[("grape", "nebbiolo")] == [91]
