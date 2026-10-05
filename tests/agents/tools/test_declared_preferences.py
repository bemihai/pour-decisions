"""Deterministic integration tests for declared preferences in taste tools."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.agents.tools import taste_profile_tools
from src.database.models import Bottle, DeclaredPreference, Wine
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


def _wine(
    wine_id: int,
    *,
    name: str,
    producer: str = "Fixture Producer",
    varietal: str = "Nebbiolo",
    region: str = "Piedmont",
    wine_type: str = "Red",
) -> Wine:
    """Build one cellar-wine fixture with joined matching fields."""
    return Wine(
        id=wine_id,
        source="manual",
        wine_name=name,
        producer_name=producer,
        varietal=varietal,
        region_name=region,
        wine_type=wine_type,
    )


def _bottle(
    bottle_id: int,
    wine_id: int,
    *,
    price: float | None = 30.0,
    currency: str = "EUR",
    quantity: int = 1,
) -> Bottle:
    """Build one owned-bottle fixture."""
    return Bottle(
        id=bottle_id,
        wine_id=wine_id,
        source="manual",
        status="in_cellar",
        purchase_price=price,
        currency=currency,
        quantity=quantity,
        location="Cellar",
    )


def _patch_recommendation_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: dict[str, object],
    wines: list[Wine],
    bottles_by_wine: dict[int, list[Bottle]],
) -> tuple[MagicMock, MagicMock]:
    """Bind recommendation logic to deterministic profile and cellar inputs."""
    profile_tool = MagicMock()
    profile_tool.invoke.return_value = profile
    wine_repository = MagicMock()
    wine_repository.get_all.return_value = wines
    bottle_repository = MagicMock()
    bottle_repository.get_by_wine.side_effect = lambda wine_id, status: bottles_by_wine.get(wine_id, [])
    monkeypatch.setattr(taste_profile_tools, "get_user_taste_profile", profile_tool)
    monkeypatch.setattr(taste_profile_tools, "WineRepository", lambda _db_path: wine_repository)
    monkeypatch.setattr(taste_profile_tools, "BottleRepository", lambda _db_path: bottle_repository)
    return wine_repository, bottle_repository


def _profile(**overrides: object) -> dict[str, object]:
    """Build the minimum released profile plus additive declared sections."""
    profile: dict[str, object] = {
        "total_wines_rated": 3,
        "average_rating": 90.0,
        "favorite_regions": [],
        "favorite_varietals": [],
        "preferred_type": None,
        "declared_preferences": {"items": [], "total": 0},
        "preference_conflicts": [],
    }
    profile.update(overrides)
    return profile


def test_recommendations_preserve_observed_only_scoring_and_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No declared preferences leave released observed behavior intact."""
    wines = [
        _wine(1, name="Region Match", region="Piedmont", varietal="Barbera"),
        _wine(2, name="No Match", region="Burgundy", varietal="Chardonnay", wine_type="White"),
    ]
    profile = _profile(favorite_regions=[{"region": "Piedmont"}])
    _patch_recommendation_dependencies(
        monkeypatch,
        profile=profile,
        wines=wines,
        bottles_by_wine={1: [_bottle(1, 1)], 2: [_bottle(2, 2)]},
    )

    result = taste_profile_tools.get_wine_recommendations_from_profile.invoke({})

    assert [item["wine_id"] for item in result] == [1]
    assert result[0]["similarity_score"] == 0.4
    assert result[0]["preference_score"] == 0.4
    assert result[0]["declared_adjustment"] == 0.0
    assert result[0]["recommendation_reason"] == "From your favorite region: Piedmont"
    assert result[0]["declared_reasons"] == []
    assert result[0]["preference_conflicts"] == []


def test_declared_like_can_recommend_with_sparse_history_without_fabricated_rating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deterministic like crosses the threshold while a rating remains unknown."""
    wine = _wine(1, name="Declared Nebbiolo")
    profile = _profile(
        total_wines_rated=0,
        average_rating=0,
        declared_preferences={
            "items": [
                {
                    "subject_kind": "grape",
                    "stance": "like",
                    "value": "Nebbiolo",
                    "provenance": "explicit_user",
                }
            ],
            "total": 1,
        },
    )
    _patch_recommendation_dependencies(
        monkeypatch,
        profile=profile,
        wines=[wine],
        bottles_by_wine={1: [_bottle(1, 1)]},
    )

    result = taste_profile_tools.get_wine_recommendations_from_profile.invoke({})

    assert len(result) == 1
    assert result[0]["predicted_rating"] is None
    assert result[0]["similarity_score"] == 0.0
    assert result[0]["declared_adjustment"] == 0.4
    assert result[0]["preference_score"] == 0.4
    assert result[0]["declared_reasons"] == ["Declared like: grape Nebbiolo"]


def test_recommendations_enforce_avoid_and_joint_price_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoids and unproven ceilings exclude while one jointly qualifying bottle passes."""
    wines = [
        _wine(1, name="Avoided", producer="Blocked Producer"),
        _wine(2, name="Too Expensive"),
        _wine(3, name="Qualifying Bottle"),
    ]
    profile = _profile(
        declared_preferences={
            "items": [
                {
                    "subject_kind": "producer",
                    "stance": "avoid",
                    "value": "Blocked Producer",
                    "provenance": "explicit_user",
                },
                {
                    "subject_kind": "wine_style",
                    "stance": "like",
                    "value": "Red",
                    "provenance": "explicit_user",
                },
                {
                    "subject_kind": "price_ceiling",
                    "stance": None,
                    "price_amount": "40.00",
                    "currency": "EUR",
                    "provenance": "explicit_user",
                },
            ],
            "total": 3,
        },
    )
    _patch_recommendation_dependencies(
        monkeypatch,
        profile=profile,
        wines=wines,
        bottles_by_wine={
            1: [_bottle(1, 1, price=20)],
            2: [_bottle(2, 2, price=50)],
            3: [
                _bottle(3, 3, price=10, currency="USD"),
                _bottle(4, 3, price=35, currency="EUR"),
            ],
        },
    )

    result = taste_profile_tools.get_wine_recommendations_from_profile.invoke({"price_max": 36})

    assert [item["wine_id"] for item in result] == [3]
    assert result[0]["applied_constraints"] == [
        {
            "constraint": "price_ceiling",
            "price_amount": "40.00",
            "currency": "EUR",
            "status": "satisfied",
        },
        {
            "constraint": "request_price_max",
            "price_amount": 36.0,
            "status": "satisfied",
        },
    ]


def test_recommendation_adjustment_is_capped_and_conflicts_are_candidate_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple likes cap at +0.6 and only applicable conflicts are returned."""
    wine = _wine(7, name="Multi Match", producer="Fixture Producer", region="Piedmont")
    items = [
        {
            "subject_kind": subject,
            "stance": "like",
            "value": value,
            "provenance": "explicit_user",
        }
        for subject, value in (
            ("grape", "Nebbiolo"),
            ("region", "Piedmont"),
            ("producer", "Fixture Producer"),
        )
    ]
    profile = _profile(
        declared_preferences={"items": items, "total": 3},
        preference_conflicts=[
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
                "value": "Burgundy",
                "declared_stance": "like",
                "observed_signal": "low_rating",
                "average_rating": 70.0,
                "count": 1,
            },
        ],
    )
    _patch_recommendation_dependencies(
        monkeypatch,
        profile=profile,
        wines=[wine],
        bottles_by_wine={7: [_bottle(7, 7)]},
    )

    result = taste_profile_tools.get_wine_recommendations_from_profile.invoke({})

    assert result[0]["declared_adjustment"] == 0.6
    assert result[0]["preference_score"] == 0.6
    assert [conflict["value"] for conflict in result[0]["preference_conflicts"]] == ["Nebbiolo"]
