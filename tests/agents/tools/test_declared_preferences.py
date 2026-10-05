"""Deterministic integration tests for declared preferences in taste tools."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agents.tools import taste_profile_tools
from src.database import get_db_connection, initialize_database
from src.database.models import Bottle, DeclaredPreference, Wine
from src.database.repository import DeclaredPreferenceRepository, PreferenceStoreBusyError


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


def _patch_comparison_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: dict[str, object],
    wine: Wine | None,
    bottles: list[Bottle] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Bind comparison logic to deterministic profile and cellar inputs."""
    profile_tool = MagicMock()
    profile_tool.invoke.return_value = profile
    wine_repository = MagicMock()
    wine_repository.get_all.return_value = [wine] if wine else []
    bottle_repository = MagicMock()
    bottle_repository.get_by_wine.return_value = bottles or []
    monkeypatch.setattr(taste_profile_tools, "get_user_taste_profile", profile_tool)
    monkeypatch.setattr(taste_profile_tools, "WineRepository", lambda _db_path: wine_repository)
    monkeypatch.setattr(taste_profile_tools, "BottleRepository", lambda _db_path: bottle_repository)
    return wine_repository, bottle_repository


def test_comparison_preserves_observed_component_without_declared_preferences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observed-only comparison retains released component scoring and label."""
    profile = _profile(
        favorite_regions=[{"region": "Piedmont"}],
        favorite_varietals=[{"varietal": "Nebbiolo"}],
        type_ratings={"Red": 90.0},
    )
    _patch_comparison_dependencies(monkeypatch, profile=profile, wine=None)

    result = taste_profile_tools.compare_wine_to_profile.invoke({"wine_name": "Piedmont Nebbiolo Red"})

    assert result["region_match"] == 100
    assert result["varietal_match"] == 100
    assert result["type_match"] == 100
    assert result["observed_match_score"] == 100
    assert result["overall_match_score"] == 100
    assert result["declared_adjustment"] == 0
    assert result["recommendation"] == "Highly Recommended"
    assert result["applied_constraints"] == []


def test_external_declared_grape_match_allows_sparse_comparison_but_producer_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminology-backed external matches work while unresolved producers never match."""
    grape_like = {
        "subject_kind": "grape",
        "stance": "like",
        "value": "Nebbiolo",
        "provenance": "explicit_user",
    }
    profile = _profile(
        total_wines_rated=0,
        average_rating=0,
        declared_preferences={"items": [grape_like], "total": 1},
    )
    _patch_comparison_dependencies(monkeypatch, profile=profile, wine=None)

    matched = taste_profile_tools.compare_wine_to_profile.invoke({"wine_name": "Langhe Nebbiolo"})
    boundary_control = taste_profile_tools.compare_wine_to_profile.invoke({"wine_name": "Nebbiolorama"})

    assert matched["overall_match_score"] == 40
    assert matched["predicted_rating"] is None
    assert matched["declared_reasons"] == ["Declared like: grape Nebbiolo"]
    assert "error" in boundary_control

    producer_profile = _profile(
        total_wines_rated=0,
        average_rating=0,
        declared_preferences={
            "items": [
                {
                    "subject_kind": "producer",
                    "stance": "like",
                    "value": "External Estate",
                    "provenance": "explicit_user",
                }
            ],
            "total": 1,
        },
    )
    _patch_comparison_dependencies(monkeypatch, profile=producer_profile, wine=None)

    unresolved_producer = taste_profile_tools.compare_wine_to_profile.invoke(
        {"wine_name": "External Estate Nebbiolo"}
    )

    assert "error" in unresolved_producer


def test_in_cellar_avoid_forces_zero_and_conflict_qualifies_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hard avoids force zero and disclosed conflicts prevent unqualified recommendations."""
    wine = _wine(11, name="Avoided Nebbiolo")
    conflict = {
        "subject_kind": "grape",
        "value": "Nebbiolo",
        "declared_stance": "avoid",
        "observed_signal": "high_rating",
        "average_rating": 95.0,
        "count": 1,
    }
    profile = _profile(
        favorite_varietals=[{"varietal": "Nebbiolo"}],
        declared_preferences={
            "items": [
                {
                    "subject_kind": "grape",
                    "stance": "avoid",
                    "value": "Nebbiolo",
                    "provenance": "explicit_user",
                }
            ],
            "total": 1,
        },
        preference_conflicts=[conflict],
    )
    _patch_comparison_dependencies(
        monkeypatch,
        profile=profile,
        wine=wine,
        bottles=[_bottle(11, 11)],
    )

    result = taste_profile_tools.compare_wine_to_profile.invoke({"wine_name": wine.wine_name})

    assert result["overall_match_score"] == 0
    assert result["recommendation"] == "Not Recommended — Declared Avoid"
    assert result["preference_conflicts"] == [conflict]
    assert "Recommended" in result["recommendation"]
    assert result["recommendation"] not in {"Recommended", "Highly Recommended"}


@pytest.mark.parametrize(
    ("wine", "bottles", "expected_status", "expected_score"),
    [
        (None, [], "unknown", 40),
        (_wine(21, name="Unpriced Nebbiolo"), [_bottle(21, 21, price=None)], "unknown", 40),
        (_wine(22, name="Expensive Nebbiolo"), [_bottle(22, 22, price=60)], "violated", 0),
        (_wine(23, name="Affordable Nebbiolo"), [_bottle(23, 23, price=35)], "satisfied", 40),
    ],
)
def test_comparison_price_constraint_is_explicit_and_never_infers_unknown_compliance(
    monkeypatch: pytest.MonkeyPatch,
    wine: Wine | None,
    bottles: list[Bottle],
    expected_status: str,
    expected_score: int,
) -> None:
    """External/unpriced prices stay unknown and proven same-currency violations force zero."""
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
                },
                {
                    "subject_kind": "price_ceiling",
                    "stance": None,
                    "price_amount": "40.00",
                    "currency": "EUR",
                    "provenance": "explicit_user",
                },
            ],
            "total": 2,
        },
    )
    _patch_comparison_dependencies(monkeypatch, profile=profile, wine=wine, bottles=bottles)

    result = taste_profile_tools.compare_wine_to_profile.invoke(
        {"wine_name": wine.wine_name if wine else "External Nebbiolo"}
    )

    assert result["applied_constraints"][0]["status"] == expected_status
    assert result["overall_match_score"] == expected_score
    if expected_status == "violated":
        assert result["recommendation"] == "Not Recommended — Price Constraint"


@pytest.fixture()
def tool_db(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a real temporary cellar for the paired deterministic phase gate."""
    db_path = temp_dir / "declared-preference-tools.db"
    assert initialize_database(db_path)
    with get_db_connection(db_path) as conn:
        regions = {}
        for primary_name, country in (
            ("Piedmont", "Italy"),
            ("Napa Valley", "USA"),
            ("Burgundy", "France"),
        ):
            regions[primary_name] = conn.execute(
                "INSERT INTO regions (primary_name, country) VALUES (?, ?)",
                (primary_name, country),
            ).lastrowid

        producers = {}
        for name, country in (
            ("Producer A", "Italy"),
            ("Producer B", "USA"),
            ("Producer C", "France"),
            ("Producer D", "Italy"),
        ):
            producers[name] = conn.execute(
                "INSERT INTO producers (name, country) VALUES (?, ?)",
                (name, country),
            ).lastrowid

        wine_rows = (
            (1, "Low Nebbiolo", "Producer A", "Red", "Nebbiolo", "Piedmont", 35.0, 75),
            (2, "High Napa Cabernet", "Producer B", "Red", "Cabernet Sauvignon", "Napa Valley", 55.0, 95),
            (3, "Unrated Chardonnay", "Producer C", "White", "Chardonnay", "Burgundy", 30.0, None),
            (4, "Solid Barbera", "Producer D", "Red", "Barbera", "Piedmont", 25.0, 88),
        )
        for wine_id, name, producer, wine_type, varietal, region, price, rating in wine_rows:
            conn.execute(
                """
                INSERT INTO wines (
                    id, source, wine_name, producer_id, wine_type, varietal, region_id
                ) VALUES (?, 'manual', ?, ?, ?, ?, ?)
                """,
                (wine_id, name, producers[producer], wine_type, varietal, regions[region]),
            )
            conn.execute(
                """
                INSERT INTO bottles (
                    wine_id, source, quantity, status, location, purchase_price, currency
                ) VALUES (?, 'manual', 1, 'in_cellar', 'Cellar', ?, 'EUR')
                """,
                (wine_id, price),
            )
            if rating is not None:
                conn.execute(
                    "INSERT INTO tastings (wine_id, personal_rating) VALUES (?, ?)",
                    (wine_id, rating),
                )
        conn.commit()

    monkeypatch.setattr(taste_profile_tools, "get_default_db_path", lambda: db_path)
    return db_path


def _without_declared_sections(profile: dict[str, object]) -> dict[str, object]:
    """Return only the released observed profile fields for paired comparison."""
    return {
        key: value
        for key, value in profile.items()
        if key not in {"declared_preferences", "preference_conflicts"}
    }


def test_real_tools_separate_declared_observed_and_apply_delete_reset_immediately(tool_db: Path) -> None:
    """Paired real-tool calls prove separation, conflicts, and lifecycle immediacy."""
    repository = DeclaredPreferenceRepository(tool_db)
    baseline_profile = taste_profile_tools.get_user_taste_profile.invoke({})
    baseline_recommendations = taste_profile_tools.get_wine_recommendations_from_profile.invoke({})

    created = repository.create("grape", stance="like", value="Nebbiolo")
    declared_profile = taste_profile_tools.get_user_taste_profile.invoke({})
    declared_recommendations = taste_profile_tools.get_wine_recommendations_from_profile.invoke({})

    assert _without_declared_sections(declared_profile) == _without_declared_sections(baseline_profile)
    assert declared_profile["declared_preferences"]["items"] == [
        {
            "subject_kind": "grape",
            "stance": "like",
            "value": "Nebbiolo",
            "provenance": "explicit_user",
        }
    ]
    assert declared_profile["preference_conflicts"] == [
        {
            "subject_kind": "grape",
            "value": "Nebbiolo",
            "declared_stance": "like",
            "observed_signal": "low_rating",
            "average_rating": 75.0,
            "count": 1,
        }
    ]
    declared_nebbiolo = next(item for item in declared_recommendations if item["wine_id"] == 1)
    baseline_nebbiolo = next(item for item in baseline_recommendations if item["wine_id"] == 1)
    assert declared_nebbiolo["preference_score"] > baseline_nebbiolo["preference_score"]
    assert declared_nebbiolo["observed_reasons"] == baseline_nebbiolo["observed_reasons"]
    assert set(declared_profile["declared_preferences"]["items"][0]) == {
        "subject_kind",
        "stance",
        "value",
        "provenance",
    }

    repository.delete(created.id, created.version)
    assert taste_profile_tools.get_user_taste_profile.invoke({}) == baseline_profile
    assert taste_profile_tools.get_wine_recommendations_from_profile.invoke({}) == baseline_recommendations

    repository.create("wine_style", stance="like", value="red")
    assert taste_profile_tools.get_user_taste_profile.invoke({})["declared_preferences"]["total"] == 1
    assert repository.reset(expected_count=1) == 1
    assert taste_profile_tools.get_user_taste_profile.invoke({}) == baseline_profile


def test_real_tools_apply_avoid_price_and_conflict_precedence(tool_db: Path) -> None:
    """Real cellar candidates obey hard precedence and qualified comparison output."""
    repository = DeclaredPreferenceRepository(tool_db)
    repository.create("producer", stance="avoid", value="Producer B")
    repository.create("price_ceiling", price_minor_units=4_000, currency="EUR")

    recommendations = taste_profile_tools.get_wine_recommendations_from_profile.invoke({})
    comparison = taste_profile_tools.compare_wine_to_profile.invoke({"wine_name": "High Napa Cabernet"})

    assert 2 not in {item["wine_id"] for item in recommendations}
    assert comparison["overall_match_score"] == 0
    assert comparison["recommendation"] == "Not Recommended — Declared Avoid"
    assert comparison["applied_constraints"] == [
        {
            "constraint": "price_ceiling",
            "price_amount": "40.00",
            "currency": "EUR",
            "status": "violated",
        }
    ]
    assert comparison["preference_conflicts"] == [
        {
            "subject_kind": "producer",
            "value": "Producer B",
            "declared_stance": "avoid",
            "observed_signal": "high_rating",
            "average_rating": 95.0,
            "count": 1,
        }
    ]


def test_top_rated_wines_never_reads_declared_preferences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The intentionally unchanged top-rated tool has zero preference reads."""
    tasting_repository = MagicMock()
    tasting_repository.get_top_rated.return_value = []
    bottle_repository = MagicMock()
    preference_repository = MagicMock(side_effect=AssertionError("unexpected preference read"))
    monkeypatch.setattr(taste_profile_tools, "TastingRepository", lambda _db_path: tasting_repository)
    monkeypatch.setattr(taste_profile_tools, "BottleRepository", lambda _db_path: bottle_repository)
    monkeypatch.setattr(taste_profile_tools, "DeclaredPreferenceRepository", preference_repository)

    assert taste_profile_tools.get_top_rated_wines.invoke({}) == []
    preference_repository.assert_not_called()
