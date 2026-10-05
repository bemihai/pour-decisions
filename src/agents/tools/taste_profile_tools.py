"""
Wine agent tools for taste profile analysis.

This module provides tools for analyzing user's wine preferences based on
tasting history from CellarTracker and Vivino imports stored in local database.
"""

from collections import defaultdict
from decimal import Decimal
import statistics
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from src.agents.tools.registry import (
    CostClass,
    LatencyClass,
    ToolCategory,
    ToolDefinition,
    ToolMetadata,
    ToolPrerequisite,
    ToolTier,
)
from src.database.models import Bottle, DeclaredPreference, PreferenceStance, PreferenceSubjectKind, Wine
from src.database.repository import (
    BottleRepository,
    DeclaredPreferenceRepository,
    TastingRepository,
    WineRepository,
    canonicalize_preference_identity,
    normalize_preference_text,
)
from src.agents.tools.utils import get_drink_status
from src.utils import get_default_db_path, logger

_OBSERVED_FIELDS: dict[PreferenceSubjectKind, str] = {
    PreferenceSubjectKind.GRAPE: "varietal",
    PreferenceSubjectKind.REGION: "region_name",
    PreferenceSubjectKind.PRODUCER: "producer_name",
    PreferenceSubjectKind.WINE_STYLE: "wine_type",
}


def _declared_preference_output(preference: DeclaredPreference) -> dict[str, Any]:
    """Project one stored preference into the bounded tool result."""
    item: dict[str, Any] = {
        "subject_kind": preference.subject_kind.value,
        "stance": preference.stance.value if preference.stance else None,
        "provenance": preference.provenance,
    }
    if preference.subject_kind == PreferenceSubjectKind.PRICE_CEILING:
        item["price_amount"] = f"{Decimal(preference.price_minor_units or 0) / Decimal(100):.2f}"
        item["currency"] = preference.currency.value if preference.currency else None
    else:
        item["value"] = preference.display_value
    return item


def _observed_preference_signals(tastings: list[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    """Index rated tasting evidence by canonical preference identity."""
    signals: dict[tuple[str, str], list[int]] = defaultdict(list)
    for tasting in tastings:
        rating = tasting.get("personal_rating")
        if rating is None:
            continue
        for subject_kind, field_name in _OBSERVED_FIELDS.items():
            value = tasting.get(field_name)
            if not value:
                continue
            raw_identity = normalize_preference_text(value)[1]
            canonical_identity = canonicalize_preference_identity(subject_kind, value)
            signals[(subject_kind.value, raw_identity)].append(int(rating))
            if canonical_identity != raw_identity:
                signals[(subject_kind.value, canonical_identity)].append(int(rating))
    return signals


def _preference_conflicts(
    preferences: list[DeclaredPreference],
    tastings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return exact declared-versus-observed rating conflicts."""
    observed_signals = _observed_preference_signals(tastings)
    conflicts: list[dict[str, Any]] = []
    for preference in preferences:
        if preference.subject_kind == PreferenceSubjectKind.PRICE_CEILING or preference.stance is None:
            continue
        ratings = observed_signals.get(
            (preference.subject_kind.value, preference.normalized_value),
            [],
        )
        if not ratings:
            continue
        average_rating = sum(ratings) / len(ratings)
        observed_signal = None
        if average_rating >= 90:
            observed_signal = "high_rating"
        elif average_rating < 80:
            observed_signal = "low_rating"

        is_conflict = (
            preference.stance == PreferenceStance.LIKE and observed_signal == "low_rating"
        ) or (
            preference.stance in {PreferenceStance.DISLIKE, PreferenceStance.AVOID}
            and observed_signal == "high_rating"
        )
        if is_conflict:
            conflicts.append(
                {
                    "subject_kind": preference.subject_kind.value,
                    "value": preference.display_value,
                    "declared_stance": preference.stance.value,
                    "observed_signal": observed_signal,
                    "average_rating": round(average_rating, 1),
                    "count": len(ratings),
                }
            )
    return conflicts


def _add_declared_sections(
    profile: dict[str, Any],
    preferences: list[DeclaredPreference],
    tastings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Add the approved declared sections without mutating observed fields."""
    return {
        **profile,
        "declared_preferences": {
            "items": [_declared_preference_output(preference) for preference in preferences],
            "total": len(preferences),
        },
        "preference_conflicts": _preference_conflicts(preferences, tastings),
    }


def _identity_candidates(subject_kind: PreferenceSubjectKind, value: str | None) -> set[str]:
    """Return exact raw and terminology-canonical identities for one value."""
    if not value:
        return set()
    raw_identity = normalize_preference_text(value)[1]
    canonical_identity = canonicalize_preference_identity(subject_kind, value)
    return {raw_identity, canonical_identity}


def _declared_item_matches_value(item: dict[str, Any], value: str | None) -> bool:
    """Match one bounded declared item to a structured wine field."""
    if item["subject_kind"] == PreferenceSubjectKind.PRICE_CEILING.value:
        return False
    subject_kind = PreferenceSubjectKind(item["subject_kind"])
    return bool(
        _identity_candidates(subject_kind, item.get("value"))
        & _identity_candidates(subject_kind, value)
    )


def _matching_declared_items(
    wine: Wine,
    declared_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return non-price preferences that exactly match structured wine fields."""
    wine_values = {
        PreferenceSubjectKind.GRAPE.value: wine.varietal,
        PreferenceSubjectKind.REGION.value: wine.region_name,
        PreferenceSubjectKind.PRODUCER.value: wine.producer_name,
        PreferenceSubjectKind.WINE_STYLE.value: wine.wine_type,
    }
    return [
        item
        for item in declared_items
        if item["subject_kind"] in wine_values
        and _declared_item_matches_value(item, wine_values[item["subject_kind"]])
    ]


def _matching_conflicts(wine: Wine, conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return profile conflicts applicable to one structured cellar wine."""
    wine_values = {
        PreferenceSubjectKind.GRAPE.value: wine.varietal,
        PreferenceSubjectKind.REGION.value: wine.region_name,
        PreferenceSubjectKind.PRODUCER.value: wine.producer_name,
        PreferenceSubjectKind.WINE_STYLE.value: wine.wine_type,
    }
    return [
        conflict
        for conflict in conflicts
        if conflict["subject_kind"] in wine_values
        and bool(
            _identity_candidates(PreferenceSubjectKind(conflict["subject_kind"]), conflict.get("value"))
            & _identity_candidates(
                PreferenceSubjectKind(conflict["subject_kind"]),
                wine_values[conflict["subject_kind"]],
            )
        )
    ]


def _recommendation_price_constraints(
    bottles: list[Bottle],
    declared_items: list[dict[str, Any]],
    price_max: float | None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Apply request-local and stored price rules to owned bottles."""
    price_item = next(
        (item for item in declared_items if item["subject_kind"] == PreferenceSubjectKind.PRICE_CEILING.value),
        None,
    )
    applied_constraints: list[dict[str, Any]] = []

    if price_item is not None:
        stored_amount = Decimal(price_item["price_amount"])
        stored_currency = price_item["currency"]
        qualifying_bottles = [
            bottle
            for bottle in bottles
            if bottle.purchase_price is not None
            and bottle.currency == stored_currency
            and Decimal(str(bottle.purchase_price)) <= stored_amount
            and (not price_max or Decimal(str(bottle.purchase_price)) <= Decimal(str(price_max)))
        ]
        if not qualifying_bottles:
            return False, [
                {
                    "constraint": "price_ceiling",
                    "price_amount": price_item["price_amount"],
                    "currency": stored_currency,
                    "status": "violated",
                }
            ]
        applied_constraints.append(
            {
                "constraint": "price_ceiling",
                "price_amount": price_item["price_amount"],
                "currency": stored_currency,
                "status": "satisfied",
            }
        )
        if price_max:
            applied_constraints.append(
                {
                    "constraint": "request_price_max",
                    "price_amount": price_max,
                    "status": "satisfied",
                }
            )
        return True, applied_constraints

    if price_max and bottles and bottles[0].purchase_price and bottles[0].purchase_price > price_max:
        return False, [
            {
                "constraint": "request_price_max",
                "price_amount": price_max,
                "status": "violated",
            }
        ]
    if price_max:
        applied_constraints.append(
            {
                "constraint": "request_price_max",
                "price_amount": price_max,
                "status": "satisfied",
            }
        )
    return True, applied_constraints


@tool
def get_user_taste_profile() -> Dict:
    """Get comprehensive user wine taste profile based on tasting history.

    Analyzes all consumed wines and ratings to build a detailed preference model.
    Uses consumption cellar-data from the local database.

    Returns:
        Dictionary containing taste profile information with summary stats,
        favorite regions/countries/varietals/producers, wine type preferences,
        rating patterns, and more.

    Example:
        >>> profile = get_user_taste_profile()
        >>> print(f"Favorite region: {profile['favorite_regions'][0]['region']}")
        >>> print(f"Average rating: {profile['average_rating']:.1f}/100")

    Notes:
        - Returns empty/default values if no tasting history exists
        - Only includes wines with personal ratings
    """
    try:
        db_path = get_default_db_path()
        preferences = DeclaredPreferenceRepository(db_path).list_all()
        tasting_repo = TastingRepository(db_path)
        tastings = tasting_repo.get_all_with_wine_info(has_rating=True)

        if not tastings:
            return _add_declared_sections(
                {
                    "total_wines_rated": 0,
                    "total_wines_consumed": 0,
                    "message": "No tasting history available",
                },
                preferences,
                tastings,
            )

        # Calculate basic stats
        ratings = [t["personal_rating"] for t in tastings if t.get("personal_rating")]
        total_rated = len(ratings)
        avg_rating = sum(ratings) / total_rated if total_rated > 0 else 0
        std_dev = statistics.stdev(ratings) if len(ratings) > 1 else 0

        # Aggregate by region
        region_stats = defaultdict(lambda: {"ratings": [], "count": 0})
        for t in tastings:
            if t.get("region_name") and t.get("personal_rating"):
                region_stats[t['region_name']]['ratings'].append(t['personal_rating'])
                region_stats[t['region_name']]['count'] += 1

        favorite_regions = [
            {
                "region": region,
                "avg_rating": sum(stats["ratings"]) / len(stats["ratings"]),
                "count": stats["count"],
                "percentage": round((stats["count"] / total_rated) * 100, 1)
            }
            for region, stats in region_stats.items()
        ]
        favorite_regions.sort(key=lambda x: (x["avg_rating"], x["count"]), reverse=True)

        # Aggregate by country
        country_stats = defaultdict(lambda: {"ratings": [], "count": 0})
        for t in tastings:
            if t.get("country") and t.get("personal_rating"):
                country_stats[t["country"]]["ratings"].append(t["personal_rating"])
                country_stats[t["country"]]["count"] += 1

        favorite_countries = [
            {
                "country": country,
                "avg_rating": sum(stats["ratings"]) / len(stats["ratings"]),
                "count": stats["count"]
            }
            for country, stats in country_stats.items()
        ]
        favorite_countries.sort(key=lambda x: (x["avg_rating"], x["count"]), reverse=True)

        # Aggregate by varietal
        varietal_stats = defaultdict(lambda: {"ratings": [], "count": 0})
        for t in tastings:
            if t.get("varietal") and t.get("personal_rating"):
                varietal_stats[t["varietal"]]["ratings"].append(t["personal_rating"])
                varietal_stats[t["varietal"]]["count"] += 1

        favorite_varietals = [
            {
                "varietal": varietal,
                "avg_rating": sum(stats["ratings"]) / len(stats["ratings"]),
                "count": stats["count"],
                "preference_strength": min(100, int((sum(stats["ratings"]) / len(stats["ratings"])) * (stats["count"] / total_rated) * 10))
            }
            for varietal, stats in varietal_stats.items()
        ]
        favorite_varietals.sort(key=lambda x: x["preference_strength"], reverse=True)

        # Aggregate by producer
        producer_stats = defaultdict(lambda: {"ratings": [], "count": 0})
        for t in tastings:
            if t.get("producer_name") and t.get("personal_rating"):
                producer_stats[t["producer_name"]]["ratings"].append(t["personal_rating"])
                producer_stats[t["producer_name"]]["count"] += 1

        favorite_producers = [
            {
                "producer": producer,
                "avg_rating": sum(stats["ratings"]) / len(stats["ratings"]),
                "count": stats["count"]
            }
            for producer, stats in producer_stats.items()
        ]
        favorite_producers.sort(key=lambda x: (x["avg_rating"], x["count"]), reverse=True)

        # Wine type distribution
        type_stats = defaultdict(lambda: {"ratings": [], "count": 0})
        for t in tastings:
            if t.get("personal_rating"):
                wine_type = t.get("wine_type", "Unknown")
                type_stats[wine_type]["ratings"].append(t["personal_rating"])
                type_stats[wine_type]["count"] += 1

        type_distribution = {wt: round((stats["count"] / total_rated) * 100, 1)
                           for wt, stats in type_stats.items()}
        type_ratings = {wt: sum(stats["ratings"]) / len(stats["ratings"])
                       for wt, stats in type_stats.items()}
        preferred_type = max(type_stats.items(), key=lambda x: x[1]["count"])[0]

        # Rating patterns
        high_rated = sum(1 for r in ratings if r >= 90)
        low_rated = sum(1 for r in ratings if r < 80)
        rating_dist = {
            '96-100': sum(1 for r in ratings if r >= 96),
            '90-95': sum(1 for r in ratings if 90 <= r < 96),
            '85-89': sum(1 for r in ratings if 85 <= r < 90),
            '80-84': sum(1 for r in ratings if 80 <= r < 85),
            '70-79': sum(1 for r in ratings if 70 <= r < 80),
            'below 70': sum(1 for r in ratings if r < 70)
        }

        profile = {
            # Summary
            'total_wines_rated': total_rated,
            'total_wines_consumed': len(tastings),
            'average_rating': round(avg_rating, 1),
            'rating_standard_deviation': round(std_dev, 1),

            # Favorites
            'favorite_regions': favorite_regions[:5],
            'favorite_countries': favorite_countries[:5],
            'favorite_varietals': favorite_varietals[:5],
            'favorite_producers': favorite_producers[:10],

            # Wine type preferences
            'type_distribution': type_distribution,
            'type_ratings': type_ratings,
            'preferred_type': preferred_type,

            # Rating patterns
            'high_rated_count': high_rated,
            'low_rated_count': low_rated,
            'rating_distribution': rating_dist
        }
        return _add_declared_sections(profile, preferences, tastings)

    except Exception:
        logger.exception("Unexpected failure while getting taste profile")
        raise


@tool
def get_top_rated_wines(
    min_rating: int = 90,
    wine_type: Optional[str] = None,
    limit: int = 10
) -> List[Dict]:
    """Get user's top-rated wines based on personal ratings.

    Retrieve wines that the user rated highly, optionally filtered by wine type.

    Args:
        min_rating: Minimum rating threshold (0-100 scale). Default 90.
        wine_type: Filter by wine type (Red, White, Rosé, Sparkling, etc.).
        limit: Maximum number of wines to return. Default 10, max 50.

    Returns:
        List of dictionaries with wine details, ratings, and cellar status.

    Example:
        >>> top_wines = get_top_rated_wines(min_rating=90)
        >>> top_reds = get_top_rated_wines(min_rating=85, wine_type="Red")

    Notes:
        - Results ordered by rating (descending), then by tasting date
        - Includes wines no longer in cellar (consumed)
    """
    try:
        tasting_repo = TastingRepository(get_default_db_path())
        bottle_repo = BottleRepository(get_default_db_path())

        top_wines = tasting_repo.get_top_rated(
            min_rating=min_rating,
            wine_type=wine_type,
            limit=min(limit, 50)
        )

        results = []
        for wine in top_wines:
            bottles = bottle_repo.get_by_wine(wine["wine_id"], status="in_cellar")
            in_cellar = len(bottles) > 0
            quantity = sum(b.quantity for b in bottles) if bottles else 0

            results.append({
                "wine_id": wine["wine_id"],
                "name": wine.get("wine_name"),
                "producer": wine.get("producer_name"),
                "vintage": wine.get("vintage"),
                "wine_type": wine.get("wine_type"),
                "varietal": wine.get("varietal"),
                "region": wine.get("region_name"),
                "country": wine.get("country"),
                "personal_rating": wine.get("personal_rating"),
                "tasting_notes": wine.get("tasting_notes"),
                "last_tasted_date": str(wine.get("last_tasted_date")) if wine.get("last_tasted_date") else None,
                "in_cellar": in_cellar,
                "quantity_owned": quantity
            })

        logger.info(f"Found {len(results)} top-rated wines")
        return results

    except Exception:
        logger.exception("Unexpected failure while getting top rated wines")
        raise


@tool
def get_wine_recommendations_from_profile(price_max: Optional[float] = None) -> List[Dict]:
    """Get personalized wine recommendations based on user's taste profile.

    Uses taste profile to recommend wines the user is likely to enjoy.

    Args:
        price_max: Maximum price per bottle

    Returns:
        List of recommended wines with predicted ratings and reasons.

    Example:
        >>> recs = get_wine_recommendations_from_profile(price_max=50.0)

    Notes:
        - Uses collaborative filtering based on rating patterns
        - Returns empty list if insufficient tasting history
        - Only recommends wines currently owned in cellar
    """
    try:
        profile = get_user_taste_profile.invoke({})
        declared_items = profile.get("declared_preferences", {}).get("items", [])
        profile_conflicts = profile.get("preference_conflicts", [])
        has_sufficient_history = profile.get("total_wines_rated", 0) >= 3

        wine_repo = WineRepository(get_default_db_path())
        bottle_repo = BottleRepository(get_default_db_path())
        cellar_wines = wine_repo.get_all()
        recommendations = []

        fav_regions = {r["region"] for r in profile.get("favorite_regions", [])[:3]}
        fav_varietals = {v["varietal"] for v in profile.get("favorite_varietals", [])[:3]}

        for wine in cellar_wines:
            bottles = bottle_repo.get_by_wine(wine.id, status="in_cellar")
            n_bottles_owned = sum(bottle.quantity for bottle in bottles)
            if not bottles or n_bottles_owned == 0:
                continue

            matching_items = _matching_declared_items(wine, declared_items)
            if any(item["stance"] == PreferenceStance.AVOID.value for item in matching_items):
                continue
            price_allowed, applied_constraints = _recommendation_price_constraints(
                bottles,
                declared_items,
                price_max,
            )
            if not price_allowed:
                continue

            similarity = 0.0
            observed_reasons = []

            if wine.region_name in fav_regions:
                similarity += 0.4
                observed_reasons.append(f"From your favorite region: {wine.region_name}")

            if wine.varietal in fav_varietals:
                similarity += 0.3
                observed_reasons.append(f"Your preferred varietal: {wine.varietal}")

            if wine.wine_type == profile.get("preferred_type"):
                similarity += 0.2
                observed_reasons.append(f"Your preferred wine type: {wine.wine_type}")

            declared_adjustment = sum(
                0.4 if item["stance"] == PreferenceStance.LIKE.value else -0.4
                for item in matching_items
                if item["stance"] in {PreferenceStance.LIKE.value, PreferenceStance.DISLIKE.value}
            )
            declared_adjustment = max(-0.6, min(0.6, declared_adjustment))
            preference_score = max(0.0, min(1.0, similarity + declared_adjustment))
            has_declared_like = any(item["stance"] == PreferenceStance.LIKE.value for item in matching_items)
            if not has_sufficient_history and not has_declared_like:
                continue
            if preference_score < 0.3:
                continue

            predicted_rating = None
            if profile.get("total_wines_rated", 0) > 0:
                predicted_rating = int(profile["average_rating"] * (0.8 + similarity * 0.2))
            drink_status = get_drink_status(wine.drink_from_year, wine.drink_to_year)
            location = bottles[0].location if bottles else None
            declared_reasons = [
                f"Declared {item['stance']}: {item['subject_kind']} {item['value']}"
                for item in matching_items
                if item["stance"] in {PreferenceStance.LIKE.value, PreferenceStance.DISLIKE.value}
            ]
            candidate_conflicts = _matching_conflicts(wine, profile_conflicts)

            recommendations.append({
                "wine_id": wine.id,
                "name": wine.wine_name,
                "producer": wine.producer_name,
                "vintage": wine.vintage,
                "wine_type": wine.wine_type,
                "varietal": wine.varietal,
                "region": wine.region_name,
                "predicted_rating": predicted_rating,
                "recommendation_reason": "; ".join(observed_reasons + declared_reasons),
                "similarity_score": round(similarity, 2),
                "preference_score": round(preference_score, 2),
                "declared_adjustment": round(declared_adjustment, 2),
                "observed_reasons": observed_reasons,
                "declared_reasons": declared_reasons,
                "preference_conflicts": candidate_conflicts,
                "applied_constraints": applied_constraints,
                "in_cellar": True,
                "quantity": n_bottles_owned,
                "location": location,
                "drinking_status": drink_status
            })

        if declared_items:
            recommendations.sort(
                key=lambda item: (
                    -item["preference_score"],
                    item["predicted_rating"] is None,
                    -(item["predicted_rating"] or 0),
                    item["wine_id"],
                )
            )
        else:
            recommendations.sort(
                key=lambda item: (item["predicted_rating"], item["similarity_score"]),
                reverse=True,
            )

        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations[:10]

    except Exception:
        logger.exception("Unexpected failure while generating recommendations")
        raise


@tool
def compare_wine_to_profile(wine_name: str) -> Dict:
    """Compare a specific wine to user's taste profile.

    Analyzes how well a wine matches user's preferences based on wine characteristics.
    Works with any wine - doesn't need to be in your cellar.

    Args:
        wine_name: Name of the wine to analyze (e.g., "Cremant de Jura", "Barolo", "Napa Cabernet")

    Returns:
        Dictionary with match scores, predicted rating, and recommendation.

    Example:
        >>> comparison = compare_wine_to_profile(wine_name='Cremant de Jura')
        >>> print(f"Match score: {comparison['overall_match_score']}/100")

    Notes:
        - Works with any wine, not just wines in cellar
        - Extracts characteristics from wine name (region, type, varietal)
        - Requires some tasting history for accurate predictions
        - For wines in cellar, provides more detailed analysis
    """
    try:
        wine_repo = WineRepository(get_default_db_path())
        profile = get_user_taste_profile.invoke({})

        if profile.get('total_wines_rated', 0) < 3:
            return {'error': 'Insufficient tasting history for comparison (need at least 3 rated wines)'}

        # First, try to find wine in cellar for detailed analysis
        wine = None
        wines = wine_repo.get_all(wine_name=wine_name, limit=1)
        if wines:
            wine = wines[0]

        # Extract wine characteristics from name if not in cellar
        wine_name_lower = wine_name.lower()

        # Determine region from wine name
        extracted_region = None
        if wine:
            extracted_region = wine.region_name
        else:
            # Try to extract region from name
            known_regions = ["burgundy", "bordeaux", "champagne", "rioja", "tuscany", "piedmont",
                           "barolo", "chianti", "napa", "sonoma", "rhone", "loire", "jura",
                           "alsace", "mosel", "rheingau"]
            for region in known_regions:
                if region in wine_name_lower:
                    extracted_region = region.title()
                    break

        # Determine wine type from name
        extracted_type = None
        if wine:
            extracted_type = wine.wine_type
        else:
            # Try to extract type from name
            if any(word in wine_name_lower for word in ["cremant", "champagne", "cava", "prosecco", "sparkling"]):
                extracted_type = "Sparkling"
            elif any(word in wine_name_lower for word in ["sauternes", "ice wine", "dessert"]):
                extracted_type = "Dessert"
            elif any(word in wine_name_lower for word in ["rose", "rosé"]):
                extracted_type = "Rosé"
            elif any(word in wine_name_lower for word in ["white", "chardonnay", "riesling", "sauvignon blanc",
                                                           "pinot grigio", "albariño", "gewurztraminer"]):
                extracted_type = "White"
            elif any(word in wine_name_lower for word in ["red", "cabernet", "merlot", "pinot noir",
                                                           "syrah", "shiraz", "malbec", "zinfandel",
                                                           "barolo", "brunello", "chianti", "rioja"]):
                extracted_type = "Red"

        # Determine varietal from name
        extracted_varietal = None
        if wine:
            extracted_varietal = wine.varietal
        else:
            known_varietals = ["chardonnay", "cabernet sauvignon", "pinot noir", "merlot",
                             "sauvignon blanc", "riesling", "syrah", "malbec", "nebbiolo",
                             "sangiovese", "tempranillo", "grenache", "zinfandel"]
            for varietal in known_varietals:
                if varietal in wine_name_lower:
                    extracted_varietal = varietal.title()
                    break

        # Calculate match scores
        region_match = 0
        varietal_match = 0
        type_match = 0

        fav_regions = profile.get('favorite_regions', [])
        if extracted_region:
            for i, fav in enumerate(fav_regions[:5]):
                if fav['region'].lower() in extracted_region.lower() or extracted_region.lower() in fav['region'].lower():
                    region_match = max(region_match, 100 - (i * 15))

        fav_varietals = profile.get('favorite_varietals', [])
        if extracted_varietal:
            for i, fav in enumerate(fav_varietals[:5]):
                if fav['varietal'].lower() in extracted_varietal.lower() or extracted_varietal.lower() in fav['varietal'].lower():
                    varietal_match = max(varietal_match, 100 - (i * 15))

        type_ratings = profile.get('type_ratings', {})
        if extracted_type and extracted_type in type_ratings:
            type_match = int((type_ratings[extracted_type] / profile['average_rating']) * 100)

        overall_match = int((region_match * 0.4 + varietal_match * 0.3 + type_match * 0.3))
        predicted_rating = int(profile['average_rating'] * (0.7 + (overall_match / 100) * 0.3))

        if overall_match >= 80:
            recommendation = "Highly Recommended"
            confidence = "high"
        elif overall_match >= 60:
            recommendation = "Recommended"
            confidence = "medium"
        elif overall_match >= 40:
            recommendation = "Worth Trying"
            confidence = "medium"
        else:
            recommendation = "May Not Match Your Taste"
            confidence = "low"

        reasons = []
        if region_match > 50:
            reasons.append(f"From a region you enjoy")
        if varietal_match > 50:
            reasons.append(f"Made with grapes you prefer")
        if type_match > 80:
            reasons.append(f"Wine type you highly rate")
        if not reasons:
            reasons.append("Based on general taste profile")

        result = {
            "wine_name": wine_name,
            "in_cellar": wine is not None,
            "overall_match_score": overall_match,
            "predicted_rating": predicted_rating,
            "confidence_level": confidence,
            "region_match": region_match,
            "varietal_match": varietal_match,
            "type_match": type_match,
            "recommendation": recommendation,
            "reasons": reasons,
            "detected_characteristics": {
                "type": extracted_type,
                "region": extracted_region,
                "varietal": extracted_varietal
            }
        }

        return result

    except Exception:
        logger.exception("Unexpected failure while comparing wine to profile")
        raise


TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        tool=get_user_taste_profile,
        metadata=ToolMetadata(
            name="get_user_taste_profile",
            category=ToolCategory.TASTE_PROFILE,
            tier=ToolTier.CORE,
            prerequisites=(ToolPrerequisite.CELLAR_SCHEMA,),
            cost_class=CostClass.FREE,
            latency_class=LatencyClass.FAST,
            idempotent=True,
            capability="Summarize the user's preferences from local tasting history.",
        ),
    ),
    ToolDefinition(
        tool=get_top_rated_wines,
        metadata=ToolMetadata(
            name="get_top_rated_wines",
            category=ToolCategory.TASTE_PROFILE,
            tier=ToolTier.EXTENDED,
            prerequisites=(ToolPrerequisite.CELLAR_SCHEMA,),
            cost_class=CostClass.FREE,
            latency_class=LatencyClass.FAST,
            idempotent=True,
            capability="List the user's highest-rated wines from tasting history.",
        ),
    ),
    ToolDefinition(
        tool=get_wine_recommendations_from_profile,
        metadata=ToolMetadata(
            name="get_wine_recommendations_from_profile",
            category=ToolCategory.TASTE_PROFILE,
            tier=ToolTier.EXTENDED,
            prerequisites=(ToolPrerequisite.CELLAR_SCHEMA,),
            cost_class=CostClass.FREE,
            latency_class=LatencyClass.FAST,
            idempotent=True,
            capability="Recommend cellar wines that fit the user's taste profile.",
        ),
    ),
    ToolDefinition(
        tool=compare_wine_to_profile,
        metadata=ToolMetadata(
            name="compare_wine_to_profile",
            category=ToolCategory.TASTE_PROFILE,
            tier=ToolTier.EXTENDED,
            prerequisites=(ToolPrerequisite.CELLAR_SCHEMA,),
            cost_class=CostClass.FREE,
            latency_class=LatencyClass.FAST,
            idempotent=True,
            capability="Compare a wine with the user's observed taste preferences.",
        ),
    ),
)
