"""
Wine agent tools for taste profile analysis.

This module provides tools for analyzing user's wine preferences based on
tasting history from CellarTracker and Vivino imports stored in local database.
"""

from typing import Dict, List, Optional
from collections import defaultdict
import statistics
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
from src.database.repository import TastingRepository, WineRepository, BottleRepository
from src.agents.tools.utils import get_drink_status
from src.utils import get_default_db_path, logger


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
        tasting_repo = TastingRepository(get_default_db_path())
        tastings = tasting_repo.get_all_with_wine_info(has_rating=True)

        if not tastings:
            return {
                "total_wines_rated": 0,
                "total_wines_consumed": 0,
                "message": "No tasting history available"
            }

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

        return {
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

        if profile.get('total_wines_rated', 0) < 3:
            return []

        wine_repo = WineRepository(get_default_db_path())
        bottle_repo = BottleRepository(get_default_db_path())
        cellar_wines = wine_repo.get_all()
        recommendations = []

        fav_regions = {r["region"] for r in profile.get("favorite_regions", [])[:3]}
        fav_varietals = {v["varietal"] for v in profile.get("favorite_varietals", [])[:3]}

        for wine in cellar_wines:
            n_bottles_owned = bottle_repo.get_owned_quantity(wine.id)
            if n_bottles_owned == 0:
                continue

            similarity = 0
            reasons = []

            if wine.region_name in fav_regions:
                similarity += 0.4
                reasons.append(f"From your favorite region: {wine.region_name}")

            if wine.varietal in fav_varietals:
                similarity += 0.3
                reasons.append(f"Your preferred varietal: {wine.varietal}")

            if wine.wine_type == profile.get('preferred_type'):
                similarity += 0.2
                reasons.append(f"Your preferred wine type: {wine.wine_type}")

            if similarity < 0.3:
                continue

            if price_max:
                bottles = bottle_repo.get_by_wine(wine.id, status="in_cellar")
                if bottles and bottles[0].purchase_price and bottles[0].purchase_price > price_max:
                    continue

            predicted_rating = int(profile["average_rating"] * (0.8 + similarity * 0.2))
            drink_status = get_drink_status(wine.drink_from_year, wine.drink_to_year)
            bottles = bottle_repo.get_by_wine(wine.id, status="in_cellar")
            location = bottles[0].location if bottles else None

            recommendations.append({
                "wine_id": wine.id,
                "name": wine.wine_name,
                "producer": wine.producer_name,
                "vintage": wine.vintage,
                "wine_type": wine.wine_type,
                "varietal": wine.varietal,
                "region": wine.region_name,
                "predicted_rating": predicted_rating,
                "recommendation_reason": "; ".join(reasons),
                "similarity_score": round(similarity, 2),
                "in_cellar": True,
                "quantity": n_bottles_owned,
                "location": location,
                "drinking_status": drink_status
            })

        recommendations.sort(key=lambda x: (x['predicted_rating'], x['similarity_score']), reverse=True)

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
