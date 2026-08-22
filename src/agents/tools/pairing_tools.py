"""
Wine agent tools for food and wine pairing.

This module provides tools for wine and food pairing recommendations
using pairing rules from the database and user's cellar inventory.
"""

from typing import Dict, Optional
from langchain_core.tools import tool

from src.agents.tools.registry import (
    ToolCategory,
    ToolDefinition,
    ToolMetadata,
    ToolPrerequisite,
    ToolTier,
)
from src.database.repository import WineRepository, BottleRepository, FoodPairingRepository
from src.utils import get_default_db_path, logger
from src.agents.tools.cellar_tools import get_drink_status


@tool
def get_food_pairing_wines(
    food: str,
    wine_type_preference: Optional[str] = None,
    from_cellar_only: bool = True,
    ready_to_drink_only: bool = False
) -> Dict:
    """Get wine recommendations for food pairing.

    Provides wine recommendations based on food pairing rules and optionally
    filters by user's cellar inventory and taste preferences.

    Args:
        food: Type of food to pair with. Examples:
              - Proteins: "steak", "beef", "lamb", "pork", "chicken", "duck",
                         "salmon", "tuna", "white fish", "shellfish"
              - Dishes: "pasta", "pizza", "risotto", "curry", "sushi"
              - Cheese: "soft cheese", "hard cheese", "blue cheese", "goat cheese"
              - Occasions: "appetizers", "dessert", "brunch"
        wine_type_preference: Preferred wine type if user has preference.
                             Options: "Red", "White", "Rosé", "Sparkling"
                             If None, suggests best matches regardless of type.
        from_cellar_only: If True, only recommend wines from user's cellar.
                         If False, provide general pairing recommendations.
        ready_to_drink_only: If True, only suggest wines currently in drinking window.

    Returns:
        Dictionary containing pairing recommendations and cellar matches.

    Example:
        >>> pairing = get_food_pairing_wines("steak")
        >>> pairing = get_food_pairing_wines(
        ...     food="salmon",
        ...     wine_type_preference="White",
        ...     ready_to_drink_only=True
        ... )

    Notes:
        - Uses classic pairing rules (weight, acid, tannin, fat)
        - Considers regional traditional pairings
        - Completely free operation (rule-based logic + local database)
        - Returns general recommendations if no cellar matches found
    """
    try:
        # Normalize food input
        food_lower = food.lower().strip()

        # Get pairing rule from database
        pairing_repo = FoodPairingRepository(get_default_db_path())
        pairing_rule = pairing_repo.find_matching_rule(food_lower)

        if not pairing_rule:
            return {
                "food_analyzed": food,
                "message": f"No specific pairing rules found for '{food}'. Try common foods like steak, salmon, pasta, etc.",
                "recommended_wine_types": ["Red", "White"],
                "general_advice": "Match wine weight to food weight, acidity to richness"
            }

        # Build recommendations from database rule
        recommended_types = pairing_rule.get_wine_types_list()
        recommended_varietals = pairing_rule.get_varietals_list()

        # Filter by wine type preference
        if wine_type_preference:
            if wine_type_preference in recommended_types:
                recommended_types = [wine_type_preference]
            else:
                recommended_types = [wine_type_preference]  # Allow user preference

        result = {
            "food_analyzed": food,
            "recommended_wine_types": recommended_types,
            "recommended_varietals": recommended_varietals,
            "pairing_principles": pairing_rule.pairing_explanation,
            "cellar_matches": []
        }

        # If from_cellar_only, find matching wines in cellar
        if from_cellar_only:
            wine_repo = WineRepository(get_default_db_path())
            bottle_repo = BottleRepository(get_default_db_path())

            cellar_matches = []

            for wine_type in recommended_types:
                wines = wine_repo.get_all(wine_type=wine_type, limit=100)

                for wine in wines:
                    # Use get_owned_quantity to check if wine is in cellar
                    n_bottles = bottle_repo.get_owned_quantity(wine.id)
                    if n_bottles == 0:
                        continue

                    # Check if ready to drink
                    if ready_to_drink_only:
                        drink_status = get_drink_status(wine.drink_from_year, wine.drink_to_year)
                        if drink_status != "ready":
                            continue

                    # Calculate pairing score based on varietal match
                    pairing_score = 0
                    if wine.varietal:
                        for rec_varietal in recommended_varietals:
                            if rec_varietal.lower() in wine.varietal.lower():
                                pairing_score = 90
                                break
                        if pairing_score == 0:
                            pairing_score = 60  # Same type, different varietal
                    else:
                        pairing_score = 50

                    # Determine drinking status
                    drink_status = get_drink_status(wine.drink_from_year, wine.drink_to_year)

                    # Get location
                    bottles = bottle_repo.get_by_wine(wine.id, status='in_cellar')
                    location = bottles[0].location if bottles else None

                    cellar_matches.append({
                        "wine_id": wine.id,
                        "name": wine.wine_name,
                        "producer": wine.producer_name,
                        "vintage": wine.vintage,
                        "wine_type": wine.wine_type,
                        "varietal": wine.varietal,
                        "region": wine.region_name,
                        "location": location,
                        "quantity": n_bottles,
                        "pairing_score": pairing_score,
                        "pairing_notes": f"Pairs well with {food} - {pairing_rule.characteristics}",
                        "drinking_status": drink_status
                    })

            # Sort by pairing score
            cellar_matches.sort(key=lambda x: x["pairing_score"], reverse=True)
            result["cellar_matches"] = cellar_matches[:10]

            if not cellar_matches:
                result["message"] = f"No wines in cellar match the pairing for {food}. Consider purchasing: {', '.join(recommended_varietals[:3])}"

        logger.info(f"Generated pairing for {food}: {len(result.get('cellar_matches', []))} cellar matches")
        return result

    except Exception as e:
        logger.error(f"Error getting food pairing: {e}")
        return {"error": str(e)}


@tool
def get_pairing_for_wine(wine_name: str) -> Dict:
    """Get food pairing recommendations for a specific wine.

    Given a wine from the cellar, suggest foods that pair well with it.
    Useful when deciding what to cook based on a wine you want to drink.

    Args:
        wine_name: The name of the wine to get pairings for.

    Returns:
        Dictionary containing food pairing recommendations.

    Example:
        >>> pairing = get_pairing_for_wine(wine_name="Smerenie")
        >>> print(f"Primary pairings: {pairing['primary_pairings']}")

    Notes:
        - Based on wine characteristics (varietal, region, style)
        - Includes traditional regional pairings
        - Completely free operation (rule-based logic)
    """
    try:
        wine_repo = WineRepository(get_default_db_path())
        wine = wine_repo.get_by_name(wine_name)

        if not wine:
            return {"error": "Wine not found"}

        wine_type = wine.wine_type
        varietal = (wine.varietal or "").lower()

        # Red wines
        if wine_type == "Red":
            if any(v in varietal for v in ["cabernet", "malbec", "syrah", "shiraz"]):
                primary_pairings = ["Grilled steak", "Beef roast", "Lamb chops"]
                secondary_pairings = ["BBQ ribs", "Beef stew", "Aged cheddar"]
                proteins = ["Beef", "Lamb", "Game"]
            elif "pinot noir" in varietal or "burgundy" in varietal:
                primary_pairings = ["Duck breast", "Roasted chicken", "Grilled salmon"]
                secondary_pairings = ["Mushroom risotto", "Pork tenderloin", "Soft cheeses"]
                proteins = ["Duck", "Chicken", "Pork", "Salmon"]
            elif any(v in varietal for v in ["sangiovese", "chianti", "barolo", "nebbiolo"]):
                primary_pairings = ["Pasta with tomato sauce", "Pizza", "Osso buco"]
                secondary_pairings = ["Risotto", "Aged parmesan", "Braised meats"]
                proteins = ["Beef", "Pork", "Game"]
            else:
                primary_pairings = ["Grilled meats", "Pasta with red sauce", "Hard cheeses"]
                secondary_pairings = ["Burgers", "Beef stew", "BBQ"]
                proteins = ["Beef", "Pork", "Lamb"]

        # White wines
        elif wine_type == "White":
            if "chardonnay" in varietal:
                primary_pairings = ["Roasted chicken", "Grilled fish", "Lobster"]
                secondary_pairings = ["Pasta with cream sauce", "Soft cheeses", "Pork chops"]
                proteins = ["Chicken", "Fish", "Shellfish", "Pork"]
            elif any(v in varietal for v in ["sauvignon blanc", "pinot grigio", "albariño"]):
                primary_pairings = ["Oysters", "White fish", "Salads"]
                secondary_pairings = ["Sushi", "Goat cheese", "Grilled vegetables"]
                proteins = ["Shellfish", "White fish", "Chicken"]
            elif "riesling" in varietal:
                primary_pairings = ["Spicy Asian cuisine", "Pork", "Duck"]
                secondary_pairings = ["Thai curry", "Indian food", "Soft cheeses"]
                proteins = ["Pork", "Chicken", "Duck"]
            else:
                primary_pairings = ["White fish", "Chicken", "Salads"]
                secondary_pairings = ["Pasta", "Soft cheeses", "Vegetables"]
                proteins = ["Fish", "Chicken", "Shellfish"]

        # Sparkling
        elif wine_type == "Sparkling":
            primary_pairings = ["Oysters", "Fried foods", "Salty snacks"]
            secondary_pairings = ["Sushi", "Soft cheeses", "Appetizers"]
            proteins = ["Shellfish", "Fish", "Chicken"]

        # Rosé
        elif wine_type == "Rosé":
            primary_pairings = ["Grilled vegetables", "Salmon", "Light pasta"]
            secondary_pairings = ["Salads", "Chicken", "Mediterranean food"]
            proteins = ["Chicken", "Fish", "Pork"]

        else:
            primary_pairings = ["Versatile pairing"]
            secondary_pairings = []
            proteins = []

        # Regional pairings
        traditional_pairings = []
        if wine.country:
            if wine.country == "France":
                traditional_pairings = ["French cuisine", "Coq au vin", "Boeuf bourguignon"]
            elif wine.country == "Italy":
                traditional_pairings = ["Italian cuisine", "Pasta", "Risotto"]
            elif wine.country == "Spain":
                traditional_pairings = ["Paella", "Tapas", "Chorizo"]

        return {
            "wine_name": wine.wine_name,
            "wine_type": wine.wine_type,
            "varietal": wine.varietal,
            "region": wine.region_name,
            "primary_pairings": primary_pairings,
            "secondary_pairings": secondary_pairings,
            "proteins": proteins,
            "traditional_pairings": traditional_pairings,
            "why_it_works": f"{wine_type} wines with {wine.varietal or 'these characteristics'} complement these foods through balance of weight, acidity, and flavor intensity",
            "serving_temperature": "Room temperature (60-68°F)" if wine_type == "Red" else "Chilled (45-55°F)"
        }

    except Exception as e:
        logger.error(f"Error getting wine pairing: {e}")
        return {"error": str(e)}


@tool
def get_wine_and_cheese_pairings(
    cheese_type: Optional[str] = None,
    from_cellar_only: bool = True
) -> Dict:
    """Get wine and cheese pairing recommendations.

    Specialized tool for wine and cheese pairings, which follow specific
    pairing principles different from general food pairings.

    Args:
        cheese_type: Type of cheese to pair. Examples:
                    - "soft" - Brie, Camembert, fresh goat cheese
                    - "hard" - Aged Cheddar, Comté, Manchego, Parmigiano
                    - "blue" - Roquefort, Gorgonzola, Stilton
                    - "washed rind" - Epoisses, Taleggio
                    - "triple cream" - Brillat-Savarin
                    Specific cheese names also accepted.
        from_cellar_only: If True, recommend wines from cellar.

    Returns:
        Dictionary containing cheese and wine pairing recommendations.

    Example:
        >>> pairing = get_wine_and_cheese_pairings("blue")
        >>> pairing = get_wine_and_cheese_pairings("Roquefort")

    Notes:
        - Specialized logic for cheese pairing rules
        - Considers sweet wines for blue cheese
        - Completely free operation (rule-based + local database)
    """
    try:
        cheese_lower = (cheese_type or "").lower().strip()

        # Define cheese pairing rules
        if any(c in cheese_lower for c in ["blue", "roquefort", "gorgonzola", "stilton"]):
            cheese_category = "Blue Cheese"
            classic_pairings = ["Sauternes", "Port", "Late Harvest Riesling", "Tawny Port"]
            wine_types = ["Dessert", "Fortified"]
            why = "Sweetness balances saltiness and pungency of blue cheese"

        elif any(c in cheese_lower for c in ["soft", "brie", "camembert", "goat"]):
            cheese_category = "Soft Cheese"
            classic_pairings = ["Champagne", "Sauvignon Blanc", "Chardonnay", "Loire Valley whites"]
            wine_types = ["White", "Sparkling"]
            why = "Acidity cuts through creamy texture and cleanses palate"

        elif any(c in cheese_lower for c in ["hard", "aged", "cheddar", "comté", "manchego", "parmigiano"]):
            cheese_category = "Hard/Aged Cheese"
            classic_pairings = ["Cabernet Sauvignon", "Rioja", "Barolo", "Chardonnay"]
            wine_types = ["Red", "White"]
            why = "Bold flavors of aged cheese need wines with structure and intensity"

        else:
            cheese_category = "Mixed/General Cheese"
            classic_pairings = ["Champagne", "Pinot Noir", "Chardonnay"]
            wine_types = ["Red", "White", "Sparkling"]
            why = "Versatile wines work with variety of cheeses"

        result = {
            "cheese_category": cheese_category,
            "classic_pairings": classic_pairings,
            "recommended_wine_types": wine_types,
            "why_it_works": why,
            "serving_tips": {
                "cheese_temperature": "Room temperature - remove from fridge 30-60 minutes before serving",
                "wine_temperature": "Appropriate for wine type (reds warmer, whites/sparklings chilled)",
                "accompaniments": ["Crackers", "Fresh fruits", "Honey", "Nuts"]
            },
            "cellar_suggestions": []
        }

        # Find wines from cellar if requested
        if from_cellar_only:
            wine_repo = WineRepository(get_default_db_path())
            bottle_repo = BottleRepository(get_default_db_path())
            suggestions = []

            for wine_type in wine_types:
                wines = wine_repo.get_all(wine_type=wine_type, limit=50)

                for wine in wines:
                    # Use get_owned_quantity
                    n_bottles = bottle_repo.get_owned_quantity(wine.id)
                    if n_bottles == 0:
                        continue

                    # Score based on varietal match
                    score = 50
                    for classic in classic_pairings:
                        if wine.varietal and classic.lower() in wine.varietal.lower():
                            score = 90
                            break
                        if wine.wine_name and classic.lower() in wine.wine_name.lower():
                            score = 85
                            break

                    suggestions.append({
                        "wine_id": wine.id,
                        "name": wine.wine_name,
                        "producer": wine.producer_name,
                        "wine_type": wine.wine_type,
                        "varietal": wine.varietal,
                        "pairing_score": score,
                        "quantity": n_bottles
                    })

            suggestions.sort(key=lambda x: x["pairing_score"], reverse=True)
            result["cellar_suggestions"] = suggestions[:5]

        logger.info(f"Generated cheese pairing for {cheese_type}")
        return result

    except Exception as e:
        logger.error(f"Error getting cheese pairing: {e}")
        return {"error": str(e)}


TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        tool=get_food_pairing_wines,
        metadata=ToolMetadata(
            name="get_food_pairing_wines",
            category=ToolCategory.PAIRING,
            tier=ToolTier.CORE,
            prerequisites=(
                ToolPrerequisite.CELLAR_SCHEMA,
                ToolPrerequisite.PAIRING_RULES,
            ),
            capability="Recommend food-pairing wines using pairing rules and cellar inventory.",
        ),
    ),
    ToolDefinition(
        tool=get_pairing_for_wine,
        metadata=ToolMetadata(
            name="get_pairing_for_wine",
            category=ToolCategory.PAIRING,
            tier=ToolTier.EXTENDED,
            prerequisites=(ToolPrerequisite.CELLAR_SCHEMA,),
            capability="Suggest foods that pair with a specified wine.",
        ),
    ),
    ToolDefinition(
        tool=get_wine_and_cheese_pairings,
        metadata=ToolMetadata(
            name="get_wine_and_cheese_pairings",
            category=ToolCategory.PAIRING,
            tier=ToolTier.EXTENDED,
            prerequisites=(ToolPrerequisite.CELLAR_SCHEMA,),
            capability="Recommend wine and cheese combinations with cellar options.",
        ),
    ),
)
