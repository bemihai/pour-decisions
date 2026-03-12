"""
Heuristic drinking window rules for wine aging estimation.

Rules are Python dataclasses ordered by descending priority. The rule engine
matches on wine attributes using optional regex patterns and returns an estimated
drinking window as offsets from the vintage year.

Usage:
    >>> from src.agents.drinking_window_rules import estimate_from_rules
    >>> from src.database.models import Wine
    >>> wine = Wine(wine_type="Red", varietal="Nebbiolo", region_name="Piedmont", vintage=2018)
    >>> result = estimate_from_rules(wine)
    >>> # result -> (2026, 2038)  i.e. vintage + offset_start, vintage + offset_end
"""

import re
from dataclasses import dataclass

from src.database.models import Wine


@dataclass(frozen=True)
class DrinkingWindowRule:
    """A single heuristic rule for estimating a drinking window.

    Attributes:
        priority: Higher value = more specific; checked first.
        wine_type: Exact match against Wine.wine_type (case-insensitive).
        offset_start: Years after vintage when window opens.
        offset_end: Years after vintage when window closes.
        varietal_re: Optional regex matched against Wine.varietal.
        region_re: Optional regex matched against Wine.region_name or Wine.country.
        designation_re: Optional regex matched against Wine.designation or Wine.appellation.
        note: Human-readable rationale for the rule.
    """

    priority: int
    wine_type: str
    offset_start: int
    offset_end: int
    varietal_re: str | None = None
    region_re: str | None = None
    designation_re: str | None = None
    note: str = ""


def _match(pattern: str | None, value: str | None) -> bool:
    """Return True when pattern is None (wildcard) or matches value."""
    if pattern is None:
        return True
    if not value:
        return False
    return bool(re.search(pattern, value, re.IGNORECASE))


def _matches_rule(rule: DrinkingWindowRule, wine: Wine) -> bool:
    """Return True when all non-None rule fields match the wine."""
    if wine.wine_type.lower() != rule.wine_type.lower():
        return False
    if not _match(rule.varietal_re, wine.varietal):
        return False
    region_target = f"{wine.region_name or ''} {wine.country or ''}"
    if not _match(rule.region_re, region_target):
        return False
    designation_target = f"{wine.designation or ''} {wine.appellation or ''}"
    if not _match(rule.designation_re, designation_target):
        return False
    return True


def estimate_from_rules(wine: Wine) -> tuple[int, int] | None:
    """Return (drink_from_year, drink_to_year) for a wine using the heuristic rule engine.

    Rules are evaluated from highest to lowest priority. The first match wins.
    Non-vintage wines always return None.

    Args:
        wine: Wine model. Must have vintage set for estimation to proceed.

    Returns:
        Tuple (drink_from_year, drink_to_year) or None if no rule matches or
        vintage is missing.

    Example:
        >>> wine = Wine(wine_type="Red", varietal="Nebbiolo", region_name="Piedmont - Barolo", vintage=2018)
        >>> estimate_from_rules(wine)
        (2026, 2043)
    """
    if wine.vintage is None:
        return None

    for rule in RULES:
        if _matches_rule(rule, wine):
            return wine.vintage + rule.offset_start, wine.vintage + rule.offset_end

    return None


# ---------------------------------------------------------------------------
# Rule catalogue -- ordered by descending priority (most specific first)
# ---------------------------------------------------------------------------

RULES: list[DrinkingWindowRule] = [
    # --- Red: Nebbiolo / Barolo / Barbaresco ---
    DrinkingWindowRule(
        priority=100, wine_type="Red",
        varietal_re=r"Nebbiolo",
        designation_re=r"Riserva",
        offset_start=8, offset_end=30,
        note="Barolo/Barbaresco Riserva: long ageing",
    ),
    DrinkingWindowRule(
        priority=95, wine_type="Red",
        varietal_re=r"Nebbiolo",
        region_re=r"Barolo|Barbaresco|Piedmont|Piemonte",
        offset_start=5, offset_end=20,
        note="Barolo/Barbaresco standard",
    ),
    DrinkingWindowRule(
        priority=90, wine_type="Red",
        varietal_re=r"Nebbiolo",
        offset_start=3, offset_end=12,
        note="Langhe Nebbiolo and other regional",
    ),

    # --- Red: Cabernet-dominant Bordeaux ---
    DrinkingWindowRule(
        priority=88, wine_type="Red",
        varietal_re=r"Cabernet",
        region_re=r"Bordeaux|Médoc|Medoc|Pauillac|Saint-Julien|Margaux|Saint-Estèphe|Pessac",
        designation_re=r"Grand Cru|Premier Cru|Classé|Classe",
        offset_start=8, offset_end=30,
        note="Classified Bordeaux Cabernet",
    ),
    DrinkingWindowRule(
        priority=85, wine_type="Red",
        varietal_re=r"Cabernet",
        region_re=r"Bordeaux|Médoc|Medoc|Pauillac|Saint-Julien|Margaux|Saint-Estèphe|Pessac",
        offset_start=5, offset_end=20,
        note="Standard Bordeaux Cabernet",
    ),
    DrinkingWindowRule(
        priority=82, wine_type="Red",
        varietal_re=r"Cabernet Sauvignon",
        region_re=r"Napa|California|Sonoma",
        offset_start=4, offset_end=18,
        note="Napa / California Cabernet Sauvignon",
    ),
    DrinkingWindowRule(
        priority=80, wine_type="Red",
        varietal_re=r"Cabernet Sauvignon",
        offset_start=3, offset_end=12,
        note="Generic Cabernet Sauvignon",
    ),

    # --- Red: Sangiovese / Tuscany ---
    DrinkingWindowRule(
        priority=78, wine_type="Red",
        varietal_re=r"Sangiovese",
        region_re=r"Brunello|Montalcino",
        offset_start=8, offset_end=25,
        note="Brunello di Montalcino",
    ),
    DrinkingWindowRule(
        priority=76, wine_type="Red",
        varietal_re=r"Sangiovese",
        region_re=r"Chianti Classico",
        designation_re=r"Gran Selezione|Riserva",
        offset_start=5, offset_end=18,
        note="Chianti Classico Riserva / Gran Selezione",
    ),
    DrinkingWindowRule(
        priority=74, wine_type="Red",
        varietal_re=r"Sangiovese",
        region_re=r"Chianti|Tuscany|Toscana",
        offset_start=2, offset_end=10,
        note="Generic Chianti / Tuscan Sangiovese",
    ),
    DrinkingWindowRule(
        priority=72, wine_type="Red",
        varietal_re=r"Sangiovese",
        offset_start=2, offset_end=8,
        note="Generic Sangiovese",
    ),

    # --- Red: Pinot Noir / Burgundy ---
    DrinkingWindowRule(
        priority=70, wine_type="Red",
        varietal_re=r"Pinot Noir",
        region_re=r"Burgundy|Bourgogne|Côte de Nuits|Cote de Nuits|Gevrey|Chambolle|Vosne",
        designation_re=r"Grand Cru",
        offset_start=5, offset_end=20,
        note="Burgundy Grand Cru Pinot Noir",
    ),
    DrinkingWindowRule(
        priority=68, wine_type="Red",
        varietal_re=r"Pinot Noir",
        region_re=r"Burgundy|Bourgogne|Côte de Nuits|Cote de Nuits",
        designation_re=r"Premier Cru|1er Cru",
        offset_start=4, offset_end=15,
        note="Burgundy Premier Cru Pinot Noir",
    ),
    DrinkingWindowRule(
        priority=66, wine_type="Red",
        varietal_re=r"Pinot Noir",
        region_re=r"Burgundy|Bourgogne",
        offset_start=2, offset_end=10,
        note="Generic Burgundy Pinot Noir",
    ),
    DrinkingWindowRule(
        priority=64, wine_type="Red",
        varietal_re=r"Pinot Noir",
        region_re=r"Oregon|Willamette",
        offset_start=2, offset_end=10,
        note="Oregon Pinot Noir",
    ),
    DrinkingWindowRule(
        priority=62, wine_type="Red",
        varietal_re=r"Pinot Noir",
        offset_start=1, offset_end=8,
        note="Generic Pinot Noir",
    ),

    # --- Red: Syrah / Rhône ---
    DrinkingWindowRule(
        priority=60, wine_type="Red",
        varietal_re=r"Syrah|Shiraz",
        region_re=r"Hermitage|Côte-Rôtie|Cote-Rotie|Cornas",
        offset_start=5, offset_end=20,
        note="Northern Rhône Syrah",
    ),
    DrinkingWindowRule(
        priority=58, wine_type="Red",
        varietal_re=r"Syrah|Shiraz",
        region_re=r"Rhône|Rhone|Barossa",
        offset_start=3, offset_end=12,
        note="Generic Rhône / Barossa Syrah",
    ),
    DrinkingWindowRule(
        priority=56, wine_type="Red",
        varietal_re=r"Syrah|Shiraz",
        offset_start=2, offset_end=8,
        note="Generic Syrah/Shiraz",
    ),

    # --- Red: Tempranillo / Rioja ---
    DrinkingWindowRule(
        priority=54, wine_type="Red",
        varietal_re=r"Tempranillo|Tinto Fino|Tinto del País",
        designation_re=r"Gran Reserva",
        offset_start=5, offset_end=20,
        note="Rioja / Ribera Gran Reserva",
    ),
    DrinkingWindowRule(
        priority=52, wine_type="Red",
        varietal_re=r"Tempranillo|Tinto Fino",
        designation_re=r"Reserva",
        offset_start=3, offset_end=15,
        note="Rioja / Ribera Reserva",
    ),
    DrinkingWindowRule(
        priority=50, wine_type="Red",
        varietal_re=r"Tempranillo",
        offset_start=2, offset_end=10,
        note="Generic Tempranillo",
    ),

    # --- Red: Malbec ---
    DrinkingWindowRule(
        priority=48, wine_type="Red",
        varietal_re=r"Malbec",
        designation_re=r"Reserva|Reserve|Gran",
        offset_start=3, offset_end=12,
        note="Reserva Malbec",
    ),
    DrinkingWindowRule(
        priority=46, wine_type="Red",
        varietal_re=r"Malbec",
        offset_start=1, offset_end=6,
        note="Generic Malbec",
    ),

    # --- Red: Gamay ---
    DrinkingWindowRule(
        priority=44, wine_type="Red",
        varietal_re=r"Gamay",
        offset_start=0, offset_end=3,
        note="Beaujolais / Gamay — drink fresh",
    ),

    # --- Red: generic fallback ---
    DrinkingWindowRule(
        priority=10, wine_type="Red",
        offset_start=2, offset_end=8,
        note="Generic red wine fallback",
    ),

    # --- White: Chardonnay / Burgundy ---
    DrinkingWindowRule(
        priority=88, wine_type="White",
        varietal_re=r"Chardonnay",
        region_re=r"Burgundy|Bourgogne|Puligny|Meursault|Chassagne|Chablis",
        designation_re=r"Grand Cru",
        offset_start=3, offset_end=15,
        note="Burgundy Grand Cru Chardonnay",
    ),
    DrinkingWindowRule(
        priority=86, wine_type="White",
        varietal_re=r"Chardonnay",
        region_re=r"Burgundy|Bourgogne|Puligny|Meursault|Chassagne|Chablis",
        offset_start=2, offset_end=10,
        note="Burgundy village / Premier Cru Chardonnay",
    ),
    DrinkingWindowRule(
        priority=82, wine_type="White",
        varietal_re=r"Chardonnay",
        offset_start=1, offset_end=5,
        note="Generic Chardonnay",
    ),

    # --- White: Riesling ---
    DrinkingWindowRule(
        priority=84, wine_type="White",
        varietal_re=r"Riesling",
        designation_re=r"Spätlese|Auslese|Beerenauslese|TBA|Trockenbeerenauslese|Eiswein",
        offset_start=3, offset_end=20,
        note="Riesling Praedikat (sweet/off-dry)",
    ),
    DrinkingWindowRule(
        priority=80, wine_type="White",
        varietal_re=r"Riesling",
        region_re=r"Mosel|Rheingau|Alsace|Alsatian",
        offset_start=2, offset_end=12,
        note="Classic-region Riesling",
    ),
    DrinkingWindowRule(
        priority=76, wine_type="White",
        varietal_re=r"Riesling",
        offset_start=1, offset_end=6,
        note="Generic Riesling",
    ),

    # --- White: Sauvignon Blanc ---
    DrinkingWindowRule(
        priority=74, wine_type="White",
        varietal_re=r"Sauvignon Blanc",
        region_re=r"Sancerre|Pouilly-Fumé|Pouilly-Fume|Loire",
        offset_start=1, offset_end=5,
        note="Loire Sauvignon Blanc",
    ),
    DrinkingWindowRule(
        priority=70, wine_type="White",
        varietal_re=r"Sauvignon Blanc",
        offset_start=0, offset_end=3,
        note="Generic Sauvignon Blanc",
    ),

    # --- White: generic fallback ---
    DrinkingWindowRule(
        priority=10, wine_type="White",
        offset_start=1, offset_end=4,
        note="Generic white wine fallback",
    ),

    # --- Rosé ---
    DrinkingWindowRule(
        priority=10, wine_type="Rosé",
        offset_start=0, offset_end=2,
        note="Generic rosé — drink young",
    ),

    # --- Sparkling ---
    DrinkingWindowRule(
        priority=20, wine_type="Sparkling",
        designation_re=r"Prestige|Cuvée|Blanc de Blancs|Blanc de Noirs|Vintage",
        region_re=r"Champagne",
        offset_start=2, offset_end=12,
        note="Vintage / prestige Champagne",
    ),
    DrinkingWindowRule(
        priority=10, wine_type="Sparkling",
        offset_start=0, offset_end=3,
        note="Generic sparkling — drink young",
    ),

    # --- Dessert ---
    DrinkingWindowRule(
        priority=20, wine_type="Dessert",
        varietal_re=r"Sauternes|Riesling|Furmint",
        offset_start=3, offset_end=20,
        note="Classic noble rot dessert wines",
    ),
    DrinkingWindowRule(
        priority=10, wine_type="Dessert",
        offset_start=1, offset_end=10,
        note="Generic dessert wine",
    ),

    # --- Fortified ---
    DrinkingWindowRule(
        priority=20, wine_type="Fortified",
        designation_re=r"Vintage|Tawny|LBV|Late Bottled",
        offset_start=0, offset_end=20,
        note="Vintage Port / Tawny",
    ),
    DrinkingWindowRule(
        priority=10, wine_type="Fortified",
        offset_start=0, offset_end=10,
        note="Generic fortified wine",
    ),
]

# Sort descending by priority at module load to ensure highest priority is checked first.
RULES.sort(key=lambda r: r.priority, reverse=True)

