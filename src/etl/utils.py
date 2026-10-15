"""Utility functions for ETL pipelines."""

import hashlib
import html
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional

from dateutil import parser

from src.utils import logger

# Wine type mapping from various formats to standardized types
WINE_TYPE_MAP = {
    # Vivino formats
    'Red Wine': 'Red',
    'White Wine': 'White',
    'Rosé Wine': 'Rosé',
    'Sparkling': 'Sparkling',
    'Dessert Wine': 'Dessert',
    'Fortified Wine': 'Fortified',

    # CellarTracker formats
    'Red': 'Red',
    'White': 'White',
    'Rosé': 'Rosé',
    'Dessert': 'Dessert',
    'Fortified': 'Fortified',
}

CELLARTRACKER_WINE_TYPE_MAP = {
    "red": "Red",
    "white": "White",
    "rosé": "Rosé",
    "sparkling": "Sparkling",
    "dessert": "Dessert",
    "fortified": "Fortified",
}

COUNTRY_MAP = {
    "ro": "Romania",
    "fr": "France",
    "it": "Italy",
    "es": "Spain",
    "de": "Germany",
    "pt": "Portugal",
    "us": "United States",
    "md": "Moldova",
}


def normalize_wine_type(type_str: str) -> str:
    """
    Standardize wine type to unified format.
    """
    if not type_str:
        return "Red"

    return WINE_TYPE_MAP.get(type_str.strip(), "Red")


def normalize_cellartracker_wine_type(type_str: str | None) -> str:
    """Normalize a CellarTracker wine type without inferring from other fields.

    CellarTracker may combine a base wine family with a style qualifier using a
    hyphen, such as ``"White - Sparkling"`` or ``"White - offdry"``.

    Args:
        type_str: Source value from CellarTracker's ``Type`` field.

    Returns:
        The canonical wine family, or ``"Unknown"`` when no supported family
        is present.
    """
    source_type = type_str.strip() if type_str else ""
    components = [component.strip().casefold() for component in source_type.split("-")]

    if "sparkling" in components:
        return "Sparkling"

    for component in components:
        if canonical_type := CELLARTRACKER_WINE_TYPE_MAP.get(component):
            return canonical_type

    logger.warning("Unrecognized CellarTracker wine type: %r", type_str)
    return "Unknown"


def clean_text(text: str) -> str | None:
    """
    Clean and normalize text.
    """
    if not text or text.strip() == "" or text.strip().lower() == "unknown":
        return None
    text = html.unescape(text)
    text = text.strip()

    return text if text else None


def parse_country(country_str: str) -> str | None:
    """
    Parse country name from various formats.
    """
    if not country_str or country_str.strip() == "":
        return None
    country_str = clean_text(country_str)
    if country_str in COUNTRY_MAP.keys():
        return COUNTRY_MAP[country_str]
    return country_str


def parse_date(date_str: str) -> str | None:
    """
    Parse various date formats to YYYY-MM-DD.
    """
    if not date_str:
        return None

    try:
        dt = parser.parse(date_str)
        return dt.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None


def parse_vintage(vintage_str: str) -> Optional[int]:
    """
    Parse vintage year from string.
    """
    if not vintage_str or vintage_str.strip() == "":
        return None

    try:
        vintage = int(vintage_str.strip())
        if 1900 <= vintage <= datetime.now().year + 2:
            return vintage
        return None
    except (ValueError, TypeError):
        return None


def parse_drinking_window(window_str: str, end_str: str | None = None) -> tuple[Optional[int], Optional[int]]:
    """
    Parse drinking window from various formats.

    Supports:
    - Vivino format: "2025 2027" (single string with space)
    - CellarTracker format: separate start and end parameters

    Args:
        window_str: Drinking window string from Vivino, or start year from CellarTracker
        end_str: Optional end year (for CellarTracker format)

    Returns:
        Tuple of (start_year, end_year) or (None, None)
    """
    if not window_str or window_str.strip() == '':
        return None, None

    try:
        # If end_str is provided, use CellarTracker format
        if end_str:
            start = int(window_str.strip()) if window_str else None
            end = int(end_str.strip()) if end_str else None
            return start, end

        # Otherwise try Vivino format (space-separated)
        parts = window_str.strip().split()
        if len(parts) == 2:
            start = int(parts[0])
            end = int(parts[1])
            return start, end
        elif len(parts) == 1:
            year = int(parts[0])
            return year, year
    except (ValueError, TypeError):
        pass

    return None, None


def string_similarity(str1: str, str2: str) -> float:
    """
    Calculate similarity ratio between two strings (0-1 scale).
    """
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def normalize_rating(rating: float, source: str) -> int | None:
    """
    Normalize ratings to 0-100 scale with interval mapping for Vivino ratings.

    Vivino (0-5) to 0-100 mapping:
    - 0-2.9   -> 70-79
    - 3.0-3.5 -> 80-85
    - 3.6-3.9 -> 86-89
    - 4.0-4.4 -> 90-93
    - 4.5-4.7 -> 94-97
    - 4.8-5.0 -> 98-100

    Args:
        rating: The rating value to normalize
        source: Source system ('vivino' or 'cellar_tracker')

    Returns:
        Normalized rating on 0-100 scale, or None if invalid
    """
    if rating is None:
        return None

    try:
        if source == 'vivino':
            # Vivino uses 0-5 scale, convert to 0-100 with interval mapping
            rating_float = float(rating)

            # Clamp to valid range
            rating_float = max(0.0, min(5.0, rating_float))

            # Apply interval mapping
            if rating_float < 3.0:
                # 0-2.9 -> 70-79 (linear interpolation)
                normalized = 70 + ((rating_float - 0.0) / 2.9) * 9
            elif rating_float < 3.6:
                # 3.0-3.5 -> 80-85 (linear interpolation)
                normalized = 80 + ((rating_float - 3.0) / 0.5) * 5
            elif rating_float < 4.0:
                # 3.6-3.9 -> 86-89 (linear interpolation)
                normalized = 86 + ((rating_float - 3.6) / 0.4) * 4
            elif rating_float < 4.5:
                # 4.0-4.4 -> 90-93 (linear interpolation)
                normalized = 90 + ((rating_float - 4.0) / 0.5) * 4
            elif rating_float < 4.8:
                # 4.5-4.7 -> 94-97 (linear interpolation)
                normalized = 94 + ((rating_float - 4.5) / 0.3) * 4
            else:
                # 4.8-5.0 -> 98-100 (linear interpolation)
                normalized = 98 + ((rating_float - 4.8) / 0.2) * 2

            return int(round(normalized))

        elif source == 'cellar_tracker':
            # CellarTracker already uses 0-100
            return int(rating)
    except (ValueError, TypeError):
        return None

    return None


def denormalize_rating(normalized_rating: int) -> float | None:
    """
    Convert normalized 0-100 rating back to Vivino's 0-5 scale.

    Uses reverse interval mapping:
    - 70-79   -> 0-2.9
    - 80-85   -> 3.0-3.5
    - 86-89   -> 3.6-3.9
    - 90-93   -> 4.0-4.4
    - 94-97   -> 4.5-4.7
    - 98-100  -> 4.8-5.0

    Args:
        normalized_rating: Rating on 0-100 scale

    Returns:
        Rating on 0-5 scale (Vivino format), or None if invalid
    """
    if normalized_rating is None:
        return None

    try:
        rating_int = int(normalized_rating)

        # Clamp to valid range
        rating_int = max(0, min(100, rating_int))

        # Apply reverse interval mapping
        if rating_int < 70:
            # Below our mapping range, extrapolate
            vivino_rating = (rating_int / 70) * 2.9
        elif rating_int < 80:
            # 70-79 -> 0-2.9
            vivino_rating = 0.0 + ((rating_int - 70) / 9) * 2.9
        elif rating_int < 86:
            # 80-85 -> 3.0-3.5
            vivino_rating = 3.0 + ((rating_int - 80) / 5) * 0.5
        elif rating_int < 90:
            # 86-89 -> 3.6-3.9
            vivino_rating = 3.6 + ((rating_int - 86) / 4) * 0.4
        elif rating_int < 94:
            # 90-93 -> 4.0-4.4
            vivino_rating = 4.0 + ((rating_int - 90) / 4) * 0.5
        elif rating_int < 98:
            # 94-97 -> 4.5-4.7
            vivino_rating = 4.5 + ((rating_int - 94) / 4) * 0.3
        else:
            # 98-100 -> 4.8-5.0
            vivino_rating = 4.8 + ((rating_int - 98) / 2) * 0.2

        # Round to 1 decimal place (typical Vivino precision)
        return round(vivino_rating, 1)
    except (ValueError, TypeError):
        return None


def get_rating_description(rating: int | float | None) -> str:
    """
    Map a numerical score (0-100) to a descriptive quality rating.

    Quality tiers based on interval mapping:
    - 70-79:  Average
    - 80-85:  Good
    - 86-89:  Very Good
    - 90-93:  Excellent
    - 94-97:  Outstanding
    - 98-100: Exceptional

    Args:
        rating: Numerical rating on 0-100 scale (or None)

    Returns:
        String description of quality level
    """
    if rating is None:
        return "Not Rated"

    try:
        score = int(rating) if isinstance(rating, float) else rating

        if score < 70:
            return "Below Average"
        elif score < 80:
            return "Average"
        elif score < 86:
            return "Good"
        elif score < 90:
            return "Very Good"
        elif score < 94:
            return "Excellent"
        elif score < 98:
            return "Outstanding"
        else:
            return "Exceptional"
    except (ValueError, TypeError):
        return "Not Rated"


def generate_external_id(winery: str, wine_name: str, vintage: int | None) -> str:
    """
    Generate a consistent external ID for Vivino wines.
    """
    vintage_str = str(vintage) if vintage else "NV"
    key = f"{winery}_{wine_name}_{vintage_str}".lower()
    md5_hash = hashlib.md5(key.encode()).hexdigest()
    return str(int(md5_hash, 16) % (2**32))


def parse_float(value: str) -> Optional[float]:
    """
    Safely parse a float from string, returning None if invalid.
    """
    if not value or value.strip() == "":
        return None

    try:
        return float(value.strip())
    except (ValueError, TypeError):
        return None


def parse_int(value: str) -> Optional[int]:
    """
    Safely parse an integer from string, returning None if invalid.
    """
    if not value or value.strip() == "":
        return None

    try:
        return int(value.strip())
    except (ValueError, TypeError):
        return None


def parse_bool(value: str) -> Optional[bool]:
    """
    Safely parse a boolean from string, returning None if invalid.
    Accepts 'true', 'false', '1', '0' (case-insensitive).
    """
    if not value or value.strip() == "":
        return None

    value_lower = value.strip().lower()
    if value_lower in ['true', '1']:
        return True
    elif value_lower in ['false', '0']:
        return False
    return None
