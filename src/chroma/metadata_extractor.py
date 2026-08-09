"""Wine metadata extraction from document content.

This module provides utilities for extracting wine-specific metadata from text,
including grape varieties, wine regions, vintage years, and producer names.
"""
import re
from dataclasses import dataclass, field

from src.utils import WINE_APPELLATIONS, GRAPE_PATTERNS, REGION_PATTERNS, CLASSIFICATION_PATTERNS, PRODUCER_PREFIXES, \
    PRODUCER_SUFFIXES


@dataclass
class WineMetadata:
    """
    Extracted wine-specific metadata from text content.

    Attributes:
        grapes: set of grape varieties mentioned in the text.
        regions: set of wine regions mentioned in the text.
        vintages: set of vintage years mentioned in the text.
        classifications: set of wine classifications (DOCG, AOC, etc.) mentioned.
        producers: set of producer/winery names mentioned.
        appellations: set of wine appellations mentioned (Barolo, Champagne, etc.).
    """
    grapes: set[str] = field(default_factory=set)
    regions: set[str] = field(default_factory=set)
    vintages: set[str] = field(default_factory=set)
    classifications: set[str] = field(default_factory=set)
    producers: set[str] = field(default_factory=set)
    appellations: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, list[str]]:
        """Convert to dictionary with sorted lists for JSON serialization."""
        return {
            "grapes": sorted(self.grapes),
            "regions": sorted(self.regions),
            "vintages": sorted(self.vintages),
            "classifications": sorted(self.classifications),
            "producers": sorted(self.producers),
            "appellations": sorted(self.appellations),
        }

    def is_empty(self) -> bool:
        """Check if no metadata was extracted."""
        return not any([
            self.grapes, self.regions, self.vintages,
            self.classifications, self.producers, self.appellations
        ])


def extract_grapes(text: str) -> set[str]:
    """
    Extract grape variety names from text.

    Uses the wine terminology dictionary to find canonical grape names
    and their synonyms in the text.

    Args:
        text: Text content to search for grape varieties.

    Returns:
        set of canonical grape variety names found in the text.
    """
    text_lower = text.lower()
    found_grapes = set()

    for pattern, canonical in GRAPE_PATTERNS.items():
        if re.search(rf'\b{re.escape(pattern)}\b', text_lower):
            found_grapes.add(canonical)

    return found_grapes


def extract_regions(text: str) -> set[str]:
    """
    Extract wine region names from text.

    Uses the wine terminology dictionary to find canonical region names
    and their variations in the text.

    Args:
        text: Text content to search for wine regions.

    Returns:
        set of canonical wine region names found in the text.
    """
    text_lower = text.lower()
    found_regions = set()

    for pattern, canonical in REGION_PATTERNS.items():
        if re.search(rf'\b{re.escape(pattern)}\b', text_lower):
            found_regions.add(canonical)

    return found_regions


def extract_vintages(text: str) -> set[str]:
    """
    Extract vintage years from text.

    Looks for 4-digit years in the range 1900-2050 that appear in
    wine-related contexts.

    Args:
        text: Text content to search for vintage years.

    Returns:
        set of vintage year strings found in the text.
    """
    # Match years from 1900-2050
    year_pattern = r'\b(19[0-9]{2}|20[0-2][0-9]|2050)\b'
    matches = re.findall(year_pattern, text)

    # Filter to likely vintages (not page numbers, etc.)
    vintages = set()
    for year in matches:
        year_int = int(year)
        if 1900 <= year_int <= 2050:
            vintages.add(year)

    return vintages


def extract_classifications(text: str) -> set[str]:
    """
    Extract wine classification terminology from text.

    Looks for wine classification abbreviations like DOCG, AOC, AVA, etc.

    Args:
        text: Text content to search for classifications.

    Returns:
        set of classification abbreviations found in the text (uppercase).
    """
    text_lower = text.lower()
    found = set()

    for classification in CLASSIFICATION_PATTERNS:
        if re.search(rf'\b{re.escape(classification)}\b', text_lower):
            found.add(classification.upper())

    return found


def extract_producers(text: str) -> set[str]:
    """
    Extract wine producer/winery names from text.

    Uses pattern matching for common producer naming conventions:
    - Prefix patterns: Château X, Domaine Y, Bodega Z
    - Suffix patterns: X Winery, Y Vineyards, Z Estate

    Args:
        text: Text content to search for producer names.

    Returns:
        set of producer names found in the text.
    """
    found = set()

    # Build prefix pattern: "Château/Domaine/etc. + Name"
    prefix_pattern = r'(?:' + '|'.join(PRODUCER_PREFIXES) + r')\s+([A-Z][a-zA-Zéèêëàâäùûüôöîïç\-\']+(?:\s+[A-Z][a-zA-Zéèêëàâäùûüôöîïç\-\']+){0,3})'

    for match in re.finditer(prefix_pattern, text, re.IGNORECASE):
        full_match = match.group(0).strip()
        if len(full_match) > 5:  # Avoid very short matches
            found.add(full_match.title())

    # Build suffix pattern: "Name + Winery/Vineyards/etc."
    suffix_pattern = r'([A-Z][a-zA-Zéèêëàâäùûüôöîïç\-\']+(?:\s+[A-Z][a-zA-Zéèêëàâäùûüôöîïç\-\']+){0,2})\s+(?:' + '|'.join(PRODUCER_SUFFIXES) + r')'

    for match in re.finditer(suffix_pattern, text, re.IGNORECASE):
        full_match = match.group(0).strip()
        if len(full_match) > 5:
            found.add(full_match.title())

    return found


def extract_appellations(text: str) -> set[str]:
    """
    Extract wine appellation names from text.

    Looks for famous wine appellations that indicate specific wine types
    (e.g., Barolo, Champagne, Châteauneuf-du-Pape).

    Args:
        text: Text content to search for appellations.

    Returns:
        set of appellation names found in the text.
    """
    text_lower = text.lower()
    found = set()

    for appellation in WINE_APPELLATIONS:
        pattern = rf'\b{re.escape(appellation)}\b'
        if re.search(pattern, text_lower):
            found.add(appellation.title())

    return found


def extract_wine_metadata(text: str) -> WineMetadata:
    """
    Extract all wine-specific metadata from text.

    This is the main entry point for wine metadata extraction. It combines
    all individual extraction functions into a single WineMetadata object.

    Args:
        text: Text content to analyze for wine metadata.

    Returns:
        WineMetadata object containing all extracted information.
    """
    return WineMetadata(
        grapes=extract_grapes(text),
        regions=extract_regions(text),
        vintages=extract_vintages(text),
        classifications=extract_classifications(text),
        producers=extract_producers(text),
        appellations=extract_appellations(text),
    )


def extract_document_context(elements: list, max_title_length: int = 200) -> dict[str, str]:
    """
    Extract document-level context from parsed elements.

    Extracts title, chapter headings, and generates a brief summary
    from the document structure. This context can be added to chunks
    to improve retrieval.

    Args:
        elements: list of legacy parser-like elements with category and text values.
        max_title_length: Maximum length for extracted title.

    Returns:
        dictionary with 'document_title', 'chapter', and 'section' keys.
    """
    context = {
        "document_title": "",
        "chapter": "",
        "section": "",
    }

    if not elements:
        return context

    # Extract title from first Title element or first element
    for elem in elements[:10]:  # Check first 10 elements
        elem_type = getattr(elem, 'category', '') or type(elem).__name__
        elem_text = str(elem).strip()

        if elem_type == 'Title' and elem_text:
            context["document_title"] = elem_text[:max_title_length]
            break
        elif not context["document_title"] and elem_text:
            # Use first non-empty element as fallback
            context["document_title"] = elem_text[:max_title_length]

    # Look for chapter/section structure
    current_chapter = ""
    current_section = ""

    for elem in elements:
        elem_type = getattr(elem, 'category', '') or type(elem).__name__
        elem_text = str(elem).strip()

        if elem_type == 'Title':
            # Check if it looks like a chapter heading
            if re.match(r'^(Chapter|Part|Section)\s+\d+', elem_text, re.IGNORECASE):
                current_chapter = elem_text
            elif len(elem_text) < 100:  # Short titles are likely section headers
                current_section = elem_text

    context["chapter"] = current_chapter
    context["section"] = current_section

    return context
