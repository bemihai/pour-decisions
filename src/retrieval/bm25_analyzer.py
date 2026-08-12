"""Deterministic wine-aware analyzer shared by BM25 documents and queries."""

from __future__ import annotations

from functools import lru_cache
import unicodedata

from src.utils import GRAPE_SYNONYMS, MISSPELLINGS, REGION_VARIATIONS


_APOSTROPHES = {"'", "’", "ʼ", "`"}
_HYPHENS = {"-", "‐", "‑", "‒", "–", "—", "―"}
_QUESTION_STOPWORDS = {
    "a",
    "an",
    "are",
    "about",
    "can",
    "characteristic",
    "characteristics",
    "could",
    "describe",
    "did",
    "do",
    "does",
    "explain",
    "flavor",
    "flavors",
    "flavour",
    "flavours",
    "give",
    "how",
    "is",
    "main",
    "me",
    "notes",
    "of",
    "primary",
    "profile",
    "should",
    "tell",
    "the",
    "typical",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "would",
}


def analyze_bm25_text(text: str) -> list[str]:
    """Normalize text, canonicalize configured wine terms, and remove query filler."""
    tokens = _unicode_tokens(text)
    canonical_tokens = _replace_terminology(tokens)
    return [token for token in canonical_tokens if token not in _QUESTION_STOPWORDS]


def _unicode_tokens(text: str) -> list[str]:
    """Tokenize Unicode text consistently across punctuation and diacritics."""
    normalized = unicodedata.normalize("NFKD", str(text or "").casefold())
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    characters: list[str] = []
    for index, character in enumerate(normalized):
        if character.isalnum():
            characters.append(character)
            continue
        if character in _APOSTROPHES and _between_alphanumeric(normalized, index):
            characters.append("'")
            continue
        if character in _HYPHENS:
            characters.append(" ")
            continue
        characters.append(" ")
    return "".join(characters).split()


def _between_alphanumeric(text: str, index: int) -> bool:
    """Return whether punctuation joins two alphanumeric characters."""
    return index > 0 and index + 1 < len(text) and text[index - 1].isalnum() and text[index + 1].isalnum()


def _replace_terminology(tokens: list[str]) -> list[str]:
    """Apply longest-match terminology aliases without substring replacements."""
    aliases = _terminology_aliases()
    if not tokens or not aliases:
        return tokens

    maximum_length = max(len(alias) for alias in aliases)
    output: list[str] = []
    index = 0
    while index < len(tokens):
        matched = False
        for length in range(min(maximum_length, len(tokens) - index), 0, -1):
            alias = tuple(tokens[index : index + length])
            canonical = aliases.get(alias)
            if canonical is None:
                continue
            output.extend(canonical)
            index += length
            matched = True
            break
        if not matched:
            output.append(tokens[index])
            index += 1
    return output


@lru_cache(maxsize=1)
def _terminology_aliases() -> dict[tuple[str, ...], tuple[str, ...]]:
    """Build deterministic normalized aliases from the configured terminology."""
    aliases: dict[tuple[str, ...], tuple[str, ...]] = {}
    for canonical, variants in [*GRAPE_SYNONYMS.items(), *REGION_VARIATIONS.items()]:
        canonical_tokens = tuple(_unicode_tokens(canonical))
        if not canonical_tokens:
            continue
        aliases.setdefault(canonical_tokens, canonical_tokens)
        for variant in variants:
            variant_tokens = tuple(_unicode_tokens(variant))
            if variant_tokens:
                aliases.setdefault(variant_tokens, canonical_tokens)

    for misspelling, correction in MISSPELLINGS.items():
        misspelled_tokens = tuple(_unicode_tokens(misspelling))
        corrected_tokens = tuple(_unicode_tokens(correction))
        if misspelled_tokens and corrected_tokens:
            aliases[misspelled_tokens] = aliases.get(corrected_tokens, corrected_tokens)
    return aliases
