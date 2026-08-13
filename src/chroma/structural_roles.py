"""Provider-neutral structural role classification for extracted and chunked text."""

from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import mean
from typing import Any, Mapping


STRUCTURAL_ROLES = {
    "prose",
    "table",
    "wine_list",
    "toc",
    "bibliography",
    "index",
    "worksheet",
    "unknown",
}
REJECTED_STRUCTURAL_ROLES = {"toc", "bibliography", "index", "worksheet"}

_BIBLIOGRAPHY_HEADING_PATTERN = re.compile(
    r"^(?:bibliograph(?:y|ies)|references|notes|endnotes|works cited)$",
    re.IGNORECASE,
)
_TOC_HEADING_PATTERN = re.compile(r"^(?:table of contents|contents)$", re.IGNORECASE)
_INDEX_HEADING_PATTERN = re.compile(r"^(?:general index|subject index|index)$", re.IGNORECASE)
_CITATION_PATTERN = re.compile(
    r"(?:\bibid\.|\bisbn\b|\bdoi\s*:|\bet\s+al\.|\bpp?\.\s*\d+|\((?:19|20)\d{2}\))",
    re.IGNORECASE,
)
_NUMBERED_REFERENCE_PATTERN = re.compile(r"^\s*\d{1,3}\.\s+.+", re.MULTILINE)
_TOC_LINE_PATTERN = re.compile(r"^.{2,80}(?:\.{3,}|\s{3,})\s*\d{1,4}\s*$")
_INDEX_LINE_PATTERN = re.compile(r"^[A-Za-zÀ-ž][^.!?]{1,80},?\s+\d+(?:\s*[-,]\s*\d+)*$")
_OCR_CID_PATTERN = re.compile(r"\(cid:\d+\)", re.IGNORECASE)
_FORM_BLANK_PATTERN = re.compile(r"_{4,}|\.{5,}")
_FORM_SCALE_PATTERN = re.compile(r"\b0(?:\s*-+\s*\d+){3,}|\b0\s+5\s+10\b")
_TASTING_FORM_FIELD_PATTERN = re.compile(
    r"\b(?:wine|producer|year)\s*_{4,}|\bvisual\s+clarity\s*:|"
    r"\btactile\s+sensations?\s*:|\btaste\s+sweetness\s*:",
    re.IGNORECASE,
)
_WORKSHEET_LABEL_PATTERN = re.compile(
    r"\b(?:food|wine)\s+(?:flavo[u]?r\s+type|observations?)\s*:|"
    r"\blevel of (?:food\s*&\s*wine )?match\s*:|\bcomments?\s*:|\bmatch based on\s*:",
    re.IGNORECASE,
)
_SENTENCE_END_PATTERN = re.compile(r"[.!?](?:[\"')\]]+)?$")


@dataclass(frozen=True)
class StructuralRoleAssessment:
    """One deterministic structural classification decision."""

    role: str
    confidence: float
    reasons: tuple[str, ...] = ()


def classify_structural_role(
    text: str,
    metadata: Mapping[str, Any] | None = None,
    *,
    element_type: str = "",
) -> StructuralRoleAssessment:
    """Classify text independently from its extraction provider."""
    normalized_text = str(text or "").strip()
    normalized_metadata = metadata or {}
    if not normalized_text:
        return StructuralRoleAssessment("unknown", 1.0, ("empty_text",))

    context_values = [
        str(normalized_metadata.get(field, "") or "").strip()
        for field in ("section", "chapter", "heading_path")
    ]
    context_parts = {
        part.strip()
        for value in context_values
        for part in value.split(">")
        if part.strip()
    }
    lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]

    declared_role = str(normalized_metadata.get("structural_role", "") or "").strip().casefold()
    if declared_role in REJECTED_STRUCTURAL_ROLES:
        return StructuralRoleAssessment(declared_role, 1.0, ("declared_structural_role",))

    if bool(normalized_metadata.get("layout_audit_required")):
        return StructuralRoleAssessment("unknown", 1.0, ("layout_audit_required",))
    if _looks_like_worksheet(normalized_text):
        return StructuralRoleAssessment("worksheet", 1.0, ("worksheet_form_signals",))
    if any(_TOC_HEADING_PATTERN.fullmatch(part) for part in context_parts) or _line_ratio(
        lines,
        _TOC_LINE_PATTERN,
    ) > 0.30:
        return StructuralRoleAssessment("toc", 1.0, ("toc_structure",))
    if any(_BIBLIOGRAPHY_HEADING_PATTERN.fullmatch(part) for part in context_parts):
        return StructuralRoleAssessment("bibliography", 1.0, ("bibliography_heading",))

    explicit_citation_matches = len(_CITATION_PATTERN.findall(normalized_text))
    numbered_reference_matches = len(_NUMBERED_REFERENCE_PATTERN.findall(normalized_text))
    citation_matches = explicit_citation_matches + numbered_reference_matches
    if explicit_citation_matches and citation_matches >= 3 and citation_matches / max(1, len(lines)) > 0.10:
        return StructuralRoleAssessment("bibliography", 0.95, ("citation_density",))
    if any(_INDEX_HEADING_PATTERN.fullmatch(part) for part in context_parts) or _line_ratio(
        lines,
        _INDEX_LINE_PATTERN,
    ) > 0.45:
        return StructuralRoleAssessment("index", 0.95, ("index_structure",))

    cid_count = len(_OCR_CID_PATTERN.findall(normalized_text))
    if cid_count >= 2 and cid_count / max(1, len(normalized_text.split())) > 0.10:
        return StructuralRoleAssessment("unknown", 1.0, ("ocr_placeholder_density",))

    if declared_role in STRUCTURAL_ROLES and declared_role != "unknown":
        return StructuralRoleAssessment(declared_role, 1.0, ("declared_structural_role",))

    normalized_element_type = element_type.strip().casefold()
    if normalized_element_type == "table":
        return StructuralRoleAssessment("table", 0.95, ("table_element",))
    if normalized_element_type == "list_item" or _looks_like_list_only(lines):
        return StructuralRoleAssessment("wine_list", 0.75, ("list_dominant",))
    if _looks_like_prose(lines):
        return StructuralRoleAssessment("prose", 0.90, ("sentence_content",))
    return StructuralRoleAssessment("unknown", 0.50, ("unclassified_structure",))


def _looks_like_worksheet(text: str) -> bool:
    """Return whether text contains multiple independent form/worksheet signals."""
    signals = 0
    signals += min(3, len(_WORKSHEET_LABEL_PATTERN.findall(text)))
    signals += min(3, len(_TASTING_FORM_FIELD_PATTERN.findall(text)))
    signals += int(bool(_FORM_BLANK_PATTERN.search(text)))
    signals += int(bool(_FORM_SCALE_PATTERN.search(text)))
    return signals >= 2


def _line_ratio(lines: list[str], pattern: re.Pattern[str]) -> float:
    """Return the proportion of non-empty lines matching a structural pattern."""
    if not lines:
        return 0.0
    return sum(bool(pattern.match(line)) for line in lines) / len(lines)


def _looks_like_list_only(lines: list[str]) -> bool:
    """Recognize line-dominant lists without treating normal short prose as lists."""
    if len(lines) < 5:
        return False
    short_lines = [line for line in lines if len(line.split()) <= 12]
    sentence_lines = [line for line in lines if _SENTENCE_END_PATTERN.search(line)]
    return len(short_lines) / len(lines) >= 0.70 and len(sentence_lines) / len(lines) < 0.30


def _looks_like_prose(lines: list[str]) -> bool:
    """Recognize readable prose using sentence and average-line evidence."""
    if not lines:
        return False
    sentence_lines = sum(bool(_SENTENCE_END_PATTERN.search(line)) for line in lines)
    average_length = mean(len(line) for line in lines)
    return sentence_lines > 0 or average_length >= 60
