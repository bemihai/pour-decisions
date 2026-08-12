"""Validated contextual search text shared by dense and sparse indexing."""

from __future__ import annotations

import re
from typing import Any, Mapping

from src.chroma.structural_roles import REJECTED_STRUCTURAL_ROLES, classify_structural_role


_OCR_PLACEHOLDER_PATTERN = re.compile(r"\(cid:\d+\)|[\ufffd]", re.IGNORECASE)
_PAGE_OR_SCORE_PATTERN = re.compile(
    r"^(?:page\s+)?\d{1,4}$|^(?:now|\d{4})\s+to\s+(?:\d{4}|now)\s+\d{1,2}(?:\.\d)?$",
    re.IGNORECASE,
)
_CONTEXT_FIELDS = ("document_title", "chapter", "entry_title", "section")


def build_contextual_search_text(body: str, metadata: Mapping[str, Any] | None = None) -> str:
    """Prefix clean body text with safe, de-duplicated structural lineage."""
    clean_body = str(body or "").strip()
    context = validated_context_parts(metadata or {})
    if not context:
        return clean_body
    return f"{' > '.join(context)}\n\n{clean_body}"


def validated_context_parts(metadata: Mapping[str, Any]) -> list[str]:
    """Return context values that are plausible headings rather than corpus noise."""
    if str(metadata.get("structural_role", "")).casefold() in REJECTED_STRUCTURAL_ROLES:
        return []
    if bool(metadata.get("layout_audit_required")):
        return []

    context: list[str] = []
    normalized_seen: set[str] = set()
    for field in _CONTEXT_FIELDS:
        value = str(metadata.get(field, "") or "").strip()
        normalized = value.casefold()
        if normalized in normalized_seen or not _is_valid_context_value(value):
            continue
        context.append(value)
        normalized_seen.add(normalized)
    return context


def _is_valid_context_value(value: str) -> bool:
    """Reject headings that are uncertain, structural, or visibly corrupted."""
    if not value or len(value) > 200 or len(value.split()) > 24:
        return False
    if _OCR_PLACEHOLDER_PATTERN.search(value) or _PAGE_OR_SCORE_PATTERN.fullmatch(value):
        return False
    alphanumeric_count = sum(character.isalnum() for character in value)
    if alphanumeric_count / max(1, len(value)) < 0.60:
        return False
    assessment = classify_structural_role(value, {"section": value})
    return assessment.role not in REJECTED_STRUCTURAL_ROLES
