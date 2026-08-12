"""Authoritative structural role and quality gate for indexable chunks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from statistics import mean
from typing import Any

from .structural_roles import REJECTED_STRUCTURAL_ROLES, classify_structural_role


_VALID_MODES = {"disabled", "audit", "enforce"}
_OCR_CID_PATTERN = re.compile(r"\(cid:\d+\)", re.IGNORECASE)


@dataclass(frozen=True)
class ChunkQualityAssessment:
    """Auditable quality and rejection decision for one chunk."""

    structural_role: str
    quality_score: float
    rejection_reasons: tuple[str, ...]
    below_threshold: bool
    should_reject: bool


class ChunkQualityFilter:
    """Classify structural roles and score ambiguous chunk quality."""

    def __init__(self, *, mode: str = "disabled", min_score: float = 0.4) -> None:
        """Configure disabled, audit, or enforce behavior."""
        normalized_mode = str(mode).strip().casefold()
        if normalized_mode not in _VALID_MODES:
            raise ValueError(f"Unsupported quality-filter mode: {mode!r}")
        if not isinstance(min_score, (int, float)) or isinstance(min_score, bool):
            raise TypeError("quality-filter min_score must be numeric")
        if not 0.0 <= float(min_score) <= 1.0:
            raise ValueError("quality-filter min_score must be between 0.0 and 1.0")
        self.mode = normalized_mode
        self.min_score = float(min_score)

    @classmethod
    def from_config(cls, indexing_config: Mapping[str, Any] | Any | None) -> ChunkQualityFilter:
        """Build a filter from the optional chroma.indexing configuration."""
        quality_config = _config_value(indexing_config, "quality_filter", {})
        return cls(
            mode=str(_config_value(quality_config, "mode", "disabled")),
            min_score=float(_config_value(quality_config, "min_score", 0.4)),
        )

    def score(self, text: str, metadata: Mapping[str, Any] | None = None) -> float:
        """Return the deterministic quality score in the inclusive range [0, 1]."""
        return self.assess(text, metadata).quality_score

    def assess(self, text: str, metadata: Mapping[str, Any] | None = None) -> ChunkQualityAssessment:
        """Return role, quality, reasons, and the mode-independent rejection decision."""
        normalized_text = str(text or "").strip()
        normalized_metadata = metadata or {}
        role_assessment = classify_structural_role(
            normalized_text,
            normalized_metadata,
            element_type=str(normalized_metadata.get("element_type", "") or ""),
        )
        reasons = list(role_assessment.reasons)
        score = 1.0

        if not normalized_text:
            score = 0.0
        else:
            word_count = len(normalized_text.split())
            if word_count < 10:
                score -= 0.60
                reasons.append("very_short_text")
            elif word_count < 40:
                score -= 0.30
                reasons.append("short_text")
            elif word_count < 80:
                score -= 0.10
                reasons.append("moderately_short_text")

            if role_assessment.role in REJECTED_STRUCTURAL_ROLES:
                score -= 0.80
            if "layout_audit_required" in reasons or "ocr_placeholder_density" in reasons:
                score -= 0.80
            lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
            if len(lines) >= 5 and mean(len(line) for line in lines) < 20:
                score -= 0.20
                reasons.append("repeated_short_lines")
            cid_count = len(_OCR_CID_PATTERN.findall(normalized_text))
            if cid_count and cid_count / max(1, word_count) > 0.05:
                score -= 0.40
                reasons.append("ocr_artifact_ratio")

        quality_score = max(0.0, min(1.0, score))
        structural_rejection = role_assessment.role in REJECTED_STRUCTURAL_ROLES or any(
            reason in {"layout_audit_required", "ocr_placeholder_density"} for reason in reasons
        )
        if role_assessment.role == "unknown" and normalized_text and len(normalized_text.split()) < 10:
            score = 0.0
            quality_score = 0.0
            reasons.append("very_short_unknown")
            structural_rejection = True
        below_threshold = quality_score < self.min_score
        return ChunkQualityAssessment(
            structural_role=role_assessment.role,
            quality_score=quality_score,
            rejection_reasons=tuple(dict.fromkeys(reasons)),
            below_threshold=below_threshold,
            should_reject=structural_rejection or below_threshold,
        )


def _config_value(config: Mapping[str, Any] | Any | None, key: str, default: Any) -> Any:
    """Read a value from a mapping, OmegaConf-like object, or absent config."""
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)
