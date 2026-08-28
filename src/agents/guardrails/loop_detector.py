"""Exact duplicate detection primitives for intelligent-agent tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import UUID

from omegaconf import DictConfig, OmegaConf


_TYPE_MARKER = "__m09a_type__"
LOOP_DETECTED_EVENT_CODE = "exact_tool_call_duplicate"
TOOL_CALL_FINGERPRINT_ERROR_CODE = "tool_call_fingerprint_failed"


@dataclass(frozen=True)
class LoopDetectionConfig:
    """Validated exact loop-detection configuration."""

    enabled: bool = True


@dataclass(frozen=True)
class ToolCallFingerprint:
    """Stable exact identity for one tool call."""

    tool_name: str
    arguments_sha256: str

    def as_history_entry(self) -> dict[str, str]:
        """Return the JSON-serializable state representation."""
        return {
            "tool_name": self.tool_name,
            "arguments_sha256": self.arguments_sha256,
        }


@dataclass(frozen=True)
class LoopDetectionResult:
    """Outcome of checking one complete pending tool-call batch."""

    allowed: bool
    history: tuple[dict[str, str], ...]
    event: dict[str, str] | None = None


def load_loop_detection_config(config: DictConfig | None) -> LoopDetectionConfig:
    """Resolve and validate the exact loop-detection rollout flag."""
    enabled = (
        OmegaConf.select(config, "agents.guardrails.loop_detection.enabled", default=True)
        if config
        else True
    )
    if type(enabled) is not bool:
        raise ValueError("agents.guardrails.loop_detection.enabled must be a boolean")
    return LoopDetectionConfig(enabled=enabled)


def _canonical_sort_key(value: object) -> str:
    """Return a stable JSON string used to order normalized values."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_json_value(value: Any) -> object:
    """Normalize JSON and reviewed non-JSON values without lossy fallback text."""
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        return {_TYPE_MARKER: "float", "value": repr(value)}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value) and _TYPE_MARKER not in value:
            return {key: _normalize_json_value(item) for key, item in value.items()}
        normalized_items = [
            [_normalize_json_value(key), _normalize_json_value(item)]
            for key, item in value.items()
        ]
        normalized_items.sort(key=_canonical_sort_key)
        return {_TYPE_MARKER: "mapping", "items": normalized_items}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return {_TYPE_MARKER: "tuple", "items": [_normalize_json_value(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        normalized_items = [_normalize_json_value(item) for item in value]
        normalized_items.sort(key=_canonical_sort_key)
        return {
            _TYPE_MARKER: "frozenset" if isinstance(value, frozenset) else "set",
            "items": normalized_items,
        }
    if isinstance(value, bytes):
        return {_TYPE_MARKER: "bytes", "hex": value.hex()}
    if isinstance(value, bytearray):
        return {_TYPE_MARKER: "bytearray", "hex": bytes(value).hex()}
    if isinstance(value, Decimal):
        return {_TYPE_MARKER: "decimal", "value": str(value)}
    if isinstance(value, UUID):
        return {_TYPE_MARKER: "uuid", "value": str(value)}
    if isinstance(value, Path):
        return {_TYPE_MARKER: "path", "value": str(value)}
    raise TypeError(f"Unsupported tool argument type: {type(value).__module__}.{type(value).__qualname__}")


def canonicalize_tool_arguments(arguments: Mapping[str, Any]) -> str:
    """Serialize tool arguments to deterministic compact canonical JSON.

    Args:
        arguments: Tool-call arguments to normalize.

    Returns:
        Canonical JSON with sorted keys and preserved Unicode.

    Raises:
        TypeError: If arguments are not a mapping or contain unsupported values.
    """
    if not isinstance(arguments, Mapping):
        raise TypeError("Tool arguments must be a mapping")
    normalized = _normalize_json_value(arguments)
    return _canonical_sort_key(normalized)


def fingerprint_tool_call(tool_name: str, arguments: Mapping[str, Any]) -> ToolCallFingerprint:
    """Build an exact tool-name and canonical-argument fingerprint."""
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("Tool name must be a non-empty string")
    canonical_arguments = canonicalize_tool_arguments(arguments)
    return ToolCallFingerprint(
        tool_name=tool_name,
        arguments_sha256=sha256(canonical_arguments.encode("utf-8")).hexdigest(),
    )


def _history_fingerprints(history: Sequence[Mapping[str, str]]) -> set[ToolCallFingerprint]:
    """Read valid fingerprint entries while tolerating older checkpoint values."""
    fingerprints = set()
    for entry in history:
        tool_name = entry.get("tool_name")
        arguments_sha256 = entry.get("arguments_sha256")
        if isinstance(tool_name, str) and isinstance(arguments_sha256, str):
            fingerprints.add(
                ToolCallFingerprint(
                    tool_name=tool_name,
                    arguments_sha256=arguments_sha256,
                )
            )
    return fingerprints


def detect_duplicate_tool_calls(
    pending_calls: Sequence[Mapping[str, Any]],
    prior_history: Sequence[Mapping[str, str]],
) -> LoopDetectionResult:
    """Check an entire pending batch for exact history or same-batch duplicates.

    Args:
        pending_calls: Model-produced tool calls with ``name`` and ``args`` keys.
        prior_history: Previously accepted fingerprint state entries.

    Returns:
        An allowed result with appended history, or a rejected result preserving
        the original history exactly.

    Raises:
        ValueError: If a pending call lacks a valid tool name.
        TypeError: If pending arguments are not a supported mapping.
    """
    original_history = tuple(dict(entry) for entry in prior_history)
    historical = _history_fingerprints(prior_history)
    pending_fingerprints: list[ToolCallFingerprint] = []
    seen_in_batch: set[ToolCallFingerprint] = set()

    for call in pending_calls:
        tool_name = call.get("name")
        arguments = call.get("args", {})
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("Pending tool call must contain a non-empty name")
        if not isinstance(arguments, Mapping):
            raise TypeError("Pending tool-call arguments must be a mapping")

        fingerprint = fingerprint_tool_call(tool_name, arguments)
        duplicate_scope = None
        if fingerprint in historical:
            duplicate_scope = "history"
        elif fingerprint in seen_in_batch:
            duplicate_scope = "pending_batch"

        if duplicate_scope:
            return LoopDetectionResult(
                allowed=False,
                history=original_history,
                event={
                    "code": LOOP_DETECTED_EVENT_CODE,
                    "tool_name": tool_name,
                    "duplicate_scope": duplicate_scope,
                },
            )

        seen_in_batch.add(fingerprint)
        pending_fingerprints.append(fingerprint)

    updated_history = original_history + tuple(
        fingerprint.as_history_entry() for fingerprint in pending_fingerprints
    )
    return LoopDetectionResult(allowed=True, history=updated_history)
