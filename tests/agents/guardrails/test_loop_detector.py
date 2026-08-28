"""Tests for exact intelligent-agent tool-call loop detection."""

from decimal import Decimal

import pytest
from omegaconf import OmegaConf

from src.agents.guardrails.loop_detector import (
    LOOP_DETECTED_EVENT_CODE,
    LoopDetectionConfig,
    canonicalize_tool_arguments,
    detect_duplicate_tool_calls,
    fingerprint_tool_call,
    load_loop_detection_config,
)


def test_argument_key_order_does_not_change_fingerprint() -> None:
    """Canonical object key ordering should make equivalent mappings identical."""
    first = fingerprint_tool_call("search_wine", {"region": "Piemonte", "vintage": 2019})
    second = fingerprint_tool_call("search_wine", {"vintage": 2019, "region": "Piemonte"})

    assert first == second


def test_tool_name_and_argument_values_are_part_of_identity() -> None:
    """Changing the tool or argument value should change the exact identity."""
    baseline = fingerprint_tool_call("search_wine", {"vintage": 2019})

    assert fingerprint_tool_call("search_reviews", {"vintage": 2019}) != baseline
    assert fingerprint_tool_call("search_wine", {"vintage": 2020}) != baseline


def test_unicode_is_preserved_and_deterministic() -> None:
    """Canonical JSON should preserve Unicode while remaining stable."""
    arguments = {"producer": "Domaine de la Romanée-Conti", "region": "Bourgogne"}

    canonical = canonicalize_tool_arguments(arguments)

    assert "Romanée-Conti" in canonical
    assert canonicalize_tool_arguments(arguments) == canonical


def test_supported_non_json_values_have_stable_type_markers() -> None:
    """Reviewed non-JSON values should normalize without fallback repr strings."""
    arguments = {
        "bytes": b"wine",
        "decimal": Decimal("1.20"),
        "set": {"Barolo", "Barbaresco"},
        "tuple": (2019, 2020),
    }

    assert canonicalize_tool_arguments(arguments) == canonicalize_tool_arguments(arguments)
    assert fingerprint_tool_call("compare", arguments) == fingerprint_tool_call("compare", arguments)


def test_type_marker_like_json_cannot_collide_with_non_json_value() -> None:
    """User JSON resembling an internal marker should retain a distinct identity."""
    native_json = {"value": {"__m09a_type__": "bytes", "hex": "77696e65"}}
    non_json = {"value": b"wine"}

    assert fingerprint_tool_call("inspect", native_json) != fingerprint_tool_call("inspect", non_json)


def test_unsupported_values_fail_closed() -> None:
    """Unknown objects should not use unstable or disclosure-prone representations."""
    with pytest.raises(TypeError, match="Unsupported tool argument type"):
        canonicalize_tool_arguments({"value": object()})


def test_prior_history_duplicate_is_rejected_without_history_change() -> None:
    """An exact call already accepted in an earlier batch should be rejected."""
    existing = fingerprint_tool_call("search_wine", {"region": "Barolo"}).as_history_entry()

    result = detect_duplicate_tool_calls(
        [{"name": "search_wine", "args": {"region": "Barolo"}}],
        [existing],
    )

    assert result.allowed is False
    assert result.history == (existing,)
    assert result.event == {
        "code": LOOP_DETECTED_EVENT_CODE,
        "tool_name": "search_wine",
        "duplicate_scope": "history",
    }


def test_same_batch_duplicate_rejects_complete_batch_atomically() -> None:
    """A same-batch duplicate should add none of that batch to history."""
    existing = fingerprint_tool_call("cellar_stats", {}).as_history_entry()
    repeated = {"name": "search_wine", "args": {"region": "Barolo"}}

    result = detect_duplicate_tool_calls([repeated, repeated], [existing])

    assert result.allowed is False
    assert result.history == (existing,)
    assert result.event is not None
    assert result.event["duplicate_scope"] == "pending_batch"


def test_different_arguments_to_same_tool_are_allowed_and_recorded() -> None:
    """Exact detection should allow the same tool with distinct arguments."""
    result = detect_duplicate_tool_calls(
        [
            {"name": "search_wine", "args": {"region": "Barolo"}},
            {"name": "search_wine", "args": {"region": "Barbaresco"}},
        ],
        [],
    )

    assert result.allowed is True
    assert result.event is None
    assert len(result.history) == 2
    assert result.history[0] != result.history[1]


def test_empty_pending_batch_preserves_history() -> None:
    """No pending calls should be a safe no-op."""
    existing = fingerprint_tool_call("cellar_stats", {}).as_history_entry()

    result = detect_duplicate_tool_calls([], [existing])

    assert result.allowed is True
    assert result.history == (existing,)


@pytest.mark.parametrize("enabled", [0, 1, "true"])
def test_loop_detection_flag_requires_boolean(enabled: object) -> None:
    """The rollout flag should reject truthy non-boolean values."""
    config = OmegaConf.create(
        {"agents": {"guardrails": {"loop_detection": {"enabled": enabled}}}}
    )

    with pytest.raises(ValueError, match="loop_detection.enabled must be a boolean"):
        load_loop_detection_config(config)


def test_loop_detection_can_be_disabled_explicitly() -> None:
    """A false rollout flag should remain false after validation."""
    config = OmegaConf.create(
        {"agents": {"guardrails": {"loop_detection": {"enabled": False}}}}
    )

    assert load_loop_detection_config(config) == LoopDetectionConfig(enabled=False)
