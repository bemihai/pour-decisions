"""Tests for bounded M13B Phase 5 arm evidence helpers."""

from __future__ import annotations

import pytest

from src.eval.scripts.m13b_preference_arm import MAX_ARM_ATTEMPTS, UsageCallback, _assess, _control_signature


def _output(tool_name: str, content: str) -> dict[str, str]:
    return {"tool_name": tool_name, "content": content}


def test_usage_callback_refuses_attempt_beyond_arm_cap() -> None:
    callback = UsageCallback()
    for _ in range(MAX_ARM_ATTEMPTS):
        callback._reserve()
    with pytest.raises(RuntimeError, match="ceiling reached"):
        callback._reserve()


def test_control_signature_removes_only_candidate_additions() -> None:
    signature = _control_signature(
        [
            _output(
                "get_user_taste_profile",
                "{'total_wines_rated': 1, 'favorite_varietals': [{'varietal': 'Nebbiolo'}], "
                "'declared_preferences': {'items': [], 'total': 0}, 'preference_conflicts': []}",
            )
        ]
    )
    assert signature == {"total_wines_rated": 1, "favorite_varietals": [{"varietal": "Nebbiolo"}]}


def test_hard_avoid_assessment_fails_when_blocked_wine_is_recommended() -> None:
    assessment = _assess(
        "m13b_pref_003",
        "I recommend Fixture Blocked Pinot.",
        [_output("get_wine_recommendations_from_profile", "[{'name': 'Fixture Blocked Pinot'}]")],
    )
    assert assessment["hard_avoid_violations"] == 1
    assert not assessment["visible_effect"]


def test_price_visible_effect_requires_bounded_answer_and_tool_evidence() -> None:
    assessment = _assess(
        "m13b_pref_006",
        "Your stored budget ceiling is 40 EUR.",
        [_output("get_wine_recommendations_from_profile", "[{'name': 'Fixture Neutral White'}]")],
    )
    assert assessment["visible_effect"]
    assert assessment["prohibited_or_fabricated_facts"] == 0
