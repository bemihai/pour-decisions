"""Tests for deterministic intelligent-agent relevance routing."""

import pytest
from omegaconf import DictConfig, OmegaConf

from src.agents.guardrails.relevance import (
    DEFAULT_OFF_TOPIC_PATTERNS,
    DEFAULT_WINE_TOPIC_ALLOWLIST,
    RelevanceConfig,
    RelevanceDecision,
    evaluate_relevance,
    load_relevance_config,
)


def _config(
    *,
    enabled: object = True,
    allowlist: object = DEFAULT_WINE_TOPIC_ALLOWLIST,
    off_topic_patterns: object = DEFAULT_OFF_TOPIC_PATTERNS,
) -> DictConfig:
    """Build a focused relevance configuration."""
    return OmegaConf.create(
        {
            "agents": {
                "guardrails": {
                    "relevance": {
                        "enabled": enabled,
                        "wine_topic_allowlist": allowlist,
                        "off_topic_patterns": off_topic_patterns,
                    }
                }
            }
        }
    )


def test_relevance_defaults_are_valid() -> None:
    """Missing configuration should resolve to the reviewed phrase defaults."""
    assert load_relevance_config(OmegaConf.create({})) == RelevanceConfig()


@pytest.mark.parametrize("enabled", [0, 1, "true"])
def test_relevance_enabled_requires_boolean(enabled: object) -> None:
    """The relevance rollout flag should reject truthy non-booleans."""
    with pytest.raises(ValueError, match="relevance.enabled must be a boolean"):
        load_relevance_config(_config(enabled=enabled))


@pytest.mark.parametrize("phrases", [None, "wine", [], ["wine", " "], ["wine", 3]])
def test_allowlist_requires_non_empty_string_collection(phrases: object) -> None:
    """Wine-topic phrases should be a usable reviewed string collection."""
    with pytest.raises(ValueError, match="wine_topic_allowlist"):
        load_relevance_config(_config(allowlist=phrases))


@pytest.mark.parametrize("phrases", [None, "weather", [], ["weather", "\t"], [False]])
def test_off_topic_patterns_require_non_empty_string_collection(phrases: object) -> None:
    """Off-topic patterns should be a usable reviewed string collection."""
    with pytest.raises(ValueError, match="off_topic_patterns"):
        load_relevance_config(_config(off_topic_patterns=phrases))


def test_phrases_are_unicode_normalized_and_deduplicated() -> None:
    """Equivalent Unicode/case phrases should be rejected as duplicates."""
    with pytest.raises(ValueError, match="duplicate normalized phrases"):
        load_relevance_config(_config(allowlist=["WINE", "ｗｉｎｅ"]))


def test_relevance_can_be_disabled_explicitly() -> None:
    """A false rollout flag should remain false after validation."""
    assert load_relevance_config(_config(enabled=False)).enabled is False


def test_wine_allowlist_wins_over_off_topic_phrase() -> None:
    """A wine-domain phrase should keep a mixed query in scope."""
    decision = evaluate_relevance(
        "How does WEATHER affect a vineyard during harvest?",
        RelevanceConfig(),
    )

    assert decision == RelevanceDecision(route="allow", reason="wine_topic")


@pytest.mark.parametrize(
    "query",
    [
        "What is the weather tomorrow?",
        "Who won the FOOTBALL match?",
        "How is the stock   market performing?",
        "Explain the ｅｌｅｃｔｉｏｎ result.",
    ],
)
def test_clear_off_topic_queries_deflect_after_normalization(query: str) -> None:
    """Case, spacing, and compatibility Unicode should normalize deterministically."""
    assert evaluate_relevance(query, RelevanceConfig()) == RelevanceDecision(
        route="deflect",
        reason="off_topic",
    )


@pytest.mark.parametrize(
    "query",
    [
        "What should I open tonight?",
        "Is the 2019 better than the 2020?",
        "Would this work with roast chicken?",
    ],
)
def test_unknown_and_ambiguous_queries_are_allowed(query: str) -> None:
    """Queries without reviewed evidence should continue to the agent."""
    assert evaluate_relevance(query, RelevanceConfig()) == RelevanceDecision(
        route="allow",
        reason="ambiguous",
    )


@pytest.mark.parametrize(
    "query",
    [
        "The leather weathered naturally.",
        "This footballer wrote a memoir.",
        "A celebrityhood memoir.",
        "The stock marketplace opens early.",
    ],
)
def test_phrase_substrings_do_not_match(query: str) -> None:
    """Reviewed phrases should require Unicode-aware word boundaries."""
    assert evaluate_relevance(query, RelevanceConfig()).route == "allow"


def test_disabled_relevance_always_allows() -> None:
    """The behavioral rollout flag should bypass deterministic deflection."""
    decision = evaluate_relevance(
        "What is the weather tomorrow?",
        RelevanceConfig(enabled=False),
    )

    assert decision == RelevanceDecision(route="allow", reason="disabled")
