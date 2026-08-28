"""Deterministic relevance configuration and matching for the wine agent."""

from dataclasses import dataclass
import re
from typing import Literal, Mapping
import unicodedata

from omegaconf import DictConfig, ListConfig, OmegaConf


DEFAULT_WINE_TOPIC_ALLOWLIST = (
    "wine",
    "cellar",
    "grape",
    "vintage",
    "bottle",
    "tasting",
    "pairing",
    "sommelier",
    "appellation",
    "terroir",
    "vineyard",
    "winery",
    "producer",
)
DEFAULT_OFF_TOPIC_PATTERNS = (
    "weather",
    "football",
    "basketball",
    "stock market",
    "cryptocurrency",
    "election",
    "celebrity",
)
RELEVANCE_DEFLECTED_EVENT_CODE = "relevance_off_topic"
RELEVANCE_REDIRECT = (
    "I can help with wine, cellars, tasting, and food pairing. "
    "Please ask me a wine-related question."
)


def normalize_relevance_text(value: str) -> str:
    """Normalize relevance text with compatibility Unicode and case folding."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


@dataclass(frozen=True)
class RelevanceConfig:
    """Validated conservative relevance-guardrail configuration."""

    enabled: bool = True
    wine_topic_allowlist: tuple[str, ...] = DEFAULT_WINE_TOPIC_ALLOWLIST
    off_topic_patterns: tuple[str, ...] = DEFAULT_OFF_TOPIC_PATTERNS


@dataclass(frozen=True)
class RelevanceDecision:
    """Stable bounded outcome of deterministic relevance evaluation."""

    route: Literal["allow", "deflect"]
    reason: Literal["disabled", "wine_topic", "off_topic", "ambiguous"]


def _contains_phrase(normalized_query: str, normalized_phrase: str) -> bool:
    """Match one normalized phrase without allowing word substrings."""
    pattern = rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)"
    return re.search(pattern, normalized_query, flags=re.UNICODE) is not None


def evaluate_relevance(query: str, config: RelevanceConfig) -> RelevanceDecision:
    """Apply conservative allowlist-first wine-scope relevance routing."""
    if not config.enabled:
        return RelevanceDecision(route="allow", reason="disabled")

    normalized_query = normalize_relevance_text(query)
    if any(_contains_phrase(normalized_query, phrase) for phrase in config.wine_topic_allowlist):
        return RelevanceDecision(route="allow", reason="wine_topic")
    if any(_contains_phrase(normalized_query, phrase) for phrase in config.off_topic_patterns):
        return RelevanceDecision(route="deflect", reason="off_topic")
    return RelevanceDecision(route="allow", reason="ambiguous")


def relevance_was_deflected(state: Mapping[str, object]) -> bool:
    """Return whether the latest bounded guardrail event is a relevance redirect."""
    events = state.get("guardrail_events", [])
    if not isinstance(events, list) or not events:
        return False
    latest_event = events[-1]
    return isinstance(latest_event, dict) and latest_event.get("code") == RELEVANCE_DEFLECTED_EVENT_CODE


def _validate_phrases(value: object, path: str) -> tuple[str, ...]:
    """Validate and normalize one reviewed relevance phrase collection."""
    if isinstance(value, ListConfig):
        phrases = list(value)
    elif type(value) in {list, tuple}:
        phrases = list(value)
    else:
        raise ValueError(f"{path} must be a non-empty list of strings")
    if not phrases:
        raise ValueError(f"{path} must be a non-empty list of strings")

    normalized_phrases = []
    for phrase in phrases:
        if not isinstance(phrase, str):
            raise ValueError(f"{path} must contain only non-empty strings")
        normalized = normalize_relevance_text(phrase)
        if not normalized:
            raise ValueError(f"{path} must contain only non-empty strings")
        normalized_phrases.append(normalized)

    if len(normalized_phrases) != len(set(normalized_phrases)):
        raise ValueError(f"{path} must not contain duplicate normalized phrases")
    return tuple(normalized_phrases)


def load_relevance_config(config: DictConfig | None) -> RelevanceConfig:
    """Resolve and validate the conservative relevance configuration."""
    enabled = (
        OmegaConf.select(config, "agents.guardrails.relevance.enabled", default=True)
        if config
        else True
    )
    allowlist = (
        OmegaConf.select(
            config,
            "agents.guardrails.relevance.wine_topic_allowlist",
            default=list(DEFAULT_WINE_TOPIC_ALLOWLIST),
        )
        if config
        else list(DEFAULT_WINE_TOPIC_ALLOWLIST)
    )
    off_topic_patterns = (
        OmegaConf.select(
            config,
            "agents.guardrails.relevance.off_topic_patterns",
            default=list(DEFAULT_OFF_TOPIC_PATTERNS),
        )
        if config
        else list(DEFAULT_OFF_TOPIC_PATTERNS)
    )

    if type(enabled) is not bool:
        raise ValueError("agents.guardrails.relevance.enabled must be a boolean")
    return RelevanceConfig(
        enabled=enabled,
        wine_topic_allowlist=_validate_phrases(
            allowlist,
            "agents.guardrails.relevance.wine_topic_allowlist",
        ),
        off_topic_patterns=_validate_phrases(
            off_topic_patterns,
            "agents.guardrails.relevance.off_topic_patterns",
        ),
    )
