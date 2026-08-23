"""Deterministic redaction for sensitive intelligent-agent output."""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass


REDACTION_TOKEN = "[internal configuration redacted]"
_MINIMUM_SENSITIVE_VALUE_LENGTH = 8

_SENSITIVE_ENVIRONMENT_NAMES = frozenset(
    {
        "CELLAR_TRACKER_PASSWORD",
        "GOOGLE_API_KEY",
        "LANGFUSE_SECRET_KEY",
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
    }
)
_SENSITIVE_IDENTIFIER_SUFFIXES = frozenset(
    {
        "API_KEY",
        "CREDENTIAL",
        "CREDENTIALS",
        "ENDPOINT",
        "HOST",
        "PASSWORD",
        "PATH",
        "PORT",
        "SECRET",
        "TOKEN",
        "USERNAME",
    }
)

_ENVIRONMENT_IDENTIFIER_PATTERN = re.compile(
    r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"
)
_ENVIRONMENT_ASSIGNMENT_PATTERN = re.compile(
    r"(?<![A-Z0-9_])(?:export\s+|set\s+)?"
    r"(?P<name>[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\s*=\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_URL_USERINFO_PATTERN = re.compile(
    r"(?P<scheme>https?://)[^\s/@:]+:[^\s/@]+@",
    flags=re.IGNORECASE,
)
_URL_CREDENTIAL_QUERY_PATTERN = re.compile(
    r"(?P<prefix>[?&](?:api[_-]?key|access[_-]?token|token|password|secret|signature)=)"
    r"[^&#\s]+",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SanitizationResult:
    """Sanitized text and the number of replacements applied."""

    text: str
    redaction_count: int


class SensitiveOutputSanitizer:
    """Redact configuration identifiers and values from final agent answers."""

    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        """Capture explicitly sensitive configured values for exact matching.

        Args:
            environment: Environment mapping used to resolve sensitive values.
                Defaults to the process environment. Tests should pass an explicit
                mapping rather than mutate or inspect real credentials.
        """
        source = os.environ if environment is None else environment
        self._sensitive_values = tuple(
            sorted(
                {
                    value
                    for name in _SENSITIVE_ENVIRONMENT_NAMES
                    if len(value := source.get(name, "")) >= _MINIMUM_SENSITIVE_VALUE_LENGTH
                    and value != REDACTION_TOKEN
                },
                key=len,
                reverse=True,
            )
        )

    def sanitize(self, text: str) -> SanitizationResult:
        """Return text with reviewed sensitive forms replaced by one neutral token."""
        sanitized = text
        redaction_count = 0

        sanitized, count = _replace_sensitive_assignments(sanitized)
        redaction_count += count

        sanitized, count = _URL_USERINFO_PATTERN.subn(
            lambda match: f"{match.group('scheme')}{REDACTION_TOKEN}@",
            sanitized,
        )
        redaction_count += count

        sanitized, count = _URL_CREDENTIAL_QUERY_PATTERN.subn(
            lambda match: f"{match.group('prefix')}{REDACTION_TOKEN}",
            sanitized,
        )
        redaction_count += count

        for value in self._sensitive_values:
            occurrences = sanitized.count(value)
            if occurrences:
                sanitized = sanitized.replace(value, REDACTION_TOKEN)
                redaction_count += occurrences

        sanitized, count = _replace_sensitive_identifiers(sanitized)
        redaction_count += count

        return SanitizationResult(text=sanitized, redaction_count=redaction_count)


def _replace_sensitive_assignments(text: str) -> tuple[str, int]:
    """Redact complete assignments when their identifier is configuration-sensitive."""
    replacement_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacement_count
        if not _is_sensitive_identifier(match.group("name")):
            return match.group(0)
        replacement_count += 1
        return REDACTION_TOKEN

    return _ENVIRONMENT_ASSIGNMENT_PATTERN.sub(replace, text), replacement_count


def _replace_sensitive_identifiers(text: str) -> tuple[str, int]:
    """Redact reviewed environment-style identifiers without matching wine terms."""
    replacement_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacement_count
        identifier = match.group(0)
        if not _is_sensitive_identifier(identifier):
            return identifier
        replacement_count += 1
        return REDACTION_TOKEN

    return _ENVIRONMENT_IDENTIFIER_PATTERN.sub(replace, text), replacement_count


def _is_sensitive_identifier(identifier: str) -> bool:
    """Return whether an uppercase underscore identifier resembles configuration."""
    if identifier in _SENSITIVE_ENVIRONMENT_NAMES:
        return True
    return any(
        identifier == suffix or identifier.endswith(f"_{suffix}")
        for suffix in _SENSITIVE_IDENTIFIER_SUFFIXES
    )
