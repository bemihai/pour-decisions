"""Tests for deterministic sensitive-output sanitization."""

import pytest

from src.agents.guardrails.sanitizer import (
    REDACTION_TOKEN,
    SensitiveOutputSanitizer,
)


def test_redacts_sensitive_environment_identifier() -> None:
    """Provider environment identifiers should not survive finalization."""
    sanitizer = SensitiveOutputSanitizer(environment={})

    result = sanitizer.sanitize(
        "Set the M09A_SYNTHETIC_PROVIDER_TOKEN environment variable and retry."
    )

    assert result.text == f"Set the {REDACTION_TOKEN} environment variable and retry."
    assert result.redaction_count == 1


@pytest.mark.parametrize(
    "assignment",
    (
        "OPENAI_API_KEY=synthetic-secret-value",
        "export OPENAI_API_KEY='synthetic-secret-value'",
        'set OPENAI_API_KEY="synthetic-secret-value"',
    ),
)
def test_redacts_complete_environment_assignments(assignment: str) -> None:
    """Assignments should not retain either the identifier or inline value."""
    sanitizer = SensitiveOutputSanitizer(environment={})

    result = sanitizer.sanitize(f"Run {assignment} before startup.")

    assert result.text == f"Run {REDACTION_TOKEN} before startup."
    assert result.redaction_count == 1
    assert "synthetic-secret-value" not in result.text


def test_redacts_exact_configured_sensitive_value() -> None:
    """Long configured credential values should be matched without naming the variable."""
    sanitizer = SensitiveOutputSanitizer(
        environment={"GOOGLE_API_KEY": "synthetic-google-secret"}
    )

    result = sanitizer.sanitize("The rejected credential was synthetic-google-secret.")

    assert result.text == f"The rejected credential was {REDACTION_TOKEN}."
    assert result.redaction_count == 1


def test_does_not_exact_match_short_configured_value() -> None:
    """Short values should not be redacted through potentially broad exact matching."""
    sanitizer = SensitiveOutputSanitizer(environment={"GOOGLE_API_KEY": "short"})

    result = sanitizer.sanitize("The word short is ordinary prose.")

    assert result.text == "The word short is ordinary prose."
    assert result.redaction_count == 0


@pytest.mark.parametrize(
    ("url", "expected"),
    (
        (
            "https://user:synthetic-password@example.com/wines",
            f"https://{REDACTION_TOKEN}@example.com/wines",
        ),
        (
            "https://example.com/search?api_key=synthetic-key&query=barolo",
            f"https://example.com/search?api_key={REDACTION_TOKEN}&query=barolo",
        ),
        (
            "https://example.com/search?query=barolo&access_token=synthetic-token",
            f"https://example.com/search?query=barolo&access_token={REDACTION_TOKEN}",
        ),
    ),
)
def test_redacts_credential_bearing_urls(url: str, expected: str) -> None:
    """URL userinfo and credential query values should be neutralized."""
    sanitizer = SensitiveOutputSanitizer(environment={})

    result = sanitizer.sanitize(url)

    assert result.text == expected
    assert result.redaction_count == 1


@pytest.mark.parametrize(
    "safe_text",
    (
        "CABERNET_SAUVIGNON is a grape synonym used in this fixture.",
        "BAROLO 2019 is drinking beautifully.",
        "https://example.com/search?region=barolo&vintage=2019",
        "The FOOD_PAIRING section contains normal wine terminology.",
    ),
)
def test_preserves_reviewed_false_positive_controls(safe_text: str) -> None:
    """Normal wine content should not match configuration-focused patterns."""
    sanitizer = SensitiveOutputSanitizer(environment={})

    result = sanitizer.sanitize(safe_text)

    assert result.text == safe_text
    assert result.redaction_count == 0


def test_counts_multiple_independent_redactions() -> None:
    """The result should expose a bounded replacement count for later observability."""
    sanitizer = SensitiveOutputSanitizer(
        environment={"TAVILY_API_KEY": "synthetic-tavily-secret"}
    )

    result = sanitizer.sanitize(
        "TAVILY_API_KEY failed with synthetic-tavily-secret at "
        "https://example.com/?token=another-secret"
    )

    assert result.text.count(REDACTION_TOKEN) == 3
    assert result.redaction_count == 3
