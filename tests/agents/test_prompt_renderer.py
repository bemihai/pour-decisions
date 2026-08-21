"""Tests for strict agent prompt rendering."""

from pathlib import Path

import pytest
from jinja2 import UndefinedError

from src.agents.prompt_renderer import render_prompt_template


def test_render_prompt_template_rejects_missing_context(
    tmp_path: Path,
) -> None:
    """An undeclared template variable should fail instead of rendering blank text."""
    (tmp_path / "strict.md.j2").write_text("Available: {{ selected_tools }}")

    with pytest.raises(UndefinedError, match="selected_tools"):
        render_prompt_template("strict.md.j2", prompt_directory=tmp_path)


def test_render_prompt_template_preserves_prompt_characters(
    tmp_path: Path,
) -> None:
    """Prompt rendering should not HTML-escape ordinary Markdown content."""
    (tmp_path / "markdown.md.j2").write_text("Rules: {{ content }}")

    rendered = render_prompt_template(
        "markdown.md.j2",
        {"content": "use <tool> & report its result"},
        prompt_directory=tmp_path,
    )

    assert rendered == "Rules: use <tool> & report its result"
