"""Tests for strict agent prompt rendering."""

from pathlib import Path

import pytest
from jinja2 import UndefinedError

from src.agents.prompt_registry import get_prompt_registry, sha256_text
from src.agents.prompt_renderer import (
    render_intelligent_agent_system_prompt,
    render_prompt_source,
    render_prompt_template,
)
from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import ToolSelectionSnapshot


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


def test_registered_prompt_source_keeps_strict_jinja_validation() -> None:
    """Registry-backed source rendering should still reject missing variables."""
    with pytest.raises(UndefinedError, match="selected_tools"):
        render_prompt_source("Available: {{ selected_tools }}")


def test_intelligent_prompt_matches_existing_file_rendering() -> None:
    """Registry integration must not change the effective intelligent prompt."""
    snapshot = ToolSelectionSnapshot(definitions=TOOL_DEFINITIONS[:3], readiness=())
    context = {
        "selected_tool_names": frozenset(
            definition.metadata.name for definition in snapshot.definitions
        ),
        "selected_categories": frozenset(
            definition.metadata.category.value for definition in snapshot.definitions
        ),
    }
    expected = render_prompt_template("intelligent_agent_system_prompt.md.j2", context)

    rendered = render_intelligent_agent_system_prompt(snapshot)

    assert rendered.content == expected
    assert rendered.source_hash == get_prompt_registry().get(
        "intelligent_agent_system"
    ).source_hash
    assert rendered.rendered_hash == sha256_text(expected)


def test_intelligent_render_hash_is_stable_for_same_snapshot() -> None:
    """Equivalent construction snapshots should produce the same render identity."""
    snapshot = ToolSelectionSnapshot(definitions=TOOL_DEFINITIONS[:2], readiness=())

    first = render_intelligent_agent_system_prompt(snapshot)
    second = render_intelligent_agent_system_prompt(snapshot)

    assert first == second


def test_tool_selection_changes_rendered_hash_but_not_source_hash() -> None:
    """Snapshot-dependent prompt content should remain distinct from source identity."""
    empty = ToolSelectionSnapshot(definitions=(), readiness=())
    selected = ToolSelectionSnapshot(definitions=TOOL_DEFINITIONS[:1], readiness=())

    empty_render = render_intelligent_agent_system_prompt(empty)
    selected_render = render_intelligent_agent_system_prompt(selected)

    assert empty_render.content != selected_render.content
    assert empty_render.rendered_hash != selected_render.rendered_hash
    assert empty_render.source_hash == selected_render.source_hash
