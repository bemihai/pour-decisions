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


def test_intelligent_prompt_has_general_multi_evidence_guidance() -> None:
    """The registered guidance covers whole requests without frozen sample answers."""
    snapshot = ToolSelectionSnapshot(definitions=TOOL_DEFINITIONS, readiness=())

    rendered = render_intelligent_agent_system_prompt(snapshot)
    planning_section = rendered.content.split("**Planning and Completion:**", 1)[1].split(
        "**Mandatory Tool Use:**", 1
    )[0]

    assert "whole request" in planning_section
    assert "each independent kind of evidence" in planning_section
    assert "cellar evidence before concluding" in planning_section
    assert "suitable specialized tool" in planning_section
    assert "equivalent repeat calls with the same arguments" in planning_section
    assert "non-empty final answer" in planning_section
    assert "multi_hop_" not in planning_section
    assert "Nebbiolo" not in planning_section
    assert "Bandol" not in planning_section


def test_intelligent_prompt_routes_explicit_history_to_complete_history_tool() -> None:
    """History wording must select the complete local history path, not top-rated wines."""
    snapshot = ToolSelectionSnapshot(definitions=TOOL_DEFINITIONS, readiness=())

    rendered = render_intelligent_agent_system_prompt(snapshot).content

    assert '**Tasting History** ("Have I tasted this?", "When did I taste it?", "Did I like it?")' in rendered
    assert "Explicit \"have I tasted\"" in rendered
    assert "requests require get_tasting_history" in rendered
    assert "Use get_top_rated_wines only for genuinely highest-rated or rating-threshold requests" in rendered


def test_intelligent_prompt_requires_verified_evidence_and_web_fallback() -> None:
    """Unsupported factual wine claims must route from local evidence to web evidence."""
    snapshot = ToolSelectionSnapshot(definitions=TOOL_DEFINITIONS, readiness=())

    rendered = render_intelligent_agent_system_prompt(snapshot).content

    assert "Ground every factual wine claim" in rendered
    assert "Never use model prior knowledge as evidence" in rendered
    assert "Use relevant local evidence before web search" in rendered
    assert "make one or more focused web searches" in rendered
    assert "Do NOT use when relevant local evidence already supports" in rendered
    assert "directly supports the claim" in rendered
    assert "missing, unavailable, or failed" in rendered
    assert "plausible explanation or speculation" in rendered
    assert "Clearly distinguish those facts from your reasoning" in rendered
    assert "You may provide general wine knowledge" not in rendered
    assert "Based on general wine knowledge" not in rendered


def test_intelligent_prompt_refuses_prior_knowledge_when_web_is_unavailable() -> None:
    """A snapshot without web tools must preserve the no-speculation contract."""
    definitions = tuple(
        definition for definition in TOOL_DEFINITIONS if definition.metadata.category.value != "web_search"
    )
    snapshot = ToolSelectionSnapshot(definitions=definitions, readiness=())

    rendered = render_intelligent_agent_system_prompt(snapshot).content

    assert "Web search is not available" in rendered
    assert "state that it could not be verified" in rendered
    assert "do not answer it from model prior knowledge" in rendered
    assert "make one or more focused web searches" not in rendered


def test_intelligent_prompt_requires_non_technical_grounding_notes() -> None:
    """Substantive answers should disclose useful sources without execution details."""
    snapshot = ToolSelectionSnapshot(definitions=TOOL_DEFINITIONS, readiness=())

    rendered = render_intelligent_agent_system_prompt(snapshot).content

    assert "End every substantive answer with a compact `Sources:` note" in rendered
    assert '"your cellar data" or "your tasting history"' in rendered
    assert '"information you provided"' in rendered
    assert "include useful web links" in rendered
    assert "Never expose tool names, internal source identifiers" in rendered


@pytest.mark.parametrize("definitions", [(), TOOL_DEFINITIONS[:2], TOOL_DEFINITIONS])
def test_intelligent_prompt_keeps_concise_voice_across_tool_selections(definitions: tuple) -> None:
    """Tone guidance must survive empty, partial, and complete capability snapshots."""
    snapshot = ToolSelectionSnapshot(definitions=definitions, readiness=())

    rendered = render_intelligent_agent_system_prompt(snapshot).content

    for rule in (
        "answer or recommendation in the first sentence",
        "warm, professional, direct, and concise",
        "120 words, including the Sources note",
        "no more than three supporting points",
        "only reasoning and evidence that materially help",
        "Avoid performative or exclamatory openings",
        "praise of the question",
        "unnecessary restatement",
        "rhetorical flourishes",
        "without apology preambles",
        "narration of your search process",
        "Do not add generic closing invitations",
        "additional detail for multi-part requests",
        "uncertainty or limitations",
        "comparisons that need meaningful reasoning",
        "explicit request for detail or a different style",
        "Do not force these answers into the simple-answer limit",
        "Brevity must preserve required cellar details",
        "do not expand a short definition with remembered facts",
        "unsupported descriptive claims",
        "certainty and intensity when paraphrasing",
        "not a request for extra tool or model calls",
    ):
        assert rule in rendered
    assert "enthusiasm" not in rendered
    assert "enthusiastic" not in rendered


def test_intelligent_prompt_provenance_tracks_concise_revision() -> None:
    """The edited registered source and rendering retain distinct stable identities."""
    snapshot = ToolSelectionSnapshot(definitions=TOOL_DEFINITIONS[:2], readiness=())
    record = get_prompt_registry().get("intelligent_agent_system")

    rendered = render_intelligent_agent_system_prompt(snapshot)

    assert rendered.label == record.label == "concise-direct-v1"
    assert rendered.source_hash == sha256_text(record.source)
    assert rendered.rendered_hash == sha256_text(rendered.content)
