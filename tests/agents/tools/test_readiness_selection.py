"""Tests for readiness aggregation, selection, and capability rendering."""

from unittest.mock import MagicMock

import pytest
from langchain_core.tools import BaseTool, tool

from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import (
    CostClass,
    LatencyClass,
    ToolCategory,
    ToolDefinition,
    ToolMetadata,
    ToolPrerequisite,
    ToolReadiness,
    ToolRegistry,
    ToolSelectionSnapshot,
    ToolTier,
    _PrerequisiteReadiness,
)


@tool
def cellar_core(query: str) -> str:
    """Return a cellar result."""
    return query


@tool
def cellar_extended(query: str) -> str:
    """Return an extended cellar result."""
    return query


@tool
def pairing_core(query: str) -> str:
    """Return a pairing result."""
    return query


@tool
def rag_core(query: str) -> str:
    """Return a RAG result."""
    return query


def _definition(
    langchain_tool: BaseTool,
    *,
    category: ToolCategory,
    tier: ToolTier,
    prerequisites: tuple[ToolPrerequisite, ...] = (),
    capability: str,
) -> ToolDefinition:
    """Build a definition for selection tests."""
    return ToolDefinition(
        tool=langchain_tool,
        metadata=ToolMetadata(
            name=langchain_tool.name,
            category=category,
            tier=tier,
            prerequisites=prerequisites,
            cost_class=CostClass.FREE,
            latency_class=LatencyClass.FAST,
            idempotent=True,
            capability=capability,
        ),
    )


DEFINITIONS = (
    _definition(
        cellar_core,
        category=ToolCategory.CELLAR,
        tier=ToolTier.CORE,
        prerequisites=(ToolPrerequisite.CELLAR_SCHEMA,),
        capability="Query the cellar.",
    ),
    _definition(
        pairing_core,
        category=ToolCategory.PAIRING,
        tier=ToolTier.CORE,
        prerequisites=(ToolPrerequisite.CELLAR_SCHEMA, ToolPrerequisite.PAIRING_RULES),
        capability="Recommend pairings.",
    ),
    _definition(
        rag_core,
        category=ToolCategory.RAG,
        tier=ToolTier.CORE,
        prerequisites=(ToolPrerequisite.CHROMA_COLLECTION,),
        capability="Search wine knowledge.",
    ),
    _definition(
        cellar_extended,
        category=ToolCategory.CELLAR,
        tier=ToolTier.EXTENDED,
        prerequisites=(ToolPrerequisite.WEB_SEARCH_CONFIG,),
        capability="Fetch current cellar context.",
    ),
)


def _evidence(
    prerequisite: ToolPrerequisite,
    *,
    available: bool,
) -> _PrerequisiteReadiness:
    """Build prerequisite evidence for selection tests."""
    return _PrerequisiteReadiness(
        prerequisite=prerequisite,
        available=available,
        reason_code=None if available else "dependency_unreachable",
        reason=None if available else "Dependency unavailable.",
    )


def test_check_readiness_probes_each_unique_prerequisite_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared dependencies should be resolved once while tool order stays stable."""
    registry = ToolRegistry(DEFINITIONS)
    results = {
        prerequisite: _evidence(prerequisite, available=True)
        for prerequisite in ToolPrerequisite
    }
    getter = MagicMock(side_effect=lambda prerequisite, **_kwargs: results[prerequisite])
    monkeypatch.setattr(registry, "_get_prerequisite_readiness", getter)

    readiness = registry.check_readiness()

    assert tuple(item.name for item in readiness) == tuple(
        definition.metadata.name for definition in DEFINITIONS
    )
    assert all(item.available for item in readiness)
    assert [call.args[0] for call in getter.call_args_list] == list(ToolPrerequisite)


def test_full_catalogue_maps_eighteen_tools_to_four_prerequisite_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The active catalogue should share one Chroma result across all five RAG tools."""
    registry = ToolRegistry(TOOL_DEFINITIONS)
    getter = MagicMock(
        side_effect=lambda prerequisite, **_kwargs: _evidence(prerequisite, available=True)
    )
    monkeypatch.setattr(registry, "_get_prerequisite_readiness", getter)

    readiness = registry.check_readiness()

    assert len(readiness) == 18
    assert all(item.available for item in readiness)
    checked = [call.args[0] for call in getter.call_args_list]
    assert checked.count(ToolPrerequisite.CHROMA_COLLECTION) == 1
    assert checked == [
        ToolPrerequisite.CELLAR_SCHEMA,
        ToolPrerequisite.CHROMA_COLLECTION,
        ToolPrerequisite.PAIRING_RULES,
        ToolPrerequisite.WEB_SEARCH_CONFIG,
    ]


def test_force_refresh_is_forwarded_once_per_unique_prerequisite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forced catalogue refresh should not probe shared dependencies per tool."""
    registry = ToolRegistry(DEFINITIONS)
    getter = MagicMock(side_effect=lambda prerequisite, **_kwargs: _evidence(prerequisite, available=True))
    monkeypatch.setattr(registry, "_get_prerequisite_readiness", getter)

    registry.check_readiness(force_refresh=True)

    assert getter.call_count == len(ToolPrerequisite)
    assert all(call.kwargs == {"force_refresh": True} for call in getter.call_args_list)


def test_available_selection_filters_tier_before_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core-only selection should not inspect extended-only prerequisites."""
    registry = ToolRegistry(DEFINITIONS)
    results = {
        ToolPrerequisite.CELLAR_SCHEMA: _evidence(ToolPrerequisite.CELLAR_SCHEMA, available=True),
        ToolPrerequisite.PAIRING_RULES: _evidence(ToolPrerequisite.PAIRING_RULES, available=False),
        ToolPrerequisite.CHROMA_COLLECTION: _evidence(ToolPrerequisite.CHROMA_COLLECTION, available=True),
    }
    getter = MagicMock(side_effect=lambda prerequisite, **_kwargs: results[prerequisite])
    monkeypatch.setattr(registry, "_get_prerequisite_readiness", getter)

    snapshot = registry.select(extended=False)

    assert tuple(item.metadata.name for item in snapshot.definitions) == ("cellar_core", "rag_core")
    assert tuple(item.name for item in snapshot.readiness) == (
        "cellar_core",
        "pairing_core",
        "rag_core",
    )
    assert ToolPrerequisite.WEB_SEARCH_CONFIG not in [call.args[0] for call in getter.call_args_list]


def test_multiple_prerequisites_must_all_be_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first unavailable prerequisite should provide safe tool-level evidence."""
    registry = ToolRegistry((DEFINITIONS[1],))
    results = {
        ToolPrerequisite.CELLAR_SCHEMA: _evidence(ToolPrerequisite.CELLAR_SCHEMA, available=True),
        ToolPrerequisite.PAIRING_RULES: _evidence(ToolPrerequisite.PAIRING_RULES, available=False),
    }
    monkeypatch.setattr(
        registry,
        "_get_prerequisite_readiness",
        lambda prerequisite, **_kwargs: results[prerequisite],
    )

    readiness = registry.check_readiness()
    snapshot = registry.select(extended=True)

    assert readiness == (
        ToolReadiness(
            name="pairing_core",
            available=False,
            reason_code="dependency_unreachable",
            reason="Dependency unavailable.",
        ),
    )
    assert snapshot.definitions == ()


def test_capability_rendering_is_deterministic_and_snapshot_scoped() -> None:
    """Rendering should group only selected definitions in stable category order."""
    registry = ToolRegistry(DEFINITIONS)
    snapshot = ToolSelectionSnapshot(
        definitions=(DEFINITIONS[2], DEFINITIONS[0], DEFINITIONS[3]),
        readiness=(),
    )

    section = registry.build_tool_context_section(snapshot)

    assert section == (
        "## Available Tool Capabilities\n\n"
        "### Cellar\n"
        "- `cellar_core`: Query the cellar.\n"
        "- `cellar_extended`: Fetch current cellar context.\n\n"
        "### RAG\n"
        "- `rag_core`: Search wine knowledge."
    )
    assert "pairing_core" not in section


def test_empty_snapshot_has_non_misleading_capability_text() -> None:
    """A completely unavailable catalogue should not advertise callable tools."""
    registry = ToolRegistry(())
    snapshot = ToolSelectionSnapshot(definitions=(), readiness=())

    assert registry.build_tool_context_section(snapshot) == (
        "## Available Tool Capabilities\n\nNo tools are currently available."
    )
