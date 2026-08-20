"""Tests for the M6 typed tool registry core."""

from dataclasses import FrozenInstanceError

import pytest
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

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
)


@tool
def first_tool(query: str) -> str:
    """Return the first test result."""
    return query


@tool
def second_tool(query: str) -> str:
    """Return the second test result."""
    return query


def _definition(
    langchain_tool: BaseTool = first_tool,
    *,
    name: str | None = None,
    category: ToolCategory = ToolCategory.CELLAR,
    tier: ToolTier = ToolTier.CORE,
) -> ToolDefinition:
    """Build a minimal valid tool definition."""
    return ToolDefinition(
        tool=langchain_tool,
        metadata=ToolMetadata(
            name=name or langchain_tool.name,
            category=category,
            tier=tier,
            capability=f"Use {langchain_tool.name} in tests.",
        ),
    )


def test_enum_values_match_the_reviewed_contract() -> None:
    """Registry enums should expose only the reviewed stable values."""
    assert {value.value for value in ToolCategory} == {
        "cellar",
        "taste_profile",
        "pairing",
        "rag",
        "web_search",
    }
    assert {value.value for value in ToolTier} == {"core", "extended"}
    assert {value.value for value in CostClass} == {"free", "cheap", "expensive"}
    assert {value.value for value in LatencyClass} == {"fast", "slow"}
    assert {value.value for value in ToolPrerequisite} == {
        "cellar_schema",
        "pairing_rules",
        "chroma_collection",
        "web_search_config",
    }


def test_metadata_defaults_are_explicit_and_immutable() -> None:
    """Metadata should use the reviewed low-cost defaults and reject mutation."""
    metadata = _definition().metadata

    assert metadata.prerequisites == ()
    assert metadata.cost_class == CostClass.FREE
    assert metadata.latency_class == LatencyClass.FAST
    assert metadata.idempotent is True
    with pytest.raises(ValidationError):
        metadata.capability = "changed"


@pytest.mark.parametrize("field", ["name", "capability"])
def test_metadata_rejects_blank_text(field: str) -> None:
    """Names and concise capability descriptions must contain useful text."""
    values = {
        "name": first_tool.name,
        "category": ToolCategory.CELLAR,
        "tier": ToolTier.CORE,
        "capability": "Use the first tool.",
    }
    values[field] = "   "

    with pytest.raises(ValidationError, match="must not be blank"):
        ToolMetadata(**values)


def test_metadata_rejects_unknown_enum_values() -> None:
    """Unknown category, tier, and prerequisite values should fail validation."""
    with pytest.raises(ValidationError):
        ToolMetadata(
            name=first_tool.name,
            category="unknown",
            tier="core",
            prerequisites=("network",),
            capability="Use the first tool.",
        )


def test_definition_and_snapshot_are_frozen() -> None:
    """Definitions and agent selection snapshots must be immutable containers."""
    definition = _definition()
    snapshot = ToolSelectionSnapshot(
        definitions=(definition,),
        readiness=(ToolReadiness(name=first_tool.name, available=True),),
        registry_enabled=True,
    )

    with pytest.raises(FrozenInstanceError):
        setattr(definition, "tool", second_tool)
    with pytest.raises(FrozenInstanceError):
        setattr(snapshot, "registry_enabled", False)


def test_registry_preserves_definition_and_category_order() -> None:
    """Lookups should preserve the explicit input order."""
    definitions = (
        _definition(first_tool),
        _definition(second_tool, tier=ToolTier.EXTENDED),
    )
    registry = ToolRegistry(definitions)

    assert registry.definitions() == definitions
    assert registry.get_by_category(ToolCategory.CELLAR) == definitions
    assert registry.get_metadata(second_tool.name) is definitions[1].metadata


def test_registry_rejects_duplicate_tool_names() -> None:
    """Duplicate catalogue names should fail construction clearly."""
    with pytest.raises(ValueError, match="Duplicate tool name"):
        ToolRegistry((_definition(), _definition()))


def test_registry_rejects_metadata_tool_name_mismatch() -> None:
    """Metadata names must match the decorated LangChain tool exactly."""
    with pytest.raises(ValueError, match="does not match tool name"):
        ToolRegistry((_definition(name="different_name"),))


def test_empty_registry_is_deterministic() -> None:
    """An empty catalogue should support stable static lookup and selection."""
    registry = ToolRegistry(())

    assert registry.definitions() == ()
    assert registry.get_by_category(ToolCategory.RAG) == ()
    assert registry.select(extended=True, available_only=False) == ToolSelectionSnapshot(
        definitions=(),
        readiness=(),
        registry_enabled=False,
    )


def test_static_tier_selection_preserves_order() -> None:
    """Disabled-mode selection should implement the existing core/all contract."""
    definitions = (
        _definition(first_tool),
        _definition(second_tool, tier=ToolTier.EXTENDED),
    )
    registry = ToolRegistry(definitions)

    assert registry.select(extended=False, available_only=False).definitions == (definitions[0],)
    assert registry.select(extended=True, available_only=False).definitions == definitions


def test_readiness_methods_are_explicitly_deferred() -> None:
    """Phase 1 must not hide readiness behavior before probes are implemented."""
    registry = ToolRegistry((_definition(),))

    with pytest.raises(NotImplementedError, match="Phase 2"):
        registry.check_readiness()
    with pytest.raises(NotImplementedError, match="Phase 2"):
        registry.select(extended=True, available_only=True)
