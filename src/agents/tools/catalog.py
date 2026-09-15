"""Explicit composition of the built-in intelligent-agent tool catalogue."""

from typing import TYPE_CHECKING

from omegaconf import DictConfig

from src.agents.tools.registry import ToolDefinition, ToolRegistry, ToolTier
from src.agents.tools.cellar_tools import TOOL_DEFINITIONS as CELLAR_TOOL_DEFINITIONS
from src.agents.tools.pairing_tools import TOOL_DEFINITIONS as PAIRING_TOOL_DEFINITIONS
from src.agents.tools.rag_tools import (
    TOOL_DEFINITIONS as RAG_TOOL_DEFINITIONS,
    build_async_rag_tool_definitions,
)
from src.agents.tools.taste_profile_tools import (
    TOOL_DEFINITIONS as TASTE_PROFILE_TOOL_DEFINITIONS,
)
from src.agents.tools.web_search_tools import TOOL_DEFINITIONS as WEB_SEARCH_TOOL_DEFINITIONS

if TYPE_CHECKING:
    from src.retrieval import AsyncRAGRuntimeResources


_CORE_MODULE_CATALOGUES = (
    CELLAR_TOOL_DEFINITIONS,
    TASTE_PROFILE_TOOL_DEFINITIONS,
    RAG_TOOL_DEFINITIONS,
    PAIRING_TOOL_DEFINITIONS,
    WEB_SEARCH_TOOL_DEFINITIONS,
)

_EXTENDED_MODULE_CATALOGUES = (
    CELLAR_TOOL_DEFINITIONS,
    TASTE_PROFILE_TOOL_DEFINITIONS,
    PAIRING_TOOL_DEFINITIONS,
    RAG_TOOL_DEFINITIONS,
    WEB_SEARCH_TOOL_DEFINITIONS,
)

CORE_DEFINITIONS: tuple[ToolDefinition, ...] = tuple(
    definition
    for module_catalogue in _CORE_MODULE_CATALOGUES
    for definition in module_catalogue
    if definition.metadata.tier == ToolTier.CORE
)

EXTENDED_DEFINITIONS: tuple[ToolDefinition, ...] = tuple(
    definition
    for module_catalogue in _EXTENDED_MODULE_CATALOGUES
    for definition in module_catalogue
    if definition.metadata.tier == ToolTier.EXTENDED
)

TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = CORE_DEFINITIONS + EXTENDED_DEFINITIONS


def build_tool_registry(
    config: DictConfig,
    *,
    async_rag_resources: "AsyncRAGRuntimeResources | None" = None,
) -> ToolRegistry:
    """Build a fresh validated registry from the authoritative catalogue.

    Args:
        config: Application configuration containing registry readiness-cache
            settings.
        async_rag_resources: Optional API-lifespan resources used to replace
            only the five RAG definitions with coroutine-backed variants.

    Returns:
        A newly constructed registry containing all active definitions.
    """
    definitions = TOOL_DEFINITIONS
    if async_rag_resources is not None:
        async_rag_by_name = {
            definition.metadata.name: definition
            for definition in build_async_rag_tool_definitions(async_rag_resources)
        }
        definitions = tuple(
            async_rag_by_name.get(definition.metadata.name, definition)
            for definition in TOOL_DEFINITIONS
        )
    return ToolRegistry(definitions, config=config)
