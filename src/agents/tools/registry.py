"""Typed catalogue models and validation for intelligent-agent tools."""

from dataclasses import dataclass
from enum import Enum

from langchain_core.tools import BaseTool
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator


class ToolCategory(str, Enum):
    """Stable functional categories for the active tool catalogue."""

    CELLAR = "cellar"
    TASTE_PROFILE = "taste_profile"
    PAIRING = "pairing"
    RAG = "rag"
    WEB_SEARCH = "web_search"


class ToolTier(str, Enum):
    """Compatibility tiers used by ``get_tools(extended=...)``."""

    CORE = "core"
    EXTENDED = "extended"


class CostClass(str, Enum):
    """Coarse external-cost classification for a tool call."""

    FREE = "free"
    CHEAP = "cheap"
    EXPENSIVE = "expensive"


class LatencyClass(str, Enum):
    """Coarse expected-latency classification for a tool call."""

    FAST = "fast"
    SLOW = "slow"


class ToolPrerequisite(str, Enum):
    """Named dependency capabilities used by later readiness checks."""

    CELLAR_SCHEMA = "cellar_schema"
    PAIRING_RULES = "pairing_rules"
    CHROMA_COLLECTION = "chroma_collection"
    WEB_SEARCH_CONFIG = "web_search_config"


class ToolMetadata(BaseModel):
    """Validated metadata associated with one LangChain tool."""

    model_config = ConfigDict(frozen=True)

    name: str
    category: ToolCategory
    tier: ToolTier
    prerequisites: tuple[ToolPrerequisite, ...] = ()
    cost_class: CostClass = CostClass.FREE
    latency_class: LatencyClass = LatencyClass.FAST
    idempotent: bool = True
    capability: str

    @field_validator("name", "capability")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        """Reject empty catalogue identifiers and capability descriptions."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


@dataclass(frozen=True)
class ToolDefinition:
    """Immutable association between a LangChain tool and its metadata."""

    tool: BaseTool
    metadata: ToolMetadata


class ToolReadiness(BaseModel):
    """Current safe readiness result for one catalogue tool."""

    model_config = ConfigDict(frozen=True)

    name: str
    available: bool
    reason_code: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ToolSelectionSnapshot:
    """Immutable record of definitions selected for one agent construction."""

    definitions: tuple[ToolDefinition, ...]
    readiness: tuple[ToolReadiness, ...]
    registry_enabled: bool


class ToolRegistry:
    """Validated, ordered catalogue of intelligent-agent tools.

    Phase 1 owns catalogue validation and static tier selection only. Dependency
    readiness, caching, and prompt rendering are added by later M6 phases.
    """

    def __init__(
        self,
        definitions: tuple[ToolDefinition, ...],
        *,
        config: DictConfig | None = None,
    ) -> None:
        """Validate and retain an immutable ordered catalogue.

        Args:
            definitions: Tool definitions in stable catalogue order.
            config: Application configuration. A missing registry section keeps
                the migration disabled and uses the reviewed default cache TTL.

        Raises:
            ValueError: If catalogue entries or registry configuration are invalid.
        """
        self._definitions = tuple(definitions)
        self._metadata_by_name: dict[str, ToolMetadata] = {}
        self._config = config
        self._registry_enabled, self._health_check_ttl_seconds = self._validate_config(config)

        for definition in self._definitions:
            tool_name = definition.tool.name
            metadata_name = definition.metadata.name
            if metadata_name != tool_name:
                raise ValueError(
                    f"Tool metadata name {metadata_name!r} does not match tool name {tool_name!r}"
                )
            if tool_name in self._metadata_by_name:
                raise ValueError(f"Duplicate tool name: {tool_name!r}")
            self._metadata_by_name[tool_name] = definition.metadata

    @staticmethod
    def _validate_config(config: DictConfig | None) -> tuple[bool, int]:
        """Resolve and validate the disabled migration settings."""
        if config is None:
            return False, 60

        enabled = OmegaConf.select(config, "agents.tool_registry.enabled", default=False)
        ttl_seconds = OmegaConf.select(
            config,
            "agents.tool_registry.health_check_ttl_seconds",
            default=60,
        )
        if type(enabled) is not bool:
            raise ValueError("agents.tool_registry.enabled must be a boolean")
        if type(ttl_seconds) is not int or ttl_seconds < 1:
            raise ValueError(
                "agents.tool_registry.health_check_ttl_seconds must be an integer of at least 1"
            )
        return enabled, ttl_seconds

    @property
    def registry_enabled(self) -> bool:
        """Return whether registry-backed runtime selection is configured."""
        return self._registry_enabled

    @property
    def health_check_ttl_seconds(self) -> int:
        """Return the validated readiness-cache TTL for later phases."""
        return self._health_check_ttl_seconds

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return all definitions in stable catalogue order."""
        return self._definitions

    def get_metadata(self, tool_name: str) -> ToolMetadata:
        """Return metadata for an exact tool name.

        Args:
            tool_name: Exact LangChain tool name.

        Returns:
            Validated metadata for the requested tool.

        Raises:
            KeyError: If the tool name is not in the catalogue.
        """
        return self._metadata_by_name[tool_name]

    def get_by_category(self, category: ToolCategory) -> tuple[ToolDefinition, ...]:
        """Return one category while preserving catalogue order."""
        return tuple(
            definition
            for definition in self._definitions
            if definition.metadata.category == category
        )

    def check_readiness(self, *, force_refresh: bool = False) -> tuple[ToolReadiness, ...]:
        """Return current dependency readiness once Phase 2 implements probes."""
        del force_refresh
        raise NotImplementedError("Tool readiness is implemented in M6 Phase 2")

    def select(self, *, extended: bool, available_only: bool) -> ToolSelectionSnapshot:
        """Select a static tier snapshot or delegate readiness filtering.

        Args:
            extended: Include extended definitions when true.
            available_only: Filter through readiness when true.

        Returns:
            Immutable selection snapshot.

        Raises:
            NotImplementedError: If readiness filtering is requested before Phase 2.
        """
        selected = tuple(
            definition
            for definition in self._definitions
            if extended or definition.metadata.tier == ToolTier.CORE
        )
        if available_only:
            readiness = self.check_readiness()
            available_names = {item.name for item in readiness if item.available}
            selected = tuple(
                definition
                for definition in selected
                if definition.metadata.name in available_names
            )
            return ToolSelectionSnapshot(
                definitions=selected,
                readiness=readiness,
                registry_enabled=True,
            )
        return ToolSelectionSnapshot(
            definitions=selected,
            readiness=(),
            registry_enabled=False,
        )

    def build_tool_context_section(self, snapshot: ToolSelectionSnapshot) -> str:
        """Build prompt context once Phase 3 implements rendering."""
        del snapshot
        raise NotImplementedError("Tool prompt rendering is implemented in M6 Phase 3")

    def invalidate_readiness_cache(self) -> None:
        """Invalidate cached readiness once Phase 2 implements caching."""
        raise NotImplementedError("Tool readiness caching is implemented in M6 Phase 2")
