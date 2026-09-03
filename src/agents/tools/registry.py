"""Typed catalogue models and validation for intelligent-agent tools."""

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from enum import Enum

from chromadb.errors import NotFoundError
from langchain_core.tools import BaseTool
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator

from src.utils import get_default_db_path, initialize_chroma_client, logger


_WEB_SEARCH_PROVIDER_KEY_PATHS = {
    "tavily": "web_search.tavily.api_key_env",
}


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
    """Named dependency capabilities used by readiness checks."""

    CELLAR_SCHEMA = "cellar_schema"
    PAIRING_RULES = "pairing_rules"
    CHROMA_COLLECTION = "chroma_collection"
    WEB_SEARCH_CONFIG = "web_search_config"


_SQLITE_PREREQUISITE_TABLES = {
    ToolPrerequisite.CELLAR_SCHEMA: frozenset(
        {"wines", "bottles", "producers", "regions", "tastings"}
    ),
    ToolPrerequisite.PAIRING_RULES: frozenset({"food_pairing_rules"}),
}

_TOOL_CATEGORY_LABELS = {
    ToolCategory.CELLAR: "Cellar",
    ToolCategory.TASTE_PROFILE: "Taste Profile",
    ToolCategory.PAIRING: "Pairing",
    ToolCategory.RAG: "RAG",
    ToolCategory.WEB_SEARCH: "Web Search",
}


class ToolMetadata(BaseModel):
    """Validated metadata associated with one LangChain tool."""

    model_config = ConfigDict(frozen=True)

    name: str
    category: ToolCategory
    tier: ToolTier
    prerequisites: tuple[ToolPrerequisite, ...] = ()
    cost_class: CostClass
    latency_class: LatencyClass
    idempotent: bool
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
class _PrerequisiteReadiness:
    """Internal readiness evidence for one shared prerequisite."""

    prerequisite: ToolPrerequisite
    available: bool
    reason_code: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class _CachedPrerequisiteReadiness:
    """One prerequisite result with a monotonic expiry timestamp."""

    result: _PrerequisiteReadiness
    expires_at: float


@dataclass(frozen=True)
class ToolSelectionSnapshot:
    """Immutable record of definitions selected for one agent construction."""

    definitions: tuple[ToolDefinition, ...]
    readiness: tuple[ToolReadiness, ...]


class ToolRegistry:
    """Validated, ordered catalogue with cached dependency readiness."""

    def __init__(
        self,
        definitions: tuple[ToolDefinition, ...],
        *,
        config: DictConfig | None = None,
    ) -> None:
        """Validate and retain an immutable ordered catalogue.

        Args:
            definitions: Tool definitions in stable catalogue order.
            config: Application configuration. A missing registry section uses
                the reviewed default readiness-cache TTL.

        Raises:
            ValueError: If catalogue entries or registry configuration are invalid.
        """
        self._definitions = tuple(definitions)
        self._metadata_by_name: dict[str, ToolMetadata] = {}
        self._config = config
        self._health_check_ttl_seconds = self._validate_config(config)
        self._readiness_cache: dict[ToolPrerequisite, _CachedPrerequisiteReadiness] = {}
        self._readiness_cache_lock = threading.Lock()
        self._readiness_refresh_locks = {
            prerequisite: threading.Lock() for prerequisite in ToolPrerequisite
        }

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
    def _validate_config(config: DictConfig | None) -> int:
        """Resolve and validate the readiness-cache TTL."""
        if config is None:
            return 60

        ttl_seconds = OmegaConf.select(
            config,
            "agents.tool_registry.health_check_ttl_seconds",
            default=60,
        )
        if type(ttl_seconds) is not int or ttl_seconds < 1:
            raise ValueError(
                "agents.tool_registry.health_check_ttl_seconds must be an integer of at least 1"
            )
        return ttl_seconds

    @property
    def health_check_ttl_seconds(self) -> int:
        """Return the validated readiness-cache TTL."""
        return self._health_check_ttl_seconds

    def _check_web_search_config(self) -> _PrerequisiteReadiness:
        """Check web-search configuration without constructing a provider client."""
        prerequisite = ToolPrerequisite.WEB_SEARCH_CONFIG

        def missing_configuration(reason: str) -> _PrerequisiteReadiness:
            """Build safe unavailable evidence for expected configuration gaps."""
            return _PrerequisiteReadiness(
                prerequisite=prerequisite,
                available=False,
                reason_code="missing_configuration",
                reason=reason,
            )

        try:
            if self._config is None:
                return missing_configuration("Web search configuration is missing.")

            provider = OmegaConf.select(self._config, "web_search.provider")
            if not isinstance(provider, str) or not provider.strip():
                return missing_configuration("Web search provider is not configured.")

            provider_name = provider.strip().lower()
            key_env_path = _WEB_SEARCH_PROVIDER_KEY_PATHS.get(provider_name)
            if key_env_path is None:
                return missing_configuration("Configured web search provider is not supported.")

            key_env = OmegaConf.select(self._config, key_env_path)
            if not isinstance(key_env, str) or not key_env.strip():
                return missing_configuration("Web search credential configuration is missing.")

            if not os.environ.get(key_env.strip(), "").strip():
                return missing_configuration("Web search credentials are not configured.")

            return _PrerequisiteReadiness(
                prerequisite=prerequisite,
                available=True,
            )
        except Exception:
            logger.exception("Unexpected failure while checking web search configuration")
            return _PrerequisiteReadiness(
                prerequisite=prerequisite,
                available=False,
                reason_code="readiness_check_failed",
                reason="Web search readiness check failed.",
            )

    def _check_sqlite_schema(
        self,
        prerequisite: ToolPrerequisite,
    ) -> _PrerequisiteReadiness:
        """Check one SQLite schema capability without creating or changing the database."""
        required_tables = _SQLITE_PREREQUISITE_TABLES.get(prerequisite)
        if required_tables is None:
            raise ValueError(f"Unsupported SQLite prerequisite: {prerequisite.value}")

        try:
            database_path = get_default_db_path()
            if not database_path.is_file():
                return _PrerequisiteReadiness(
                    prerequisite=prerequisite,
                    available=False,
                    reason_code="database_missing",
                    reason="Cellar database is missing.",
                )

            database_uri = f"{database_path.resolve().as_uri()}?mode=ro"
            with sqlite3.connect(database_uri, uri=True) as connection:
                rows = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            existing_tables = {str(row[0]) for row in rows}
            if not required_tables.issubset(existing_tables):
                return _PrerequisiteReadiness(
                    prerequisite=prerequisite,
                    available=False,
                    reason_code="database_schema_incomplete",
                    reason="Cellar database schema is incomplete.",
                )

            return _PrerequisiteReadiness(
                prerequisite=prerequisite,
                available=True,
            )
        except Exception:
            logger.exception("Unexpected failure while checking cellar database schema")
            return _PrerequisiteReadiness(
                prerequisite=prerequisite,
                available=False,
                reason_code="readiness_check_failed",
                reason="Cellar database readiness check failed.",
            )

    def _check_chroma_collection(self) -> _PrerequisiteReadiness:
        """Check the configured Chroma collection without loading retrieval resources."""
        prerequisite = ToolPrerequisite.CHROMA_COLLECTION

        def missing_configuration(reason: str) -> _PrerequisiteReadiness:
            """Build safe unavailable evidence for incomplete Chroma settings."""
            return _PrerequisiteReadiness(
                prerequisite=prerequisite,
                available=False,
                reason_code="missing_configuration",
                reason=reason,
            )

        try:
            if self._config is None:
                return missing_configuration("Chroma configuration is missing.")

            host = OmegaConf.select(self._config, "chroma.client.host")
            port = OmegaConf.select(self._config, "chroma.client.port")
            collection_name = OmegaConf.select(self._config, "chroma.collections.0.name")
            if not isinstance(host, str) or not host.strip():
                return missing_configuration("Chroma host is not configured.")
            if type(port) is int:
                resolved_port = port
            elif isinstance(port, str) and port.strip().isdigit():
                resolved_port = int(port.strip())
            else:
                return missing_configuration("Chroma port is not configured.")
            if resolved_port < 1:
                return missing_configuration("Chroma port is not configured.")
            if not isinstance(collection_name, str) or not collection_name.strip():
                return missing_configuration("Chroma collection is not configured.")
        except Exception:
            logger.exception("Unexpected failure while resolving Chroma readiness configuration")
            return _PrerequisiteReadiness(
                prerequisite=prerequisite,
                available=False,
                reason_code="readiness_check_failed",
                reason="Chroma readiness check failed.",
            )

        try:
            client = initialize_chroma_client(host.strip(), resolved_port)
        except Exception:
            return _PrerequisiteReadiness(
                prerequisite=prerequisite,
                available=False,
                reason_code="dependency_unreachable",
                reason="Chroma service is unavailable.",
            )

        try:
            client.get_collection(collection_name.strip())
            return _PrerequisiteReadiness(
                prerequisite=prerequisite,
                available=True,
            )
        except NotFoundError:
            return _PrerequisiteReadiness(
                prerequisite=prerequisite,
                available=False,
                reason_code="collection_missing",
                reason="Required Chroma collection is missing.",
            )
        except Exception:
            logger.exception("Unexpected failure while checking Chroma collection")
            return _PrerequisiteReadiness(
                prerequisite=prerequisite,
                available=False,
                reason_code="readiness_check_failed",
                reason="Chroma readiness check failed.",
            )

    def _probe_prerequisite(self, prerequisite: ToolPrerequisite) -> _PrerequisiteReadiness:
        """Run one uncached prerequisite probe."""
        if prerequisite == ToolPrerequisite.WEB_SEARCH_CONFIG:
            return self._check_web_search_config()
        if prerequisite in _SQLITE_PREREQUISITE_TABLES:
            return self._check_sqlite_schema(prerequisite)
        if prerequisite == ToolPrerequisite.CHROMA_COLLECTION:
            return self._check_chroma_collection()
        raise ValueError(f"Unsupported tool prerequisite: {prerequisite.value}")

    def _get_prerequisite_readiness(
        self,
        prerequisite: ToolPrerequisite,
        *,
        force_refresh: bool = False,
    ) -> _PrerequisiteReadiness:
        """Return cached prerequisite evidence or refresh one dependency safely."""
        now = time.monotonic()
        if not force_refresh:
            with self._readiness_cache_lock:
                cached = self._readiness_cache.get(prerequisite)
                if cached is not None and cached.expires_at > now:
                    return cached.result

        refresh_lock = self._readiness_refresh_locks[prerequisite]
        with refresh_lock:
            if not force_refresh:
                now = time.monotonic()
                with self._readiness_cache_lock:
                    cached = self._readiness_cache.get(prerequisite)
                    if cached is not None and cached.expires_at > now:
                        return cached.result

            try:
                result = self._probe_prerequisite(prerequisite)
            except Exception:
                logger.exception("Unexpected failure while probing tool prerequisite")
                result = _PrerequisiteReadiness(
                    prerequisite=prerequisite,
                    available=False,
                    reason_code="readiness_check_failed",
                    reason="Tool prerequisite readiness check failed.",
                )
            cached_result = _CachedPrerequisiteReadiness(
                result=result,
                expires_at=time.monotonic() + self._health_check_ttl_seconds,
            )
            with self._readiness_cache_lock:
                self._readiness_cache[prerequisite] = cached_result
            return result

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

    def _build_readiness(
        self,
        definitions: tuple[ToolDefinition, ...],
        *,
        force_refresh: bool,
    ) -> tuple[ToolReadiness, ...]:
        """Build ordered tool readiness from one result per unique prerequisite."""
        ordered_prerequisites = tuple(
            dict.fromkeys(
                prerequisite
                for definition in definitions
                for prerequisite in definition.metadata.prerequisites
            )
        )
        prerequisite_results = {
            prerequisite: self._get_prerequisite_readiness(
                prerequisite,
                force_refresh=force_refresh,
            )
            for prerequisite in ordered_prerequisites
        }

        readiness: list[ToolReadiness] = []
        for definition in definitions:
            unavailable = next(
                (
                    prerequisite_results[prerequisite]
                    for prerequisite in definition.metadata.prerequisites
                    if not prerequisite_results[prerequisite].available
                ),
                None,
            )
            readiness.append(
                ToolReadiness(
                    name=definition.metadata.name,
                    available=unavailable is None,
                    reason_code=unavailable.reason_code if unavailable is not None else None,
                    reason=unavailable.reason if unavailable is not None else None,
                )
            )
        return tuple(readiness)

    def check_readiness(self, *, force_refresh: bool = False) -> tuple[ToolReadiness, ...]:
        """Return current readiness for the complete catalogue in stable order."""
        return self._build_readiness(
            self._definitions,
            force_refresh=force_refresh,
        )

    def select(self, *, extended: bool) -> ToolSelectionSnapshot:
        """Select a readiness-filtered snapshot for the requested tier.

        Args:
            extended: Include extended definitions when true.

        Returns:
            Immutable selection snapshot.
        """
        selected = tuple(
            definition
            for definition in self._definitions
            if extended or definition.metadata.tier == ToolTier.CORE
        )
        readiness = self._build_readiness(selected, force_refresh=False)
        available_names = {item.name for item in readiness if item.available}
        selected = tuple(
            definition
            for definition in selected
            if definition.metadata.name in available_names
        )
        return ToolSelectionSnapshot(
            definitions=selected,
            readiness=readiness,
        )

    def build_tool_context_section(self, snapshot: ToolSelectionSnapshot) -> str:
        """Render deterministic category-grouped capabilities from one snapshot."""
        heading = "## Available Tool Capabilities"
        if not snapshot.definitions:
            return f"{heading}\n\nNo tools are currently available."

        sections = [heading]
        for category in ToolCategory:
            definitions = tuple(
                definition
                for definition in snapshot.definitions
                if definition.metadata.category == category
            )
            if not definitions:
                continue
            lines = [f"### {_TOOL_CATEGORY_LABELS[category]}"]
            lines.extend(
                f"- `{definition.metadata.name}`: {definition.metadata.capability}"
                for definition in definitions
            )
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def invalidate_readiness_cache(self) -> None:
        """Invalidate all cached prerequisite evidence."""
        with self._readiness_cache_lock:
            self._readiness_cache.clear()
