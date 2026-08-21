"""Read-only tool catalogue and readiness introspection endpoint."""

from fastapi import APIRouter, HTTPException, Request

from src.agents.tools.registry import ToolCategory, ToolReadiness, ToolRegistry
from src.api.schemas.tools import ToolStatusResponse, ToolsResponse
from src.utils import logger


router = APIRouter(prefix="/api/tools", tags=["tools"])

_PUBLIC_REASON_CODES = frozenset(
    {
        "missing_configuration",
        "database_missing",
        "database_schema_incomplete",
        "dependency_unreachable",
        "collection_missing",
        "readiness_check_failed",
    }
)
_UNAVAILABLE_REASON_BY_CATEGORY = {
    ToolCategory.CELLAR: "Wine cellar service is unavailable.",
    ToolCategory.TASTE_PROFILE: "Wine cellar service is unavailable.",
    ToolCategory.PAIRING: "Wine pairing service is unavailable.",
    ToolCategory.RAG: "Wine knowledge service is unavailable.",
    ToolCategory.WEB_SEARCH: "Web search service is unavailable.",
}


def _public_reason_code(readiness: ToolReadiness | None) -> str | None:
    """Return a stable public reason code for an unavailable tool."""
    if readiness is None:
        return "readiness_check_failed"
    if readiness.available:
        return None
    if readiness.reason_code in _PUBLIC_REASON_CODES:
        return readiness.reason_code
    return "readiness_check_failed"


def _public_unavailable_reason(
    category: ToolCategory,
    reason_code: str | None,
) -> str | None:
    """Map internal readiness evidence to a safe category-level message."""
    if reason_code is None:
        return None
    if category == ToolCategory.WEB_SEARCH and reason_code == "missing_configuration":
        return "Web search is not configured."
    return _UNAVAILABLE_REASON_BY_CATEGORY[category]


@router.get("", response_model=ToolsResponse)
def get_tools_status(request: Request) -> ToolsResponse:
    """Return the complete catalogue, current readiness, and startup selection."""
    registry: ToolRegistry | None = getattr(request.app.state, "tool_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Tool registry is unavailable.")

    try:
        definitions = registry.definitions()
        readiness_by_name = {
            readiness.name: readiness for readiness in registry.check_readiness()
        }

        cloud_agent = getattr(request.app.state, "cloud_intelligent_agent", None)
        startup_snapshot = getattr(cloud_agent, "tool_selection_snapshot", None)
        selected_names = (
            {
                definition.metadata.name
                for definition in startup_snapshot.definitions
            }
            if startup_snapshot is not None
            else set()
        )

        tool_rows: list[ToolStatusResponse] = []
        for definition in definitions:
            metadata = definition.metadata
            readiness = readiness_by_name.get(metadata.name)
            available = readiness.available if readiness is not None else False
            reason_code = _public_reason_code(readiness)
            tool_rows.append(
                ToolStatusResponse(
                    name=metadata.name,
                    category=metadata.category.value,
                    tier=metadata.tier.value,
                    available=available,
                    selected_for_agent=metadata.name in selected_names,
                    reason_code=reason_code,
                    unavailable_reason=_public_unavailable_reason(
                        metadata.category,
                        reason_code,
                    ),
                    cost_class=metadata.cost_class.value,
                    latency_class=metadata.latency_class.value,
                    idempotent=metadata.idempotent,
                    capability=metadata.capability,
                )
            )

        available_count = sum(row.available for row in tool_rows)
        selected_count = sum(row.selected_for_agent for row in tool_rows)
        return ToolsResponse(
            total=len(tool_rows),
            available=available_count,
            unavailable=len(tool_rows) - available_count,
            selected=selected_count,
            registry_enabled=registry.registry_enabled,
            tools=tool_rows,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to build public tool registry status")
        raise HTTPException(
            status_code=503,
            detail="Tool registry status is unavailable.",
        ) from None
