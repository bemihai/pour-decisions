"""Pydantic response schemas for tool-registry introspection."""

from pydantic import BaseModel


class ToolStatusResponse(BaseModel):
    """Public catalogue, readiness, and agent-selection state for one tool."""

    name: str
    category: str
    tier: str
    available: bool
    selected_for_agent: bool
    reason_code: str | None
    unavailable_reason: str | None
    cost_class: str
    latency_class: str
    idempotent: bool
    capability: str


class ToolsResponse(BaseModel):
    """Ordered public status response for the complete active tool catalogue."""

    total: int
    available: int
    unavailable: int
    selected: int
    tools: list[ToolStatusResponse]
