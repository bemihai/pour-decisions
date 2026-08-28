"""M9A Gate 0 reproduction for the known environment-name disclosure path."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.agents.guardrails import REDACTION_TOKEN, SafeToolErrorCode
from src.agents.intelligent.agent import WineAgent
from src.agents.tools.registry import ToolRegistry, ToolSelectionSnapshot
from src.agents.tools.web_search_tools import TOOL_DEFINITIONS, search_web_for_wine


SYNTHETIC_ENVIRONMENT_IDENTIFIER = "M09A_SYNTHETIC_PROVIDER_TOKEN"


def _tool_snapshot() -> ToolSelectionSnapshot:
    """Create a deterministic snapshot containing the active general web-search tool."""
    definition = next(
        definition for definition in TOOL_DEFINITIONS if definition.tool is search_web_for_wine
    )
    return ToolSelectionSnapshot(definitions=(definition,), readiness=())


def test_tool_error_environment_name_is_safe_and_model_repetition_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close both stages of the Gate 0 disclosure path using synthetic data only."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: "Test system prompt.",
    )

    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = _tool_snapshot()

    engine = MagicMock()
    engine.search.side_effect = ValueError(
        f"Provider credential missing. Set the {SYNTHETIC_ENVIRONMENT_IDENTIFIER} environment variable."
    )
    monkeypatch.setattr("src.agents.tools.web_search_tools._engine", engine)

    bound_model = MagicMock()
    bound_model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": search_web_for_wine.name,
                    "args": {"query": "synthetic wine news"},
                    "id": "synthetic-tool-call",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content=(
                "Web search could not run because "
                f"{SYNTHETIC_ENVIRONMENT_IDENTIFIER} is not configured."
            )
        ),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model

    agent = WineAgent(llm=llm, tool_registry=registry)
    result = agent.invoke("Find current synthetic wine news.")

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content.startswith(
        f"[{SafeToolErrorCode.WEB_SEARCH_UNAVAILABLE.value}]"
    )
    assert SYNTHETIC_ENVIRONMENT_IDENTIFIER not in tool_messages[0].content
    assert SYNTHETIC_ENVIRONMENT_IDENTIFIER not in result["final_answer"]
    assert REDACTION_TOKEN in result["final_answer"]
    assert bound_model.invoke.call_count == 2
