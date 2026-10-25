"""Bounded full-versus-core intelligent-agent tool portfolio experiment."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agents.intelligent.agent import WineAgent
from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import ToolDefinition, ToolRegistry
from src.eval.models import GoldenSample
from src.eval.utils import load_execution_model, run_agent_sample_sync
from src.utils import get_config

DEFAULT_PORTFOLIO_COHORT_PATH = Path("src/eval/tool_portfolio_cohort.json")


class ToolPortfolioCase(BaseModel):
    """One discriminating tool-selection and argument-contract case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    question: str
    expected_tool: str
    expected_arguments: dict[str, Any]
    accepted_lean_tools: tuple[str, ...] = ()


class ToolPortfolioCohort(BaseModel):
    """Validated portfolio cohort covering the complete active catalogue."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    repetitions: int = Field(ge=1, le=3)
    cases: tuple[ToolPortfolioCase, ...]

    @model_validator(mode="after")
    def validate_inventory_coverage(self) -> "ToolPortfolioCohort":
        """Require exactly one case for every active catalogue tool."""
        expected_names = {definition.metadata.name for definition in TOOL_DEFINITIONS}
        case_names = [case.expected_tool for case in self.cases]
        if len(case_names) != len(set(case_names)):
            raise ValueError("Portfolio cohort expected tools must be unique")
        if set(case_names) != expected_names:
            missing = sorted(expected_names - set(case_names))
            extra = sorted(set(case_names) - expected_names)
            raise ValueError(f"Portfolio cohort mismatch: missing={missing}, extra={extra}")
        known_names = expected_names
        unknown_fallbacks = sorted(
            {name for case in self.cases for name in case.accepted_lean_tools if name not in known_names}
        )
        if unknown_fallbacks:
            raise ValueError(f"Unknown lean fallback tools: {unknown_fallbacks}")
        return self


def load_tool_portfolio_cohort(
    path: Path = DEFAULT_PORTFOLIO_COHORT_PATH,
) -> ToolPortfolioCohort:
    """Load and validate the synthetic portfolio cohort."""
    return ToolPortfolioCohort.model_validate_json(path.read_text(encoding="utf-8"))


def _stub_result(tool_name: str, arguments: dict[str, Any]) -> str:
    """Return deterministic evidence without accessing a real dependency."""
    return json.dumps(
        {
            "fixture_tool": tool_name,
            "fixture_arguments": arguments,
            "fixture_evidence": "Representative local evidence is available for this request.",
        },
        sort_keys=True,
    )


def build_stub_tool_registry(config: DictConfig) -> ToolRegistry:
    """Build an all-ready registry that preserves production model contracts."""
    definitions: list[ToolDefinition] = []
    for definition in TOOL_DEFINITIONS:
        tool_name = definition.metadata.name

        def invoke_stub(_tool_name: str = tool_name, **kwargs: Any) -> str:
            """Return one deterministic fixture result."""
            return _stub_result(_tool_name, kwargs)

        stub = StructuredTool.from_function(
            func=invoke_stub,
            name=tool_name,
            description=definition.tool.description,
            args_schema=definition.tool.args_schema,
        )
        definitions.append(
            ToolDefinition(
                tool=stub,
                metadata=definition.metadata.model_copy(update={"prerequisites": ()}),
            )
        )
    return ToolRegistry(tuple(definitions), config=config)


def build_portfolio_agents(config: DictConfig | None = None) -> tuple[WineAgent, WineAgent]:
    """Build full and core-only arms around one configured execution model."""
    resolved_config = config or get_config()
    model = load_execution_model(resolved_config)
    registry = build_stub_tool_registry(resolved_config)
    return (
        WineAgent(llm=model, tool_registry=registry, extended_tools=True),
        WineAgent(llm=model, tool_registry=registry, extended_tools=False),
    )


def _sample(case: ToolPortfolioCase) -> GoldenSample:
    """Adapt one portfolio case to the shared agent execution helper."""
    return GoldenSample(
        id="multi_hop_001",
        question=case.question,
        category="multi_hop",
        difficulty="medium",
        expected_facts=[],
        expected_tool_calls=[case.expected_tool],
        ground_truth="Synthetic portfolio selection case.",
        ground_truth_chunk_ids=[],
        tags=["tool_portfolio"],
        notes=case.id,
    )


def _argument_value_matches(expected: Any, actual: Any) -> bool:
    """Compare one expected semantic argument against a model-produced value."""
    if expected == "__present__":
        return isinstance(actual, str) and bool(actual.strip())
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.casefold() in actual.casefold()
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(expected) == float(actual)
    return expected == actual


def arguments_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Return whether actual arguments contain the expected semantic subset."""
    return all(
        (value is None and key not in actual) or (key in actual and _argument_value_matches(value, actual[key]))
        for key, value in expected.items()
    )


def _capture_arm(
    arm: Literal["full", "lean"],
    agent: WineAgent,
    cohort: ToolPortfolioCohort,
) -> dict[str, Any]:
    """Capture one sequential portfolio arm with deterministic local tool bodies."""
    executions: list[dict[str, Any]] = []
    for repetition in range(1, cohort.repetitions + 1):
        for case in cohort.cases:
            started = time.perf_counter()
            result = run_agent_sample_sync(agent, _sample(case))
            calls = [call.model_dump(mode="json") for call in result.tool_call_records]
            expected_calls = [call for call in calls if call["tool_name"] == case.expected_tool]
            accepted_names = {case.expected_tool}
            if arm == "lean":
                accepted_names.update(case.accepted_lean_tools)
            executions.append(
                {
                    "case_id": case.id,
                    "repetition": repetition,
                    "expected_tool": case.expected_tool,
                    "expected_arguments": case.expected_arguments,
                    "tool_calls": calls,
                    "exact_tool_trajectory": [call["tool_name"] for call in calls] == [case.expected_tool],
                    "expected_tool_called": bool(expected_calls),
                    "arguments_correct": any(
                        arguments_match(case.expected_arguments, call["arguments"]) for call in expected_calls
                    ),
                    "capability_evidence_completed": any(call["tool_name"] in accepted_names for call in calls),
                    "llm_call_count": result.llm_call_count,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "token_usage": result.token_usage,
                    "terminal_outcome": result.terminal_outcome,
                }
            )
    return {
        "arm": arm,
        "tool_names": [tool.name for tool in agent.tools],
        "tool_contract": (
            agent.execution_provenance.tools.model_dump(mode="json")
            if agent.execution_provenance.tools is not None
            else None
        ),
        "prompt": agent.rendered_system_prompt.model_dump(mode="json"),
        "executions": executions,
    }


def _static_contract_metrics(agent: WineAgent) -> dict[str, int]:
    """Measure the exact compact LangChain contract and rendered prompt."""
    serialized = json.dumps(
        [convert_to_openai_tool(tool) for tool in agent.tools],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "tool_count": len(agent.tools),
        "serialized_tool_schema_bytes": len(serialized.encode("utf-8")),
        "rendered_prompt_characters": len(agent.system_prompt),
        "static_token_estimate": (len(serialized) + len(agent.system_prompt) + 3) // 4,
    }


def _arm_metrics(arm: dict[str, Any]) -> dict[str, Any]:
    """Summarize automatic selection, argument, cost, and reliability signals."""
    executions = arm["executions"]
    usages = [item["token_usage"] for item in executions if item["token_usage"] is not None]
    count = len(executions)
    return {
        "execution_count": count,
        "exact_tool_trajectory_rate": sum(item["exact_tool_trajectory"] for item in executions) / count,
        "expected_tool_call_rate": sum(item["expected_tool_called"] for item in executions) / count,
        "argument_correctness_rate": sum(item["arguments_correct"] for item in executions) / count,
        "capability_evidence_completion_rate": sum(item["capability_evidence_completed"] for item in executions)
        / count,
        "model_attempts": sum(item["llm_call_count"] for item in executions),
        "mean_model_attempts": statistics.mean(item["llm_call_count"] for item in executions),
        "mean_latency_ms": round(statistics.mean(item["latency_ms"] for item in executions), 3),
        "terminal_failures": sum(item["terminal_outcome"] is not None for item in executions),
        "token_coverage": len(usages),
        "input_tokens_when_reported": sum(item["input_tokens"] for item in usages),
        "output_tokens_when_reported": sum(item["output_tokens"] for item in usages),
    }


def capture_tool_portfolio_pair(
    cohort: ToolPortfolioCohort,
    full_agent: WineAgent,
    lean_agent: WineAgent,
) -> dict[str, Any]:
    """Capture and summarize both same-process portfolio arms."""
    full = _capture_arm("full", full_agent, cohort)
    lean = _capture_arm("lean", lean_agent, cohort)
    full["static_contract"] = _static_contract_metrics(full_agent)
    lean["static_contract"] = _static_contract_metrics(lean_agent)
    return {
        "schema_version": 1,
        "cohort_repetitions": cohort.repetitions,
        "case_count": len(cohort.cases),
        "external_tool_calls": 0,
        "full": full,
        "lean": lean,
        "metrics": {
            "full": _arm_metrics(full),
            "lean": _arm_metrics(lean),
        },
    }


def estimate_tool_portfolio_pair(cohort: ToolPortfolioCohort) -> dict[str, int]:
    """Return a conservative model-attempt bound without provider calls."""
    executions = len(cohort.cases) * cohort.repetitions * 2
    return {
        "cases": len(cohort.cases),
        "repetitions": cohort.repetitions,
        "paired_executions": executions,
        "maximum_logical_model_calls": executions * 5,
        "external_tool_calls": 0,
    }
