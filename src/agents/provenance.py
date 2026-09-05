"""Deterministic, content-safe execution provenance for prompt-bearing paths."""

import math
from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

from src.agents.guardrails import (
    CallBudgetConfig,
    LoopDetectionConfig,
    RelevanceConfig,
    ToolExecutionConfig,
)
from src.agents.prompt_registry import (
    PromptRecord,
    PromptRegistry,
    RenderedPrompt,
    get_prompt_registry,
    sha256_canonical,
)
from src.agents.tools.registry import ToolDefinition, ToolSelectionSnapshot


ExecutionMode = Literal["intelligent", "rag", "description_wine", "description_producer"]
ModelRole = Literal["planning", "generation"]
ModelProvider = Literal["google", "ollama", "unknown"]


class PromptProvenance(BaseModel):
    """Content identities for one prompt used by an execution."""

    model_config = ConfigDict(frozen=True)

    name: str
    source_hash: str
    rendered_hash: str | None = None
    label: str = ""


class ModelProvenance(BaseModel):
    """Allowlisted identity and parameters for one instantiated model role."""

    model_config = ConfigDict(frozen=True)

    role: ModelRole
    model_class: str
    provider: ModelProvider
    name: str | None = None
    temperature: int | float | None = None
    top_p: int | float | None = None
    top_k: int | None = None
    max_retries: int | None = None
    num_predict: int | None = None
    reasoning: bool | str | None = None


class ToolReadinessProvenance(BaseModel):
    """Bounded readiness evidence for one catalogue tool."""

    model_config = ConfigDict(frozen=True)

    name: str
    available: bool
    reason_code: str | None = None


class ToolContractProvenance(BaseModel):
    """Identity and bounded inventory of one selected tool contract."""

    model_config = ConfigDict(frozen=True)

    contract_hash: str
    selected_names: tuple[str, ...]
    readiness: tuple[ToolReadinessProvenance, ...]


class CallBudgetPolicy(BaseModel):
    """Behavioral call-budget settings."""

    model_config = ConfigDict(frozen=True)

    enabled: bool
    max_llm_calls_per_query: int
    max_graph_steps_per_query: int


class LoopDetectionPolicy(BaseModel):
    """Behavioral duplicate-call detection settings."""

    model_config = ConfigDict(frozen=True)

    enabled: bool


class RelevancePolicy(BaseModel):
    """Behavioral relevance routing settings."""

    model_config = ConfigDict(frozen=True)

    enabled: bool
    wine_topic_allowlist: tuple[str, ...]
    off_topic_patterns: tuple[str, ...]


class ToolTimeoutPolicy(BaseModel):
    """Behavioral tool deadlines by latency class."""

    model_config = ConfigDict(frozen=True)

    fast: float
    slow: float


class ToolRetryPolicy(BaseModel):
    """Behavioral bounded retry settings."""

    model_config = ConfigDict(frozen=True)

    enabled: bool
    max_attempts: int
    delay_seconds: float
    min_remaining_seconds: float
    allowed_cost_classes: tuple[str, ...]


class ToolExecutionPolicy(BaseModel):
    """Behavioral asynchronous tool-execution settings."""

    model_config = ConfigDict(frozen=True)

    enabled: bool
    max_concurrent_calls: int
    timeout_seconds: ToolTimeoutPolicy
    retry: ToolRetryPolicy


class AgentPolicyConfig(BaseModel):
    """Canonical readable configuration that affects agent behavior."""

    model_config = ConfigDict(frozen=True)

    call_budget: CallBudgetPolicy
    loop_detection: LoopDetectionPolicy
    relevance: RelevancePolicy
    tool_execution: ToolExecutionPolicy


class AgentPolicyProvenance(BaseModel):
    """Readable agent policy and its canonical behavior identity."""

    model_config = ConfigDict(frozen=True)

    hash: str
    config: AgentPolicyConfig


class ExecutionProvenance(BaseModel):
    """Immutable provenance retained for one prompt-bearing execution path."""

    model_config = ConfigDict(frozen=True)

    mode: ExecutionMode
    prompts: tuple[PromptProvenance, ...]
    prompt_bundle_hash: str
    models: tuple[ModelProvenance, ...]
    tools: ToolContractProvenance | None = None
    agent_policy: AgentPolicyProvenance | None = None

    def to_trace_attributes(self) -> dict[str, str | int | float | bool]:
        """Return bounded scalar attributes without prompt or request content."""
        attributes: dict[str, str | int | float | bool] = {
            "pour_decisions.execution.mode": self.mode,
            "pour_decisions.prompt.bundle_hash": self.prompt_bundle_hash,
        }
        for prompt in self.prompts:
            prefix = f"pour_decisions.prompt.{prompt.name}"
            attributes[f"{prefix}.source_hash"] = prompt.source_hash
            if prompt.rendered_hash is not None:
                attributes[f"{prefix}.rendered_hash"] = prompt.rendered_hash
            if prompt.label:
                attributes[f"{prefix}.label"] = prompt.label

        for model in self.models:
            prefix = f"pour_decisions.model.{model.role}"
            model_values = model.model_dump(exclude={"role"}, exclude_none=True)
            for name, value in model_values.items():
                attributes[f"{prefix}.{name}"] = value
        if self.tools is not None:
            attributes["pour_decisions.tools.contract_hash"] = self.tools.contract_hash
            attributes["pour_decisions.tools.selected_count"] = len(self.tools.selected_names)
        if self.agent_policy is not None:
            attributes["pour_decisions.agent.policy_hash"] = self.agent_policy.hash
        return attributes

    def to_eval_dict(self) -> dict[str, object]:
        """Return complete JSON-safe run-level evidence."""
        prompts = {
            prompt.name: prompt.model_dump(exclude={"name"}, exclude_none=True)
            for prompt in self.prompts
        }
        models = {
            model.role: model.model_dump(exclude={"role"}, exclude_none=True)
            for model in self.models
        }
        snapshot: dict[str, object] = {
            "mode": self.mode,
            "prompts": prompts,
            "prompt_bundle_hash": self.prompt_bundle_hash,
            "models": models,
        }
        if self.tools is not None:
            snapshot["tools"] = self.tools.model_dump(mode="json")
        if self.agent_policy is not None:
            snapshot["agent_policy"] = self.agent_policy.model_dump(mode="json")
        return snapshot


def _qualified_class_name(value: object) -> str:
    """Return the stable fully qualified class name for an instance."""
    model_class = type(value)
    return f"{model_class.__module__}.{model_class.__qualname__}"


def _infer_provider(model_class: str, provider_hint: str | None) -> ModelProvider:
    """Infer a supported provider without inspecting arbitrary model state."""
    normalized_class = model_class.casefold()
    if "google" in normalized_class and "genai" in normalized_class:
        return "google"
    if "ollama" in normalized_class:
        return "ollama"
    normalized_hint = provider_hint.strip().casefold() if isinstance(provider_hint, str) else ""
    if normalized_hint in {"google", "ollama"}:
        return normalized_hint  # type: ignore[return-value]
    return "unknown"


def _read_public_value(model: object, name: str) -> object | None:
    """Read one reviewed public field while tolerating unusual model wrappers."""
    try:
        return getattr(model, name, None)
    except Exception:
        return None


def _read_model_name(model: object) -> str | None:
    """Read the first nonblank reviewed model-name field."""
    for field_name in ("model", "model_name"):
        value = _read_public_value(model, field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _read_number(model: object, name: str, *, integer_only: bool = False) -> int | float | None:
    """Read one finite numeric model parameter without coercion."""
    value = _read_public_value(model, name)
    if type(value) is int:
        return value
    if not integer_only and type(value) is float and math.isfinite(value):
        return value
    return None


def describe_model(
    model: BaseChatModel,
    *,
    role: ModelRole,
    provider_hint: str | None = None,
) -> ModelProvenance:
    """Describe an instantiated model from a strict public-field allowlist.

    Unknown or test-injected model classes retain only their class identity plus
    safe allowlisted primitive fields. Attribute inspection failures are ignored
    so provenance cannot block an otherwise valid invocation.
    """
    model_class = _qualified_class_name(model)
    reasoning = _read_public_value(model, "reasoning")
    if type(reasoning) not in {bool, str}:
        reasoning = None
    return ModelProvenance(
        role=role,
        model_class=model_class,
        provider=_infer_provider(model_class, provider_hint),
        name=_read_model_name(model),
        temperature=_read_number(model, "temperature"),
        top_p=_read_number(model, "top_p"),
        top_k=_read_number(model, "top_k", integer_only=True),
        max_retries=_read_number(model, "max_retries", integer_only=True),
        num_predict=_read_number(model, "num_predict", integer_only=True),
        reasoning=reasoning,
    )


def describe_prompt(prompt: PromptRecord | RenderedPrompt) -> PromptProvenance:
    """Reduce a registered prompt record to content identities only."""
    return PromptProvenance(
        name=prompt.name,
        source_hash=prompt.source_hash,
        rendered_hash=prompt.rendered_hash if isinstance(prompt, RenderedPrompt) else None,
        label=prompt.label,
    )


def build_prompt_bundle_hash(prompts: Sequence[PromptProvenance]) -> str:
    """Hash the ordered logical prompt names and their exact source identities."""
    bundle = [
        {"name": prompt.name, "source_hash": prompt.source_hash}
        for prompt in prompts
    ]
    return sha256_canonical(bundle)


def _tool_contract_entry(definition: ToolDefinition) -> dict[str, object]:
    """Return the canonical model-visible contract for one selected tool."""
    schema_model = definition.tool.tool_call_schema
    if schema_model is None or not hasattr(schema_model, "model_json_schema"):
        raise TypeError(f"Tool {definition.metadata.name!r} has no Pydantic call schema")
    return {
        "name": definition.metadata.name,
        "description": definition.tool.description,
        "input_schema": schema_model.model_json_schema(),
        "metadata": definition.metadata.model_dump(mode="json"),
    }


def build_tool_contract_provenance(
    snapshot: ToolSelectionSnapshot,
) -> ToolContractProvenance:
    """Describe and hash the immutable contract selected for one agent.

    Tool definition and readiness ordering are semantically irrelevant. Human
    readiness text is intentionally excluded because it is not a behavioral
    contract and may change independently of readiness reason codes.
    """
    ordered_definitions = sorted(
        snapshot.definitions,
        key=lambda definition: definition.metadata.name,
    )
    contract = [_tool_contract_entry(definition) for definition in ordered_definitions]
    readiness = tuple(
        ToolReadinessProvenance(
            name=item.name,
            available=item.available,
            reason_code=item.reason_code,
        )
        for item in sorted(snapshot.readiness, key=lambda item: item.name)
    )
    return ToolContractProvenance(
        contract_hash=sha256_canonical(contract),
        selected_names=tuple(
            definition.metadata.name for definition in ordered_definitions
        ),
        readiness=readiness,
    )


def build_agent_policy_provenance(
    *,
    call_budget: CallBudgetConfig,
    loop_detection: LoopDetectionConfig,
    relevance: RelevanceConfig,
    tool_execution: ToolExecutionConfig,
) -> AgentPolicyProvenance:
    """Build canonical provenance from validated M9A/M9B policy objects."""
    config = AgentPolicyConfig(
        call_budget=CallBudgetPolicy(
            enabled=call_budget.enabled,
            max_llm_calls_per_query=call_budget.max_llm_calls_per_query,
            max_graph_steps_per_query=call_budget.max_graph_steps_per_query,
        ),
        loop_detection=LoopDetectionPolicy(enabled=loop_detection.enabled),
        relevance=RelevancePolicy(
            enabled=relevance.enabled,
            wine_topic_allowlist=tuple(sorted(relevance.wine_topic_allowlist)),
            off_topic_patterns=tuple(sorted(relevance.off_topic_patterns)),
        ),
        tool_execution=ToolExecutionPolicy(
            enabled=tool_execution.enabled,
            max_concurrent_calls=tool_execution.max_concurrent_calls,
            timeout_seconds=ToolTimeoutPolicy(
                fast=tool_execution.timeout_seconds.fast,
                slow=tool_execution.timeout_seconds.slow,
            ),
            retry=ToolRetryPolicy(
                enabled=tool_execution.retry.enabled,
                max_attempts=tool_execution.retry.max_attempts,
                delay_seconds=tool_execution.retry.delay_seconds,
                min_remaining_seconds=tool_execution.retry.min_remaining_seconds,
                allowed_cost_classes=tuple(
                    sorted(
                        cost_class.value
                        for cost_class in tool_execution.retry.allowed_cost_classes
                    )
                ),
            ),
        ),
    )
    return AgentPolicyProvenance(
        hash=sha256_canonical(config.model_dump(mode="json")),
        config=config,
    )


def build_execution_provenance(
    *,
    mode: ExecutionMode,
    prompts: Sequence[PromptRecord | RenderedPrompt],
    models: Sequence[ModelProvenance],
    tools: ToolContractProvenance | None = None,
    agent_policy: AgentPolicyProvenance | None = None,
) -> ExecutionProvenance:
    """Compose immutable base provenance for one prompt-bearing path."""
    prompt_provenance = tuple(describe_prompt(prompt) for prompt in prompts)
    return ExecutionProvenance(
        mode=mode,
        prompts=prompt_provenance,
        prompt_bundle_hash=build_prompt_bundle_hash(prompt_provenance),
        models=tuple(models),
        tools=tools,
        agent_policy=agent_policy,
    )


def build_intelligent_execution_provenance(
    *,
    rendered_prompt: RenderedPrompt,
    planning_model: BaseChatModel,
    generation_model: BaseChatModel,
    tool_snapshot: ToolSelectionSnapshot,
    call_budget: CallBudgetConfig,
    loop_detection: LoopDetectionConfig,
    relevance: RelevanceConfig,
    tool_execution: ToolExecutionConfig,
) -> ExecutionProvenance:
    """Compose complete immutable provenance for one intelligent agent."""
    return build_execution_provenance(
        mode="intelligent",
        prompts=(rendered_prompt,),
        models=(
            describe_model(planning_model, role="planning"),
            describe_model(generation_model, role="generation"),
        ),
        tools=build_tool_contract_provenance(tool_snapshot),
        agent_policy=build_agent_policy_provenance(
            call_budget=call_budget,
            loop_detection=loop_detection,
            relevance=relevance,
            tool_execution=tool_execution,
        ),
    )


def build_rag_execution_provenance(
    model: BaseChatModel,
    *,
    prompt_registry: PromptRegistry | None = None,
) -> ExecutionProvenance:
    """Compose RAG-only provenance from its two cached templates and model."""
    registry = prompt_registry or get_prompt_registry()
    return build_execution_provenance(
        mode="rag",
        prompts=(registry.get("rag_only_system"), registry.get("rag_only_user")),
        models=(describe_model(model, role="generation"),),
    )


def build_description_execution_provenance(
    *,
    entity_type: Literal["wine", "producer"],
    prompt: PromptRecord,
    model: BaseChatModel,
) -> ExecutionProvenance:
    """Compose provenance for one description-generation operation type."""
    mode: ExecutionMode = (
        "description_wine" if entity_type == "wine" else "description_producer"
    )
    return build_execution_provenance(
        mode=mode,
        prompts=(prompt,),
        models=(describe_model(model, role="generation"),),
    )
