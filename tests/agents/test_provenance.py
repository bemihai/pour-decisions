"""Tests for deterministic runtime execution provenance."""

from dataclasses import replace
from types import SimpleNamespace

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src.agents.guardrails import (
    CallBudgetConfig,
    LoopDetectionConfig,
    RelevanceConfig,
    ToolExecutionConfig,
)
from src.agents.guardrails.tool_execution import ToolRetryConfig, ToolTimeoutConfig
from src.agents.prompt_registry import get_prompt_registry, sha256_canonical
from src.agents.provenance import (
    build_agent_policy_provenance,
    build_description_execution_provenance,
    build_execution_provenance,
    build_intelligent_execution_provenance,
    build_prompt_bundle_hash,
    build_rag_execution_provenance,
    build_tool_contract_provenance,
    describe_model,
    describe_prompt,
)
from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import (
    CostClass,
    ToolDefinition,
    ToolReadiness,
    ToolSelectionSnapshot,
)


class UnknownModel:
    """Test model exposing both approved and denied public state."""

    model = "test-model"
    temperature = 0.25
    callbacks = ["secret-callback"]
    base_url = "https://secret.invalid"
    model_kwargs = {"api_key": "secret"}


class HostileUnknownModel:
    """Unknown wrapper whose reviewed attributes cannot be inspected."""

    @property
    def model(self) -> str:
        raise RuntimeError("unavailable")

    @property
    def temperature(self) -> float:
        raise RuntimeError("unavailable")


def test_canonical_json_hash_is_key_order_independent() -> None:
    """Canonical hashing should ignore mapping insertion order."""
    assert sha256_canonical({"a": 1, "b": 2}) == sha256_canonical({"b": 2, "a": 1})


def test_describe_ollama_model_uses_actual_allowlisted_values() -> None:
    """Ollama provenance should report the instantiated sampling parameters."""
    model = ChatOllama(
        model="gemma4:test",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        reasoning=True,
    )

    provenance = describe_model(model, role="generation")

    assert provenance.provider == "ollama"
    assert provenance.name == "gemma4:test"
    assert provenance.temperature == 1.0
    assert provenance.top_p == 0.95
    assert provenance.top_k == 64
    assert provenance.reasoning is True


def test_describe_google_model_uses_actual_allowlisted_values() -> None:
    """Google provenance should report the instantiated model and retry policy."""
    model = ChatGoogleGenerativeAI(
        model="gemini-test",
        temperature=0.0,
        max_retries=2,
        google_api_key="test-key",
    )

    provenance = describe_model(model, role="planning")

    assert provenance.provider == "google"
    assert provenance.name == "gemini-test"
    assert provenance.temperature == 0.0
    assert provenance.max_retries == 2


def test_unknown_model_exposes_only_allowlisted_fields() -> None:
    """Unknown injected models must not leak arbitrary or sensitive attributes."""
    provenance = describe_model(UnknownModel(), role="generation")  # type: ignore[arg-type]
    serialized = provenance.model_dump(exclude_none=True)

    assert provenance.provider == "unknown"
    assert provenance.model_class.endswith(".UnknownModel")
    assert provenance.name == "test-model"
    assert provenance.temperature == 0.25
    assert "callbacks" not in serialized
    assert "base_url" not in serialized
    assert "model_kwargs" not in serialized
    assert "secret" not in str(serialized)


def test_provider_hint_is_bounded_and_unknown_models_fail_open() -> None:
    """Only supported provider hints should enrich an otherwise unknown class."""
    model = SimpleNamespace(model="fixture")

    hinted = describe_model(model, role="generation", provider_hint="OLLAMA")  # type: ignore[arg-type]
    unsupported = describe_model(model, role="generation", provider_hint="other")  # type: ignore[arg-type]

    assert hinted.provider == "ollama"
    assert unsupported.provider == "unknown"


def test_unknown_model_attribute_errors_fail_open() -> None:
    """Unusual wrappers must retain class identity without blocking execution."""
    provenance = describe_model(HostileUnknownModel(), role="generation")  # type: ignore[arg-type]

    assert provenance.provider == "unknown"
    assert provenance.name is None
    assert provenance.temperature is None


def test_prompt_bundle_hash_uses_ordered_names_and_source_hashes_only() -> None:
    """Bundle identity should exclude labels and request-specific rendered content."""
    registry = get_prompt_registry()
    system = describe_prompt(registry.get("rag_only_system"))
    user = describe_prompt(registry.get("rag_only_user"))

    bundle_hash = build_prompt_bundle_hash((system, user))

    assert bundle_hash == sha256_canonical(
        [
            {"name": system.name, "source_hash": system.source_hash},
            {"name": user.name, "source_hash": user.source_hash},
        ]
    )
    assert bundle_hash != build_prompt_bundle_hash((user, system))


def test_base_execution_serializers_exclude_prompt_content() -> None:
    """Trace and eval representations should contain identities, never prompt text."""
    registry = get_prompt_registry()
    record = registry.get("wine_description")
    model = describe_model(UnknownModel(), role="generation")  # type: ignore[arg-type]

    provenance = build_execution_provenance(
        mode="description_wine",
        prompts=(record,),
        models=(model,),
    )

    trace_attributes = provenance.to_trace_attributes()
    eval_snapshot = provenance.to_eval_dict()
    serialized = f"{trace_attributes!r}{eval_snapshot!r}"
    assert record.source not in serialized
    assert trace_attributes["pour_decisions.execution.mode"] == "description_wine"
    assert eval_snapshot["models"]["generation"]["provider"] == "unknown"  # type: ignore[index]


def _tool_snapshot(
    definitions: tuple[ToolDefinition, ...],
    *,
    reason: str = "First wording",
) -> ToolSelectionSnapshot:
    """Build one selection with deliberately non-contractual readiness text."""
    readiness = tuple(
        ToolReadiness(
            name=definition.metadata.name,
            available=True,
            reason_code="ready",
            reason=reason,
        )
        for definition in reversed(definitions)
    )
    return ToolSelectionSnapshot(definitions=definitions, readiness=readiness)


def test_tool_contract_hash_ignores_definition_order_and_readiness_wording() -> None:
    """Semantically unordered inputs and human reason text must not alter identity."""
    definitions = TOOL_DEFINITIONS[:3]

    first = build_tool_contract_provenance(_tool_snapshot(definitions))
    reordered = build_tool_contract_provenance(
        _tool_snapshot(tuple(reversed(definitions)), reason="Rephrased wording")
    )

    assert first.contract_hash == reordered.contract_hash
    assert first.selected_names == reordered.selected_names
    assert first.readiness == reordered.readiness


def test_tool_description_change_changes_contract_hash() -> None:
    """The model-visible description participates in contract identity."""
    definition = TOOL_DEFINITIONS[0]
    changed_tool = definition.tool.model_copy(update={"description": "Changed description"})
    changed = ToolDefinition(tool=changed_tool, metadata=definition.metadata)

    original_hash = build_tool_contract_provenance(_tool_snapshot((definition,))).contract_hash
    changed_hash = build_tool_contract_provenance(_tool_snapshot((changed,))).contract_hash

    assert original_hash != changed_hash


def test_tool_name_change_changes_contract_hash() -> None:
    """The selected model-visible tool name participates in contract identity."""
    definition = TOOL_DEFINITIONS[0]
    changed_name = f"{definition.metadata.name}_changed"
    changed = ToolDefinition(
        tool=definition.tool.model_copy(update={"name": changed_name}),
        metadata=definition.metadata.model_copy(update={"name": changed_name}),
    )

    original_hash = build_tool_contract_provenance(_tool_snapshot((definition,))).contract_hash
    changed_hash = build_tool_contract_provenance(_tool_snapshot((changed,))).contract_hash

    assert original_hash != changed_hash


def test_tool_input_schema_change_changes_contract_hash() -> None:
    """The JSON input schema participates in contract identity."""
    class ChangedInput(BaseModel):
        value: str

    definition = TOOL_DEFINITIONS[0]
    changed_tool = definition.tool.model_copy(update={"args_schema": ChangedInput})
    changed = ToolDefinition(tool=changed_tool, metadata=definition.metadata)

    original_hash = build_tool_contract_provenance(_tool_snapshot((definition,))).contract_hash
    changed_hash = build_tool_contract_provenance(_tool_snapshot((changed,))).contract_hash

    assert original_hash != changed_hash


def test_tool_metadata_change_changes_contract_hash() -> None:
    """Validated catalogue metadata participates in contract identity."""
    definition = TOOL_DEFINITIONS[0]
    changed = ToolDefinition(
        tool=definition.tool,
        metadata=definition.metadata.model_copy(update={"capability": "Changed capability"}),
    )

    original_hash = build_tool_contract_provenance(_tool_snapshot((definition,))).contract_hash
    changed_hash = build_tool_contract_provenance(_tool_snapshot((changed,))).contract_hash

    assert original_hash != changed_hash


def test_selected_tool_membership_changes_contract_hash() -> None:
    """Adding or removing a selected tool must change the aggregate identity."""
    one_tool = build_tool_contract_provenance(_tool_snapshot(TOOL_DEFINITIONS[:1]))
    two_tools = build_tool_contract_provenance(_tool_snapshot(TOOL_DEFINITIONS[:2]))

    assert one_tool.contract_hash != two_tools.contract_hash


def _agent_policy_hash(
    *,
    call_budget: CallBudgetConfig | None = None,
    loop_detection: LoopDetectionConfig | None = None,
    relevance: RelevanceConfig | None = None,
    tool_execution: ToolExecutionConfig | None = None,
) -> str:
    """Return one policy hash using reviewed defaults for omitted groups."""
    return build_agent_policy_provenance(
        call_budget=call_budget or CallBudgetConfig(),
        loop_detection=loop_detection or LoopDetectionConfig(),
        relevance=relevance or RelevanceConfig(),
        tool_execution=tool_execution or ToolExecutionConfig(),
    ).hash


def test_each_agent_behavior_group_changes_policy_hash() -> None:
    """Every M9A/M9B behavior group must participate in policy identity."""
    baseline = _agent_policy_hash()

    assert _agent_policy_hash(call_budget=CallBudgetConfig(enabled=False)) != baseline
    assert _agent_policy_hash(loop_detection=LoopDetectionConfig(enabled=False)) != baseline
    assert _agent_policy_hash(relevance=RelevanceConfig(enabled=False)) != baseline
    assert _agent_policy_hash(tool_execution=ToolExecutionConfig(enabled=False)) != baseline


def test_each_tool_execution_subgroup_changes_policy_hash() -> None:
    """Concurrency, deadlines, and retry behavior must all affect policy identity."""
    baseline_config = ToolExecutionConfig()
    baseline = _agent_policy_hash(tool_execution=baseline_config)

    assert _agent_policy_hash(
        tool_execution=replace(baseline_config, max_concurrent_calls=9)
    ) != baseline
    assert _agent_policy_hash(
        tool_execution=replace(
            baseline_config,
            timeout_seconds=ToolTimeoutConfig(fast=4.0, slow=20.0),
        )
    ) != baseline
    assert _agent_policy_hash(
        tool_execution=replace(
            baseline_config,
            retry=replace(baseline_config.retry, max_attempts=1),
        )
    ) != baseline


def test_reordering_unordered_policy_collections_does_not_change_hash() -> None:
    """Relevance phrases and retry cost classes should have set semantics."""
    first_relevance = RelevanceConfig(
        wine_topic_allowlist=("wine", "cellar"),
        off_topic_patterns=("weather", "football"),
    )
    second_relevance = RelevanceConfig(
        wine_topic_allowlist=("cellar", "wine"),
        off_topic_patterns=("football", "weather"),
    )
    first_execution = ToolExecutionConfig(
        retry=ToolRetryConfig(
            allowed_cost_classes=frozenset({CostClass.CHEAP, CostClass.FREE})
        )
    )
    second_execution = ToolExecutionConfig(
        retry=ToolRetryConfig(
            allowed_cost_classes=frozenset({CostClass.FREE, CostClass.CHEAP})
        )
    )

    first = _agent_policy_hash(relevance=first_relevance, tool_execution=first_execution)
    second = _agent_policy_hash(relevance=second_relevance, tool_execution=second_execution)

    assert first == second


def test_agent_policy_is_json_safe_and_immutable() -> None:
    """The readable policy should serialize without implementation-only objects."""
    provenance = build_agent_policy_provenance(
        call_budget=CallBudgetConfig(),
        loop_detection=LoopDetectionConfig(),
        relevance=RelevanceConfig(),
        tool_execution=ToolExecutionConfig(),
    )

    payload = provenance.model_dump(mode="json")

    assert payload["hash"] == sha256_canonical(payload["config"])
    assert payload["config"]["tool_execution"]["retry"]["allowed_cost_classes"] == ["free"]


def test_intelligent_execution_composes_all_runtime_dimensions() -> None:
    """Intelligent provenance should retain prompt, model, tool, and policy evidence."""
    from src.agents.prompt_renderer import render_intelligent_agent_system_prompt

    snapshot = _tool_snapshot(TOOL_DEFINITIONS[:2])
    rendered_prompt = render_intelligent_agent_system_prompt(snapshot)
    model = ChatOllama(model="gemma4:test", temperature=1.0, top_p=0.95, top_k=64)

    provenance = build_intelligent_execution_provenance(
        rendered_prompt=rendered_prompt,
        planning_model=model,
        generation_model=model,
        tool_snapshot=snapshot,
        call_budget=CallBudgetConfig(),
        loop_detection=LoopDetectionConfig(),
        relevance=RelevanceConfig(),
        tool_execution=ToolExecutionConfig(),
    )

    assert provenance.mode == "intelligent"
    assert provenance.prompts[0].rendered_hash == rendered_prompt.rendered_hash
    assert [model.role for model in provenance.models] == ["planning", "generation"]
    assert provenance.models[0].model_dump(exclude={"role"}) == provenance.models[1].model_dump(
        exclude={"role"}
    )
    assert provenance.tools is not None
    assert provenance.tools.selected_names == tuple(sorted(provenance.tools.selected_names))
    assert provenance.agent_policy is not None

    trace_attributes = provenance.to_trace_attributes()
    eval_snapshot = provenance.to_eval_dict()
    assert all(isinstance(value, (str, int, float, bool)) for value in trace_attributes.values())
    assert "pour_decisions.tools.selected_names" not in trace_attributes
    assert "pour_decisions.agent.policy.config" not in trace_attributes
    assert eval_snapshot["tools"]["selected_names"] == list(  # type: ignore[index]
        provenance.tools.selected_names
    )
    assert eval_snapshot["agent_policy"]["config"] == (  # type: ignore[index]
        provenance.agent_policy.config.model_dump(mode="json")
    )


def test_hybrid_intelligent_execution_records_distinct_model_roles() -> None:
    """Hybrid provenance should describe actual planning and generation models separately."""
    from src.agents.prompt_renderer import render_intelligent_agent_system_prompt

    snapshot = _tool_snapshot(TOOL_DEFINITIONS[:1])
    rendered_prompt = render_intelligent_agent_system_prompt(snapshot)
    planning_model = ChatGoogleGenerativeAI(
        model="gemini-planner",
        google_api_key="test-key",
    )
    generation_model = ChatOllama(model="local-generator")

    provenance = build_intelligent_execution_provenance(
        rendered_prompt=rendered_prompt,
        planning_model=planning_model,
        generation_model=generation_model,
        tool_snapshot=snapshot,
        call_budget=CallBudgetConfig(),
        loop_detection=LoopDetectionConfig(),
        relevance=RelevanceConfig(),
        tool_execution=ToolExecutionConfig(),
    )

    assert provenance.models[0].role == "planning"
    assert provenance.models[0].provider == "google"
    assert provenance.models[0].name == "gemini-planner"
    assert provenance.models[1].role == "generation"
    assert provenance.models[1].provider == "ollama"
    assert provenance.models[1].name == "local-generator"


def test_rag_execution_uses_only_two_rag_prompts_and_generation_model() -> None:
    """RAG provenance should not claim intelligent or description resources."""
    registry = get_prompt_registry()
    model = ChatOllama(model="rag-model")

    provenance = build_rag_execution_provenance(model, prompt_registry=registry)

    assert provenance.mode == "rag"
    assert [prompt.name for prompt in provenance.prompts] == ["rag_only_system", "rag_only_user"]
    assert [model.role for model in provenance.models] == ["generation"]
    assert provenance.tools is None
    assert provenance.agent_policy is None


def test_description_execution_uses_only_applicable_prompt() -> None:
    """Each description operation should identify only its own template."""
    registry = get_prompt_registry()
    model = ChatOllama(model="description-model")

    wine = build_description_execution_provenance(
        entity_type="wine",
        prompt=registry.get("wine_description"),
        model=model,
    )
    producer = build_description_execution_provenance(
        entity_type="producer",
        prompt=registry.get("producer_description"),
        model=model,
    )

    assert wine.mode == "description_wine"
    assert [prompt.name for prompt in wine.prompts] == ["wine_description"]
    assert producer.mode == "description_producer"
    assert [prompt.name for prompt in producer.prompts] == ["producer_description"]
