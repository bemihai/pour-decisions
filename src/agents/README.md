# Agents Module

> **Project version:** 0.8.4 — last verified 2026-09-05.
> The current baseline includes the Milestone 6 dynamic tool registry, Milestone 9A guardrails,
> Milestone 6A minimum async runtime, Milestone 9B tool-execution reliability, and Milestone 5
> prompt and execution provenance. The agentic layer remains subject to future native-async
> completion, streaming, session memory, planner, multi-agent, and corrective-RAG work.
> Update this README after each milestone.

The `agents` module implements the agentic LLM layer for Pour Decisions. It provides the intelligent agent architecture and a set of LangChain tools for wine-related tasks.

## Components

| File / Directory | Purpose |
|------------------|---------|
| `intelligent/agent.py` | `WineAgent` - LangGraph ReAct agent with LLM-driven tool selection |
| `guardrails/` | Deterministic relevance, call-budget, loop, safe-error, tool-execution, sanitization, and trace helpers |
| `prompt_registry.py` | Validated, process-cached prompt assets and content identities |
| `prompt_renderer.py` | Strict Jinja rendering for snapshot-aware agent prompts |
| `provenance.py` | Deterministic prompt, model, tool-contract, and agent-policy provenance |
| `tools/` | LangChain `@tool` functions organised by category |
| `llm.py` | LLM loading (Ollama / Google), prompt chain, invocation |
| `description_service.py` | Lazy LLM generation of wine/producer descriptions with RAG context |
| `prompts/` | Markdown prompts and Jinja prompt templates |

## Agent Architecture

### Intelligent Agent (`intelligent/agent.py`)

Uses one compiled LangGraph `StateGraph` with explicit safety routing and paired sync/async model
callables:

1. **Relevance** - Clear off-topic requests are deterministically redirected before any model or tool call
2. **Budget** - Every planning, ReAct, and hybrid generation attempt is reserved before model invocation
3. **Planning** - The LLM analyses the query and selects zero or more tools
4. **Loop check** - An exact repeated tool name and canonical argument set terminates before the pending batch runs
5. **Execution** - On the async path, ready tools run under snapshot-derived admission, deadlines, and narrow retry policy; unexpected failures become stable safe messages
6. **Generation** - The standard loop or hybrid generation model produces the answer within the remaining budget
7. **Finalization** - Every returned answer passes mandatory sensitive-output sanitization

Standard requests typically use 1-3 LLM calls. The reviewed default hard limit is five attempted
calls and 30 graph steps per request; hybrid planning and generation count separately.

```python
from src.agents import create_wine_agent

agent = create_wine_agent(verbose=True)
result = agent.invoke("What wines in my cellar pair with lamb?")
print(result["final_answer"])

# Inside an async request or task:
result = await agent.ainvoke("What wines in my cellar pair with lamb?")
```

`invoke()` and `ainvoke()` share history conversion, initial state, graph limits, trace metadata,
final sanitization, and result shaping. M9B execution policy applies only to tools reached through
`ainvoke()`; synchronous `invoke()`, `stream()`, and current eval paths retain M9A behavior. The
FastAPI chat route awaits `ainvoke()` directly. The RAG-only production pipeline remains
synchronous and is temporarily bridged with `asyncio.to_thread()` at the API boundary until M6B.

### Keyword Agent (`keyword/agent.py`) — **Deprecated, removed**

The keyword agent has been removed. Use the intelligent agent or `rag_only` mode instead.

## Runtime Guardrails (`guardrails/`)

See [`guardrails/README.md`](guardrails/README.md) for the complete runtime policy, internal event
and trace schema, synchronous-worker limitations, and local timing evidence. M9A and M9B protect
only the active intelligent-agent graph; `rag_only` retains its existing bounded production-RAG
path and API error mapping.

- Call-budget, exact-loop, and relevance behavior each has an independent configuration flag.
- Safe tool-error normalization and final-answer sanitization are always active and have no bypass.
- Internal state records `llm_call_count`, hashed `tool_call_history`, and bounded
  `guardrail_events`; these fields do not change the public chat response schema.
- Existing request spans receive only low-cardinality trigger booleans, counts, configured limits,
  and the catalogue tool name for an exact duplicate. User text, tool arguments, exception details,
  and matched sensitive text are not attached.
- `ToolNode` keeps the M9A wrappers. Its async wrapper adds one total metadata-derived deadline that
  starts before shared app-worker admission and covers the optional retry.
- One extra attempt is allowed only for structured SQLite busy/locked failures on explicitly
  idempotent tools whose cost class is allowed and whose original deadline has useful time left.
- Caller cancellation and LangGraph control flow propagate. An upstream `TimeoutError` is a
  terminal safe failure, not an M9B deadline or retry candidate.
- All current built-in tools are synchronous. A deadline stops waiting but cannot terminate the
  framework worker thread; timed-out work may continue and accumulate beyond admission capacity.

```yaml
agents:
  guardrails:
    call_budget:
      enabled: true
      max_llm_calls_per_query: 5
      max_graph_steps_per_query: 30
    loop_detection:
      enabled: true
    relevance:
      enabled: true
    tool_execution:
      enabled: true
      max_concurrent_calls: 4
      timeout_seconds:
        fast: 10
        slow: 30
      retry:
        enabled: true
        max_attempts: 2
        delay_seconds: 0.1
        min_remaining_seconds: 1.0
        allowed_cost_classes:
          - free
```

## Tools (`tools/`)

All tools are LangChain `@tool` decorated functions with explicit metadata in module-local
catalogues. The composed registry checks shared prerequisites once per TTL window and binds only
the tools ready when an agent is constructed.

| File | Tools | Description |
|------|-------|-------------|
| `cellar_tools.py` | `get_cellar_wines`, `get_wine_details`, `get_cellar_statistics` | Cellar inventory queries (SQLite) |
| `taste_profile_tools.py` | `get_user_taste_profile`, `get_top_rated_wines`, `get_wine_recommendations_from_profile`, `compare_wine_to_profile` | Taste preference analysis |
| `pairing_tools.py` | `get_food_pairing_wines`, `get_pairing_for_wine`, `get_wine_and_cheese_pairings` | Food and wine pairing |
| `rag_tools.py` | `search_wine_knowledge`, `search_wine_region_info`, `search_grape_variety_info`, `search_wine_term_definition`, `search_wine_producer_info` | Shared production-path RAG search with tool-local generation disabled |
| `web_search_tools.py` | `search_web_for_wine`, `search_wine_price`, `search_wine_reviews` | Thin wrappers over the shared cached Tavily service |
| `utils.py` | `get_drink_status` | Shared helper for drinking window calculation |

### Tool Collections

```python
from src.agents.tools import get_tools, CORE_TOOLS, EXTENDED_TOOLS, ALL_TOOLS

# CORE_TOOLS (5): essential tools for basic queries
# EXTENDED_TOOLS (13): additional specialised tools
# ALL_TOOLS (18): CORE_TOOLS + EXTENDED_TOOLS

tools = get_tools(extended=True)   # returns ALL_TOOLS
tools = get_tools(extended=False)  # returns CORE_TOOLS
```

### Registry Readiness and Introspection

`ToolRegistry` evaluates shared Tavily, Chroma, and SQLite prerequisites when an intelligent agent
is constructed. The agent binds one immutable readiness-filtered snapshot, and the Jinja system
prompt is rendered from that same snapshot. Readiness results are cached for
`agents.tool_registry.health_check_ttl_seconds` (60 seconds by default).

`GET /api/tools` returns the complete catalogue with current readiness and the default cloud
agent's startup selection. A refreshed readiness result does not mutate an already constructed
agent; reconstruction or restart is required to change its bound tools.

## LLM Integration (`llm.py`)

Supports two providers configured in `app_config.yml`:

- **Google Gemini (production default)**: `gemini-2.5-flash`
- **Ollama (local opt-in)**: `gemma3:4b` (configured via `OLLAMA_MODEL` env var or `model.ollama.name` in `app_config.yml`)

```python
from src.agents.llm import load_base_model
from src.utils import get_config

cfg = get_config()
model = load_base_model(cfg.model.provider, cfg.model.name)
```

Key functions:
- `load_base_model(provider, name)` - Load an LLM instance
- `invoke_llm(question, context, model, history)` - RAG-only invocation with prompt chain
- `process_user_prompt(model, prompt, context, history)` - Wrapper with error handling

## Description Service (`description_service.py`)

Generates and persists wine/producer descriptions:

- **Lazy**: Only generates when `description` column is NULL
- **RAG-enhanced**: Retrieves wine book context for grounded descriptions
- **Cached**: Persists to SQLite so the same wine is never described twice
- **Fallback**: Uses LLM general knowledge when no RAG context is available

```python
from src.agents.description_service import get_description_service

service = get_description_service(use_rag_context=True)
description = service.get_wine_description(wine)
```

## Prompts (`prompts/`)

`prompts/versions.yml` is the authoritative manifest for the five production prompt assets. Each
entry declares a stable logical name, exact file, renderer, optional human label, and description.
`PromptRegistry` loads the manifest and every asset once per process through
`get_prompt_registry()`. Application startup and eval preflight construct the registry before
prompt consumers, so packaging or manifest mistakes fail before requests or concurrent samples
run. There are no unversioned inline fallback prompts.

Startup rejects missing or blank assets, invalid UTF-8, unknown manifest fields, duplicate file
references, undeclared prompt files, incomplete logical-name coverage, absolute paths, path
traversal, and OmegaConf interpolation. The manifest is prompt metadata rather than application
configuration; changing it does not add an `app_config.yml` setting.

Prompt assets used by the agent and description services:

| Logical name | File | Renderer | Used By |
|--------------|------|----------|---------|
| `intelligent_agent_system` | `intelligent_agent_system_prompt.md.j2` | strict Jinja | Intelligent-agent system message rendered from its readiness-filtered tool snapshot |
| `rag_only_system` | `rag_only_system_prompt.md` | static | RAG-only system message |
| `rag_only_user` | `rag_only_user_prompt.md` | token replacement | RAG-only context + question template |
| `wine_description` | `wine_description_prompt.md` | Python formatting | Description service (wine) |
| `producer_description` | `producer_description_prompt.md` | Python formatting | Description service (producer) |

### Prompt and execution hashes

- A **source hash** is the full `sha256:` digest of the UTF-8 source read from one prompt file,
  before consumer-specific stripping or formatting.
- A **rendered hash** identifies the exact intelligent-agent system prompt produced from the Jinja
  source and its immutable tool snapshot. Request-specific RAG and description prompts are not
  hashed because they contain questions, retrieved context, or entity data.
- A **bundle hash** identifies the ordered logical prompt names and source hashes used by one
  execution path. RAG-only generation therefore identifies its system and user templates without
  including request content.

Human-readable manifest labels are optional aliases, not identities. Runtime provenance also
records allowlisted properties of the instantiated planning/generation models and, for intelligent
execution, the selected model-visible tool contract and guardrail policy. Complete nested evidence
is stored once per eval run; traces receive only bounded scalar hashes, counts, labels, and model
fields. Prompt content, template variables, user input, credentials, endpoints, and absolute paths
are excluded.

Prompt sources and the cached registry are startup-scoped. Restart the API or eval process after
editing a prompt or `versions.yml`; hot reload and directory watching are intentionally unsupported.
