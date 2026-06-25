# Agents Module

> **Doc version**: 0.7.1 — last verified 2026-06-16.
> The agentic layer is subject to significant changes across milestones:
> Milestone 4 (advanced RAG architectures), Milestone 5 (prompt config versioning),
> Milestone 6 (dynamic tool registry), Milestone 7 (streaming), Milestone 8 (session memory),
> Milestone 10 (planner-executor), Milestone 11 (multi-agent), Milestone 12 (corrective loops).
> Update this README after each milestone. See `design/roadmap/agentic-ai/milestones/` for plans.

The `agents` module implements the agentic LLM layer for Pour Decisions. It provides the intelligent agent architecture and a set of LangChain tools for wine-related tasks.

## Components

| File / Directory | Purpose |
|------------------|---------|
| `intelligent/agent.py` | `WineAgent` - LangGraph ReAct agent with LLM-driven tool selection |
| `tools/` | LangChain `@tool` functions organised by category |
| `llm.py` | LLM loading (Ollama / Google), prompt chain, invocation |
| `description_service.py` | Lazy LLM generation of wine/producer descriptions with RAG context |
| `prompts/` | Markdown prompt files loaded at module import time |

## Agent Architecture

### Intelligent Agent (`intelligent/agent.py`)

Uses a LangGraph `StateGraph` to implement a ReAct loop:

1. **Planning** - LLM analyses the query and selects tool(s) (1 LLM call)
2. **Execution** - Tools run locally against SQLite / ChromaDB (free)
3. **Generation** - LLM synthesises a natural-language answer from tool outputs (1 LLM call)
4. **Correction** - If a tool call fails, the LLM retries (0-1 LLM call)

**Cost**: 2-3 LLM calls per query.

```python
from src.agents import create_wine_agent

agent = create_wine_agent(verbose=True)
result = agent.invoke("What wines in my cellar pair with lamb?")
print(result["final_answer"])
```

### Keyword Agent (`keyword/agent.py`) — **Deprecated, removed**

The keyword agent has been removed. Use the intelligent agent or `rag_only` mode instead.

## Tools (`tools/`)

All tools are LangChain `@tool` decorated functions registered in `tools/__init__.py`.

| File | Tools | Description |
|------|-------|-------------|
| `cellar_tools.py` | `get_cellar_wines`, `get_wine_details`, `get_cellar_statistics` | Cellar inventory queries (SQLite) |
| `taste_profile_tools.py` | `get_user_taste_profile`, `get_top_rated_wines`, `get_wine_recommendations_from_profile`, `compare_wine_to_profile` | Taste preference analysis |
| `pairing_tools.py` | `get_food_pairing_wines`, `get_pairing_for_wine`, `get_wine_and_cheese_pairings`, `suggest_dinner_menu_with_wines` | Food and wine pairing |
| `rag_tools.py` | `search_wine_knowledge`, `search_wine_region_info`, `search_grape_variety_info`, `search_wine_term_definition`, `search_wine_producer_info` | RAG knowledge base search |
| `web_search_tools.py` | `search_web_for_wine`, `search_wine_price`, `search_wine_reviews` | Web search via Tavily with SQLite cache |
| `utils.py` | `get_drink_status` | Shared helper for drinking window calculation |

### Tool Collections

```python
from src.agents.tools import get_tools, CORE_TOOLS, EXTENDED_TOOLS, ALL_TOOLS

# CORE_TOOLS (5): essential tools for basic queries
# EXTENDED_TOOLS (12): additional specialised tools
# ALL_TOOLS (17): CORE_TOOLS + EXTENDED_TOOLS

tools = get_tools(extended=True)   # returns ALL_TOOLS
tools = get_tools(extended=False)  # returns CORE_TOOLS
```

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

Markdown files loaded at module import time in `llm.py`:

| File | Used By |
|------|---------|
| `intelligent_agent_system_prompt.md` | Intelligent agent system message |
| `rag_only_system_prompt.md` | RAG-only mode system message |
| `rag_only_user_prompt.md` | RAG-only mode context + question template |
| `wine_description_prompt.md` | Description service (wine) |
| `producer_description_prompt.md` | Description service (producer) |
