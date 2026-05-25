# Agents Module

The `agents` module implements the agentic LLM layer for Pour Decisions. It provides two agent architectures and a set of LangChain tools for wine-related tasks.

## Components

| File / Directory | Purpose |
|------------------|---------|
| `intelligent/agent.py` | `WineAgent` - LangGraph ReAct agent with LLM-driven tool selection |
| `keyword/agent.py` | `KeywordWineAgent` - Pattern-matching router (no LLM for routing) |
| `tools/` | LangChain `@tool` functions organised by category |
| `llm.py` | LLM loading (Ollama / Google), prompt chain, invocation |
| `description_service.py` | Lazy LLM generation of wine/producer descriptions with RAG context |
| `prompts/` | Markdown prompt files loaded at module import time |

## Agent Architectures

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

### Keyword Agent (`keyword/agent.py`)

Routes queries using keyword pattern matching instead of an LLM:

1. **Routing** - Pattern matching classifies query into: `cellar`, `taste`, `knowledge`, `pairing`, `web_search`
2. **Execution** - Runs the matching tool(s) locally
3. **Generation** - LLM generates answer from tool output (1 LLM call)

**Cost**: 1 LLM call per query. Better for testing and cost-sensitive usage.

```python
from src.agents import create_keyword_agent

agent = create_keyword_agent(verbose=True)
result = agent.invoke("What is malolactic fermentation?")
```

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

- **Ollama (local, default)**: `gemma2:2b`
- **Google Gemini (optional fallback)**: `gemini-2.5-flash`

```python
from src.agents.llm import load_base_model

model = load_base_model("ollama", "gemma2:2b")
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
| `keyword_agent_generation_prompt.md` | Keyword agent answer generation |
| `rag_only_system_prompt.md` | RAG-only mode system message |
| `rag_only_user_prompt.md` | RAG-only mode context + question template |
| `wine_description_prompt.md` | Description service (wine) |
| `producer_description_prompt.md` | Description service (producer) |

