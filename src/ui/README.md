# UI Module

The `ui` module implements a multi-page Streamlit application for Pour Decisions.

## Components

| File / Directory | Purpose |
|------------------|---------|
| `app.py` | Main entry point - configures multi-page navigation |
| `sidebar.py` | Sidebar with agent mode selector and RAG settings |
| `resources.py` | `@st.cache_resource` loaders for LLM, agents, retriever, reranker |
| `pages/` | Individual page implementations |
| `helper/` | Reusable display widgets and analytics components |

## Pages

### Chatbot (`pages/chatbot.py`)

Default page. Wine Q&A with three agent modes:

- **Intelligent Agent** - LLM-driven tool selection (2-3 LLM calls)
- **Keyword Agent** - Pattern-matching routing (1 LLM call)
- **No Agent (RAG Only)** - Traditional RAG retrieval without agents

Displays source citations from RAG results and web search URLs when applicable.

### Cellar (`pages/cellar.py`)

Wine cellar dashboard:

- Inventory browser with filters
- Cellar statistics (by type, country, region)
- CellarTracker sync button with progress feedback
- Wine detail view with LLM description generation

### Taste Profile (`pages/taste_profile.py`)

Analytics dashboard with Plotly visualizations:

- Key insights overview (average rating, wines rated, etc.)
- Rating distribution and trends over time
- Wine type performance analysis
- Top varietals and varietal breakdown
- Producer loyalty metrics
- Favourite regions, countries, vintages, appellations
- Consumed wines inventory

## Sidebar (`sidebar.py`)

Renders:
1. App description
2. Agent mode selector (dropdown)
3. RAG settings (only in "No Agent" mode): system status, RAG toggle, source count slider
4. Reset chat button

## Cached Resources (`resources.py`)

All expensive-to-initialize objects are wrapped with `@st.cache_resource`:

| Function | Returns |
|----------|---------|
| `load_llm()` | `BaseChatModel` from config |
| `load_intelligent_agent()` | `WineAgent` instance |
| `load_keyword_agent()` | `KeywordWineAgent` instance |
| `load_chroma_client()` | `chromadb.ClientAPI` (HTTP) |
| `load_vector_retriever()` | `ChromaRetriever` |
| `load_bm25_index()` | `BM25Index` (loads or builds from collection) |
| `load_reranker()` | `DocumentReranker` (cross-encoder) |
| `load_retriever()` | `HybridRetriever` or `ChromaRetriever` (best available) |

## Helper Widgets (`helper/`)

### `display.py`

General-purpose UI utilities:

- `get_drinking_status()` / `render_drinking_index_bar()` - Drinking window visualisation
- `display_message()` - Render chat messages with markdown
- `make_page_title()` / `make_compact_page_title()` - Page header formatting
- CSS constants: `CONTENT_STYLE`, `TABS_DISPLAY`

### `cellar_stats.py`

Cellar page widgets:

- `show_cellar_metrics()` - Key metric cards
- `show_cellar_inventory()` - Paginated wine inventory with filters and detail modal
- `show_cellar_statistics()` - Charts for type/country/region distribution

### `taste_profile_stats.py`

Taste profile page widgets (Plotly charts):

- `show_taste_profile_overview()` - Summary metrics
- `show_rating_distribution()` / `show_rating_trends()` - Rating analytics
- `show_wine_type_distribution()` / `show_wine_type_performance()` - Type breakdown
- `show_top_varietals()` / `show_varietal_analysis()` - Grape variety insights
- `show_producer_loyalty()` - Repeat-purchase producers
- `show_favorite_regions()` / `show_favorite_countries()` / `show_favorite_vintages()` / `show_favorite_appellations()`
- `show_consumed_wines_inventory()` - Consumed wines table

## Running

```bash
make run   # Starts Streamlit on http://localhost:8501
```

The app auto-starts ChromaDB if needed.

