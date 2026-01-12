# GitHub Copilot Instructions for Pour Decisions Wine RAG Project

## Project Overview
**Pour Decisions** - A RAG-powered wine chatbot for personal wine knowledge and cellar management. 
Almost zero-cost hobby project using free-tier services as the first choice.

### Project's general guidelines for code generation and architecture
1. **Understand wine domain** - Research wine concept before coding and design.
2. **Analyze cost impact** - How does this affect free-tier usage and costs?
3. **Design local-first** - Prefer local solutions over cloud/external APIs.
4. **Optimize LLM usage** - Minimize calls, batch requests, cache results.
5. **Modular architecture** - Ensure components are decoupled and testable.

## Tech Stack
- **RAG**: ChromaDB vector store + LangChain + Hybrid Search (BM25 + Vector)
- **Embeddings**: sentence transformers local models
- **Reranking**: Cross-encoder models for precision
- **UI**: Streamlit and CSS enhancements
- **Backend**: Python 3.11+, SQLite, Pydantic

## Project Structure
```
src/
├── agents/        # LLM integration, prompts
├── database/      # SQLite models for wine cellar data
├── etl/           # Import from Vivino, CellarTracker
├── rag/           # Vector store, retrieval, chunking, reranking
├── ui/            # Streamlit pages and components
└── utils/         # Config, logging, helpers
```

### Python Requirements
- **Type hints**: REQUIRED for all function parameters and returns
- **Docstrings**: Google-style for all public functions, classes, modules, include usage examples if complex
- **Logging**: Use Python logging module, no print statements
- **Formatting**: Black (120 chars), isort for imports
- **Code Style**: Follow PEP 8 guidelines
- **Imports**: All imports must be at the top of the file, grouped by standard library, third-party, local modules

### Code Generation Constraints 
- Always follow the project's general guidelines for code generation and architecture.
- DO NOT generate summary documents to explain the changes unless explicitly requested. Instead, update
  the existing documentation in the codebase.
- Do not use emojis in code comments, docstrings, logs.
- Generated code must be less verbose, only use comments where necessary for clarity.
- Avoid redundant code; use helper functions or classes to encapsulate repeated logic.
- Always update the documentation when the code is changed significantly or new features are added.
  If there is no documentation for a component, create a readme file explaining its purpose and usage.
- Ensure all functions and classes have clear and concise docstrings. If a function or class is complex, 
  include usage examples in the docstring.
- New features should be modular, easily testable, and integrated into the existing architecture without major refactoring.
- Focus on code quality over quantity; prioritize maintainability and readability.

### Critical Constraints
1. **Minimize LLM calls** - Use local solutions where possible, batch requests
2. **Local-first** - Prefer database queries and calculations over external APIs
3. **No paid services first** - All tools must be free or have a generous free tier to minimize cost. 
   Paid tools only if absolutely necessary.
4. **Cache results** - Avoid repeated expensive operations (LLM calls, database queries)

