# GitHub Copilot Instructions for Pour Decisions Wine RAG Project

## Project Reference

For full architecture, key patterns, data flow, development commands, testing, and environment setup, see [`AGENTS.md`](../AGENTS.md) at the project root.

## Code Generation Guidelines

### General Principles
1. **Understand wine domain** - Research wine concepts before coding and design.
2. **Analyze cost impact** - How does this affect free-tier usage and costs?
3. **Design local-first** - Prefer local solutions over cloud/external APIs.
4. **Optimize LLM usage** - Minimize calls, batch requests, cache results.
5. **Modular architecture** - Ensure components are decoupled and testable.

### Python Requirements
- **Type hints**: REQUIRED for all function parameters and returns.
- **Docstrings**: Google-style for all public functions, classes, and modules. Include usage examples for complex ones.
- **Logging**: Use `from src.utils import logger`. Never use `print()`.
- **Formatting**: Black (120 chars), isort for imports.
- **Imports**: Grouped and separated - standard library, third-party, then local (`src.*`).

### Constraints
- DO NOT generate summary documents for changes unless explicitly requested. Update existing docs instead.
- Do not use emojis in code comments, docstrings, or logs.
- Keep generated code concise; use comments only where necessary for clarity.
- Avoid redundant code; encapsulate repeated logic in helper functions or classes.
- Always update or create component-level `README.md` when making significant changes.
- New features must be modular, testable, and integrated without major refactoring.
- **Step-by-step execution**: When given a design document, implement only ONE phase/step at a time. Do not write the entire implementation at once.
- **Living documentation**: If coding reality requires deviating from the design plan, stop and update the design document first.

### Critical Cost Constraints
1. **Minimize LLM calls** - Use local solutions where possible; batch requests.
2. **Local-first** - Prefer database queries and calculations over external APIs.
3. **No paid services first** - All tools must be free or have a generous free tier.
4. **Cache results** - Avoid repeated expensive operations (LLM calls, DB queries).
