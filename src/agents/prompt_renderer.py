"""Render agent prompts from checked-in Jinja templates."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from src.agents.tools.registry import ToolSelectionSnapshot
from src.utils import find_project_root


_PROMPT_DIRECTORY = Path(find_project_root()) / "src/agents/prompts"
_INTELLIGENT_AGENT_PROMPT = "intelligent_agent_system_prompt.md.j2"


def create_prompt_environment(prompt_directory: Path = _PROMPT_DIRECTORY) -> Environment:
    """Create the strict Jinja environment used for agent prompts.

    Args:
        prompt_directory: Directory containing prompt templates.

    Returns:
        A Jinja environment that fails on missing template variables.
    """
    return Environment(
        loader=FileSystemLoader(prompt_directory),
        autoescape=select_autoescape(default_for_string=False, default=False),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_prompt_template(
    template_name: str,
    context: Mapping[str, Any] | None = None,
    *,
    prompt_directory: Path = _PROMPT_DIRECTORY,
) -> str:
    """Render one prompt template with strict variable validation.

    Args:
        template_name: Filename relative to the prompt directory.
        context: Values exposed to the Jinja template.
        prompt_directory: Directory containing prompt templates.

    Returns:
        The rendered prompt without surrounding whitespace.

    Raises:
        jinja2.TemplateError: If the template is invalid or references missing data.
    """
    environment = create_prompt_environment(prompt_directory)
    template = environment.get_template(template_name)
    return template.render(**(context or {})).strip()


def render_intelligent_agent_system_prompt(snapshot: ToolSelectionSnapshot) -> str:
    """Render the intelligent-agent prompt for one immutable tool snapshot.

    Args:
        snapshot: Tool selection captured during agent construction.

    Returns:
        A system prompt matching the tools bound to the agent.
    """
    context = {
        "selected_tool_names": frozenset(
            definition.metadata.name for definition in snapshot.definitions
        ),
        "selected_categories": frozenset(
            definition.metadata.category.value for definition in snapshot.definitions
        ),
    }
    return render_prompt_template(_INTELLIGENT_AGENT_PROMPT, context)
