"""Render agent prompts from checked-in Jinja templates."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from src.agents.prompt_registry import (
    PromptRegistry,
    RenderedPrompt,
    get_prompt_registry,
    sha256_text,
)
from src.agents.tools.registry import ToolSelectionSnapshot
from src.utils import find_project_root


_PROMPT_DIRECTORY = Path(find_project_root()) / "src/agents/prompts"


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


def render_prompt_source(
    source: str,
    context: Mapping[str, Any] | None = None,
) -> str:
    """Render registered Jinja source with the existing strict environment.

    Args:
        source: Registered Jinja template source.
        context: Values exposed to the Jinja template.

    Returns:
        The rendered prompt without surrounding whitespace.

    Raises:
        jinja2.TemplateError: If the source is invalid or references missing data.
    """
    environment = create_prompt_environment()
    template = environment.from_string(source)
    return template.render(**(context or {})).strip()


def render_intelligent_agent_system_prompt(
    snapshot: ToolSelectionSnapshot,
    *,
    prompt_registry: PromptRegistry | None = None,
) -> RenderedPrompt:
    """Render the intelligent-agent prompt for one immutable tool snapshot.

    Args:
        snapshot: Tool selection captured during agent construction.
        prompt_registry: Optional registry injection for isolated tests.

    Returns:
        Immutable source and rendered identities plus prompt content matching the
        tools bound to the agent.
    """
    registry = prompt_registry or get_prompt_registry()
    prompt_record = registry.get("intelligent_agent_system")
    context = {
        "selected_tool_names": frozenset(
            definition.metadata.name for definition in snapshot.definitions
        ),
        "selected_categories": frozenset(
            definition.metadata.category.value for definition in snapshot.definitions
        ),
    }
    content = render_prompt_source(prompt_record.source, context)
    return RenderedPrompt(
        name=prompt_record.name,
        content=content,
        source_hash=prompt_record.source_hash,
        rendered_hash=sha256_text(content),
        label=prompt_record.label,
    )
