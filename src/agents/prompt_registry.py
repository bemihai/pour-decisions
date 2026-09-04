"""Validated prompt assets and deterministic content identities."""

import hashlib
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeAlias

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator

from src.utils import logger


PromptRenderer: TypeAlias = Literal["static", "token_replace", "python_format", "jinja2"]

_PROMPT_DIRECTORY = Path(__file__).parent / "prompts"
_DEFAULT_MANIFEST_PATH = _PROMPT_DIRECTORY / "versions.yml"
_PROMPT_ASSET_SUFFIXES = (".md", ".md.j2")
_EXPECTED_PROMPT_SPECS: Mapping[str, tuple[str, PromptRenderer]] = MappingProxyType(
    {
        "intelligent_agent_system": ("intelligent_agent_system_prompt.md.j2", "jinja2"),
        "rag_only_system": ("rag_only_system_prompt.md", "static"),
        "rag_only_user": ("rag_only_user_prompt.md", "token_replace"),
        "wine_description": ("wine_description_prompt.md", "python_format"),
        "producer_description": ("producer_description_prompt.md", "python_format"),
    }
)


def sha256_text(content: str) -> str:
    """Return a complete, algorithm-qualified digest for UTF-8 text."""
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


def sha256_canonical(value: object) -> str:
    """Hash a JSON-compatible value using deterministic serialization.

    Args:
        value: A value composed only of JSON-compatible primitives.

    Returns:
        A complete, algorithm-qualified SHA-256 digest.

    Raises:
        TypeError: If ``value`` cannot be serialized without an implicit conversion.
    """
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256_text(payload)


class PromptManifestEntry(BaseModel):
    """Validated metadata for one registered prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str
    renderer: PromptRenderer
    label: str = ""
    description: str

    @field_validator("file", "description")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        """Reject blank required manifest strings."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        """Normalize optional human-readable labels while allowing blanks."""
        return value.strip()


class _PromptManifest(BaseModel):
    """Validated top-level prompt manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompts: dict[str, PromptManifestEntry]


class PromptRecord(BaseModel):
    """Immutable source content and identity for one prompt asset."""

    model_config = ConfigDict(frozen=True)

    name: str
    file_path: Path
    renderer: PromptRenderer
    label: str
    description: str
    source: str
    source_hash: str


class RenderedPrompt(BaseModel):
    """One effective prompt rendered from a registered source."""

    model_config = ConfigDict(frozen=True)

    name: str
    content: str
    source_hash: str
    rendered_hash: str
    label: str


class PromptRegistry:
    """Validated immutable registry of checked-in prompt assets."""

    def __init__(self, records: Mapping[str, PromptRecord]) -> None:
        """Retain a defensive immutable copy of fully validated records."""
        self._records: Mapping[str, PromptRecord] = MappingProxyType(dict(records))

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "PromptRegistry":
        """Load and validate one prompt manifest and all declared assets.

        Args:
            manifest_path: YAML manifest whose directory contains the prompt assets.

        Returns:
            A fully constructed immutable prompt registry.

        Raises:
            FileNotFoundError: If the manifest or a declared prompt does not exist.
            UnicodeDecodeError: If a prompt asset is not valid UTF-8.
            ValueError: If the manifest or prompt inventory violates the contract.
        """
        manifest_path = Path(manifest_path).resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Prompt manifest does not exist: {manifest_path}")
        if not manifest_path.is_file():
            raise ValueError(f"Prompt manifest is not a file: {manifest_path}")

        raw_manifest = OmegaConf.load(manifest_path)
        manifest_data = OmegaConf.to_container(raw_manifest, resolve=False)
        cls._reject_interpolation(manifest_data)
        manifest = _PromptManifest.model_validate(manifest_data)

        expected_names = set(_EXPECTED_PROMPT_SPECS)
        declared_names = set(manifest.prompts)
        if declared_names != expected_names:
            missing = sorted(expected_names - declared_names)
            unexpected = sorted(declared_names - expected_names)
            raise ValueError(
                "Prompt manifest logical-name coverage mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

        declared_files = [entry.file for entry in manifest.prompts.values()]
        duplicate_files = sorted(
            file_name for file_name in set(declared_files) if declared_files.count(file_name) > 1
        )
        if duplicate_files:
            raise ValueError(f"Prompt manifest contains duplicate file references: {duplicate_files}")

        prompt_directory = manifest_path.parent.resolve()
        records: dict[str, PromptRecord] = {}
        for name, entry in manifest.prompts.items():
            file_path = cls._resolve_prompt_path(prompt_directory, entry.file)
            expected_file, expected_renderer = _EXPECTED_PROMPT_SPECS[name]
            if entry.file != expected_file:
                raise ValueError(
                    f"Prompt {name!r} must reference {expected_file!r}, got {entry.file!r}"
                )
            if entry.renderer != expected_renderer:
                raise ValueError(
                    f"Prompt {name!r} must use renderer {expected_renderer!r}, "
                    f"got {entry.renderer!r}"
                )
            if not file_path.exists():
                raise FileNotFoundError(f"Prompt asset does not exist: {file_path}")
            if not file_path.is_file():
                raise ValueError(f"Prompt asset is not a file: {file_path}")

            source = file_path.read_text(encoding="utf-8")
            if not source.strip():
                raise ValueError(f"Prompt asset is blank: {file_path}")
            records[name] = PromptRecord(
                name=name,
                file_path=file_path,
                renderer=entry.renderer,
                label=entry.label,
                description=entry.description,
                source=source,
                source_hash=sha256_text(source),
            )

        declared_assets = {record.file_path.name for record in records.values()}
        discovered_assets = {
            path.name
            for path in prompt_directory.iterdir()
            if path.is_file() and path.name.endswith(_PROMPT_ASSET_SUFFIXES)
        }
        undeclared_assets = sorted(discovered_assets - declared_assets)
        if undeclared_assets:
            raise ValueError(f"Prompt directory contains undeclared assets: {undeclared_assets}")

        return cls(records)

    @staticmethod
    def _reject_interpolation(value: object, location: str = "manifest") -> None:
        """Reject OmegaConf interpolation syntax anywhere in prompt metadata."""
        if isinstance(value, str):
            if "${" in value:
                raise ValueError(f"Prompt manifest interpolation is not supported at {location}")
            return
        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                PromptRegistry._reject_interpolation(nested_value, f"{location}.{key}")
            return
        if isinstance(value, list):
            for index, nested_value in enumerate(value):
                PromptRegistry._reject_interpolation(nested_value, f"{location}[{index}]")

    @staticmethod
    def _resolve_prompt_path(prompt_directory: Path, file_name: str) -> Path:
        """Resolve a manifest path while rejecting absolute and traversal paths."""
        relative_path = Path(file_name)
        if relative_path.is_absolute():
            raise ValueError(f"Prompt asset path must be relative: {file_name!r}")
        if ".." in relative_path.parts:
            raise ValueError(f"Prompt asset path must not contain traversal: {file_name!r}")

        file_path = (prompt_directory / relative_path).resolve()
        try:
            file_path.relative_to(prompt_directory)
        except ValueError as exc:
            raise ValueError(f"Prompt asset path escapes prompt directory: {file_name!r}") from exc
        return file_path

    def get(self, name: str) -> PromptRecord:
        """Return one prompt record by logical name."""
        try:
            return self._records[name]
        except KeyError as exc:
            raise KeyError(f"Unknown prompt logical name: {name!r}") from exc

    def get_source_version_map(self) -> dict[str, str]:
        """Return a detached logical-name to full source-hash mapping."""
        return {name: record.source_hash for name, record in self._records.items()}


@cache
def get_prompt_registry() -> PromptRegistry:
    """Return the process-cached registry for the package prompt manifest."""
    registry = PromptRegistry.from_manifest(_DEFAULT_MANIFEST_PATH)
    summary = ", ".join(
        f"{name}={source_hash.removeprefix('sha256:')[:12]}"
        for name, source_hash in registry.get_source_version_map().items()
    )
    logger.info("Loaded prompt registry: %s", summary)
    return registry
