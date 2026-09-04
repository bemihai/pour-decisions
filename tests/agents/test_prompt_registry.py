"""Tests for validated prompt loading and deterministic source identities."""

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.agents.prompt_registry import (
    PromptRegistry,
    get_prompt_registry,
    sha256_canonical,
    sha256_text,
)


_PROJECT_ROOT = Path(__file__).parents[2]
_REAL_MANIFEST = _PROJECT_ROOT / "src/agents/prompts/versions.yml"
_EXPECTED_PROMPTS: dict[str, dict[str, str]] = {
    "intelligent_agent_system": {
        "file": "intelligent_agent_system_prompt.md.j2",
        "renderer": "jinja2",
        "label": "",
        "description": "Tool-aware system prompt for the intelligent agent",
    },
    "rag_only_system": {
        "file": "rag_only_system_prompt.md",
        "renderer": "static",
        "label": "",
        "description": "System prompt for RAG-only generation",
    },
    "rag_only_user": {
        "file": "rag_only_user_prompt.md",
        "renderer": "token_replace",
        "label": "",
        "description": "RAG-only user template containing context and question tokens",
    },
    "wine_description": {
        "file": "wine_description_prompt.md",
        "renderer": "python_format",
        "label": "",
        "description": "Wine description and drinking-window prompt",
    },
    "producer_description": {
        "file": "producer_description_prompt.md",
        "renderer": "python_format",
        "label": "",
        "description": "Producer description prompt",
    },
}


def _valid_entries() -> dict[str, dict[str, str]]:
    """Return a mutable copy of the supported manifest entries."""
    return copy.deepcopy(_EXPECTED_PROMPTS)


def _render_manifest(entries: dict[str, dict[str, Any]]) -> str:
    """Render simple JSON-compatible test entries as valid YAML."""
    lines = ["prompts:"]
    for name, entry in entries.items():
        lines.append(f"  {name}:")
        for key, value in entry.items():
            lines.append(f"    {key}: {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def _write_manifest(manifest_path: Path, entries: dict[str, dict[str, Any]]) -> None:
    """Write a temporary prompt manifest."""
    manifest_path.write_text(_render_manifest(entries), encoding="utf-8")


def _create_prompt_tree(
    directory: Path,
    *,
    content_prefix: str = "fixture",
) -> tuple[Path, dict[str, dict[str, str]]]:
    """Create one complete valid temporary manifest and prompt inventory."""
    directory.mkdir(parents=True, exist_ok=True)
    entries = _valid_entries()
    for name, entry in entries.items():
        (directory / entry["file"]).write_text(f"{content_prefix}: {name}\n", encoding="utf-8")
    manifest_path = directory / "versions.yml"
    _write_manifest(manifest_path, entries)
    return manifest_path, entries


def test_real_manifest_loads_exact_supported_inventory() -> None:
    """The checked-in manifest should describe every production prompt exactly once."""
    registry = PromptRegistry.from_manifest(_REAL_MANIFEST)

    assert set(registry.get_source_version_map()) == set(_EXPECTED_PROMPTS)
    for name, expected in _EXPECTED_PROMPTS.items():
        record = registry.get(name)
        assert record.name == name
        assert record.file_path == (_REAL_MANIFEST.parent / expected["file"]).resolve()
        assert record.renderer == expected["renderer"]
        assert record.label == ""
        assert record.source == record.file_path.read_text(encoding="utf-8")


def test_source_hashes_are_full_qualified_and_deterministic() -> None:
    """Repeated loads should produce full SHA-256 identities for identical source text."""
    first = PromptRegistry.from_manifest(_REAL_MANIFEST).get_source_version_map()
    second = PromptRegistry.from_manifest(_REAL_MANIFEST).get_source_version_map()

    assert first == second
    assert all(re.fullmatch(r"sha256:[0-9a-f]{64}", source_hash) for source_hash in first.values())


def test_source_edit_changes_only_affected_hash(tmp_path: Path) -> None:
    """A source edit should not perturb identities for unrelated prompt assets."""
    manifest_path, entries = _create_prompt_tree(tmp_path)
    before = PromptRegistry.from_manifest(manifest_path).get_source_version_map()

    changed_name = "rag_only_user"
    changed_file = tmp_path / entries[changed_name]["file"]
    changed_file.write_text("changed source\n", encoding="utf-8")
    after = PromptRegistry.from_manifest(manifest_path).get_source_version_map()

    assert before[changed_name] != after[changed_name]
    assert {
        name: source_hash for name, source_hash in before.items() if name != changed_name
    } == {name: source_hash for name, source_hash in after.items() if name != changed_name}


def test_unknown_logical_name_raises_key_error_with_name() -> None:
    """Lookup failures should identify the unsupported logical name."""
    registry = PromptRegistry.from_manifest(_REAL_MANIFEST)

    with pytest.raises(KeyError, match="missing_prompt"):
        registry.get("missing_prompt")


def test_prompt_records_are_immutable() -> None:
    """Callers should not be able to mutate published prompt records."""
    record = PromptRegistry.from_manifest(_REAL_MANIFEST).get("rag_only_system")

    with pytest.raises(ValidationError, match="frozen"):
        record.label = "changed"


def test_missing_manifest_fails_clearly(tmp_path: Path) -> None:
    """A nonexistent manifest should be reported before any asset loading."""
    missing_manifest = tmp_path / "versions.yml"

    with pytest.raises(FileNotFoundError, match="Prompt manifest does not exist"):
        PromptRegistry.from_manifest(missing_manifest)


def test_manifest_directory_is_rejected(tmp_path: Path) -> None:
    """A manifest path must identify a regular file."""
    manifest_directory = tmp_path / "versions.yml"
    manifest_directory.mkdir()

    with pytest.raises(ValueError, match="Prompt manifest is not a file"):
        PromptRegistry.from_manifest(manifest_directory)


@pytest.mark.parametrize(
    "manifest_text",
    [
        "prompts: [unterminated\n",
        "prompts:\n  rag_only_system:\n    file: one.md\n  rag_only_system:\n    file: two.md\n",
    ],
    ids=["malformed-yaml", "duplicate-yaml-key"],
)
def test_yaml_parse_failures_are_rejected(tmp_path: Path, manifest_text: str) -> None:
    """Malformed YAML and duplicate mapping keys should fail during parsing."""
    manifest_path = tmp_path / "versions.yml"
    manifest_path.write_text(manifest_text, encoding="utf-8")

    with pytest.raises(Exception, match="duplicate key|expected|while parsing"):
        PromptRegistry.from_manifest(manifest_path)


def test_unknown_manifest_fields_are_rejected(tmp_path: Path) -> None:
    """Manifest typos should not be silently ignored."""
    manifest_path, entries = _create_prompt_tree(tmp_path)
    entries["rag_only_system"]["unexpected"] = "value"
    _write_manifest(manifest_path, entries)

    with pytest.raises(ValidationError, match="unexpected"):
        PromptRegistry.from_manifest(manifest_path)


@pytest.mark.parametrize("missing_name", sorted(_EXPECTED_PROMPTS))
def test_incomplete_logical_name_coverage_is_rejected(tmp_path: Path, missing_name: str) -> None:
    """Every supported logical prompt name is required."""
    manifest_path, entries = _create_prompt_tree(tmp_path)
    entries.pop(missing_name)
    _write_manifest(manifest_path, entries)

    with pytest.raises(ValueError, match=rf"coverage mismatch.*{missing_name}"):
        PromptRegistry.from_manifest(manifest_path)


def test_unexpected_logical_name_is_rejected(tmp_path: Path) -> None:
    """Future prompt names require an explicit registry contract change."""
    manifest_path, entries = _create_prompt_tree(tmp_path)
    entries["unreviewed_prompt"] = {
        "file": "unreviewed.md",
        "renderer": "static",
        "label": "",
        "description": "Unreviewed prompt",
    }
    (tmp_path / "unreviewed.md").write_text("unreviewed\n", encoding="utf-8")
    _write_manifest(manifest_path, entries)

    with pytest.raises(ValueError, match=r"coverage mismatch.*unreviewed_prompt"):
        PromptRegistry.from_manifest(manifest_path)


def test_undeclared_prompt_asset_is_rejected(tmp_path: Path) -> None:
    """Every Markdown prompt asset in the manifest directory must be registered."""
    manifest_path, _ = _create_prompt_tree(tmp_path)
    (tmp_path / "orphan.md").write_text("orphan prompt\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"undeclared assets.*orphan\.md"):
        PromptRegistry.from_manifest(manifest_path)


def test_duplicate_file_reference_is_rejected(tmp_path: Path) -> None:
    """Two logical prompt names must not share one source file."""
    manifest_path, entries = _create_prompt_tree(tmp_path)
    entries["producer_description"]["file"] = entries["wine_description"]["file"]
    _write_manifest(manifest_path, entries)

    with pytest.raises(ValueError, match="duplicate file references"):
        PromptRegistry.from_manifest(manifest_path)


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    """Relative traversal must fail even when the target exists."""
    prompt_directory = tmp_path / "prompts"
    manifest_path, entries = _create_prompt_tree(prompt_directory)
    (tmp_path / "outside.md").write_text("outside\n", encoding="utf-8")
    entries["rag_only_system"]["file"] = "../outside.md"
    _write_manifest(manifest_path, entries)

    with pytest.raises(ValueError, match="traversal"):
        PromptRegistry.from_manifest(manifest_path)


def test_absolute_prompt_path_is_rejected(tmp_path: Path) -> None:
    """Prompt files must remain relative to their manifest directory."""
    manifest_path, entries = _create_prompt_tree(tmp_path / "prompts")
    absolute_file = tmp_path / "absolute.md"
    absolute_file.write_text("absolute\n", encoding="utf-8")
    entries["rag_only_system"]["file"] = str(absolute_file)
    _write_manifest(manifest_path, entries)

    with pytest.raises(ValueError, match="must be relative"):
        PromptRegistry.from_manifest(manifest_path)


def test_missing_prompt_file_is_rejected(tmp_path: Path) -> None:
    """Every declared source must exist when the registry is constructed."""
    manifest_path, entries = _create_prompt_tree(tmp_path)
    (tmp_path / entries["rag_only_system"]["file"]).unlink()

    with pytest.raises(FileNotFoundError, match="Prompt asset does not exist"):
        PromptRegistry.from_manifest(manifest_path)


def test_prompt_directory_entry_is_rejected(tmp_path: Path) -> None:
    """A declared prompt path cannot refer to a directory."""
    manifest_path, entries = _create_prompt_tree(tmp_path)
    prompt_path = tmp_path / entries["rag_only_system"]["file"]
    prompt_path.unlink()
    prompt_path.mkdir()

    with pytest.raises(ValueError, match="Prompt asset is not a file"):
        PromptRegistry.from_manifest(manifest_path)


def test_invalid_utf8_prompt_is_rejected(tmp_path: Path) -> None:
    """Source identities are defined only over decoded UTF-8 text."""
    manifest_path, entries = _create_prompt_tree(tmp_path)
    (tmp_path / entries["rag_only_system"]["file"]).write_bytes(b"\xff")

    with pytest.raises(UnicodeDecodeError):
        PromptRegistry.from_manifest(manifest_path)


@pytest.mark.parametrize("blank_source", ["", " \n\t"])
def test_blank_prompt_is_rejected(tmp_path: Path, blank_source: str) -> None:
    """Empty and whitespace-only prompt sources are deployment errors."""
    manifest_path, entries = _create_prompt_tree(tmp_path)
    (tmp_path / entries["rag_only_system"]["file"]).write_text(blank_source, encoding="utf-8")

    with pytest.raises(ValueError, match="Prompt asset is blank"):
        PromptRegistry.from_manifest(manifest_path)


def test_wrong_renderer_is_rejected(tmp_path: Path) -> None:
    """The manifest must preserve each reviewed prompt's rendering contract."""
    manifest_path, entries = _create_prompt_tree(tmp_path)
    entries["rag_only_system"]["renderer"] = "jinja2"
    _write_manifest(manifest_path, entries)

    with pytest.raises(ValueError, match="must use renderer"):
        PromptRegistry.from_manifest(manifest_path)


def test_manifest_interpolation_is_rejected_without_resolution(tmp_path: Path) -> None:
    """Prompt metadata must not vary through OmegaConf environment interpolation."""
    manifest_path, entries = _create_prompt_tree(tmp_path)
    entries["rag_only_system"]["label"] = "${oc.env:PROMPT_LABEL,default}"
    _write_manifest(manifest_path, entries)

    with pytest.raises(ValueError, match="interpolation is not supported"):
        PromptRegistry.from_manifest(manifest_path)


def test_blank_optional_label_is_accepted_and_does_not_replace_hash(tmp_path: Path) -> None:
    """A blank alias remains valid while source content supplies the identity."""
    manifest_path, _ = _create_prompt_tree(tmp_path)

    record = PromptRegistry.from_manifest(manifest_path).get("rag_only_system")

    assert record.label == ""
    assert record.source_hash == sha256_text(record.source)
    assert record.source_hash != record.label


def test_default_registry_factory_is_process_cached() -> None:
    """Production callers should share one validated default registry."""
    get_prompt_registry.cache_clear()
    try:
        first = get_prompt_registry()
        second = get_prompt_registry()
        assert first is second
    finally:
        get_prompt_registry.cache_clear()


def test_direct_registries_are_isolated(tmp_path: Path) -> None:
    """Injected manifests should construct independent registries for tests."""
    first_manifest, _ = _create_prompt_tree(tmp_path / "first", content_prefix="first")
    second_manifest, _ = _create_prompt_tree(tmp_path / "second", content_prefix="second")

    first = PromptRegistry.from_manifest(first_manifest)
    second = PromptRegistry.from_manifest(second_manifest)

    assert first is not second
    assert first.get_source_version_map() != second.get_source_version_map()


def test_canonical_hash_is_key_order_independent_and_strict() -> None:
    """Canonical helpers should sort keys and reject implicit object stringification."""
    assert sha256_canonical({"b": 2, "a": 1}) == sha256_canonical({"a": 1, "b": 2})

    with pytest.raises(TypeError):
        sha256_canonical({"path": Path("prompt.md")})
