"""Tests for dependency-level readiness caching and concurrency."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from src.agents.tools.registry import ToolPrerequisite, ToolRegistry, _PrerequisiteReadiness


def _registry(ttl_seconds: int = 60) -> ToolRegistry:
    """Build an empty registry with an explicit cache TTL."""
    config = OmegaConf.create(
        {
            "agents": {
                "tool_registry": {
                    "enabled": False,
                    "health_check_ttl_seconds": ttl_seconds,
                }
            }
        }
    )
    return ToolRegistry((), config=config)


def _ready(prerequisite: ToolPrerequisite) -> _PrerequisiteReadiness:
    """Build available prerequisite evidence for cache tests."""
    return _PrerequisiteReadiness(prerequisite=prerequisite, available=True)


def test_repeated_readiness_within_ttl_probes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated callers should share fresh prerequisite evidence."""
    registry = _registry()
    probe = MagicMock(return_value=_ready(ToolPrerequisite.WEB_SEARCH_CONFIG))
    monkeypatch.setattr(registry, "_probe_prerequisite", probe)

    first = registry._get_prerequisite_readiness(ToolPrerequisite.WEB_SEARCH_CONFIG)
    second = registry._get_prerequisite_readiness(ToolPrerequisite.WEB_SEARCH_CONFIG)

    assert first is second
    probe.assert_called_once_with(ToolPrerequisite.WEB_SEARCH_CONFIG)


def test_expired_readiness_is_refreshed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A result at its expiry boundary should be probed again."""
    registry = _registry(ttl_seconds=10)
    clock = [100.0]
    probe = MagicMock(return_value=_ready(ToolPrerequisite.CELLAR_SCHEMA))
    monkeypatch.setattr("src.agents.tools.registry.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(registry, "_probe_prerequisite", probe)

    registry._get_prerequisite_readiness(ToolPrerequisite.CELLAR_SCHEMA)
    clock[0] = 109.9
    registry._get_prerequisite_readiness(ToolPrerequisite.CELLAR_SCHEMA)
    clock[0] = 110.0
    registry._get_prerequisite_readiness(ToolPrerequisite.CELLAR_SCHEMA)

    assert probe.call_count == 2


def test_force_refresh_bypasses_fresh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Internal diagnostics should be able to force new evidence."""
    registry = _registry()
    probe = MagicMock(return_value=_ready(ToolPrerequisite.PAIRING_RULES))
    monkeypatch.setattr(registry, "_probe_prerequisite", probe)

    registry._get_prerequisite_readiness(ToolPrerequisite.PAIRING_RULES)
    registry._get_prerequisite_readiness(
        ToolPrerequisite.PAIRING_RULES,
        force_refresh=True,
    )

    assert probe.call_count == 2


def test_explicit_invalidation_clears_all_prerequisites(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalidation should force every previously cached prerequisite to refresh."""
    registry = _registry()
    probe = MagicMock(side_effect=lambda prerequisite: _ready(prerequisite))
    monkeypatch.setattr(registry, "_probe_prerequisite", probe)

    registry._get_prerequisite_readiness(ToolPrerequisite.CELLAR_SCHEMA)
    registry._get_prerequisite_readiness(ToolPrerequisite.WEB_SEARCH_CONFIG)
    registry.invalidate_readiness_cache()
    registry._get_prerequisite_readiness(ToolPrerequisite.CELLAR_SCHEMA)
    registry._get_prerequisite_readiness(ToolPrerequisite.WEB_SEARCH_CONFIG)

    assert probe.call_count == 4


def test_concurrent_callers_share_one_prerequisite_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent callers for one prerequisite should execute one probe."""
    registry = _registry()
    probe_started = threading.Event()
    allow_probe_to_finish = threading.Event()
    probe_count = 0
    probe_count_lock = threading.Lock()

    def blocking_probe(prerequisite: ToolPrerequisite) -> _PrerequisiteReadiness:
        """Hold one probe open while a second caller reaches the refresh lock."""
        nonlocal probe_count
        with probe_count_lock:
            probe_count += 1
        probe_started.set()
        assert allow_probe_to_finish.wait(timeout=2)
        return _ready(prerequisite)

    monkeypatch.setattr(registry, "_probe_prerequisite", blocking_probe)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            registry._get_prerequisite_readiness,
            ToolPrerequisite.CHROMA_COLLECTION,
        )
        assert probe_started.wait(timeout=2)
        second = executor.submit(
            registry._get_prerequisite_readiness,
            ToolPrerequisite.CHROMA_COLLECTION,
        )
        allow_probe_to_finish.set()
        assert first.result(timeout=2).available is True
        assert second.result(timeout=2).available is True

    assert probe_count == 1


def test_slow_prerequisite_does_not_block_unrelated_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-prerequisite locks should allow independent probes to run concurrently."""
    registry = _registry()
    chroma_started = threading.Event()
    release_chroma = threading.Event()

    def probe(prerequisite: ToolPrerequisite) -> _PrerequisiteReadiness:
        """Block only the Chroma prerequisite."""
        if prerequisite == ToolPrerequisite.CHROMA_COLLECTION:
            chroma_started.set()
            assert release_chroma.wait(timeout=2)
        return _ready(prerequisite)

    monkeypatch.setattr(registry, "_probe_prerequisite", probe)
    with ThreadPoolExecutor(max_workers=2) as executor:
        chroma_future = executor.submit(
            registry._get_prerequisite_readiness,
            ToolPrerequisite.CHROMA_COLLECTION,
        )
        assert chroma_started.wait(timeout=2)
        web_future = executor.submit(
            registry._get_prerequisite_readiness,
            ToolPrerequisite.WEB_SEARCH_CONFIG,
        )
        assert web_future.result(timeout=1).available is True
        release_chroma.set()
        assert chroma_future.result(timeout=2).available is True
