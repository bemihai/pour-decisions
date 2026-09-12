"""Tests for durable conversation-memory configuration and storage."""

import asyncio
import inspect
import re
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint
from omegaconf import OmegaConf

from src.agents.memory import (
    ConversationMemoryManager,
    SessionMemoryConfig,
    TurnExecution,
    load_session_memory_config,
)
from src.utils import get_config


@pytest.fixture
async def memory_manager(tmp_path: Path) -> AsyncIterator[ConversationMemoryManager]:
    """Yield an enabled manager backed by a temporary real SQLite saver."""
    config = SessionMemoryConfig(enabled=True, db_path=str(tmp_path / "memory.db"), lock_stripes=8)
    manager = await ConversationMemoryManager.open(config)
    assert manager is not None
    try:
        yield manager
    finally:
        await manager.close()


async def _save_terminal_checkpoint(
    manager: ConversationMemoryManager,
    starting_config: RunnableConfig,
    value: str = "complete",
) -> TurnExecution[str]:
    """Persist a child checkpoint through the real saver and return its config."""
    checkpoint_config = await manager.saver.aput(
        starting_config,
        empty_checkpoint(),
        {"source": "loop", "step": 0, "parents": {}},
        {},
    )
    return TurnExecution(value=value, checkpoint_config=checkpoint_config)


def test_session_memory_defaults_are_enabled_after_rollout() -> None:
    """The application config should enable the reviewed session-memory rollout."""
    assert load_session_memory_config(get_config()) == SessionMemoryConfig(enabled=True)


async def test_opening_disabled_config_does_not_create_database(tmp_path: Path) -> None:
    """Opening disabled storage must not create its configured database."""
    db_path = tmp_path / "conversation_memory.db"
    config = OmegaConf.create({"session_memory": {"enabled": False, "db_path": str(db_path)}})

    loaded = load_session_memory_config(config)
    manager = await ConversationMemoryManager.open(loaded)

    assert loaded.enabled is False
    assert manager is None
    assert not db_path.exists()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_prior_turns", 0),
        ("retention_days", -1),
        ("cleanup_interval_hours", 0),
        ("lock_stripes", True),
    ],
)
def test_session_memory_config_rejects_nonpositive_numbers(field_name: str, value: object) -> None:
    """Numeric settings must be positive integers rather than booleans."""
    config = OmegaConf.create({"session_memory": {field_name: value}})

    with pytest.raises(ValueError, match=field_name):
        load_session_memory_config(config)


@pytest.mark.parametrize("db_path", ["", "   ", None])
def test_session_memory_config_rejects_empty_database_path(db_path: object) -> None:
    """The conversation database path must be a non-empty string."""
    config = OmegaConf.create({"session_memory": {"db_path": db_path}})

    with pytest.raises(ValueError, match="db_path"):
        load_session_memory_config(config)


async def test_setup_and_close_manage_real_saver_lifecycle(tmp_path: Path) -> None:
    """Enabled setup should create both schemas and closing should be idempotent."""
    db_path = tmp_path / "nested" / "memory.db"
    config = SessionMemoryConfig(enabled=True, db_path=str(db_path))

    manager = await ConversationMemoryManager.open(config)

    assert manager is not None
    assert db_path.exists()
    async with manager._connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ) as cursor:
        tables = {row[0] for row in await cursor.fetchall()}
    assert {"checkpoints", "writes", "conversation_threads"}.issubset(tables)

    await manager.close()
    await manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        await manager.get_thread("closed-thread")


def test_manager_sql_does_not_query_saver_owned_tables() -> None:
    """Application SQL must leave LangGraph checkpoint and write tables private."""
    manager_source = inspect.getsource(ConversationMemoryManager)

    for table_name in ("checkpoints", "writes"):
        saver_table_query = (
            rf"\b(?:FROM|INTO|UPDATE|JOIN|DELETE FROM|TABLE(?: IF NOT EXISTS)?)\s+{table_name}\b"
        )
        assert re.search(saver_table_query, manager_source, flags=re.IGNORECASE) is None


async def test_append_advances_successful_checkpoint_pointers(
    memory_manager: ConversationMemoryManager,
) -> None:
    """A successful append should commit exactly one completed turn."""
    result = await memory_manager.run_turn(
        "append-thread",
        "append",
        lambda config: _save_terminal_checkpoint(memory_manager, config),
    )

    thread = await memory_manager.get_thread("append-thread")
    assert result == "complete"
    assert thread is not None
    assert thread.completed_turns == 1
    assert thread.previous_checkpoint_id == thread.base_checkpoint_id
    assert thread.active_checkpoint_id not in (None, thread.base_checkpoint_id)


async def test_failed_branch_does_not_advance_committed_pointers(
    memory_manager: ConversationMemoryManager,
) -> None:
    """A persisted intermediate branch must remain uncommitted when the runner fails."""
    await memory_manager.run_turn(
        "failed-thread",
        "append",
        lambda config: _save_terminal_checkpoint(memory_manager, config),
    )
    committed = await memory_manager.get_thread("failed-thread")

    async def fail_after_checkpoint(config: RunnableConfig) -> TurnExecution[str]:
        await _save_terminal_checkpoint(memory_manager, config)
        raise RuntimeError("graph failed")

    with pytest.raises(RuntimeError, match="graph failed"):
        await memory_manager.run_turn("failed-thread", "append", fail_after_checkpoint)

    assert await memory_manager.get_thread("failed-thread") == committed


async def test_replace_last_uses_predecessor_without_incrementing_turn_count(
    memory_manager: ConversationMemoryManager,
) -> None:
    """Replacement should branch from the predecessor and preserve completed-turn count."""
    await memory_manager.run_turn(
        "replace-thread",
        "append",
        lambda config: _save_terminal_checkpoint(memory_manager, config, "original"),
    )
    original = await memory_manager.get_thread("replace-thread")
    assert original is not None
    seen_start: dict[str, Any] = {}

    async def replace(config: RunnableConfig) -> TurnExecution[str]:
        seen_start.update(config["configurable"])
        return await _save_terminal_checkpoint(memory_manager, config, "replacement")

    result = await memory_manager.run_turn("replace-thread", "replace_last", replace)
    replaced = await memory_manager.get_thread("replace-thread")

    assert result == "replacement"
    assert replaced is not None
    assert seen_start["checkpoint_id"] == original.previous_checkpoint_id
    assert replaced.previous_checkpoint_id == original.previous_checkpoint_id
    assert replaced.active_checkpoint_id != original.active_checkpoint_id
    assert replaced.completed_turns == 1


async def test_replace_last_rejects_empty_thread(
    memory_manager: ConversationMemoryManager,
) -> None:
    """A thread with no completed turns has nothing to replace."""
    with pytest.raises(RuntimeError, match="before the thread has a completed turn"):
        await memory_manager.run_turn(
            "empty-thread",
            "replace_last",
            lambda config: _save_terminal_checkpoint(memory_manager, config),
        )


async def test_delete_removes_metadata_and_saver_state(
    memory_manager: ConversationMemoryManager,
) -> None:
    """Thread deletion should remove saver content before metadata and remain idempotent."""
    await memory_manager.run_turn(
        "delete-thread",
        "append",
        lambda config: _save_terminal_checkpoint(memory_manager, config),
    )

    await memory_manager.delete_thread("delete-thread")
    await memory_manager.delete_thread("delete-thread")

    assert await memory_manager.get_thread("delete-thread") is None
    assert (
        await memory_manager.saver.aget_tuple(
            {"configurable": {"thread_id": "delete-thread", "checkpoint_ns": ""}}
        )
        is None
    )


async def test_retention_deletes_only_expired_threads(
    memory_manager: ConversationMemoryManager,
) -> None:
    """Cleanup should delete inactive threads while preserving the retention boundary."""
    for thread_id in ("expired-thread", "current-thread"):
        await memory_manager.run_turn(
            thread_id,
            "append",
            lambda config: _save_terminal_checkpoint(memory_manager, config),
        )
    now = datetime.now(timezone.utc)
    expired_at = (now - timedelta(days=31)).isoformat()
    await memory_manager._connection.execute(
        "UPDATE conversation_threads SET updated_at = ? WHERE thread_id = ?",
        (expired_at, "expired-thread"),
    )
    await memory_manager._connection.commit()

    removed = await memory_manager.cleanup_expired(now=now)

    assert removed == 1
    assert await memory_manager.get_thread("expired-thread") is None
    assert await memory_manager.get_thread("current-thread") is not None


async def test_retention_failure_keeps_metadata_for_retry(
    memory_manager: ConversationMemoryManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saver deletion failure must retain application metadata for a later retry."""
    await memory_manager.run_turn(
        "retry-thread",
        "append",
        lambda config: _save_terminal_checkpoint(memory_manager, config),
    )
    now = datetime.now(timezone.utc)
    await memory_manager._connection.execute(
        "UPDATE conversation_threads SET updated_at = ? WHERE thread_id = ?",
        ((now - timedelta(days=31)).isoformat(), "retry-thread"),
    )
    await memory_manager._connection.commit()

    async def fail_delete(_thread_id: str) -> None:
        raise RuntimeError("delete failed")

    monkeypatch.setattr(memory_manager.saver, "adelete_thread", fail_delete)

    assert await memory_manager.cleanup_expired(now=now) == 0
    assert await memory_manager.get_thread("retry-thread") is not None


async def test_scheduled_cleanup_failure_does_not_fail_the_caller(
    memory_manager: ConversationMemoryManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected cleanup failures should be logged and isolated from startup or turns."""

    async def fail_cleanup(*, now: datetime | None = None) -> int:
        raise RuntimeError(f"cleanup failed at {now}")

    monkeypatch.setattr(memory_manager, "cleanup_expired", fail_cleanup)

    assert await memory_manager.maybe_cleanup(force=True) == 0


async def test_thread_metadata_survives_manager_restart(tmp_path: Path) -> None:
    """Committed pointers should be available after closing and reopening storage."""
    config = SessionMemoryConfig(enabled=True, db_path=str(tmp_path / "restart.db"))
    first = await ConversationMemoryManager.open(config)
    assert first is not None
    await first.run_turn(
        "restart-thread",
        "append",
        lambda runnable_config: _save_terminal_checkpoint(first, runnable_config),
    )
    expected = await first.get_thread("restart-thread")
    await first.close()

    second = await ConversationMemoryManager.open(config)
    assert second is not None
    try:
        assert await second.get_thread("restart-thread") == expected
    finally:
        await second.close()


async def test_same_thread_turns_are_serialized(
    memory_manager: ConversationMemoryManager,
) -> None:
    """A second operation for one thread must wait for the first complete sequence."""
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first_runner(config: RunnableConfig) -> TurnExecution[str]:
        first_entered.set()
        await release_first.wait()
        return await _save_terminal_checkpoint(memory_manager, config, "first")

    async def second_runner(config: RunnableConfig) -> TurnExecution[str]:
        second_entered.set()
        return await _save_terminal_checkpoint(memory_manager, config, "second")

    first_task = asyncio.create_task(memory_manager.run_turn("serial-thread", "append", first_runner))
    await first_entered.wait()
    second_task = asyncio.create_task(memory_manager.run_turn("serial-thread", "append", second_runner))
    await asyncio.sleep(0)
    assert not second_entered.is_set()

    release_first.set()
    assert await first_task == "first"
    assert await second_task == "second"
    assert second_entered.is_set()


async def test_different_lock_stripes_can_progress_concurrently(
    memory_manager: ConversationMemoryManager,
) -> None:
    """Unrelated threads on different stripes should enter their runners together."""
    first_thread = "parallel-0"
    second_thread = next(
        f"parallel-{index}"
        for index in range(1, 100)
        if memory_manager._lock_for(f"parallel-{index}") is not memory_manager._lock_for(first_thread)
    )
    both_entered = asyncio.Event()
    entered = 0

    async def concurrent_runner(config: RunnableConfig) -> TurnExecution[str]:
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        return await _save_terminal_checkpoint(memory_manager, config)

    results = await asyncio.gather(
        memory_manager.run_turn(first_thread, "append", concurrent_runner),
        memory_manager.run_turn(second_thread, "append", concurrent_runner),
    )

    assert results == ["complete", "complete"]
