"""Phase 4 storage measurements and disposable recovery rehearsal."""

import json
import sqlite3
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint

from src.agents.conversation_service import ConversationService
from src.agents.memory import (
    ConversationMemoryManager,
    ConversationThread,
    SessionMemoryConfig,
    ThreadAction,
    TurnExecution,
)
from src.database.migrations.create_chat_sessions import SCHEMA, run_migration, validate_schema
from src.utils import logger


async def _save_checkpoint(
    manager: ConversationMemoryManager,
    config: RunnableConfig,
    answer: str,
) -> TurnExecution[str]:
    """Persist one deterministic terminal checkpoint without model inference."""
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"messages": ["question", answer]}
    saved = await manager.saver.aput(
        config,
        checkpoint,
        {"source": "loop", "step": 1, "parents": {}},
        {},
    )
    return TurnExecution(value=answer, checkpoint_config=saved)


async def _complete_turn(
    manager: ConversationMemoryManager,
    service: ConversationService,
    session_id: str,
    sequence: int,
) -> None:
    """Reserve and atomically publish one minimal intelligent turn."""
    operation_id = f"{session_id}-operation-{sequence}"
    session = await service.get(session_id)
    await service.reserve_operation(
        session_id,
        operation_id,
        "intelligent",
        "append",
        f"execution input {sequence}",
        f"display input {sequence}",
        session.version,
        "cloud",
        True,
        5,
    )

    async def publish(
        thread: ConversationThread,
        _action: ThreadAction,
        terminal_checkpoint_id: str,
        answer: str,
    ) -> None:
        await service.repository.publish_operation(
            session_id,
            operation_id,
            thread.active_checkpoint_id,
            terminal_checkpoint_id,
            {
                "answer": answer,
                "sources": [{"name": "fixture", "page": sequence}],
                "web_sources": [],
                "agent_mode": "intelligent",
                "model_provider": "cloud",
            },
        )

    async with manager.lifecycle_lock(session_id):
        await manager.run_turn_locked(
            session_id,
            "append",
            lambda config: _save_checkpoint(manager, config, f"answer {sequence}"),
            publish,
        )


def _allocated_bytes(path: Path) -> int:
    """Return SQLite's allocated main-database bytes after a checkpoint."""
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        page_count = connection.execute("PRAGMA page_count").fetchone()[0]
        page_size = connection.execute("PRAGMA page_size").fetchone()[0]
    return page_count * page_size


async def _measure_database(path: Path, session_turns: dict[str, int]) -> dict[str, object]:
    """Build deterministic saved sessions and return reproducible size evidence."""
    path.touch()
    run_migration(path)
    manager = await ConversationMemoryManager.open(SessionMemoryConfig(enabled=True, db_path=str(path)))
    assert manager is not None
    service = await ConversationService.open(manager)
    samples: dict[int, int] = {}
    try:
        for session_id, turn_count in session_turns.items():
            await service.create(session_id, "intelligent", f"First message for {session_id}")
            for sequence in range(1, turn_count + 1):
                await _complete_turn(manager, service, session_id, sequence)
                if len(session_turns) == 1 and sequence in {10, 50, 200}:
                    samples[sequence] = _allocated_bytes(path)
        with sqlite3.connect(path) as connection:
            row_counts = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("chat_sessions", "chat_turns", "chat_operations", "conversation_threads")
            }
    finally:
        await service.close()
        await manager.close()
    return {"allocated_bytes": _allocated_bytes(path), "samples": samples, "row_counts": row_counts}


async def test_saved_conversation_storage_measurements_are_bounded_and_reproducible(tmp_path: Path) -> None:
    """Record 10/50/200-turn and multi-session storage without inference."""
    single = await _measure_database(tmp_path / "single.db", {"measurement": 200})
    multiple = await _measure_database(
        tmp_path / "multiple.db",
        {f"session-{index}": 10 for index in range(5)},
    )

    samples = single["samples"]
    assert isinstance(samples, dict)
    assert set(samples) == {10, 50, 200}
    assert samples[10] <= samples[50] <= samples[200]
    assert single["row_counts"] == {
        "chat_sessions": 1,
        "chat_turns": 200,
        "chat_operations": 200,
        "conversation_threads": 1,
    }
    assert multiple["row_counts"] == {
        "chat_sessions": 5,
        "chat_turns": 50,
        "chat_operations": 50,
        "conversation_threads": 5,
    }
    assert multiple["allocated_bytes"] > 0
    bytes_per_turn = {turns: round(size / turns, 2) for turns, size in samples.items()}
    logger.info(
        "Saved conversation Phase 4 storage measurement: %s",
        json.dumps(
            {
                "single_session": {**single, "allocated_bytes_per_turn": bytes_per_turn},
                "five_sessions": multiple,
            },
            sort_keys=True,
        ),
    )


def _backup(source_path: Path, backup_path: Path) -> None:
    """Create a consistent SQLite backup, including committed WAL content."""
    with sqlite3.connect(source_path) as source, sqlite3.connect(backup_path) as destination:
        source.backup(destination)


def _validate_database(path: Path, *, expect_chat_schema: bool) -> None:
    """Validate integrity, foreign keys, and optional chat schema identity."""
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        objects = dict(connection.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"))
        if expect_chat_schema:
            version = connection.execute("SELECT version FROM chat_schema_version WHERE id = 1").fetchone()[0]
            validate_schema(objects, version)
        else:
            assert set(SCHEMA).isdisjoint(objects)


def test_wal_backup_migration_and_restore_rehearsal_preserves_each_compatibility_point(tmp_path: Path) -> None:
    """A disposable rehearsal restores exact pre-upgrade and upgraded database pairs."""
    database = tmp_path / "conversation.db"
    pre_upgrade_backup = tmp_path / "pre-upgrade.db"
    upgraded_backup = tmp_path / "upgraded.db"
    restored_pre_upgrade = tmp_path / "restored-pre-upgrade.db"
    restored_upgraded = tmp_path / "restored-upgraded.db"

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute("CREATE TABLE conversation_threads (thread_id TEXT PRIMARY KEY, payload TEXT)")
        connection.execute("INSERT INTO conversation_threads VALUES ('legacy', 'committed in WAL')")
        connection.commit()
        assert database.with_name(database.name + "-wal").is_file()
        _backup(database, pre_upgrade_backup)

    _validate_database(pre_upgrade_backup, expect_chat_schema=False)
    run_migration(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO chat_sessions VALUES (?, 'rag_only', 'Saved', 1, 'active', ?, ?, ?, NULL)",
            ("saved", "2026-10-28T00:00:00+00:00", "2026-10-28T00:00:00+00:00", "fingerprint"),
        )
        connection.commit()
    _backup(database, upgraded_backup)
    _validate_database(upgraded_backup, expect_chat_schema=True)

    _backup(pre_upgrade_backup, restored_pre_upgrade)
    _backup(upgraded_backup, restored_upgraded)
    _validate_database(restored_pre_upgrade, expect_chat_schema=False)
    _validate_database(restored_upgraded, expect_chat_schema=True)
    with sqlite3.connect(restored_pre_upgrade) as connection:
        assert connection.execute("SELECT * FROM conversation_threads").fetchall() == [("legacy", "committed in WAL")]
    with sqlite3.connect(restored_upgraded) as connection:
        assert connection.execute("SELECT id, title FROM chat_sessions").fetchall() == [("saved", "Saved")]
        assert connection.execute("SELECT * FROM conversation_threads").fetchall() == [("legacy", "committed in WAL")]


def test_restored_unsupported_schema_is_rejected_before_saved_storage_opens(tmp_path: Path) -> None:
    """New code refuses an unknown schema instead of silently repairing it."""
    database = tmp_path / "unsupported.db"
    database.touch()
    run_migration(database)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute("UPDATE chat_schema_version SET version = 2")
        connection.commit()
    with pytest.raises(ValueError, match="missing or incompatible"):
        run_migration(database)
