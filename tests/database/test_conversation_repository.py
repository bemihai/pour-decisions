"""Repository and lifecycle tests against disposable real SQLite storage."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.agents.conversation_service import ConversationService
from src.agents.memory import ConversationMemoryManager, SavedSessionThreadError, SessionMemoryConfig
from src.database.migrations.create_chat_sessions import run_migration
from src.database.repository.conversation import ConversationConflict, ConversationNotFound


@pytest.fixture()
async def lifecycle(tmp_path: Path):
    """Yield a manager and service backed by separate connections."""
    path = tmp_path / "conversation.db"
    path.touch()
    run_migration(path)
    manager = await ConversationMemoryManager.open(SessionMemoryConfig(enabled=True, db_path=str(path)))
    assert manager is not None
    service = await ConversationService.open(manager)
    try:
        yield manager, service
    finally:
        await service.close()
        await manager.close()


async def test_create_is_idempotent_and_title_uses_normalized_display_text(lifecycle) -> None:
    """Creation fingerprints exact display input while titles collapse whitespace."""
    _manager, service = lifecycle
    session, created = await service.create("one", "intelligent", "  Compare\n these wines  ")
    repeated, repeated_created = await service.create("one", "intelligent", "  Compare\n these wines  ")
    assert created is True
    assert repeated_created is False
    assert session == repeated
    assert session.title == "Compare these wines"
    with pytest.raises(ConversationConflict, match="conversation_id_conflict"):
        await service.create("one", "intelligent", "Changed")


async def test_metadata_and_turn_pagination_are_bounded_and_stable(lifecycle) -> None:
    """Metadata uses keyset cursors and transcript pages retain ascending order."""
    _manager, service = lifecycle
    for index in range(4):
        await service.create(f"session-{index}", "rag_only", f"Question {index}")
        timestamp = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)).isoformat()
        await service.repository._connection.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (timestamp, f"session-{index}")
        )
    await service.repository._connection.commit()
    first = await service.repository.list(limit=2)
    second = await service.repository.list(limit=2, cursor=first.next_cursor)
    assert [item.id for item in first.items] == ["session-3", "session-2"]
    assert [item.id for item in second.items] == ["session-1", "session-0"]
    assert second.next_cursor is None
    with pytest.raises(ValueError, match="Malformed"):
        await service.repository.list(cursor="not-a-cursor")

    now = datetime.now(timezone.utc).isoformat()
    for sequence in range(1, 5):
        await service.repository._connection.execute(
            "INSERT INTO chat_turns VALUES (?, ?, ?, ?, ?, '[]', '[]', 'completed', ?, ?)",
            ("session-3", sequence, f"execution {sequence}", f"display {sequence}", f"answer {sequence}", now, now),
        )
    await service.repository._connection.commit()
    newest = await service.repository.list_turns("session-3", limit=2)
    older = await service.repository.list_turns("session-3", limit=2, before_sequence=newest.next_before_sequence)
    assert [turn.sequence for turn in newest.items] == [3, 4]
    assert [turn.sequence for turn in older.items] == [1, 2]
    assert newest.conversation_version == 1


async def test_rename_conflicts_expose_current_metadata_and_busy_state(lifecycle) -> None:
    """Rename enforces expected version and refuses accepted running work."""
    _manager, service = lifecycle
    session, _ = await service.create("rename", "intelligent", "Original")
    renamed = await service.rename("rename", "New title", session.version)
    assert renamed.title == "New title"
    assert renamed.version == 2
    with pytest.raises(ConversationConflict) as stale:
        await service.rename("rename", "Stale", 1)
    assert stale.value.code == "stale_version"
    assert stale.value.session == renamed

    now = datetime.now(timezone.utc).isoformat()
    await service.repository._connection.execute(
        "INSERT INTO chat_turns VALUES ('rename', 1, 'execution', 'display', NULL, '[]', '[]', 'running', ?, ?)",
        (now, now),
    )
    await service.repository._connection.execute(
        "INSERT INTO chat_operations VALUES ('operation', 'rename', 1, 'append', 'fingerprint', "
        "'running', NULL, NULL, 3, ?, ?)",
        (now, now),
    )
    await service.repository._connection.commit()
    with pytest.raises(ConversationConflict) as busy:
        await service.rename("rename", "Busy", 2)
    assert busy.value.code == "conversation_busy"


async def test_operation_reservation_reconciles_identity_before_stale_version(lifecycle) -> None:
    """The same semantic operation never accepts inference twice."""
    _manager, service = lifecycle
    await service.create("saved", "rag_only", "First")
    reserved = await service.reserve_operation(
        "saved", "op-1", "rag_only", "append", "exact query", "Visible query", 1, None, True, 5
    )
    assert reserved.accepted is True
    assert reserved.operation.turn_sequence == 1
    assert reserved.operation.conversation_version == 2

    duplicate = await service.reserve_operation(
        "saved", "op-1", "rag_only", "append", "exact query", "Visible query", 1, None, True, 5
    )
    assert duplicate.accepted is False
    assert duplicate.operation == reserved.operation
    with pytest.raises(ConversationConflict) as reused:
        await service.reserve_operation(
            "saved", "op-1", "rag_only", "append", "changed", "Visible query", 2, None, True, 5
        )
    assert reused.value.code == "operation_id_conflict"

    with pytest.raises(ConversationConflict) as busy:
        await service.reserve_operation("saved", "op-2", "rag_only", "append", "next", "Next", 2, None, True, 5)
    assert busy.value.code == "conversation_busy"


async def test_operation_failure_is_terminal_and_advances_version_once(lifecycle) -> None:
    """Failure records a repeatable terminal result without enabling replay."""
    _manager, service = lifecycle
    await service.create("saved", "intelligent", "First")
    reserved = await service.reserve_operation(
        "saved", "op-1", "intelligent", "append", "query", "Query", 1, "ollama", True, 5
    )
    failed = await service.repository.record_operation_failure("saved", "op-1", "failed", "inference_failed")
    repeated = await service.repository.record_operation_failure("saved", "op-1", "failed", "ignored")
    assert failed == repeated
    assert failed.status == "failed"
    assert failed.error_code == "inference_failed"
    assert failed.conversation_version == 3
    assert (await service.get("saved")).version == 3
    assert (await service.list_turns("saved", 10, None)).items[0].status == "failed"
    reconciled = await service.reserve_operation(
        "saved", "op-1", "intelligent", "append", "query", "Query", 1, "ollama", True, 5
    )
    assert reconciled.accepted is False
    assert reconciled.operation == failed
    assert reserved.operation.conversation_version == 2


async def test_replace_failure_preserves_completed_turn_and_history_is_bounded(lifecycle) -> None:
    """Replacement is exact-input only and failures keep the old public answer."""
    _manager, service = lifecycle
    await service.create("saved", "rag_only", "First")
    now = datetime.now(timezone.utc).isoformat()
    for sequence, status in ((1, "completed"), (2, "failed"), (3, "completed")):
        answer = f"answer {sequence}" if status == "completed" else None
        await service.repository._connection.execute(
            "INSERT INTO chat_turns VALUES (?, ?, ?, ?, ?, '[]', '[]', ?, ?, ?)",
            ("saved", sequence, f"execution {sequence}", f"display {sequence}", answer, status, now, now),
        )
    await service.repository._connection.commit()
    history = await service.repository.completed_history("saved", 4, 1)
    assert history == [
        {"role": "user", "content": "execution 3"},
        {"role": "assistant", "content": "answer 3"},
    ]

    with pytest.raises(ConversationConflict) as mismatch:
        await service.reserve_operation(
            "saved", "replace-bad", "rag_only", "replace_last", "changed", "display 3", 1, None, True, 5
        )
    assert mismatch.value.code == "replace_input_mismatch"
    reserved = await service.reserve_operation(
        "saved", "replace", "rag_only", "replace_last", "execution 3", "display 3", 1, None, True, 5
    )
    assert reserved.operation.turn_sequence == 3
    await service.repository.record_operation_failure("saved", "replace", "interrupted", "client_disconnected")
    latest = (await service.list_turns("saved", 10, None)).items[-1]
    assert latest.status == "completed"
    assert latest.answer == "answer 3"


async def test_restart_interrupts_running_once_without_promoting_context(tmp_path: Path) -> None:
    """Startup recovery reaches a repeatable terminal state and advances version once."""
    path = tmp_path / "conversation.db"
    path.touch()
    run_migration(path)
    manager = await ConversationMemoryManager.open(SessionMemoryConfig(enabled=True, db_path=str(path)))
    assert manager is not None
    service = await ConversationService.open(manager)
    await service.create("recover", "rag_only", "Recover")
    now = datetime.now(timezone.utc).isoformat()
    await service.repository._connection.execute("UPDATE chat_sessions SET version = 2 WHERE id = 'recover'")
    await service.repository._connection.execute(
        "INSERT INTO chat_turns VALUES ('recover', 1, 'execution', 'display', NULL, '[]', '[]', 'running', ?, ?)",
        (now, now),
    )
    await service.repository._connection.execute(
        "INSERT INTO chat_operations VALUES ('operation', 'recover', 1, 'append', 'fingerprint', "
        "'running', NULL, NULL, 2, ?, ?)",
        (now, now),
    )
    await service.repository._connection.commit()
    await service.close()
    reopened = await ConversationService.open(manager)
    assert (await reopened.repository.get("recover")).version == 3
    await reopened.repository.recover_running()
    assert (await reopened.repository.get("recover")).version == 3
    async with reopened.repository._connection.execute(
        "SELECT status, error_code, conversation_version FROM chat_operations"
    ) as cursor:
        assert tuple(await cursor.fetchone()) == ("interrupted", "startup_interrupted", 3)
    await reopened.close()
    await manager.close()


async def test_delete_retries_after_saver_failure_and_tombstone_blocks_reuse(lifecycle) -> None:
    """A failed cleanup remains deleting and startup retry leaves a permanent marker."""
    manager, service = lifecycle
    session, _ = await service.create("delete", "intelligent", "Delete me")
    original_delete = manager.saver.adelete_thread
    manager.saver.adelete_thread = AsyncMock(side_effect=RuntimeError("injected"))
    with pytest.raises(RuntimeError, match="injected"):
        await service.delete("delete", session.version)
    assert (await service.repository.get("delete")).state == "deleting"
    manager.saver.adelete_thread = original_delete
    await service.delete("delete", None)
    with pytest.raises(ConversationNotFound):
        await service.repository.get("delete")
    assert await service.repository.is_owned("delete") is True
    with pytest.raises(ConversationConflict):
        await service.create("delete", "intelligent", "Resurrect")


async def test_saved_ids_reject_legacy_mutation_and_legacy_cleanup(lifecycle) -> None:
    """Saved ownership protects active and deleted IDs while legacy IDs still work."""
    manager, service = lifecycle
    session, _ = await service.create("saved", "intelligent", "Saved")
    with pytest.raises(SavedSessionThreadError):
        await manager.delete_thread("saved")
    await service.delete("saved", session.version)
    with pytest.raises(SavedSessionThreadError):
        await manager.delete_thread("saved")

    old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    await manager._connection.execute(
        "INSERT INTO conversation_threads VALUES ('legacy', 'base', 'base', NULL, 0, ?, ?)", (old, old)
    )
    await manager._connection.commit()
    assert await manager.cleanup_expired() == 1
    assert await manager.get_thread("legacy") is None
