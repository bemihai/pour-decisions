"""Durable conversation-memory configuration and storage primitives."""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generic, Literal, TypeVar

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint, get_checkpoint_id
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from omegaconf import DictConfig, OmegaConf

from src.utils import get_project_root, logger


DEFAULT_CONVERSATION_MEMORY_DB_PATH = "cellar-data/conversation_memory.db"
DEFAULT_MAX_PRIOR_TURNS = 10
DEFAULT_RETENTION_DAYS = 30
DEFAULT_CLEANUP_INTERVAL_HOURS = 24
DEFAULT_LOCK_STRIPES = 64
MAX_CLEANUP_BATCH_SIZE = 100

ThreadAction = Literal["append", "replace_last"]
TurnValue = TypeVar("TurnValue")


@dataclass(frozen=True)
class SessionMemoryConfig:
    """Validated configuration for durable conversation memory."""

    enabled: bool = False
    db_path: str = DEFAULT_CONVERSATION_MEMORY_DB_PATH
    max_prior_turns: int = DEFAULT_MAX_PRIOR_TURNS
    retention_days: int = DEFAULT_RETENTION_DAYS
    cleanup_interval_hours: int = DEFAULT_CLEANUP_INTERVAL_HOURS
    lock_stripes: int = DEFAULT_LOCK_STRIPES


@dataclass(frozen=True)
class ConversationThread:
    """Application-owned pointers to a thread's committed checkpoints."""

    thread_id: str
    base_checkpoint_id: str
    active_checkpoint_id: str | None
    previous_checkpoint_id: str | None
    completed_turns: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class TurnExecution(Generic[TurnValue]):
    """A graph result paired with its successful terminal checkpoint config."""

    value: TurnValue
    checkpoint_config: RunnableConfig


TurnRunner = Callable[[RunnableConfig], Awaitable[TurnExecution[TurnValue]]]


class ConversationMemoryManager:
    """Own async checkpoint storage and committed conversation pointers."""

    def __init__(
        self,
        config: SessionMemoryConfig,
        connection: aiosqlite.Connection,
        saver: AsyncSqliteSaver,
    ) -> None:
        """Initialize a manager around a lifespan-owned SQLite connection."""
        self.config = config
        self.saver = saver
        self._connection = connection
        self._locks = tuple(asyncio.Lock() for _ in range(config.lock_stripes))
        self._cleanup_lock = asyncio.Lock()
        self._last_cleanup_at: datetime | None = None
        self._is_closed = False

    @classmethod
    async def open(
        cls,
        config: SessionMemoryConfig,
        *,
        project_root: Path | None = None,
    ) -> "ConversationMemoryManager | None":
        """Open enabled conversation storage without touching disk when disabled.

        Args:
            config: Validated session-memory settings.
            project_root: Base for a relative database path. Defaults to the repository root.

        Returns:
            A ready manager when enabled, otherwise ``None``.
        """
        if not config.enabled:
            return None

        db_path = Path(config.db_path)
        if not db_path.is_absolute():
            db_path = (project_root or get_project_root()) / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        connection = await aiosqlite.connect(db_path)
        manager = cls(config, connection, AsyncSqliteSaver(connection))
        try:
            await manager.setup()
        except BaseException:
            await connection.close()
            raise
        return manager

    async def setup(self) -> None:
        """Create saver and application metadata schemas, then run startup cleanup."""
        self._ensure_open()
        await self.saver.setup()
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_threads (
                thread_id TEXT PRIMARY KEY,
                base_checkpoint_id TEXT NOT NULL,
                active_checkpoint_id TEXT,
                previous_checkpoint_id TEXT,
                completed_turns INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        await self._connection.commit()
        await self.maybe_cleanup(force=True)

    async def close(self) -> None:
        """Close the lifespan-owned SQLite connection once."""
        if self._is_closed:
            return
        self._is_closed = True
        await self._connection.close()

    async def run_turn(
        self,
        thread_id: str,
        action: ThreadAction,
        runner: TurnRunner[TurnValue],
    ) -> TurnValue:
        """Run and commit one serialized threaded graph operation.

        The runner must return the terminal checkpoint config only after graph success. Exceptions
        and cancellations leave the application's committed pointers unchanged.

        Args:
            thread_id: Opaque external conversation identifier.
            action: Whether to append or replace the last completed turn.
            runner: Async graph operation receiving the committed starting checkpoint.

        Returns:
            The successful graph result supplied by the runner.

        Raises:
            ValueError: If the action or terminal checkpoint config is invalid.
            RuntimeError: If replacement is requested before a completed turn exists.
        """
        self._ensure_open()
        if action not in ("append", "replace_last"):
            raise ValueError(f"Unsupported thread action: {action}")

        await self.maybe_cleanup()
        async with self._lock_for(thread_id):
            thread = await self._get_or_create_thread(thread_id)
            if action == "replace_last" and thread.completed_turns == 0:
                raise RuntimeError("Cannot replace a turn before the thread has a completed turn")

            starting_checkpoint_id = (
                thread.active_checkpoint_id if action == "append" else thread.previous_checkpoint_id
            )
            if starting_checkpoint_id is None:
                raise RuntimeError("Conversation thread has no committed starting checkpoint")

            starting_config = self._checkpoint_config(thread_id, starting_checkpoint_id)
            execution = await runner(starting_config)
            terminal_checkpoint_id = await self._validate_terminal_checkpoint(
                thread_id,
                starting_checkpoint_id,
                execution.checkpoint_config,
            )
            await self._commit_turn(thread, action, terminal_checkpoint_id)
            return execution.value

    async def get_thread(self, thread_id: str) -> ConversationThread | None:
        """Return application-owned lifecycle metadata for one known thread."""
        self._ensure_open()
        async with self._connection.execute(
            """
            SELECT thread_id, base_checkpoint_id, active_checkpoint_id,
                   previous_checkpoint_id, completed_turns, created_at, updated_at
            FROM conversation_threads
            WHERE thread_id = ?
            """,
            (thread_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_thread(row) if row is not None else None

    async def delete_thread(self, thread_id: str) -> None:
        """Delete saver state before removing application metadata for a thread."""
        self._ensure_open()
        async with self._lock_for(thread_id):
            await self.saver.adelete_thread(thread_id)
            await self._connection.execute(
                "DELETE FROM conversation_threads WHERE thread_id = ?",
                (thread_id,),
            )
            await self._connection.commit()

    async def maybe_cleanup(self, *, force: bool = False, now: datetime | None = None) -> int:
        """Run bounded retention cleanup no more often than the configured interval."""
        self._ensure_open()
        current_time = self._as_utc(now or datetime.now(timezone.utc))
        async with self._cleanup_lock:
            if not force and self._last_cleanup_at is not None:
                interval = timedelta(hours=self.config.cleanup_interval_hours)
                if current_time - self._last_cleanup_at < interval:
                    return 0
            try:
                return await self.cleanup_expired(now=current_time)
            except Exception:
                logger.warning("Conversation-memory retention cleanup failed", exc_info=True)
                return 0
            finally:
                self._last_cleanup_at = current_time

    async def cleanup_expired(self, *, now: datetime | None = None) -> int:
        """Delete one bounded batch of inactive threads through public saver APIs."""
        self._ensure_open()
        current_time = self._as_utc(now or datetime.now(timezone.utc))
        cutoff = current_time - timedelta(days=self.config.retention_days)
        async with self._connection.execute(
            """
            SELECT thread_id
            FROM conversation_threads
            WHERE updated_at < ?
            ORDER BY updated_at
            LIMIT ?
            """,
            (cutoff.isoformat(), MAX_CLEANUP_BATCH_SIZE),
        ) as cursor:
            candidates = [row[0] for row in await cursor.fetchall()]

        removed = 0
        for thread_id in candidates:
            async with self._lock_for(thread_id):
                thread = await self.get_thread(thread_id)
                if thread is None or thread.updated_at >= cutoff:
                    continue
                try:
                    await self.saver.adelete_thread(thread_id)
                except Exception:
                    logger.warning("Conversation-memory retention cleanup failed", exc_info=True)
                    continue
                await self._connection.execute(
                    "DELETE FROM conversation_threads WHERE thread_id = ?",
                    (thread_id,),
                )
                await self._connection.commit()
                removed += 1
        return removed

    async def _get_or_create_thread(self, thread_id: str) -> ConversationThread:
        """Return thread metadata, creating an empty saver checkpoint when absent."""
        thread = await self.get_thread(thread_id)
        if thread is not None:
            return thread

        checkpoint = empty_checkpoint()
        saved_config = await self.saver.aput(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
            checkpoint,
            {"source": "input", "step": -1, "parents": {}},
            {},
        )
        base_checkpoint_id = get_checkpoint_id(saved_config)
        if base_checkpoint_id is None:
            await self.saver.adelete_thread(thread_id)
            raise RuntimeError("Saver did not return an empty base checkpoint ID")

        now = datetime.now(timezone.utc).isoformat()
        try:
            await self._connection.execute(
                """
                INSERT INTO conversation_threads (
                    thread_id, base_checkpoint_id, active_checkpoint_id,
                    previous_checkpoint_id, completed_turns, created_at, updated_at
                ) VALUES (?, ?, ?, NULL, 0, ?, ?)
                """,
                (thread_id, base_checkpoint_id, base_checkpoint_id, now, now),
            )
            await self._connection.commit()
        except BaseException:
            await self.saver.adelete_thread(thread_id)
            raise

        thread = await self.get_thread(thread_id)
        if thread is None:
            raise RuntimeError("Conversation metadata was not persisted")
        return thread

    async def _validate_terminal_checkpoint(
        self,
        thread_id: str,
        starting_checkpoint_id: str,
        checkpoint_config: RunnableConfig,
    ) -> str:
        """Verify that a runner returned a new persisted checkpoint for this thread."""
        configurable = checkpoint_config.get("configurable", {})
        terminal_thread_id = configurable.get("thread_id")
        terminal_checkpoint_id = get_checkpoint_id(checkpoint_config)
        if terminal_thread_id != thread_id or terminal_checkpoint_id is None:
            raise ValueError("Runner returned an invalid terminal checkpoint config")
        if terminal_checkpoint_id == starting_checkpoint_id:
            raise ValueError("Runner did not create a terminal checkpoint")
        persisted = await self.saver.aget_tuple(checkpoint_config)
        if persisted is None:
            raise ValueError("Runner terminal checkpoint was not persisted")
        return terminal_checkpoint_id

    async def _commit_turn(
        self,
        thread: ConversationThread,
        action: ThreadAction,
        terminal_checkpoint_id: str,
    ) -> None:
        """Advance successful checkpoint pointers in one metadata update."""
        now = datetime.now(timezone.utc).isoformat()
        if action == "append":
            await self._connection.execute(
                """
                UPDATE conversation_threads
                SET previous_checkpoint_id = active_checkpoint_id,
                    active_checkpoint_id = ?,
                    completed_turns = completed_turns + 1,
                    updated_at = ?
                WHERE thread_id = ? AND active_checkpoint_id = ?
                """,
                (terminal_checkpoint_id, now, thread.thread_id, thread.active_checkpoint_id),
            )
        else:
            await self._connection.execute(
                """
                UPDATE conversation_threads
                SET active_checkpoint_id = ?, updated_at = ?
                WHERE thread_id = ? AND active_checkpoint_id = ?
                """,
                (terminal_checkpoint_id, now, thread.thread_id, thread.active_checkpoint_id),
            )
        await self._connection.commit()

    def _lock_for(self, thread_id: str) -> asyncio.Lock:
        """Map an external thread identifier to one fixed lock stripe."""
        digest = hashlib.sha256(thread_id.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], byteorder="big") % len(self._locks)
        return self._locks[index]

    def _ensure_open(self) -> None:
        """Reject operations after the manager's lifespan has ended."""
        if self._is_closed:
            raise RuntimeError("Conversation memory manager is closed")

    @staticmethod
    def _checkpoint_config(thread_id: str, checkpoint_id: str) -> RunnableConfig:
        """Build the saver config for one committed checkpoint."""
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
            }
        }

    @staticmethod
    def _row_to_thread(row: tuple[object, ...]) -> ConversationThread:
        """Convert one application metadata row to its typed representation."""
        return ConversationThread(
            thread_id=str(row[0]),
            base_checkpoint_id=str(row[1]),
            active_checkpoint_id=str(row[2]) if row[2] is not None else None,
            previous_checkpoint_id=str(row[3]) if row[3] is not None else None,
            completed_turns=int(row[4]),
            created_at=datetime.fromisoformat(str(row[5])),
            updated_at=datetime.fromisoformat(str(row[6])),
        )

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        """Normalize testable timestamps to timezone-aware UTC."""
        if value.tzinfo is None:
            raise ValueError("Conversation-memory timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)


def load_session_memory_config(config: DictConfig | None) -> SessionMemoryConfig:
    """Resolve and validate the session-memory configuration.

    Args:
        config: Application configuration. Defaults are used when it is absent.

    Returns:
        Validated session-memory settings.

    Raises:
        ValueError: If a setting has an invalid type or value.
    """
    values = {
        "enabled": _select(config, "session_memory.enabled", False),
        "db_path": _select(config, "session_memory.db_path", DEFAULT_CONVERSATION_MEMORY_DB_PATH),
        "max_prior_turns": _select(config, "session_memory.max_prior_turns", DEFAULT_MAX_PRIOR_TURNS),
        "retention_days": _select(config, "session_memory.retention_days", DEFAULT_RETENTION_DAYS),
        "cleanup_interval_hours": _select(
            config,
            "session_memory.cleanup_interval_hours",
            DEFAULT_CLEANUP_INTERVAL_HOURS,
        ),
        "lock_stripes": _select(config, "session_memory.lock_stripes", DEFAULT_LOCK_STRIPES),
    }

    if type(values["enabled"]) is not bool:
        raise ValueError("session_memory.enabled must be a boolean")
    if not isinstance(values["db_path"], str) or not values["db_path"].strip():
        raise ValueError("session_memory.db_path must be a non-empty string")

    for field_name in (
        "max_prior_turns",
        "retention_days",
        "cleanup_interval_hours",
        "lock_stripes",
    ):
        value = values[field_name]
        if type(value) is not int or value < 1:
            raise ValueError(f"session_memory.{field_name} must be a positive integer")

    return SessionMemoryConfig(
        enabled=values["enabled"],
        db_path=values["db_path"].strip(),
        max_prior_turns=values["max_prior_turns"],
        retention_days=values["retention_days"],
        cleanup_interval_hours=values["cleanup_interval_hours"],
        lock_stripes=values["lock_stripes"],
    )


def _select(config: DictConfig | None, key: str, default: object) -> object:
    """Select a configuration value without requiring the section to exist."""
    return OmegaConf.select(config, key, default=default) if config is not None else default
