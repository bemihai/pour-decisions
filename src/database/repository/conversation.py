"""Async raw-SQL repository for application-owned saved conversations."""

from __future__ import annotations

import base64
import hashlib
import json
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import aiosqlite

from src.database.migrations.create_chat_sessions import validate_schema

AgentMode = Literal["intelligent", "rag_only"]
SessionState = Literal["active", "deleting"]


class ConversationStorageError(RuntimeError):
    """Base error for saved-conversation storage."""


class ConversationStorageUnavailable(ConversationStorageError):
    """The explicit application schema is absent or incompatible."""


class ConversationNotFound(ConversationStorageError):
    """The requested active conversation does not exist."""


class ConversationConflict(ConversationStorageError):
    """A mutation conflicts with current conversation state."""

    def __init__(self, code: str, session: "ConversationSession | None" = None) -> None:
        """Record a stable conflict code and optional current metadata."""
        super().__init__(code)
        self.code = code
        self.session = session


@dataclass(frozen=True)
class ConversationSession:
    """Public metadata for one non-deleted conversation."""

    id: str
    agent_mode: AgentMode
    title: str
    version: int
    state: SessionState
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ConversationTurn:
    """One public transcript turn."""

    sequence: int
    message: str
    display_message: str
    answer: str | None
    sources: list[dict]
    web_sources: list[dict]
    status: Literal["running", "completed", "failed", "interrupted"]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ConversationPage:
    """A bounded recent-first metadata page."""

    items: list[ConversationSession]
    next_cursor: str | None


@dataclass(frozen=True)
class ConversationTurnPage:
    """A bounded ascending transcript slice."""

    items: list[ConversationTurn]
    next_before_sequence: int | None
    conversation_version: int


def normalize_first_message(value: str) -> str:
    """Validate and NFC-normalize saved display text without collapsing content."""
    normalized = unicodedata.normalize("NFC", value)
    if not normalized.strip() or len(normalized) > 32_000:
        raise ValueError("first_message must contain 1 to 32000 non-whitespace characters")
    return normalized


def derive_title(first_message: str) -> str:
    """Derive a zero-inference title from normalized display text."""
    return " ".join(first_message.split())[:80]


class ConversationRepository:
    """Own one lifespan connection and explicit application transactions."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        """Initialize a validated repository connection."""
        self._connection = connection
        self._is_closed = False

    @classmethod
    async def open(cls, db_path: str | Path) -> "ConversationRepository":
        """Open existing version-1 storage without running migrations."""
        connection = await aiosqlite.connect(Path(db_path))
        connection.row_factory = aiosqlite.Row
        try:
            await connection.execute("PRAGMA foreign_keys = ON")
            async with connection.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL") as cursor:
                objects = {row[0]: row[1] for row in await cursor.fetchall()}
            version: int | None = None
            if "chat_schema_version" in objects:
                async with connection.execute("SELECT id, version FROM chat_schema_version") as cursor:
                    rows = await cursor.fetchall()
                if len(rows) == 1 and rows[0][0] == 1:
                    version = int(rows[0][1])
            validate_schema(objects, version)
        except (ValueError, aiosqlite.Error) as error:
            await connection.close()
            raise ConversationStorageUnavailable(str(error)) from error
        return cls(connection)

    async def close(self) -> None:
        """Close the lifespan-owned connection once."""
        if self._is_closed:
            return
        self._is_closed = True
        await self._connection.close()

    async def create(
        self, session_id: str, agent_mode: AgentMode, first_message: str
    ) -> tuple[ConversationSession, bool]:
        """Create idempotently and return whether a new row was inserted."""
        message = normalize_first_message(first_message)
        fingerprint = hashlib.sha256(
            json.dumps([agent_mode, message], ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        existing = await self._get_any(session_id)
        if existing is not None:
            if existing["state"] != "deleted" and existing["creation_fingerprint"] == fingerprint:
                return self._session(existing), False
            raise ConversationConflict(
                "conversation_id_conflict", self._session(existing) if existing["state"] != "deleted" else None
            )
        now = datetime.now(timezone.utc).isoformat()
        try:
            await self._connection.execute(
                "INSERT INTO chat_sessions (id, agent_mode, title, version, state, created_at, updated_at, "
                "creation_fingerprint) VALUES (?, ?, ?, 1, 'active', ?, ?, ?)",
                (session_id, agent_mode, derive_title(message), now, now, fingerprint),
            )
            await self._connection.commit()
        except aiosqlite.IntegrityError as error:
            await self._connection.rollback()
            raise ConversationConflict("conversation_id_conflict") from error
        return await self.get(session_id), True

    async def get(self, session_id: str) -> ConversationSession:
        """Return public metadata, treating deleted markers as absent."""
        row = await self._get_any(session_id)
        if row is None or row["state"] == "deleted":
            raise ConversationNotFound(session_id)
        return self._session(row)

    async def is_owned(self, session_id: str) -> bool:
        """Return whether an active, deleting or deleted marker owns this UUID."""
        return await self._get_any(session_id) is not None

    async def list(self, limit: int = 30, cursor: str | None = None) -> ConversationPage:
        """List active/deleting sessions by stable recent-first keyset pagination."""
        size = min(max(limit, 1), 100)
        args: list[object] = []
        where = "state != 'deleted'"
        if cursor is not None:
            updated_at, session_id = self._decode_cursor(cursor)
            where += " AND (updated_at < ? OR (updated_at = ? AND id < ?))"
            args.extend([updated_at, updated_at, session_id])
        args.append(size + 1)
        async with self._connection.execute(
            f"SELECT * FROM chat_sessions WHERE {where} ORDER BY updated_at DESC, id DESC LIMIT ?", args
        ) as query:
            rows = await query.fetchall()
        visible = rows[:size]
        next_cursor = self._encode_cursor(visible[-1]["updated_at"], visible[-1]["id"]) if len(rows) > size else None
        return ConversationPage([self._session(row) for row in visible], next_cursor)

    async def rename(self, session_id: str, title: str, expected_version: int) -> ConversationSession:
        """Rename one idle active session with optimistic concurrency."""
        normalized = unicodedata.normalize("NFC", title).strip()
        if not 1 <= len(normalized) <= 120:
            raise ValueError("title must contain 1 to 120 characters")
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = await self.get(session_id)
            await self._ensure_idle(session_id, current, expected_version)
            now = datetime.now(timezone.utc).isoformat()
            await self._connection.execute(
                "UPDATE chat_sessions SET title = ?, version = version + 1, updated_at = ? WHERE id = ?",
                (normalized, now, session_id),
            )
            await self._connection.commit()
        except BaseException:
            await self._connection.rollback()
            raise
        return await self.get(session_id)

    async def list_turns(
        self, session_id: str, limit: int = 50, before_sequence: int | None = None
    ) -> ConversationTurnPage:
        """Return the latest eligible slice in ascending display order."""
        session = await self.get(session_id)
        size = min(max(limit, 1), 100)
        boundary = before_sequence if before_sequence is not None else 2**63 - 1
        async with self._connection.execute(
            "SELECT * FROM chat_turns WHERE session_id = ? AND sequence < ? " "ORDER BY sequence DESC LIMIT ?",
            (session_id, boundary, size + 1),
        ) as cursor:
            rows = await cursor.fetchall()
        visible = rows[:size]
        visible.reverse()
        next_before = int(visible[0]["sequence"]) if len(rows) > size else None
        return ConversationTurnPage([self._turn(row) for row in visible], next_before, session.version)

    async def begin_delete(self, session_id: str, expected_version: int | None) -> ConversationSession | None:
        """Mark an active session deleting, or return current deleting metadata."""
        row = await self._get_any(session_id)
        if row is None or row["state"] == "deleted":
            return None
        current = self._session(row)
        if current.state == "deleting":
            return current
        if expected_version is None:
            raise ConversationConflict("expected_version_required", current)
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            await self._ensure_idle(session_id, current, expected_version)
            now = datetime.now(timezone.utc).isoformat()
            await self._connection.execute(
                "UPDATE chat_sessions SET state = 'deleting', version = version + 1, updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            await self._connection.commit()
        except BaseException:
            await self._connection.rollback()
            raise
        return await self.get(session_id)

    async def complete_delete(self, session_id: str) -> None:
        """Remove content and retain a permanent deletion marker atomically."""
        now = datetime.now(timezone.utc).isoformat()
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            await self._connection.execute("DELETE FROM chat_operations WHERE session_id = ?", (session_id,))
            await self._connection.execute("DELETE FROM chat_turns WHERE session_id = ?", (session_id,))
            await self._connection.execute("DELETE FROM conversation_threads WHERE thread_id = ?", (session_id,))
            await self._connection.execute(
                "UPDATE chat_sessions SET agent_mode = NULL, title = NULL, version = NULL, state = 'deleted', "
                "created_at = NULL, updated_at = NULL, creation_fingerprint = NULL, deleted_at = ? WHERE id = ?",
                (now, session_id),
            )
            await self._connection.commit()
        except BaseException:
            await self._connection.rollback()
            raise

    async def pending_deletions(self) -> list[str]:
        """Return durable deletion work for startup retry."""
        async with self._connection.execute("SELECT id FROM chat_sessions WHERE state = 'deleting'") as cursor:
            return [str(row[0]) for row in await cursor.fetchall()]

    async def recover_running(self) -> int:
        """Interrupt accepted work once and advance affected session versions once."""
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            async with self._connection.execute(
                "SELECT DISTINCT session_id FROM chat_operations WHERE status = 'running'"
            ) as cursor:
                session_ids = [str(row[0]) for row in await cursor.fetchall()]
            if session_ids:
                placeholders = ",".join("?" for _ in session_ids)
                now = datetime.now(timezone.utc).isoformat()
                await self._connection.execute(
                    f"UPDATE chat_sessions SET version = version + 1, updated_at = ? WHERE id IN ({placeholders})",
                    (now, *session_ids),
                )
                await self._connection.execute(
                    f"UPDATE chat_turns SET status = 'interrupted', updated_at = ? "
                    f"WHERE status = 'running' AND session_id IN ({placeholders})",
                    (now, *session_ids),
                )
                await self._connection.execute(
                    f"UPDATE chat_operations SET status = 'interrupted', error_code = 'startup_interrupted', "
                    f"conversation_version = (SELECT version FROM chat_sessions "
                    f"WHERE id = chat_operations.session_id), updated_at = ? "
                    f"WHERE status = 'running' AND session_id IN ({placeholders})",
                    (now, *session_ids),
                )
            await self._connection.commit()
            return len(session_ids)
        except BaseException:
            await self._connection.rollback()
            raise

    async def _ensure_idle(self, session_id: str, current: ConversationSession, expected_version: int) -> None:
        """Enforce lifecycle, version and running-operation preconditions."""
        if current.state != "active":
            raise ConversationConflict("conversation_deleting", current)
        if current.version != expected_version:
            raise ConversationConflict("stale_version", current)
        async with self._connection.execute(
            "SELECT 1 FROM chat_operations WHERE session_id = ? AND status = 'running'", (session_id,)
        ) as cursor:
            if await cursor.fetchone() is not None:
                raise ConversationConflict("conversation_busy", current)

    async def _get_any(self, session_id: str) -> aiosqlite.Row | None:
        """Read lifecycle storage including permanent deleted markers."""
        async with self._connection.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)) as cursor:
            return await cursor.fetchone()

    @staticmethod
    def _session(row: aiosqlite.Row) -> ConversationSession:
        """Validate a non-deleted row into public metadata."""
        return ConversationSession(
            id=str(row["id"]),
            agent_mode=row["agent_mode"],
            title=str(row["title"]),
            version=int(row["version"]),
            state=row["state"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _turn(row: aiosqlite.Row) -> ConversationTurn:
        """Decode one public transcript row."""
        return ConversationTurn(
            sequence=int(row["sequence"]),
            message=str(row["message"]),
            display_message=str(row["display_message"]),
            answer=row["answer"],
            sources=json.loads(row["sources"]),
            web_sources=json.loads(row["web_sources"]),
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _encode_cursor(updated_at: str, session_id: str) -> str:
        """Encode the exact keyset pair as an opaque URL-safe token."""
        return base64.urlsafe_b64encode(json.dumps([updated_at, session_id], separators=(",", ":")).encode()).decode()

    @staticmethod
    def _decode_cursor(cursor: str) -> tuple[str, str]:
        """Decode and validate one opaque metadata cursor."""
        try:
            value = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
            if not isinstance(value, list) or len(value) != 2 or not all(isinstance(item, str) for item in value):
                raise ValueError
            datetime.fromisoformat(value[0])
            return value[0], value[1]
        except Exception as error:
            raise ValueError("Malformed conversation cursor") from error
