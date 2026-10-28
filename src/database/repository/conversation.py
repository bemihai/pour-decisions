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
OperationStatus = Literal["running", "completed", "failed", "interrupted"]


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


@dataclass(frozen=True)
class ConversationOperation:
    """Durable identity and terminal result for one accepted mutation."""

    id: str
    session_id: str
    turn_sequence: int
    action: Literal["append", "replace_last"]
    fingerprint: str
    status: OperationStatus
    response: dict | None
    error_code: str | None
    conversation_version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class OperationReservation:
    """An operation lookup result, including whether inference was newly accepted."""

    operation: ConversationOperation
    accepted: bool


def normalize_first_message(value: str) -> str:
    """Validate and NFC-normalize saved display text without collapsing content."""
    normalized = unicodedata.normalize("NFC", value)
    if not normalized.strip() or len(normalized) > 32_000:
        raise ValueError("first_message must contain 1 to 32000 non-whitespace characters")
    return normalized


def normalize_saved_message(value: str, field_name: str) -> str:
    """Validate and NFC-normalize one saved execution or display message."""
    normalized = unicodedata.normalize("NFC", value)
    if not normalized.strip() or len(normalized) > 32_000:
        raise ValueError(f"{field_name} must contain 1 to 32000 non-whitespace characters")
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

    async def reserve_operation(
        self,
        session_id: str,
        operation_id: str,
        agent_mode: AgentMode,
        action: Literal["append", "replace_last"],
        message: str,
        display_message: str,
        fingerprint: str,
        expected_version: int,
    ) -> OperationReservation:
        """Reconcile an operation identity or durably reserve new inference work."""
        execution = normalize_saved_message(message, "message")
        display = normalize_saved_message(display_message, "display_message")
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            existing = await self._get_operation_any(operation_id)
            if existing is not None:
                if existing["session_id"] == session_id and existing["fingerprint"] == fingerprint:
                    await self._connection.commit()
                    return OperationReservation(self._operation(existing), False)
                raise ConversationConflict("operation_id_conflict")

            current = await self.get(session_id)
            if current.agent_mode != agent_mode:
                raise ConversationConflict("agent_mode_mismatch", current)
            await self._ensure_idle(session_id, current, expected_version)
            now = datetime.now(timezone.utc).isoformat()
            if action == "append":
                async with self._connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM chat_turns WHERE session_id = ?", (session_id,)
                ) as cursor:
                    sequence = int((await cursor.fetchone())[0])
                await self._connection.execute(
                    "INSERT INTO chat_turns (session_id, sequence, message, display_message, status, created_at, "
                    "updated_at) VALUES (?, ?, ?, ?, 'running', ?, ?)",
                    (session_id, sequence, execution, display, now, now),
                )
            else:
                async with self._connection.execute(
                    "SELECT * FROM chat_turns WHERE session_id = ? AND status = 'completed' "
                    "ORDER BY sequence DESC LIMIT 1",
                    (session_id,),
                ) as cursor:
                    target = await cursor.fetchone()
                if target is None:
                    raise ConversationConflict("no_turn_to_replace", current)
                if target["message"] != execution or target["display_message"] != display:
                    raise ConversationConflict("replace_input_mismatch", current)
                sequence = int(target["sequence"])

            accepted_version = current.version + 1
            await self._connection.execute(
                "UPDATE chat_sessions SET version = ?, updated_at = ? WHERE id = ?",
                (accepted_version, now, session_id),
            )
            await self._connection.execute(
                "INSERT INTO chat_operations (id, session_id, turn_sequence, action, fingerprint, status, "
                "conversation_version, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)",
                (operation_id, session_id, sequence, action, fingerprint, accepted_version, now, now),
            )
            await self._connection.commit()
        except BaseException:
            await self._connection.rollback()
            raise
        return OperationReservation(await self.get_operation(session_id, operation_id), True)

    async def get_operation(self, session_id: str, operation_id: str) -> ConversationOperation:
        """Read one operation scoped to its owning conversation."""
        row = await self._get_operation_any(operation_id)
        if row is None or row["session_id"] != session_id:
            raise ConversationNotFound(operation_id)
        return self._operation(row)

    async def record_operation_failure(
        self,
        session_id: str,
        operation_id: str,
        status: Literal["failed", "interrupted"],
        error_code: str,
    ) -> ConversationOperation:
        """Finish accepted work once while preserving a replaced public turn."""
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = await self._get_operation_any(operation_id)
            if row is None or row["session_id"] != session_id:
                raise ConversationNotFound(operation_id)
            if row["status"] != "running":
                await self._connection.commit()
                return self._operation(row)
            now = datetime.now(timezone.utc).isoformat()
            await self._connection.execute(
                "UPDATE chat_sessions SET version = version + 1, updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            if row["action"] == "append":
                await self._connection.execute(
                    "UPDATE chat_turns SET status = ?, updated_at = ? WHERE session_id = ? AND sequence = ?",
                    (status, now, session_id, row["turn_sequence"]),
                )
            await self._connection.execute(
                "UPDATE chat_operations SET status = ?, error_code = ?, conversation_version = "
                "(SELECT version FROM chat_sessions WHERE id = ?), updated_at = ? WHERE id = ?",
                (status, error_code, session_id, now, operation_id),
            )
            await self._connection.commit()
        except BaseException:
            await self._connection.rollback()
            raise
        return await self.get_operation(session_id, operation_id)

    async def publish_operation(
        self,
        session_id: str,
        operation_id: str,
        starting_checkpoint_id: str,
        terminal_checkpoint_id: str,
        response: dict,
    ) -> ConversationOperation:
        """Atomically publish a finalized response and its committed checkpoint."""
        answer, sources, web_sources = self._validate_public_response(response)
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            operation_row = await self._get_operation_any(operation_id)
            if (
                operation_row is None
                or operation_row["session_id"] != session_id
                or operation_row["status"] != "running"
            ):
                raise ConversationConflict("operation_not_running")
            session = await self.get(session_id)
            if session.state != "active" or session.version != operation_row["conversation_version"]:
                raise ConversationConflict("publication_conflict", session)

            if operation_row["action"] == "append":
                cursor = await self._connection.execute(
                    "UPDATE conversation_threads SET previous_checkpoint_id = active_checkpoint_id, "
                    "active_checkpoint_id = ?, completed_turns = completed_turns + 1, updated_at = ? "
                    "WHERE thread_id = ? AND active_checkpoint_id = ?",
                    (
                        terminal_checkpoint_id,
                        datetime.now(timezone.utc).isoformat(),
                        session_id,
                        starting_checkpoint_id,
                    ),
                )
            else:
                cursor = await self._connection.execute(
                    "UPDATE conversation_threads SET active_checkpoint_id = ?, updated_at = ? "
                    "WHERE thread_id = ? AND active_checkpoint_id = ?",
                    (
                        terminal_checkpoint_id,
                        datetime.now(timezone.utc).isoformat(),
                        session_id,
                        starting_checkpoint_id,
                    ),
                )
            if cursor.rowcount != 1:
                raise ConversationConflict("checkpoint_pointer_conflict", session)

            now = datetime.now(timezone.utc).isoformat()
            terminal_version = session.version + 1
            public_response = {
                **response,
                "conversation_id": session_id,
                "operation_id": operation_id,
                "turn_sequence": int(operation_row["turn_sequence"]),
                "conversation_version": terminal_version,
            }
            encoded_response = json.dumps(public_response, ensure_ascii=False, separators=(",", ":"))
            await self._connection.execute(
                "UPDATE chat_turns SET answer = ?, sources = ?, web_sources = ?, status = 'completed', "
                "updated_at = ? WHERE session_id = ? AND sequence = ?",
                (
                    answer,
                    json.dumps(sources, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(web_sources, ensure_ascii=False, separators=(",", ":")),
                    now,
                    session_id,
                    operation_row["turn_sequence"],
                ),
            )
            await self._connection.execute(
                "UPDATE chat_sessions SET version = ?, updated_at = ? WHERE id = ?",
                (terminal_version, now, session_id),
            )
            await self._connection.execute(
                "UPDATE chat_operations SET status = 'completed', response = ?, error_code = NULL, "
                "conversation_version = ?, updated_at = ? WHERE id = ?",
                (encoded_response, terminal_version, now, operation_id),
            )
            await self._connection.commit()
        except BaseException:
            await self._connection.rollback()
            raise
        return await self.get_operation(session_id, operation_id)

    async def completed_history(self, session_id: str, before_sequence: int, max_turns: int) -> list[dict[str, str]]:
        """Build bounded model history from committed turns before a target sequence."""
        size = min(max(max_turns, 0), 100)
        async with self._connection.execute(
            "SELECT message, answer FROM chat_turns WHERE session_id = ? AND sequence < ? "
            "AND status = 'completed' ORDER BY sequence DESC LIMIT ?",
            (session_id, before_sequence, size),
        ) as cursor:
            rows = await cursor.fetchall()
        history: list[dict[str, str]] = []
        for row in reversed(rows):
            history.extend(
                ({"role": "user", "content": str(row["message"])}, {"role": "assistant", "content": str(row["answer"])})
            )
        return history

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

    async def _get_operation_any(self, operation_id: str) -> aiosqlite.Row | None:
        """Read an operation globally so identity conflicts precede version checks."""
        async with self._connection.execute("SELECT * FROM chat_operations WHERE id = ?", (operation_id,)) as cursor:
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
    def _operation(row: aiosqlite.Row) -> ConversationOperation:
        """Decode one durable operation record."""
        return ConversationOperation(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            turn_sequence=int(row["turn_sequence"]),
            action=row["action"],
            fingerprint=str(row["fingerprint"]),
            status=row["status"],
            response=json.loads(row["response"]) if row["response"] is not None else None,
            error_code=row["error_code"],
            conversation_version=int(row["conversation_version"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _validate_public_response(response: dict) -> tuple[str, list[dict], list[dict]]:
        """Validate saved response bounds before any publication write."""
        answer = response.get("answer")
        sources = response.get("sources", [])
        web_sources = response.get("web_sources", [])
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 128_000:
            raise ValueError("answer must contain 1 to 128000 non-whitespace characters")
        if not isinstance(sources, list) or len(sources) > 100 or not all(isinstance(item, dict) for item in sources):
            raise ValueError("sources must be an array of at most 100 objects")
        if (
            not isinstance(web_sources, list)
            or len(web_sources) > 100
            or not all(isinstance(item, dict) for item in web_sources)
        ):
            raise ValueError("web_sources must be an array of at most 100 objects")
        if any(not isinstance(item.get("name"), str) or len(item["name"]) > 1_024 for item in sources):
            raise ValueError("source names must contain at most 1024 characters")
        if any(
            not isinstance(item.get("title"), str)
            or len(item["title"]) > 1_024
            or not isinstance(item.get("url"), str)
            or len(item["url"]) > 2_048
            for item in web_sources
        ):
            raise ValueError("web source titles and URLs exceed saved response limits")
        try:
            json.dumps(response, ensure_ascii=False)
        except (TypeError, ValueError) as error:
            raise ValueError("response must be JSON serializable") from error
        return answer, sources, web_sources

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
