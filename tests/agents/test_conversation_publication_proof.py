"""Disposable proof of separate-connection publication, not a production repository.

The graph uses a fake model and the installed SQLite saver. Application SQL touches
only proof-owned tables. Closing an uncommitted connection models abandoned writes;
this is not a power-loss or filesystem-durability qualification.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypedDict

import aiosqlite
import pytest
from langchain_core.language_models.fake import FakeListLLM
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph


class ProofState(TypedDict):
    """Minimal graph state whose answer is also the public transcript."""

    prompt: str
    answer: str


class InjectedFailure(Exception):
    """Interrupt a selected persistence boundary."""


class PublicationProof:
    """Own disposable connections and an explicit candidate publication protocol."""

    def __init__(self, app: aiosqlite.Connection, saver: AsyncSqliteSaver) -> None:
        """Build a deterministic graph without external calls."""
        self.app = app
        self.saver = saver
        self.calls = 0
        self.lock = asyncio.Lock()
        model = FakeListLLM(responses=["old answer", "new answer", "other answer"])

        async def generate(state: ProofState) -> dict[str, str]:
            """Count execution independently of publication and reconciliation."""
            self.calls += 1
            return {"answer": await model.ainvoke(state["prompt"])}

        graph = StateGraph(ProofState)
        graph.add_node("generate", generate)
        graph.add_edge(START, "generate")
        graph.add_edge("generate", END)
        self.graph = graph.compile(checkpointer=saver)

    async def row(self, sql: str, args: tuple = ()) -> tuple:
        """Read one proof record and close its cursor."""
        async with self.app.execute(sql, args) as cursor:
            result = await cursor.fetchone()
        assert result is not None
        return result

    async def execute(self, config: dict[str, Any], prompt: str) -> tuple[str, str]:
        """Await graph completion and verify its terminal checkpoint via public APIs."""
        if config["configurable"]["thread_id"] == "session":
            assert not self.app.in_transaction
        result = await self.graph.ainvoke({"prompt": prompt}, config)
        latest = {"configurable": {"thread_id": config["configurable"]["thread_id"]}}
        snapshot = await self.graph.aget_state(latest)
        persisted = await self.saver.aget_tuple(snapshot.config)
        assert persisted is not None
        assert persisted.checkpoint["channel_values"]["answer"] == result["answer"]
        return snapshot.config["configurable"]["checkpoint_id"], result["answer"]

    async def seed(self) -> str:
        """Seed a completed answer and reserve one subsequent operation durably."""
        checkpoint, answer = await self.execute({"configurable": {"thread_id": "session"}}, "first")
        await self.app.executescript("""
            CREATE TABLE proof_session (id TEXT PRIMARY KEY, version INTEGER NOT NULL,
                active TEXT NOT NULL, previous TEXT);
            CREATE TABLE proof_turn (session TEXT PRIMARY KEY REFERENCES proof_session(id),
                answer TEXT NOT NULL, sources TEXT NOT NULL);
            CREATE TABLE proof_operation (id TEXT PRIMARY KEY,
                session TEXT NOT NULL REFERENCES proof_session(id), status TEXT NOT NULL, response TEXT);
            """)
        await self.app.execute("INSERT INTO proof_session VALUES ('session', 2, ?, NULL)", (checkpoint,))
        await self.app.execute("INSERT INTO proof_turn VALUES ('session', ?, '[]')", (answer,))
        await self.app.execute("INSERT INTO proof_operation VALUES ('operation', 'session', 'running', NULL)")
        await self.app.commit()
        return checkpoint

    async def publish(
        self, old: str, new: str, answer: str, fault: str, entered: asyncio.Event, release: asyncio.Event
    ) -> None:
        """Publish all application fields with a checked compare-and-swap and rollback."""
        response = json.dumps({"answer": answer, "sources": [{"name": "fixture"}]})
        async with self.lock:
            await self.app.execute("BEGIN IMMEDIATE")
            try:
                async with self.app.execute(
                    "UPDATE proof_session SET active = ?, previous = active, version = version + 1 "
                    "WHERE id = 'session' AND active = ? AND version = 2",
                    (new, old),
                ) as cursor:
                    if cursor.rowcount != 1:
                        raise InjectedFailure("conflict")
                entered.set()
                await release.wait()
                if fault == "after_pointer":
                    raise InjectedFailure(fault)
                await self.app.execute(
                    "UPDATE proof_turn SET answer = ?, sources = ? WHERE session = 'session'",
                    (answer, json.dumps([{"name": "fixture"}])),
                )
                await self.app.execute(
                    "UPDATE proof_operation SET status = 'completed', response = ? WHERE id = 'operation'",
                    (response,),
                )
                if fault == "after_result":
                    raise InjectedFailure(fault)
                await self.app.commit()
            except BaseException:
                await self.app.rollback()
                raise
        if fault == "after_commit":
            raise InjectedFailure(fault)

    async def recover(self) -> None:
        """Interrupt accepted work without promoting raw checkpoints or executing it."""
        await self.app.execute("BEGIN IMMEDIATE")
        await self.app.execute(
            "UPDATE proof_session SET version = version + 1 WHERE id IN "
            "(SELECT session FROM proof_operation WHERE status = 'running')"
        )
        await self.app.execute("UPDATE proof_operation SET status = 'interrupted' WHERE status = 'running'")
        await self.app.commit()


@asynccontextmanager
async def open_proof(path: Path) -> AsyncIterator[PublicationProof]:
    """Keep application transactions isolated from saver-owned commits."""
    async with aiosqlite.connect(path) as saver_connection, aiosqlite.connect(path) as app:
        saver = AsyncSqliteSaver(saver_connection)
        await saver.setup()
        await app.execute("PRAGMA foreign_keys = ON")
        assert (await (await app.execute("PRAGMA foreign_keys")).fetchone())[0] == 1
        yield PublicationProof(app, saver)
        assert not app.in_transaction
        assert not saver_connection.in_transaction


@pytest.mark.parametrize(
    "fault",
    [
        "before_checkpoint",
        "after_checkpoint",
        "before_publication",
        "after_pointer",
        "after_result",
        "after_commit",
        "none",
    ],
)
async def test_faults_reopen_with_matching_transcript_and_context(tmp_path: Path, fault: str) -> None:
    """No failure publishes a partial result, and reconciliation never repeats execution."""
    path = tmp_path / "proof.db"
    async with open_proof(path) as proof:
        old = await proof.seed()
        new = old
        try:
            if fault == "before_checkpoint":
                raise InjectedFailure(fault)
            new, answer = await proof.execute(
                {"configurable": {"thread_id": "session", "checkpoint_id": old}}, "second"
            )
            if fault in {"after_checkpoint", "before_publication"}:
                raise InjectedFailure(fault)
            release = asyncio.Event()
            release.set()
            await proof.publish(old, new, answer, fault, asyncio.Event(), release)
        except InjectedFailure:
            assert fault != "none"
        assert proof.calls == (1 if fault == "before_checkpoint" else 2)

    async with open_proof(path) as reopened:
        await reopened.recover()
        await reopened.recover()  # Recovery is repeatable, including its version effect.
        active, previous, version = await reopened.row("SELECT active, previous, version FROM proof_session")
        answer, sources = await reopened.row("SELECT answer, sources FROM proof_turn")
        status, response = await reopened.row("SELECT status, response FROM proof_operation")
        published = fault in {"none", "after_commit"}
        assert active == (new if published else old)
        assert previous == (old if published else None)
        assert version == 3
        assert answer == ("new answer" if published else "old answer")
        assert json.loads(sources) == ([{"name": "fixture"}] if published else [])
        assert status == ("completed" if published else "interrupted")
        assert (json.loads(response)["answer"] if response else None) == (answer if published else None)
        checkpoint = await reopened.saver.aget_tuple(
            {"configurable": {"thread_id": "session", "checkpoint_id": active}}
        )
        assert checkpoint is not None
        assert checkpoint.checkpoint["channel_values"]["answer"] == answer
        assert reopened.calls == 0


@pytest.mark.parametrize("outcome", ["commit", "rollback", "cancel", "conflict"])
async def test_interleaved_saver_cannot_publish_application_writes(tmp_path: Path, outcome: str) -> None:
    """A second session's saver waits for publication and cannot commit its partial state."""
    async with open_proof(tmp_path / "proof.db") as proof:
        old = await proof.seed()
        new, answer = await proof.execute({"configurable": {"thread_id": "session", "checkpoint_id": old}}, "second")
        entered, release = asyncio.Event(), asyncio.Event()
        if outcome == "conflict":
            release.set()
            with pytest.raises(InjectedFailure, match="conflict"):
                await proof.publish("stale", new, answer, "none", entered, release)
        else:
            publication = asyncio.create_task(
                proof.publish(old, new, answer, "after_pointer" if outcome == "rollback" else "none", entered, release)
            )
            await asyncio.wait_for(entered.wait(), 2)
            # Trace callback signals that SQLite actually attempted the competing INSERT.
            attempted = asyncio.Event()
            loop = asyncio.get_running_loop()

            def trace(statement: str) -> None:
                """Observe an attempted saver write without inspecting its tables."""
                if statement.startswith("INSERT"):
                    loop.call_soon_threadsafe(attempted.set)

            await proof.saver.conn.set_trace_callback(trace)
            other = asyncio.create_task(proof.execute({"configurable": {"thread_id": "other"}}, "other"))
            try:
                await asyncio.wait_for(attempted.wait(), 2)
                assert not other.done()
                async with aiosqlite.connect(tmp_path / "proof.db") as observer:
                    async with observer.execute("SELECT active FROM proof_session") as cursor:
                        assert await cursor.fetchone() == (old,)
                if outcome == "cancel":
                    publication.cancel()
                release.set()
                if outcome == "commit":
                    await publication
                else:
                    with pytest.raises(asyncio.CancelledError if outcome == "cancel" else InjectedFailure):
                        await publication
                other_checkpoint, other_answer = await asyncio.wait_for(other, 2)
                persisted = await proof.saver.aget_tuple(
                    {"configurable": {"thread_id": "other", "checkpoint_id": other_checkpoint}}
                )
                assert persisted is not None
                assert persisted.checkpoint["channel_values"]["answer"] == other_answer
            finally:
                release.set()
                await asyncio.gather(publication, other, return_exceptions=True)
                await proof.saver.conn.set_trace_callback(None)
        (active,) = await proof.row("SELECT active FROM proof_session")
        assert active == (new if outcome == "commit" else old)
        assert await proof.row("SELECT answer FROM proof_turn") == (
            "new answer" if outcome == "commit" else "old answer",
        )
        assert await proof.row("SELECT status FROM proof_operation") == (
            "completed" if outcome == "commit" else "running",
        )
        assert not proof.app.in_transaction
        # A new transaction succeeds after every terminal path.
        await proof.app.execute("BEGIN IMMEDIATE")
        await proof.app.rollback()


async def test_abandoned_partial_transaction_rolls_back_on_reopen(tmp_path: Path) -> None:
    """Connection loss without application rollback preserves the prior public answer."""
    path = tmp_path / "proof.db"
    async with open_proof(path) as proof:
        old = await proof.seed()
    async with aiosqlite.connect(path) as abandoned:
        await abandoned.execute("BEGIN IMMEDIATE")
        await abandoned.execute("UPDATE proof_session SET active = 'unpublished', version = 3")
        await abandoned.execute("UPDATE proof_turn SET answer = 'unpublished'")
        # Deliberately close with no commit or rollback.
    async with open_proof(path) as reopened:
        await reopened.recover()
        assert await reopened.row("SELECT active, version FROM proof_session") == (old, 3)
        assert await reopened.row("SELECT answer FROM proof_turn") == ("old answer",)
        assert await reopened.row("SELECT status FROM proof_operation") == ("interrupted",)
        assert reopened.calls == 0
