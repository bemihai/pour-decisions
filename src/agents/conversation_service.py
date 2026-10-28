"""Lifecycle coordination for server-owned saved conversations."""

from pathlib import Path

from src.agents.memory import ConversationMemoryManager
from src.database.repository.conversation import (
    AgentMode,
    ConversationPage,
    ConversationRepository,
    ConversationSession,
    ConversationTurnPage,
)
from src.utils import get_project_root, logger


class ConversationService:
    """Coordinate lifecycle storage with supported saver operations."""

    def __init__(self, repository: ConversationRepository, memory_manager: ConversationMemoryManager) -> None:
        """Initialize around lifespan-owned resources."""
        self.repository = repository
        self.memory_manager = memory_manager
        self._is_closed = False

    @classmethod
    async def open(cls, memory_manager: ConversationMemoryManager) -> "ConversationService":
        """Open validated application storage and perform bounded startup recovery."""
        path = Path(memory_manager.config.db_path)
        if not path.is_absolute():
            path = get_project_root() / path
        repository = await ConversationRepository.open(path)
        service = cls(repository, memory_manager)
        try:
            memory_manager.bind_saved_session_checker(repository.is_owned)
            await repository.recover_running()
            for session_id in await repository.pending_deletions():
                try:
                    await service.delete(session_id, None)
                except Exception:
                    logger.warning("Saved conversation deletion recovery remains pending", exc_info=True)
        except BaseException:
            memory_manager.bind_saved_session_checker(None)
            await repository.close()
            raise
        return service

    async def close(self) -> None:
        """Detach the ownership guard and close application storage once."""
        if self._is_closed:
            return
        self._is_closed = True
        self.memory_manager.bind_saved_session_checker(None)
        await self.repository.close()

    async def delete(self, session_id: str, expected_version: int | None) -> None:
        """Make deletion durable before saver cleanup and permanent tombstoning."""
        async with self.memory_manager.lifecycle_lock(session_id):
            session = await self.repository.begin_delete(session_id, expected_version)
            if session is None:
                return
            await self.memory_manager.saver.adelete_thread(session_id)
            await self.repository.complete_delete(session_id)

    async def create(
        self, session_id: str, agent_mode: AgentMode, first_message: str
    ) -> tuple[ConversationSession, bool]:
        """Create one session without creating a turn or executing inference."""
        async with self.memory_manager.lifecycle_lock(session_id):
            return await self.repository.create(session_id, agent_mode, first_message)

    async def rename(self, session_id: str, title: str, expected_version: int) -> ConversationSession:
        """Rename under the same lifecycle lock used by future saved operations."""
        async with self.memory_manager.lifecycle_lock(session_id):
            return await self.repository.rename(session_id, title, expected_version)

    async def get(self, session_id: str) -> ConversationSession:
        """Read one session without mutating its activity or version."""
        return await self.repository.get(session_id)

    async def list(self, limit: int, cursor: str | None) -> ConversationPage:
        """Read one stable bounded metadata page."""
        return await self.repository.list(limit, cursor)

    async def list_turns(self, session_id: str, limit: int, before_sequence: int | None) -> ConversationTurnPage:
        """Read one coherent transcript page."""
        return await self.repository.list_turns(session_id, limit, before_sequence)

    async def is_owned(self, session_id: str) -> bool:
        """Check active, deleting and deleted ownership for legacy-route guards."""
        return await self.repository.is_owned(session_id)
