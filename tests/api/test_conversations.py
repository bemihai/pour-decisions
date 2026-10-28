"""API contract tests for saved-conversation lifecycle endpoints."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.agents.conversation_service import ConversationService
from src.agents.memory import ConversationMemoryManager, SessionMemoryConfig
from src.database.migrations.create_chat_sessions import run_migration

CONVERSATION_ID = "123e4567-e89b-12d3-a456-426614174000"


@pytest.fixture()
async def conversation_client(tmp_path: Path):
    """Yield an API client with real disposable lifecycle storage."""
    from src.api.main import app

    path = tmp_path / "conversation.db"
    path.touch()
    run_migration(path)
    manager = await ConversationMemoryManager.open(SessionMemoryConfig(enabled=True, db_path=str(path)))
    assert manager is not None
    service = await ConversationService.open(manager)
    app.state.conversation_memory_manager = manager
    app.state.conversation_service = service
    try:
        yield TestClient(app), service
    finally:
        app.state.conversation_service = None
        app.state.conversation_memory_manager = None
        await service.close()
        await manager.close()


async def test_create_read_list_rename_and_delete_contract(conversation_client) -> None:
    """Lifecycle endpoints preserve idempotency, versions and deletion markers."""
    client, _service = conversation_client
    payload = {
        "id": CONVERSATION_ID,
        "agent_mode": "intelligent",
        "first_message": "  Compare\nthese wines  ",
    }
    created = client.post("/api/conversations", json=payload)
    repeated = client.post("/api/conversations", json=payload)
    assert created.status_code == 201
    assert repeated.status_code == 200
    assert created.json()["title"] == "Compare these wines"
    assert client.post("/api/conversations", json={**payload, "first_message": "Changed"}).status_code == 409

    listed = client.get("/api/conversations", params={"limit": 1})
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["items"]] == [CONVERSATION_ID]
    assert client.get(f"/api/conversations/{CONVERSATION_ID}").json()["version"] == 1

    renamed = client.patch(
        f"/api/conversations/{CONVERSATION_ID}", json={"title": "Dinner shortlist", "expected_version": 1}
    )
    assert renamed.status_code == 200
    assert renamed.json()["version"] == 2
    stale = client.patch(f"/api/conversations/{CONVERSATION_ID}", json={"title": "Stale", "expected_version": 1})
    assert stale.status_code == 409
    assert stale.json()["detail"]["current"]["version"] == 2

    assert client.delete(f"/api/conversations/{CONVERSATION_ID}", params={"expected_version": 2}).status_code == 204
    assert client.delete(f"/api/conversations/{CONVERSATION_ID}").status_code == 204
    assert client.get(f"/api/conversations/{CONVERSATION_ID}").status_code == 404
    assert client.get("/api/conversations").json()["items"] == []


async def test_turn_page_returns_saved_display_and_execution_input(conversation_client) -> None:
    """Transcript reads expose both approved CHAT-P1 representations."""
    client, service = conversation_client
    client.post(
        "/api/conversations",
        json={"id": CONVERSATION_ID, "agent_mode": "rag_only", "first_message": "/compare one two"},
    )
    now = datetime.now(timezone.utc).isoformat()
    await service.repository._connection.execute(
        "INSERT INTO chat_turns VALUES (?, 1, ?, ?, ?, ?, ?, 'completed', ?, ?)",
        (
            CONVERSATION_ID,
            "Expanded comparison instruction",
            "/compare one two",
            "Answer",
            '[{"name":"cellar","page":null,"relevance":null}]',
            '[{"title":"Reference","url":"https://example.test"}]',
            now,
            now,
        ),
    )
    await service.repository._connection.commit()
    response = client.get(f"/api/conversations/{CONVERSATION_ID}/turns")
    assert response.status_code == 200
    assert response.json()["items"][0]["message"] == "Expanded comparison instruction"
    assert response.json()["items"][0]["display_message"] == "/compare one two"
    assert response.json()["conversation_version"] == 1


def test_unavailable_storage_is_actionable_and_stateless_routes_remain_registered() -> None:
    """Saved lifecycle fails explicitly while unrelated application routes remain available."""
    from src.api.main import app

    app.state.conversation_service = None
    client = TestClient(app)
    response = client.get("/api/conversations")
    assert response.status_code == 503
    assert "explicit conversation migration" in response.json()["detail"]
    assert client.get("/api/chat/initial-message").status_code == 200


def test_lifecycle_schemas_forbid_extra_fields_and_validate_bounds() -> None:
    """New write contracts reject ambiguous or unbounded input."""
    from pydantic import ValidationError

    from src.api.schemas.conversations import ConversationCreate

    with pytest.raises(ValidationError):
        ConversationCreate.model_validate(
            {"id": CONVERSATION_ID, "agent_mode": "rag_only", "first_message": " ", "extra": True}
        )


def test_typescript_lifecycle_contract_tracks_backend() -> None:
    """Frontend contracts retain fields needed by the later UI phase."""
    types = Path("frontend/src/lib/types.ts").read_text()
    for contract in (
        "ConversationCreate",
        "ConversationMetadata",
        "ConversationPage",
        "ConversationTurnPage",
        "display_message: string",
        "conversation_version: number",
        "ConversationOperation",
        "response: ChatResponse | null",
    ):
        assert contract in types
