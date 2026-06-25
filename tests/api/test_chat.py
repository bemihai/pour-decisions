"""Tests for chat API endpoints.

Uses FastAPI TestClient with patched agents, model, and retriever
to avoid loading real LLMs or hitting external services.
"""
from unittest.mock import MagicMock, patch

import pytest

from fastapi.testclient import TestClient

from src.api.schemas.chat import ChatResponse, InitialMessageResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _populate_state(app, *, local_model=None, cloud_model=None,
                    local_intelligent_agent=None, cloud_intelligent_agent=None,
                    retriever=None, reranker=None, config=None):
    """Set all app.state attributes that lifespan normally provides."""
    app.state.local_model = local_model
    app.state.cloud_model = cloud_model
    app.state.model = local_model or cloud_model
    app.state.local_intelligent_agent = local_intelligent_agent
    app.state.cloud_intelligent_agent = cloud_intelligent_agent
    app.state.intelligent_agent = local_intelligent_agent or cloud_intelligent_agent
    app.state.retriever = retriever
    app.state.reranker = reranker
    app.state.config = config or MagicMock()


@pytest.fixture()
def client():
    """TestClient with all state attributes set to None (no real resources)."""
    from src.api.main import app

    _populate_state(app)
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /api/chat/initial-message
# ---------------------------------------------------------------------------

class TestGetInitialMessage:

    def test_returns_welcome_message(self, client):
        resp = client.get("/api/chat/initial-message")

        assert resp.status_code == 200
        body = InitialMessageResponse(**resp.json())
        assert body.role == "assistant"
        assert body.content == "Hello. How can I help you with wine today?"


# ---------------------------------------------------------------------------
# POST /api/chat/ - intelligent mode
# ---------------------------------------------------------------------------

class TestSendMessageIntelligent:

    def test_missing_agent_returns_503(self, client):
        """When both local and cloud intelligent agents are None, 503 is returned."""
        # client fixture already sets all agents to None
        resp = client.post("/api/chat/", json={
            "message": "What wine with steak?",
            "agent_mode": "intelligent",
        })

        assert resp.status_code == 503

    def test_successful_intelligent_invocation(self, client):
        from src.api.main import app
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "final_answer": "Try a Cabernet Sauvignon.",
            "messages": [],
        }
        # Simulate local model + agent both available
        app.state.local_model = MagicMock()
        app.state.local_intelligent_agent = mock_agent

        resp = client.post("/api/chat/", json={
            "message": "What wine with steak?",
            "agent_mode": "intelligent",
            "model_provider": "local",
        })

        assert resp.status_code == 200
        body = ChatResponse(**resp.json())
        assert "Cabernet" in body.answer
        assert body.agent_mode == "intelligent"
        assert body.model_provider == "local"
        assert body.error is None

        app.state.local_model = None
        app.state.local_intelligent_agent = None

    def test_intelligent_reports_local_provider_with_local_agent_even_without_local_model(self, client):
        """Intelligent mode reports local when local intelligent agent handles the request."""
        from src.api.main import app
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "final_answer": "Try a Cabernet Sauvignon.",
            "messages": [],
        }
        app.state.local_model = None
        app.state.local_intelligent_agent = mock_agent
        app.state.cloud_intelligent_agent = None

        resp = client.post("/api/chat/", json={
            "message": "What wine with steak?",
            "agent_mode": "intelligent",
            "model_provider": "local",
        })

        assert resp.status_code == 200
        body = ChatResponse(**resp.json())
        assert body.model_provider == "local"

        app.state.local_intelligent_agent = None

    def test_successful_cloud_invocation(self, client):
        from src.api.main import app
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "final_answer": "Try a Merlot.",
            "messages": [],
        }
        app.state.cloud_intelligent_agent = mock_agent

        resp = client.post("/api/chat/", json={
            "message": "What wine with steak?",
            "agent_mode": "intelligent",
            "model_provider": "cloud",
        })

        assert resp.status_code == 200
        body = ChatResponse(**resp.json())
        assert "Merlot" in body.answer
        assert body.model_provider == "cloud"

        app.state.cloud_intelligent_agent = None

    def test_local_falls_back_to_cloud_when_local_unavailable(self, client):
        """model_provider='local' falls back to cloud agent when local is None."""
        from src.api.main import app
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"final_answer": "Barolo.", "messages": []}
        app.state.local_intelligent_agent = None
        app.state.cloud_intelligent_agent = mock_agent

        resp = client.post("/api/chat/", json={
            "message": "Recommend a wine",
            "agent_mode": "intelligent",
            "model_provider": "local",
        })

        assert resp.status_code == 200
        body = ChatResponse(**resp.json())
        assert body.model_provider == "cloud"  # reported as cloud (fallback used)

        app.state.cloud_intelligent_agent = None

    def test_agent_exception_returns_friendly_error(self, client):
        from src.api.main import app
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = RuntimeError("Unexpected error")
        app.state.local_intelligent_agent = mock_agent

        resp = client.post("/api/chat/", json={
            "message": "What wine?",
            "agent_mode": "intelligent",
            "model_provider": "local",
        })

        assert resp.status_code == 200
        body = ChatResponse(**resp.json())
        assert body.error is not None

        app.state.local_intelligent_agent = None

    def test_quota_error_returns_quota_message(self, client):
        from src.api.main import app
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")
        app.state.local_intelligent_agent = mock_agent

        resp = client.post("/api/chat/", json={
            "message": "What wine?",
            "agent_mode": "intelligent",
            "model_provider": "local",
        })

        body = ChatResponse(**resp.json())
        assert "quota" in body.error.lower()

        app.state.local_intelligent_agent = None



# ---------------------------------------------------------------------------
# POST /api/chat/ - rag_only mode
# ---------------------------------------------------------------------------

class TestSendMessageRagOnly:

    @patch("src.api.routes.chat._invoke_rag_only")
    def test_successful_rag_invocation(self, mock_rag, client):
        from src.api.main import app
        mock_rag.return_value = ("Pinot Noir is a red grape.", [], [])
        app.state.cloud_model = MagicMock()
        app.state.model = app.state.cloud_model

        resp = client.post("/api/chat/", json={
            "message": "What is Pinot Noir?",
            "agent_mode": "rag_only",
        })

        assert resp.status_code == 200
        body = ChatResponse(**resp.json())
        assert "Pinot Noir" in body.answer
        assert body.agent_mode == "rag_only"

        app.state.cloud_model = None
        app.state.model = None

    @patch("src.api.routes.chat._invoke_rag_only")
    def test_rag_with_sources(self, mock_rag, client):
        from src.api.main import app
        from src.api.schemas.chat import Source
        mock_rag.return_value = (
            "Pinot Noir [1]",
            [Source(name="wine_bible", page=42, relevance=0.95)],
            [],
        )
        app.state.cloud_model = MagicMock()
        app.state.model = app.state.cloud_model

        resp = client.post("/api/chat/", json={
            "message": "What is Pinot Noir?",
            "agent_mode": "rag_only",
        })

        body = ChatResponse(**resp.json())
        assert len(body.sources) == 1
        assert body.sources[0].name == "wine_bible"

        app.state.cloud_model = None
        app.state.model = None


# ---------------------------------------------------------------------------
# POST /api/chat/ - validation
# ---------------------------------------------------------------------------

class TestChatValidation:

    def test_empty_message_rejected(self, client):
        resp = client.post("/api/chat/", json={
            "message": "",
            "agent_mode": "rag_only",
        })
        assert resp.status_code == 422

    def test_missing_message_rejected(self, client):
        resp = client.post("/api/chat/", json={
            "agent_mode": "rag_only",
        })
        assert resp.status_code == 422

    def test_unknown_mode_rejected_with_422(self, client):
        """Invalid agent_mode returns 422 (enforced by Literal type)."""
        resp = client.post("/api/chat/", json={
            "message": "Hello",
            "agent_mode": "unknown_mode",
        })

        assert resp.status_code == 422

    def test_unknown_model_provider_rejected_with_422(self, client):
        """Invalid model_provider returns 422 (enforced by Literal type)."""
        resp = client.post("/api/chat/", json={
            "message": "Hello",
            "agent_mode": "rag_only",
            "model_provider": "openai",
        })

        assert resp.status_code == 422

    @patch("src.api.routes.chat._invoke_rag_only")
    def test_message_history_forwarded(self, mock_rag, client):
        from src.api.main import app
        mock_rag.return_value = ("Answer.", [], [])
        app.state.cloud_model = MagicMock()
        app.state.model = app.state.cloud_model
        message_history = [
            {"role": "human", "content": "Previous question"},
            {"role": "ai", "content": "Previous answer"},
        ]

        resp = client.post("/api/chat/", json={
            "message": "Follow up question",
            "agent_mode": "rag_only",
            "message_history": message_history,
        })

        assert resp.status_code == 200
        assert mock_rag.call_args.kwargs["message_history"] == message_history

        app.state.cloud_model = None
        app.state.model = None
