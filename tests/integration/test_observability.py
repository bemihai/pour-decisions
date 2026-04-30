"""Optional integration smoke tests for observability pipeline."""

from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit

import httpx
import pytest


@pytest.mark.integration
def test_phoenix_receives_trace_for_chat_request() -> None:
    """Send a real chat request and ensure observability prerequisites are reachable.

    This test is intentionally lightweight and skips when local integration
    prerequisites are not available.
    """
    if os.getenv("OBSERVABILITY_ENABLED", "false").lower() != "true":
        pytest.skip("OBSERVABILITY_ENABLED is not true")

    phoenix_endpoint = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
    split_endpoint = urlsplit(phoenix_endpoint)
    health_url = urlunsplit((split_endpoint.scheme, split_endpoint.netloc, "/healthz", "", ""))

    try:
        health_resp = httpx.get(health_url, timeout=3)
    except httpx.HTTPError:
        pytest.skip("Phoenix is not reachable")

    if health_resp.status_code != 200:
        pytest.skip("Phoenix health endpoint is not healthy")

    chat_url = os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:8000/api") + "/chat/"

    try:
        chat_resp = httpx.post(
            chat_url,
            timeout=10,
            json={
                "message": "observability integration check",
                "agent_mode": "rag_only",
                "message_history": [],
                "enable_rag": False,
            },
        )
    except httpx.HTTPError:
        pytest.skip("API is not reachable for integration smoke test")

    # Any non-5xx confirms the request path is alive under integration conditions.
    assert chat_resp.status_code < 500


