"""Safety checks for Phase 3's persistent Cloud SDK request cap."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from ollama import AsyncClient, Client

from src.eval.cloud_call_meter import CloudCallMeter
from src.eval.planning_baseline import SDKCallCapExhausted


def test_sync_and_async_sdk_calls_share_a_persistent_hard_cap(tmp_path: Path) -> None:
    """The third request never reaches either SDK transport after two reservations."""
    path = tmp_path / "meter.json"
    response = SimpleNamespace(prompt_eval_count=7, eval_count=3)
    with (
        patch.object(Client, "chat", return_value=response) as sync_chat,
        patch.object(AsyncClient, "chat", new_callable=AsyncMock, return_value=response) as async_chat,
    ):
        meter = CloudCallMeter(path, limit=2)
        with meter.count_sdk_chat():
            Client.chat(object())
            asyncio.run(AsyncClient.chat(object()))
            with pytest.raises(SDKCallCapExhausted):
                Client.chat(object())

        assert sync_chat.call_count == 1
        assert async_chat.await_count == 1
        assert meter.snapshot() == {
            "limit": 2,
            "attempts": 2,
            "token_reports": 2,
            "input_tokens_when_reported": 14,
            "output_tokens_when_reported": 6,
            "actual_billed_cost_usd": None,
        }
        with pytest.raises(SDKCallCapExhausted):
            CloudCallMeter(path, limit=2).reserve()


def test_failed_request_reservation_survives_process_restart(tmp_path: Path) -> None:
    """A transport failure still consumes its approved attempt."""
    path = tmp_path / "meter.json"
    with patch.object(Client, "chat", side_effect=RuntimeError("synthetic transport failure")):
        meter = CloudCallMeter(path, limit=1)
        with meter.count_sdk_chat(), pytest.raises(RuntimeError):
            Client.chat(object())
    assert CloudCallMeter(path, limit=1).snapshot()["attempts"] == 1
    with pytest.raises(SDKCallCapExhausted):
        CloudCallMeter(path, limit=1).reserve()
