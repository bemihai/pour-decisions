"""Private SDK-call accounting for bounded direct Cloud evaluation runs."""

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from ollama import AsyncClient, Client

from src.eval.planning_baseline import SDKCallCapExhausted


class CloudCallMeter:
    """Persist conservative chat-attempt counts across sequential eval commands.

    The caller must run only one live eval process at a time. A reservation is
    persisted before the SDK request, so interrupted attempts remain counted.
    """

    def __init__(self, path: Path, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("The Cloud SDK call cap must be positive")
        self.path = path
        self.limit = limit
        self._lock = threading.Lock()
        if path.exists():
            state = self.snapshot()
            if state["attempts"] > limit:
                raise ValueError("The Cloud SDK call cap is below already reserved attempts")

    def snapshot(self) -> dict[str, int | None]:
        """Read only safe counters; no prompts, responses, or credentials are stored."""
        if not self.path.exists():
            return {
                "limit": self.limit,
                "attempts": 0,
                "token_reports": 0,
                "input_tokens_when_reported": 0,
                "output_tokens_when_reported": 0,
                "actual_billed_cost_usd": None,
            }
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(state.get("attempts"), int):
            raise ValueError("Cloud SDK call meter is malformed")
        return state

    def _write(self, state: dict[str, int | None]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def reserve(self) -> None:
        """Persist an attempt before allowing the next SDK request."""
        with self._lock:
            state = self.snapshot()
            if state["attempts"] >= self.limit:
                raise SDKCallCapExhausted("Phase 3 Cloud SDK call cap exhausted")
            state["limit"] = self.limit
            state["attempts"] += 1
            self._write(state)

    def observe(self, response: Any) -> None:
        """Record token counts only if the provider reported both values."""
        input_count = getattr(response, "prompt_eval_count", None)
        output_count = getattr(response, "eval_count", None)
        if not isinstance(input_count, int) or not isinstance(output_count, int):
            return
        with self._lock:
            state = self.snapshot()
            state["token_reports"] += 1
            state["input_tokens_when_reported"] += input_count
            state["output_tokens_when_reported"] += output_count
            self._write(state)

    @contextmanager
    def count_sdk_chat(self) -> Iterator[None]:
        """Count sync and async native Ollama chat requests within one run."""
        original_sync = Client.chat
        original_async = AsyncClient.chat

        def counted_sync(client: Client, *args: Any, **kwargs: Any) -> Any:
            self.reserve()
            response = original_sync(client, *args, **kwargs)
            self.observe(response)
            return response

        async def counted_async(client: AsyncClient, *args: Any, **kwargs: Any) -> Any:
            self.reserve()
            response = await original_async(client, *args, **kwargs)
            self.observe(response)
            return response

        with patch.object(Client, "chat", counted_sync), patch.object(AsyncClient, "chat", counted_async):
            yield
