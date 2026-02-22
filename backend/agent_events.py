"""In-memory agent status tracking and SSE streaming helpers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from collections.abc import AsyncGenerator
from collections.abc import Callable

agent_status: dict = {"steps": [], "done": False}


def reset_agent_status() -> None:
    agent_status["steps"] = []
    agent_status["done"] = False


def push_agent_step(step_type: str, message: str, detail: str = "") -> None:
    agent_status["steps"].append(
        {
            "type": step_type,
            "message": message,
            "detail": detail,
        }
    )


def mark_agent_done() -> None:
    agent_status["done"] = True


async def stream_agent_status(
    should_stop: Callable[[], Awaitable[bool]] | None = None,
) -> AsyncGenerator[str, None]:
    """Yield server-sent events for the latest search run."""
    last_index = 0
    try:
        while True:
            if should_stop is not None and await should_stop():
                break

            steps = agent_status.get("steps", [])
            while last_index < len(steps):
                if should_stop is not None and await should_stop():
                    break
                yield f"data: {json.dumps(steps[last_index])}\n\n"
                last_index += 1

            if agent_status.get("done", False):
                done_event = {"type": "done", "message": "Search complete", "detail": ""}
                yield f"data: {json.dumps(done_event)}\n\n"
                break

            await asyncio.sleep(0.3)
    except asyncio.CancelledError:
        return
    except Exception:
        return
