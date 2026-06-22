"""Unit tests for the cancel tool handler."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from aios.harness import runtime
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.tools.cancel import CancelArgumentError, cancel_handler


async def _sleeper() -> None:
    await asyncio.sleep(3600)


@pytest.fixture
def inflight_reg() -> Any:
    """Install a InflightToolRegistry on the runtime module, restore after."""
    previous = runtime.inflight_tool_registry
    reg = InflightToolRegistry()
    runtime.inflight_tool_registry = reg
    try:
        yield reg
    finally:
        runtime.inflight_tool_registry = previous


class TestCancelSpecificTask:
    async def test_cancel_existing_task(self, inflight_reg: InflightToolRegistry) -> None:
        task = asyncio.create_task(_sleeper())
        inflight_reg.add("sess_01TEST", "call_abc", task)
        result = await cancel_handler("sess_01TEST", {"tool_call_id": "call_abc"})
        assert result["cancelled"] is True
        assert result["tool_call_id"] == "call_abc"
        await asyncio.sleep(0)  # let the task process CancelledError
        assert task.cancelled()

    async def test_cancel_missing_task(self, inflight_reg: InflightToolRegistry) -> None:
        result = await cancel_handler("sess_01TEST", {"tool_call_id": "call_nope"})
        assert result["cancelled"] is False

    async def test_non_string_tool_call_id_raises(self, inflight_reg: InflightToolRegistry) -> None:
        with pytest.raises(CancelArgumentError):
            await cancel_handler("sess_01TEST", {"tool_call_id": 42})


class TestCancelAllTasks:
    async def test_cancel_all(self, inflight_reg: InflightToolRegistry) -> None:
        t1 = asyncio.create_task(_sleeper())
        t2 = asyncio.create_task(_sleeper())
        inflight_reg.add("sess_01TEST", "call_a", t1)
        inflight_reg.add("sess_01TEST", "call_b", t2)
        result = await cancel_handler("sess_01TEST", {})
        assert result["cancelled"] is True
        assert result["count"] == 2

    async def test_cancel_all_empty(self, inflight_reg: InflightToolRegistry) -> None:
        result = await cancel_handler("sess_01TEST", {})
        assert result["cancelled"] is True
        assert result["count"] == 0

    async def test_cancel_excludes_own_running_task(
        self, inflight_reg: InflightToolRegistry
    ) -> None:
        """The in-band cancel tool runs as a registered task in its own
        session set, so cancel-all must skip it. If it cancels itself the
        count is inflated by one and a CancelledError fired at the
        post-handler await overwrites the tool's real {cancelled, count}
        result with a generic "cancelled" error. The interrupt-listener
        caller runs from a non-registered task, so it is unaffected."""
        sid = "sess_01TEST"
        sibling = asyncio.create_task(_sleeper())
        inflight_reg.add(sid, "call_bash", sibling)

        captured: dict[str, Any] = {}

        async def run_cancel() -> None:
            # Mirrors dispatch: this coroutine *is* the registered cancel
            # task, so asyncio.current_task() inside the handler is the
            # call_cancel entry in the session set.
            captured["result"] = await cancel_handler(sid, {})
            # The post-handler await where a self-inflicted cancel lands.
            await asyncio.sleep(0)
            captured["reached"] = True

        own = asyncio.create_task(run_cancel())
        inflight_reg.add(sid, "call_cancel", own)

        with contextlib.suppress(asyncio.CancelledError):
            await own
        with contextlib.suppress(asyncio.CancelledError):
            await sibling

        assert not own.cancelled()  # the cancel tool must not cancel itself
        assert captured.get("reached") is True  # ran past the post-handler await
        assert captured["result"] == {"cancelled": True, "count": 1}  # sibling only
        assert sibling.cancelled()  # the real target is still cancelled
