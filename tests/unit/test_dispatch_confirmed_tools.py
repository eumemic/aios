"""Unit tests for ``_dispatch_confirmed_tools`` — the harness orchestration
that filters in-flight tools out of the unwindowed confirmed-unresolved set
resolved by ``sessions_service.list_confirmed_unresolved_tool_calls``.

The resolver's SQL — which mirrors the sweep's case-(c) predicate and recovers
the parent ``tool_call`` regardless of window position or which assistant turn
carries it — is covered by the integration test
``tests/integration/test_confirmed_unresolved_dispatch.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aios.harness.loop import _dispatch_confirmed_tools
from aios.harness.task_registry import TaskRegistry


def _tool_call(tool_call_id: str, name: str = "bash") -> dict[str, Any]:
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


class TestDispatchConfirmedTools:
    async def test_returns_empty_when_none_unresolved(self) -> None:
        pool = MagicMock()
        with patch(
            "aios.harness.loop.sessions_service.list_confirmed_unresolved_tool_calls",
            AsyncMock(return_value=[]),
        ):
            pending = await _dispatch_confirmed_tools(
                pool, "sess_x", account_id="acc_test_stub", task_registry=TaskRegistry()
            )
        assert pending == []

    async def test_returns_unwindowed_dispatchable(self) -> None:
        """The confirmed-unresolved tool_calls from the resolver are returned
        as-is when nothing is in flight — independent of window position or
        which assistant turn carries them (#737)."""
        pool = MagicMock()
        with patch(
            "aios.harness.loop.sessions_service.list_confirmed_unresolved_tool_calls",
            AsyncMock(return_value=[_tool_call("tc_X"), _tool_call("tc_Y")]),
        ):
            pending = await _dispatch_confirmed_tools(
                pool, "sess_x", account_id="acc_test_stub", task_registry=TaskRegistry()
            )
        assert [tc["id"] for tc in pending] == ["tc_X", "tc_Y"]

    async def test_skips_in_flight(self) -> None:
        """An in-flight task blocks re-dispatch of the same tool_call_id — no
        second asyncio task, no duplicate ``tool_result`` (CLAUDE.md
        invariant #4)."""
        pool = MagicMock()
        registry = TaskRegistry()
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        registry.add("sess_x", "tc_X", fut)  # type: ignore[arg-type]
        with patch(
            "aios.harness.loop.sessions_service.list_confirmed_unresolved_tool_calls",
            AsyncMock(return_value=[_tool_call("tc_X"), _tool_call("tc_Y")]),
        ):
            pending = await _dispatch_confirmed_tools(
                pool, "sess_x", account_id="acc_test_stub", task_registry=registry
            )
        assert [tc["id"] for tc in pending] == ["tc_Y"]
