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

from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.loop import _dispatch_confirmed_tools


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
                pool,
                "sess_x",
                account_id="acc_test_stub",
                inflight_tool_registry=InflightToolRegistry(),
            )
        assert pending == []

    async def test_returns_unwindowed_dispatchable(self) -> None:
        """The confirmed-unresolved tool_calls from the resolver are returned
        as-is when nothing is in flight — independent of window position or
        which assistant turn carries them (#737)."""
        pool = MagicMock()
        with (
            patch(
                "aios.harness.loop.sessions_service.list_confirmed_unresolved_tool_calls",
                AsyncMock(return_value=[_tool_call("tc_X"), _tool_call("tc_Y")]),
            ),
            patch(
                "aios.harness.loop.sessions_service.find_latest_interrupt_seq",
                AsyncMock(return_value=None),
            ),
        ):
            pending = await _dispatch_confirmed_tools(
                pool,
                "sess_x",
                account_id="acc_test_stub",
                inflight_tool_registry=InflightToolRegistry(),
            )
        assert [tc["id"] for tc in pending] == ["tc_X", "tc_Y"]

    async def test_skips_in_flight(self) -> None:
        """An in-flight task blocks re-dispatch of the same tool_call_id — no
        second asyncio task, no duplicate ``tool_result`` (CLAUDE.md
        invariant #4)."""
        pool = MagicMock()
        registry = InflightToolRegistry()
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        registry.add("sess_x", "tc_X", fut)  # type: ignore[arg-type]
        with (
            patch(
                "aios.harness.loop.sessions_service.list_confirmed_unresolved_tool_calls",
                AsyncMock(return_value=[_tool_call("tc_X"), _tool_call("tc_Y")]),
            ),
            patch(
                "aios.harness.loop.sessions_service.find_latest_interrupt_seq",
                AsyncMock(return_value=None),
            ),
        ):
            pending = await _dispatch_confirmed_tools(
                pool, "sess_x", account_id="acc_test_stub", inflight_tool_registry=registry
            )
        assert [tc["id"] for tc in pending] == ["tc_Y"]


class TestConfirmThenInterruptGuard:
    """Confirm-then-interrupt (#1756): a ``tool_confirmed``/``allow`` whose
    confirm-event seq is OLDER than the session's latest ``interrupt`` event
    must not cold-dispatch — it is resolved in-place as ``cancelled`` instead
    of being returned for ``launch_tool_calls``."""

    async def test_no_interrupt_dispatches_normally(self) -> None:
        """No interrupt on the session at all: every confirmed-unresolved call
        dispatches, and the confirm-seq lookup is never even needed."""
        pool = MagicMock()
        confirmed_seqs_mock = AsyncMock(return_value={})
        with (
            patch(
                "aios.harness.loop.sessions_service.list_confirmed_unresolved_tool_calls",
                AsyncMock(return_value=[_tool_call("tc_X")]),
            ),
            patch(
                "aios.harness.loop.sessions_service.find_latest_interrupt_seq",
                AsyncMock(return_value=None),
            ),
            patch(
                "aios.harness.loop.sessions_service.find_tool_confirmed_seqs",
                confirmed_seqs_mock,
            ),
        ):
            pending = await _dispatch_confirmed_tools(
                pool,
                "sess_x",
                account_id="acc_test_stub",
                inflight_tool_registry=InflightToolRegistry(),
            )
        assert [tc["id"] for tc in pending] == ["tc_X"]
        confirmed_seqs_mock.assert_not_awaited()

    async def test_confirm_before_interrupt_is_cancelled_not_dispatched(self) -> None:
        """A call confirmed at seq 5, with the session's latest interrupt at
        seq 10 (AFTER the confirm): the call must NOT be returned for
        dispatch — it is resolved in-place as ``cancelled`` instead."""
        pool = MagicMock()
        resolve_mock = AsyncMock()
        with (
            patch(
                "aios.harness.loop.sessions_service.list_confirmed_unresolved_tool_calls",
                AsyncMock(return_value=[_tool_call("tc_X")]),
            ),
            patch(
                "aios.harness.loop.sessions_service.find_latest_interrupt_seq",
                AsyncMock(return_value=10),
            ),
            patch(
                "aios.harness.loop.sessions_service.find_tool_confirmed_seqs",
                AsyncMock(return_value={"tc_X": 5}),
            ),
            patch(
                "aios.harness.loop.resolve_confirmed_call_as_cancelled",
                resolve_mock,
            ),
        ):
            pending = await _dispatch_confirmed_tools(
                pool,
                "sess_x",
                account_id="acc_test_stub",
                inflight_tool_registry=InflightToolRegistry(),
            )
        assert pending == []
        resolve_mock.assert_awaited_once()
        # The cancelled call is the one passed through — verify its id.
        await_args = resolve_mock.await_args
        assert await_args is not None
        _, _, call_arg = await_args.args
        assert call_arg["id"] == "tc_X"

    async def test_confirm_after_interrupt_still_dispatches(self) -> None:
        """A FRESH confirmation (seq 15) issued AFTER the interrupt (seq 10)
        must still dispatch normally — the #746 'fresh confirm of an old
        proposal is fresh intent' rule, applied at the interrupt boundary."""
        pool = MagicMock()
        resolve_mock = AsyncMock()
        with (
            patch(
                "aios.harness.loop.sessions_service.list_confirmed_unresolved_tool_calls",
                AsyncMock(return_value=[_tool_call("tc_X")]),
            ),
            patch(
                "aios.harness.loop.sessions_service.find_latest_interrupt_seq",
                AsyncMock(return_value=10),
            ),
            patch(
                "aios.harness.loop.sessions_service.find_tool_confirmed_seqs",
                AsyncMock(return_value={"tc_X": 15}),
            ),
            patch(
                "aios.harness.loop.resolve_confirmed_call_as_cancelled",
                resolve_mock,
            ),
        ):
            pending = await _dispatch_confirmed_tools(
                pool,
                "sess_x",
                account_id="acc_test_stub",
                inflight_tool_registry=InflightToolRegistry(),
            )
        assert [tc["id"] for tc in pending] == ["tc_X"]
        resolve_mock.assert_not_awaited()

    async def test_mixed_batch_partitions_correctly(self) -> None:
        """A batch with one stale-confirmed call and one fresh-confirmed call:
        only the stale one is cancelled, the fresh one dispatches."""
        pool = MagicMock()
        resolve_mock = AsyncMock()
        with (
            patch(
                "aios.harness.loop.sessions_service.list_confirmed_unresolved_tool_calls",
                AsyncMock(return_value=[_tool_call("tc_stale"), _tool_call("tc_fresh")]),
            ),
            patch(
                "aios.harness.loop.sessions_service.find_latest_interrupt_seq",
                AsyncMock(return_value=10),
            ),
            patch(
                "aios.harness.loop.sessions_service.find_tool_confirmed_seqs",
                AsyncMock(return_value={"tc_stale": 5, "tc_fresh": 20}),
            ),
            patch(
                "aios.harness.loop.resolve_confirmed_call_as_cancelled",
                resolve_mock,
            ),
        ):
            pending = await _dispatch_confirmed_tools(
                pool,
                "sess_x",
                account_id="acc_test_stub",
                inflight_tool_registry=InflightToolRegistry(),
            )
        assert [tc["id"] for tc in pending] == ["tc_fresh"]
        resolve_mock.assert_awaited_once()
