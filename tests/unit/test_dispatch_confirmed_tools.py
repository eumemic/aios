"""Unit tests for confirmed-tool dispatch.

Two layers:

* ``_dispatch_confirmed_tools`` (harness) — orchestration: detect
  confirmations from the unwindowed lifecycle tail, drop in-flight ids,
  delegate dispatch resolution.
* ``resolve_confirmed_dispatchable`` (service) — the unwindowed resolver
  that recovers each confirmed tool_call by ``tool_call_id`` from the log,
  independent of the token window.  This is the heart of the #737 fix.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from aios.harness.loop import _dispatch_confirmed_tools
from aios.harness.task_registry import TaskRegistry
from aios.services import sessions as sessions_service
from tests.unit.conftest import fake_pool_yielding_conn


def _confirmed(tool_call_id: str, result: str = "allow") -> SimpleNamespace:
    return SimpleNamespace(
        data={"event": "tool_confirmed", "result": result, "tool_call_id": tool_call_id}
    )


def _tool_call(tool_call_id: str, name: str = "bash") -> dict[str, Any]:
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


class TestDispatchConfirmedTools:
    """Orchestration: lifecycle-tail detection + in-flight filter, then
    delegate dispatch resolution to the unwindowed resolver."""

    async def test_returns_empty_when_no_confirmations(self) -> None:
        pool = MagicMock()
        resolve = AsyncMock(return_value=[])
        with (
            patch(
                "aios.harness.loop.sessions_service.read_events",
                AsyncMock(return_value=[]),
            ),
            patch(
                "aios.harness.loop.sessions_service.resolve_confirmed_dispatchable",
                resolve,
            ),
        ):
            pending = await _dispatch_confirmed_tools(
                pool, "sess_x", account_id="acc_test_stub", task_registry=TaskRegistry()
            )
        assert pending == []
        resolve.assert_not_awaited()

    async def test_returns_confirmed_dispatchable(self) -> None:
        """A confirmed tool with no result is resolved and returned."""
        pool = MagicMock()
        resolve = AsyncMock(return_value=[_tool_call("tc1")])
        with (
            patch(
                "aios.harness.loop.sessions_service.read_events",
                AsyncMock(return_value=[_confirmed("tc1")]),
            ),
            patch(
                "aios.harness.loop.sessions_service.resolve_confirmed_dispatchable",
                resolve,
            ),
        ):
            pending = await _dispatch_confirmed_tools(
                pool, "sess_x", account_id="acc_test_stub", task_registry=TaskRegistry()
            )
        assert [tc["id"] for tc in pending] == ["tc1"]
        assert resolve.await_args.kwargs["tool_call_ids"] == ["tc1"]

    async def test_reads_lifecycle_tail_newest_first(self) -> None:
        """Regression for #155: the lifecycle tail must be read newest-first
        so a fresh ``tool_confirmed`` isn't hidden behind 200 ancient
        ``turn_ended`` rows on a long session."""
        pool = MagicMock()
        mock_read = AsyncMock(return_value=[_confirmed("tc1")])
        with (
            patch("aios.harness.loop.sessions_service.read_events", mock_read),
            patch(
                "aios.harness.loop.sessions_service.resolve_confirmed_dispatchable",
                AsyncMock(return_value=[_tool_call("tc1")]),
            ),
        ):
            await _dispatch_confirmed_tools(
                pool, "sess_x", account_id="acc_test_stub", task_registry=TaskRegistry()
            )
        assert mock_read.call_args.kwargs["newest_first"] is True

    async def test_skips_in_flight_before_resolving(self) -> None:
        """An in-flight task blocks re-dispatch — and short-circuits before
        the resolver runs (no redundant DB work, no second asyncio task for
        the same tool_call_id; CLAUDE.md invariant #4)."""
        pool = MagicMock()
        registry = TaskRegistry()
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        registry.add("sess_x", "tc1", fut)  # type: ignore[arg-type]
        resolve = AsyncMock(return_value=[])
        with (
            patch(
                "aios.harness.loop.sessions_service.read_events",
                AsyncMock(return_value=[_confirmed("tc1")]),
            ),
            patch(
                "aios.harness.loop.sessions_service.resolve_confirmed_dispatchable",
                resolve,
            ),
        ):
            pending = await _dispatch_confirmed_tools(
                pool, "sess_x", account_id="acc_test_stub", task_registry=registry
            )
        assert pending == []
        resolve.assert_not_awaited()

    async def test_dedupes_confirmed_ids_preserving_order(self) -> None:
        """Multiple allow events for the same id collapse to one candidate;
        order follows first appearance in the (newest-first) tail."""
        pool = MagicMock()
        resolve = AsyncMock(return_value=[])
        with (
            patch(
                "aios.harness.loop.sessions_service.read_events",
                AsyncMock(
                    return_value=[
                        _confirmed("tc2"),
                        _confirmed("tc1"),
                        _confirmed("tc2"),  # duplicate
                    ]
                ),
            ),
            patch(
                "aios.harness.loop.sessions_service.resolve_confirmed_dispatchable",
                resolve,
            ),
        ):
            await _dispatch_confirmed_tools(
                pool, "sess_x", account_id="acc_test_stub", task_registry=TaskRegistry()
            )
        assert resolve.await_args.kwargs["tool_call_ids"] == ["tc2", "tc1"]

    async def test_dispatches_confirmed_when_parent_scrolled_out_of_window(self) -> None:
        """End-to-end regression for #737: a confirmed ``always_ask`` tool
        whose parent assistant has scrolled out of the token window is still
        dispatched.

        Exercises the REAL resolver (only the DB queries are mocked), so a
        regression that re-sources dispatch from the windowed message slice —
        which by construction lacks the scrolled-out parent — would surface
        here as an empty dispatch.  ``lookup_tool_call_by_call_id`` returning
        the call models 'parent gone from the window but still in the log';
        the operator's allow is visible via the unwindowed lifecycle tail."""
        pool = cast("Any", fake_pool_yielding_conn(MagicMock()))
        with (
            patch(
                "aios.harness.loop.sessions_service.read_events",
                AsyncMock(return_value=[_confirmed("tc_X")]),
            ),
            patch(
                "aios.services.sessions.queries.find_tool_result_event",
                AsyncMock(return_value=None),
            ),
            patch(
                "aios.services.sessions.queries.lookup_tool_call_by_call_id",
                AsyncMock(return_value=_tool_call("tc_X")),
            ),
        ):
            pending = await _dispatch_confirmed_tools(
                pool, "sess_x", account_id="acc_test_stub", task_registry=TaskRegistry()
            )
        assert [tc["id"] for tc in pending] == ["tc_X"], (
            "confirmed always_ask tool_call whose parent assistant scrolled "
            "out of the token window was never dispatched (#737)"
        )


class TestResolveConfirmedDispatchable:
    """The unwindowed resolver — the heart of the #737 fix.

    Dispatch is sourced by ``tool_call_id`` from the full log, NOT from the
    token-windowed slice, so a confirmed ``always_ask`` tool whose parent
    assistant has scrolled past ``window_max`` is still recovered."""

    async def test_recovers_parent_tool_call_unwindowed(self) -> None:
        """Regression for #737: a confirmed tool whose parent assistant has
        scrolled out of the token window is recovered via the unwindowed
        parent-by-tool_call_id lookup and returned for dispatch.

        ``lookup_tool_call_by_call_id`` returning the call models 'A1 is
        gone from the window but still in the log'; ``find_tool_result_event``
        returning None models 'no result yet'.  Pre-fix, dispatch read the
        windowed slice (which lacks A1) and returned nothing."""
        pool = cast("Any", fake_pool_yielding_conn(MagicMock()))
        with (
            patch(
                "aios.services.sessions.queries.find_tool_result_event",
                AsyncMock(return_value=None),
            ),
            patch(
                "aios.services.sessions.queries.lookup_tool_call_by_call_id",
                AsyncMock(return_value=_tool_call("tc_X")),
            ),
        ):
            dispatchable = await sessions_service.resolve_confirmed_dispatchable(
                pool, "sess_x", tool_call_ids=["tc_X"], account_id="acc_test_stub"
            )
        assert [tc["id"] for tc in dispatchable] == ["tc_X"]

    async def test_skips_when_result_already_exists(self) -> None:
        """A confirmed id whose ``tool_result`` exists in the log (even if it
        scrolled out of the window) is NOT re-dispatched — no duplicate
        ``tool_result`` (CLAUDE.md invariant #4).  The parent lookup is not
        even reached."""
        pool = cast("Any", fake_pool_yielding_conn(MagicMock()))
        lookup = AsyncMock(return_value=_tool_call("tc_X"))
        with (
            patch(
                "aios.services.sessions.queries.find_tool_result_event",
                AsyncMock(return_value=SimpleNamespace(data={})),
            ),
            patch("aios.services.sessions.queries.lookup_tool_call_by_call_id", lookup),
        ):
            dispatchable = await sessions_service.resolve_confirmed_dispatchable(
                pool, "sess_x", tool_call_ids=["tc_X"], account_id="acc_test_stub"
            )
        assert dispatchable == []
        lookup.assert_not_awaited()

    async def test_skips_when_no_parent_found(self) -> None:
        """No parent assistant for the id (should not happen in practice) →
        nothing to dispatch, no crash."""
        pool = cast("Any", fake_pool_yielding_conn(MagicMock()))
        with (
            patch(
                "aios.services.sessions.queries.find_tool_result_event",
                AsyncMock(return_value=None),
            ),
            patch(
                "aios.services.sessions.queries.lookup_tool_call_by_call_id",
                AsyncMock(return_value=None),
            ),
        ):
            dispatchable = await sessions_service.resolve_confirmed_dispatchable(
                pool, "sess_x", tool_call_ids=["tc_missing"], account_id="acc_test_stub"
            )
        assert dispatchable == []

    async def test_resolves_multiple_in_order(self) -> None:
        """Several confirmed ids resolve to their calls, order preserved."""
        pool = cast("Any", fake_pool_yielding_conn(MagicMock()))
        with (
            patch(
                "aios.services.sessions.queries.find_tool_result_event",
                AsyncMock(return_value=None),
            ),
            patch(
                "aios.services.sessions.queries.lookup_tool_call_by_call_id",
                AsyncMock(side_effect=[_tool_call("tc_a"), _tool_call("tc_b")]),
            ),
        ):
            dispatchable = await sessions_service.resolve_confirmed_dispatchable(
                pool,
                "sess_x",
                tool_call_ids=["tc_a", "tc_b"],
                account_id="acc_test_stub",
            )
        assert [tc["id"] for tc in dispatchable] == ["tc_a", "tc_b"]
