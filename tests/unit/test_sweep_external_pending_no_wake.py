"""The sweep must NOT re-fire a paid model step for a session parked on an
externally-executed tool call (#1710).

A session parked on an externally-executed tool call — a client ``custom`` tool
awaiting the client's result POST, or an ``always_ask`` call awaiting operator
confirmation — keeps ``open_tool_call_count > 0``, so it stays a wake candidate.
The empty-unreacted / no-in-flight branch of ``_filter_incomplete_batches`` used
to wake it on EVERY 30s sweep, each wake running a full paid model step with only
a ``_PENDING_EXTERNAL`` placeholder as new context. The fix narrows that branch
to wake only when at least one open call was actually *dispatched*.

**Wedge guard (fail toward waking too much, never too little):** the narrowing
must NEVER hold back a genuinely-dispatched call (a crashed built-in ghost) or a
CONFIRMED ``always_ask`` — a missed wake permanently wedges a months-long
session. Cases (3)-(5) pin those.

These tests drive ``_filter_incomplete_batches`` directly with a stubbed pool +
``InflightToolRegistry`` (the existing sweep-test pattern). The filter issues
five batched fetches in order: ``UNREACTED_ROWS_SQL``, ``ALL_RESULT_ROWS_SQL``,
``ALL_ASST_ROWS_SQL``, ``AGENT_SURFACE_SQL``, ``GHOST_LIFECYCLE_SQL``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.sweep import _filter_incomplete_batches
from tests.unit.conftest import fake_pool_yielding_conn

_SID = "sess_a"


def _asst_row(tool_name: str, *, call_id: str, arguments: Any = "{}") -> dict[str, Any]:
    return {
        "session_id": _SID,
        "data": {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": arguments},
                }
            ],
        },
    }


def _agent_row(tools: list[dict[str, Any]]) -> dict[str, Any]:
    return {"session_id": _SID, "tools": tools, "http_servers": []}


def _pool_for(
    *,
    asst_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]] | None = None,
    agent_rows: list[dict[str, Any]] | None = None,
    lifecycle_rows: list[dict[str, Any]] | None = None,
) -> Any:
    """Stub a pool whose connection returns the filter's five fetches in order:
    UNREACTED (always empty here — the sessions are parked with no new stimulus),
    ALL_RESULT, ALL_ASST, AGENT_SURFACE, GHOST_LIFECYCLE.
    """
    conn = MagicMock()
    conn.fetch = AsyncMock(
        side_effect=[
            [],  # UNREACTED_ROWS_SQL — no unreacted stimulus (parked)
            result_rows or [],  # ALL_RESULT_ROWS_SQL
            asst_rows,  # ALL_ASST_ROWS_SQL
            agent_rows if agent_rows is not None else [_agent_row([])],  # AGENT_SURFACE_SQL
            lifecycle_rows or [],  # GHOST_LIFECYCLE_SQL
        ]
    )
    return fake_pool_yielding_conn(conn)


# ── (1) client custom-pending → NOT woken ────────────────────────────────────


async def test_custom_pending_not_woken() -> None:
    """A client-executed ``custom`` tool awaiting the client's result POST is
    not dispatched — it must NOT be woken (the standing sweep wake is pure
    waste; the resolution POST already ``defer_wake``s)."""
    pool = _pool_for(
        asst_rows=[_asst_row("my_client_tool", call_id="tc_custom")],
        agent_rows=[
            _agent_row(
                [
                    {
                        "type": "custom",
                        "name": "my_client_tool",
                        "description": "d",
                        "input_schema": {"type": "object"},
                    }
                ]
            )
        ],
    )
    woken = await _filter_incomplete_batches(pool, InflightToolRegistry(), {_SID})
    assert woken == set(), "a client custom-pending session must not be woken"


# ── (2) unconfirmed always_ask → NOT woken ───────────────────────────────────


async def test_unconfirmed_always_ask_not_woken() -> None:
    """An ``always_ask`` call awaiting operator confirmation is not dispatched
    (no ``tool_confirmed allow`` lifecycle event) — it must NOT be woken."""
    pool = _pool_for(
        asst_rows=[_asst_row("bash", call_id="tc_ask")],
        agent_rows=[_agent_row([{"type": "bash", "permission": "always_ask"}])],
        lifecycle_rows=[],  # not confirmed
    )
    woken = await _filter_incomplete_batches(pool, InflightToolRegistry(), {_SID})
    assert woken == set(), "an unconfirmed always_ask session must not be woken"


# ── (3) WEDGE GUARD: crashed built-in ghost → STILL woken ────────────────────


async def test_crashed_builtin_ghost_still_woken() -> None:
    """A crashed-worker built-in ghost (dispatched, no result, no in-flight
    task) classifies as dispatched → STILL woken. A missed wake here would
    permanently wedge the session."""
    pool = _pool_for(
        asst_rows=[_asst_row("bash", call_id="tc_ghost")],
        agent_rows=[_agent_row([{"type": "bash"}])],  # always_allow default
    )
    woken = await _filter_incomplete_batches(pool, InflightToolRegistry(), {_SID})
    assert woken == {_SID}, "a crashed built-in ghost must STILL be woken"


# ── (4) WEDGE GUARD: confirmed always_ask → STILL woken ──────────────────────


async def test_confirmed_always_ask_still_woken() -> None:
    """Once the operator confirms (``tool_confirmed allow`` present), the
    ``always_ask`` call IS dispatched → STILL woken. This is the confirmed-
    dispatch path; the narrowing must not swallow it."""
    pool = _pool_for(
        asst_rows=[_asst_row("bash", call_id="tc_confirmed")],
        agent_rows=[_agent_row([{"type": "bash", "permission": "always_ask"}])],
        lifecycle_rows=[{"session_id": _SID, "tool_call_id": "tc_confirmed"}],
    )
    woken = await _filter_incomplete_batches(pool, InflightToolRegistry(), {_SID})
    assert woken == {_SID}, "a confirmed always_ask call must STILL be woken"


# ── (5) WEDGE GUARD: mixed open custom + open dispatched ghost → STILL woken ──


async def test_mixed_custom_and_dispatched_ghost_still_woken() -> None:
    """A session with BOTH an open client ``custom`` call and an open dispatched
    ghost must STILL be woken — ``any(...)`` over the open calls is True because
    the ghost is dispatched. The custom call alone would not wake it."""
    pool = _pool_for(
        asst_rows=[
            _asst_row("my_client_tool", call_id="tc_custom"),
            _asst_row("bash", call_id="tc_ghost"),
        ],
        agent_rows=[
            _agent_row(
                [
                    {
                        "type": "custom",
                        "name": "my_client_tool",
                        "description": "d",
                        "input_schema": {"type": "object"},
                    },
                    {"type": "bash"},
                ]
            )
        ],
    )
    woken = await _filter_incomplete_batches(pool, InflightToolRegistry(), {_SID})
    assert woken == {_SID}, "a mixed batch with a dispatched ghost must STILL be woken"
