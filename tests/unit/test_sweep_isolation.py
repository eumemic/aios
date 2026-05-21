"""The cross-session sweep loop in ``wake_sessions_needing_inference``
and ``find_and_repair_ghosts`` must isolate per-session failures: a
single bad row (DB transient, archived-mid-sweep, etc.) must not abort
the whole batch — the remaining sessions in the same scan still need
their wake deferred or their ghost repaired."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aios.harness.sweep import find_and_repair_ghosts, wake_sessions_needing_inference
from aios.harness.task_registry import TaskRegistry
from tests.unit.conftest import fake_pool_yielding_conn


async def test_defer_wake_failure_does_not_abort_sweep_batch() -> None:
    woken_sids = ["sess_a", "sess_b", "sess_c"]

    defer_wake_mock = AsyncMock(side_effect=[None, RuntimeError("db blip"), None])
    load_account_mock = AsyncMock(return_value="acc_test")
    repair_mock = AsyncMock(return_value=[])
    find_mock = AsyncMock(return_value=set(woken_sids))

    with (
        patch("aios.harness.sweep.find_and_repair_ghosts", repair_mock),
        patch("aios.harness.sweep.find_sessions_needing_inference", find_mock),
        patch("aios.harness.sweep.sessions_service.load_session_account_id", load_account_mock),
        patch("aios.harness.sweep.defer_wake", defer_wake_mock),
    ):
        result = await wake_sessions_needing_inference(MagicMock(), TaskRegistry())

    called_sids = {call.args[1] for call in defer_wake_mock.await_args_list}
    assert called_sids == set(woken_sids), (
        f"sweep aborted mid-batch: defer_wake was called for {called_sids}, "
        f"expected {set(woken_sids)} — a per-session failure must not skip the rest"
    )
    assert result.woken_sessions == 2


async def test_ghost_repair_failure_does_not_abort_batch(monkeypatch: Any) -> None:
    ghost_rows = [
        {
            "session_id": "sess_a",
            "data": {
                "role": "assistant",
                "tool_calls": [{"id": "tc_a", "type": "function", "function": {"name": "bash"}}],
            },
        },
        {
            "session_id": "sess_b",
            "data": {
                "role": "assistant",
                "tool_calls": [{"id": "tc_b", "type": "function", "function": {"name": "bash"}}],
            },
        },
    ]
    # Mark both tool_calls as confirmed-via-lifecycle so ``_was_dispatched``
    # returns True for the (default ``always_ask``) bash builtin — both
    # candidates become ghosts.
    lifecycle_rows = [
        {"session_id": "sess_a", "tool_call_id": "tc_a"},
        {"session_id": "sess_b", "tool_call_id": "tc_b"},
    ]
    agent_rows = [
        {"session_id": "sess_a", "tools": "[]"},
        {"session_id": "sess_b", "tools": "[]"},
    ]
    # find_and_repair_ghosts issues four ``conn.fetch`` calls in order:
    # GHOST_ASST_SQL, ALL_RESULT_ROWS_SQL, GHOST_LIFECYCLE_SQL, agents.
    conn = MagicMock()
    conn.fetch = AsyncMock(side_effect=[ghost_rows, [], lifecycle_rows, agent_rows])
    pool = fake_pool_yielding_conn(conn)

    append_mock = AsyncMock(side_effect=[RuntimeError("first ghost db blip"), None])
    monkeypatch.setattr(
        "aios.harness.sweep.sessions_service.load_session_account_id",
        AsyncMock(return_value="acc_test"),
    )
    monkeypatch.setattr("aios.harness.sweep.sessions_service.append_tool_result", append_mock)

    repaired = await find_and_repair_ghosts(pool, TaskRegistry())

    assert append_mock.await_count == 2, (
        f"sweep aborted mid-batch: append_tool_result was called {append_mock.await_count} "
        f"times, expected 2 — a per-ghost failure must not skip the rest"
    )
    # The returned list reflects ghosts whose repair actually committed
    # (the second one); the first is logged-and-skipped.
    assert len(repaired) == 1
