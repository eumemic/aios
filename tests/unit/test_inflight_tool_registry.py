"""Unit tests for the InflightToolRegistry."""

from __future__ import annotations

import asyncio

from aios.harness.inflight_tool_registry import InflightToolRegistry


def _future() -> asyncio.Future[None]:
    """Standalone Future for sync tests — Python 3.14 dropped the implicit
    loop that ``asyncio.get_event_loop()`` used to autocreate."""
    return asyncio.new_event_loop().create_future()


async def _sleeper() -> None:
    await asyncio.sleep(3600)


class TestBasicLifecycle:
    def test_remove(self) -> None:
        reg = InflightToolRegistry()
        task = _future()
        reg.add("sess_1", "call_a", task)  # type: ignore[arg-type]
        reg.remove("sess_1", "call_a")
        assert reg.in_flight_tool_call_ids("sess_1") == set()

    def test_remove_nonexistent_is_noop(self) -> None:
        reg = InflightToolRegistry()
        reg.remove("sess_1", "call_nope")  # no error


class TestCancellation:
    async def test_cancel_session(self) -> None:
        reg = InflightToolRegistry()
        t1 = asyncio.create_task(_sleeper())
        t2 = asyncio.create_task(_sleeper())
        reg.add("sess_1", "call_a", t1)
        reg.add("sess_1", "call_b", t2)
        count = reg.cancel_session("sess_1")
        assert count == 2
        await asyncio.sleep(0)  # let tasks process CancelledError
        assert t1.cancelled()
        assert t2.cancelled()

    async def test_cancel_empty_session_returns_zero(self) -> None:
        reg = InflightToolRegistry()
        assert reg.cancel_session("sess_nope") == 0


class TestInFlightQueries:
    def test_in_flight_tool_call_ids_empty(self) -> None:
        reg = InflightToolRegistry()
        assert reg.in_flight_tool_call_ids("sess_nope") == set()

    def test_in_flight_tool_call_ids(self) -> None:
        reg = InflightToolRegistry()
        f1 = _future()
        f2 = _future()
        reg.add("sess_1", "call_a", f1)  # type: ignore[arg-type]
        reg.add("sess_1", "call_b", f2)  # type: ignore[arg-type]
        assert reg.in_flight_tool_call_ids("sess_1") == {"call_a", "call_b"}

    def test_in_flight_tool_call_ids_excludes_done(self) -> None:
        reg = InflightToolRegistry()
        f1 = _future()
        f2 = _future()
        f2.set_result(None)  # mark as done
        reg.add("sess_1", "call_a", f1)  # type: ignore[arg-type]
        reg.add("sess_1", "call_b", f2)  # type: ignore[arg-type]
        assert reg.in_flight_tool_call_ids("sess_1") == {"call_a"}

    def test_all_in_flight_tool_call_ids(self) -> None:
        reg = InflightToolRegistry()
        f1 = _future()
        f2 = _future()
        f3 = _future()
        f3.set_result(None)  # done — should be excluded
        reg.add("sess_1", "call_a", f1)  # type: ignore[arg-type]
        reg.add("sess_2", "call_b", f2)  # type: ignore[arg-type]
        reg.add("sess_2", "call_c", f3)  # type: ignore[arg-type]
        result = reg.all_in_flight_tool_call_ids()
        assert result == {"sess_1": {"call_a"}, "sess_2": {"call_b"}}

    def test_all_in_flight_excludes_sessions_with_no_active(self) -> None:
        reg = InflightToolRegistry()
        f1 = _future()
        f1.set_result(None)  # done
        reg.add("sess_1", "call_a", f1)  # type: ignore[arg-type]
        assert reg.all_in_flight_tool_call_ids() == {}


class TestShutdown:
    async def test_shutdown_cancels_all(self) -> None:
        reg = InflightToolRegistry()
        t1 = asyncio.create_task(_sleeper())
        t2 = asyncio.create_task(_sleeper())
        reg.add("sess_1", "call_a", t1)
        reg.add("sess_2", "call_b", t2)
        await reg.shutdown()
        assert t1.cancelled()
        assert t2.cancelled()
        assert reg.in_flight_tool_call_ids("sess_1") == set()
        assert reg.in_flight_tool_call_ids("sess_2") == set()


# ── step-task tracking ───────────────────────────────────────────────


class TestStepTracking:
    async def test_register_and_cancel_step(self) -> None:
        reg = InflightToolRegistry()
        task = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", task)
        assert reg.cancel_step("sess_1") is True
        await asyncio.sleep(0)
        assert task.cancelled()

    async def test_cancel_step_unknown_session_returns_false(self) -> None:
        reg = InflightToolRegistry()
        assert reg.cancel_step("sess_nope") is False

    async def test_unregister_step_clears(self) -> None:
        reg = InflightToolRegistry()
        task = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", task)
        reg.unregister_step("sess_1")
        assert reg.cancel_step("sess_1") is False
        task.cancel()  # cleanup
        await asyncio.sleep(0)

    async def test_register_step_replaces_prior(self) -> None:
        reg = InflightToolRegistry()
        old = asyncio.create_task(_sleeper())
        new = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", old)
        reg.register_step("sess_1", new)
        assert reg.cancel_step("sess_1") is True
        await asyncio.sleep(0)
        assert new.cancelled()
        assert not old.cancelled()
        old.cancel()
        await asyncio.sleep(0)

    async def test_step_tracking_is_independent_of_tool_tasks(self) -> None:
        reg = InflightToolRegistry()
        step = asyncio.create_task(_sleeper())
        tool = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", step)
        reg.add("sess_1", "call_a", tool)
        assert reg.cancel_step("sess_1") is True
        await asyncio.sleep(0)
        assert step.cancelled()
        assert not tool.cancelled()
        assert reg.in_flight_tool_call_ids("sess_1") == {"call_a"}
        tool.cancel()
        await asyncio.sleep(0)


# ── seq-bounded cancellation (#1756 reconnect re-drive) ────────────────


class TestSeqBoundedStepCancel:
    """``cancel_step``'s ``min_start_seq`` gate: only cancel a step that
    began BEFORE the given seq (an interrupt seq, for the reconnect
    re-drive) — never a step that began at or after it (a legitimate
    post-interrupt follow-up step)."""

    async def test_no_min_start_seq_is_unconditional(self) -> None:
        """Sole caller unaffected: the live pg-notify listener passes no
        ``min_start_seq`` and gets the pre-#1756 unconditional-cancel."""
        reg = InflightToolRegistry()
        task = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", task, start_seq=100)
        assert reg.cancel_step("sess_1") is True
        await asyncio.sleep(0)
        assert task.cancelled()

    async def test_step_older_than_min_start_seq_is_cancelled(self) -> None:
        reg = InflightToolRegistry()
        task = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", task, start_seq=5)
        assert reg.cancel_step("sess_1", min_start_seq=10) is True
        await asyncio.sleep(0)
        assert task.cancelled()

    async def test_step_at_or_after_min_start_seq_is_not_cancelled(self) -> None:
        """A step that began AT the interrupt seq (or after) is a legitimate
        post-interrupt follow-up — must never be cancelled by the redrive."""
        reg = InflightToolRegistry()
        task = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", task, start_seq=10)
        assert reg.cancel_step("sess_1", min_start_seq=10) is False
        await asyncio.sleep(0)
        assert not task.cancelled()
        task.cancel()
        await asyncio.sleep(0)

    async def test_step_with_no_start_seq_is_always_cancellable(self) -> None:
        """A step registered without ``start_seq`` (every pre-#1756 call
        site) has no ordering info — a seq-bounded cancel is conservative
        and still cancels it."""
        reg = InflightToolRegistry()
        task = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", task)
        assert reg.cancel_step("sess_1", min_start_seq=10) is True
        await asyncio.sleep(0)
        assert task.cancelled()


class TestSeqBoundedSessionCancel:
    """``cancel_session``'s ``min_start_seq`` gate, applied per-tool-task via
    the ``dispatch_seq`` captured at :meth:`InflightToolRegistry.add` time."""

    async def test_tool_dispatched_by_stale_step_is_cancelled(self) -> None:
        reg = InflightToolRegistry()
        step = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", step, start_seq=5)
        tool = asyncio.create_task(_sleeper())
        reg.add("sess_1", "call_a", tool)  # captures dispatch_seq=5
        count = reg.cancel_session("sess_1", min_start_seq=10)
        assert count == 1
        await asyncio.sleep(0)
        assert tool.cancelled()

    async def test_tool_dispatched_by_fresh_step_is_not_cancelled(self) -> None:
        """A tool launched by a step that began AT/AFTER the interrupt seq —
        the legitimate post-interrupt follow-up dispatching its own tool —
        must survive the redrive."""
        reg = InflightToolRegistry()
        step = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", step, start_seq=15)
        tool = asyncio.create_task(_sleeper())
        reg.add("sess_1", "call_a", tool)  # captures dispatch_seq=15
        count = reg.cancel_session("sess_1", min_start_seq=10)
        assert count == 0
        await asyncio.sleep(0)
        assert not tool.cancelled()
        tool.cancel()
        step.cancel()
        await asyncio.sleep(0)

    async def test_tool_with_no_dispatch_seq_is_always_cancellable(self) -> None:
        """A tool dispatched with no registered step in scope (a cold ghost
        re-park) has ``dispatch_seq=None`` — conservatively cancellable."""
        reg = InflightToolRegistry()
        tool = asyncio.create_task(_sleeper())
        reg.add("sess_1", "call_a", tool)  # no register_step call first
        count = reg.cancel_session("sess_1", min_start_seq=10)
        assert count == 1
        await asyncio.sleep(0)
        assert tool.cancelled()

    async def test_no_min_start_seq_is_unconditional(self) -> None:
        """The live pg-notify listener's call site (no ``min_start_seq``)
        keeps the pre-#1756 unconditional-cancel behavior regardless of
        dispatch_seq."""
        reg = InflightToolRegistry()
        step = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", step, start_seq=999)
        tool = asyncio.create_task(_sleeper())
        reg.add("sess_1", "call_a", tool)
        count = reg.cancel_session("sess_1")
        assert count == 1
        await asyncio.sleep(0)
        assert tool.cancelled()
        step.cancel()
        await asyncio.sleep(0)


class TestTrackedSessionIds:
    async def test_empty_registry_returns_empty_set(self) -> None:
        reg = InflightToolRegistry()
        assert reg.tracked_session_ids() == set()

    async def test_includes_sessions_with_inflight_tool_tasks(self) -> None:
        reg = InflightToolRegistry()
        tool = asyncio.create_task(_sleeper())
        reg.add("sess_1", "call_a", tool)
        assert reg.tracked_session_ids() == {"sess_1"}
        tool.cancel()
        await asyncio.sleep(0)

    async def test_includes_sessions_with_registered_steps(self) -> None:
        reg = InflightToolRegistry()
        step = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", step)
        assert reg.tracked_session_ids() == {"sess_1"}
        step.cancel()
        await asyncio.sleep(0)

    async def test_excludes_done_tasks(self) -> None:
        reg = InflightToolRegistry()
        f1 = _future()
        f1.set_result(None)
        reg.add("sess_1", "call_a", f1)  # type: ignore[arg-type]
        assert reg.tracked_session_ids() == set()

    async def test_union_across_sessions_and_kinds(self) -> None:
        reg = InflightToolRegistry()
        step = asyncio.create_task(_sleeper())
        tool = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", step)
        reg.add("sess_2", "call_a", tool)
        assert reg.tracked_session_ids() == {"sess_1", "sess_2"}
        step.cancel()
        tool.cancel()
        await asyncio.sleep(0)
