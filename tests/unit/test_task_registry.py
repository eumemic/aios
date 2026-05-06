"""Unit tests for the TaskRegistry."""

from __future__ import annotations

import asyncio

from aios.harness.task_registry import TaskRegistry


def _future() -> asyncio.Future[None]:
    """Standalone Future for sync tests — Python 3.14 dropped the implicit
    loop that ``asyncio.get_event_loop()`` used to autocreate."""
    return asyncio.new_event_loop().create_future()


async def _sleeper() -> None:
    await asyncio.sleep(3600)


class TestBasicLifecycle:
    def test_add_and_count(self) -> None:
        reg = TaskRegistry()
        task = _future()
        reg.add("sess_1", "call_a", task)  # type: ignore[arg-type]
        assert reg.in_flight_count("sess_1") == 1

    def test_remove(self) -> None:
        reg = TaskRegistry()
        task = _future()
        reg.add("sess_1", "call_a", task)  # type: ignore[arg-type]
        reg.remove("sess_1", "call_a")
        assert reg.in_flight_count("sess_1") == 0

    def test_remove_nonexistent_is_noop(self) -> None:
        reg = TaskRegistry()
        reg.remove("sess_1", "call_nope")  # no error

    def test_count_unknown_session_is_zero(self) -> None:
        reg = TaskRegistry()
        assert reg.in_flight_count("sess_nope") == 0


class TestCancellation:
    async def test_cancel_task(self) -> None:
        reg = TaskRegistry()
        task = asyncio.create_task(_sleeper())
        reg.add("sess_1", "call_a", task)
        assert reg.cancel_task("sess_1", "call_a") is True
        await asyncio.sleep(0)  # let the task process CancelledError
        assert task.cancelled()

    async def test_cancel_unknown_returns_false(self) -> None:
        reg = TaskRegistry()
        assert reg.cancel_task("sess_1", "call_nope") is False

    async def test_cancel_session(self) -> None:
        reg = TaskRegistry()
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
        reg = TaskRegistry()
        assert reg.cancel_session("sess_nope") == 0


class TestInFlightQueries:
    def test_in_flight_tool_call_ids_empty(self) -> None:
        reg = TaskRegistry()
        assert reg.in_flight_tool_call_ids("sess_nope") == set()

    def test_in_flight_tool_call_ids(self) -> None:
        reg = TaskRegistry()
        f1 = _future()
        f2 = _future()
        reg.add("sess_1", "call_a", f1)  # type: ignore[arg-type]
        reg.add("sess_1", "call_b", f2)  # type: ignore[arg-type]
        assert reg.in_flight_tool_call_ids("sess_1") == {"call_a", "call_b"}

    def test_in_flight_tool_call_ids_excludes_done(self) -> None:
        reg = TaskRegistry()
        f1 = _future()
        f2 = _future()
        f2.set_result(None)  # mark as done
        reg.add("sess_1", "call_a", f1)  # type: ignore[arg-type]
        reg.add("sess_1", "call_b", f2)  # type: ignore[arg-type]
        assert reg.in_flight_tool_call_ids("sess_1") == {"call_a"}

    def test_all_in_flight_tool_call_ids(self) -> None:
        reg = TaskRegistry()
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
        reg = TaskRegistry()
        f1 = _future()
        f1.set_result(None)  # done
        reg.add("sess_1", "call_a", f1)  # type: ignore[arg-type]
        assert reg.all_in_flight_tool_call_ids() == {}


class TestShutdown:
    async def test_shutdown_cancels_all(self) -> None:
        reg = TaskRegistry()
        t1 = asyncio.create_task(_sleeper())
        t2 = asyncio.create_task(_sleeper())
        reg.add("sess_1", "call_a", t1)
        reg.add("sess_2", "call_b", t2)
        await reg.shutdown()
        assert t1.cancelled()
        assert t2.cancelled()
        assert reg.in_flight_count("sess_1") == 0
        assert reg.in_flight_count("sess_2") == 0


# ── step-task tracking ───────────────────────────────────────────────


class TestStepTracking:
    async def test_register_and_cancel_step(self) -> None:
        reg = TaskRegistry()
        task = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", task)
        assert reg.cancel_step("sess_1") is True
        await asyncio.sleep(0)
        assert task.cancelled()

    async def test_cancel_step_unknown_session_returns_false(self) -> None:
        reg = TaskRegistry()
        assert reg.cancel_step("sess_nope") is False

    async def test_unregister_step_clears(self) -> None:
        reg = TaskRegistry()
        task = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", task)
        reg.unregister_step("sess_1")
        assert reg.cancel_step("sess_1") is False
        task.cancel()  # cleanup
        await asyncio.sleep(0)

    async def test_register_step_replaces_prior(self) -> None:
        reg = TaskRegistry()
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
        reg = TaskRegistry()
        step = asyncio.create_task(_sleeper())
        tool = asyncio.create_task(_sleeper())
        reg.register_step("sess_1", step)
        reg.add("sess_1", "call_a", tool)
        assert reg.cancel_step("sess_1") is True
        await asyncio.sleep(0)
        assert step.cancelled()
        assert not tool.cancelled()
        assert reg.in_flight_count("sess_1") == 1
        tool.cancel()
        await asyncio.sleep(0)
