"""Unit tests for the TaskRegistry."""

from __future__ import annotations

import asyncio

from aios.harness.task_registry import TaskRegistry


async def _sleeper() -> None:
    await asyncio.sleep(3600)


class TestBasicLifecycle:
    def test_add_and_count(self) -> None:
        reg = TaskRegistry()
        task = asyncio.get_event_loop().create_future()
        reg.add("sess_1", "call_a", task)  # type: ignore[arg-type]
        assert reg.in_flight_count("sess_1") == 1

    def test_remove(self) -> None:
        reg = TaskRegistry()
        task = asyncio.get_event_loop().create_future()
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
        f1 = asyncio.get_event_loop().create_future()
        f2 = asyncio.get_event_loop().create_future()
        reg.add("sess_1", "call_a", f1)  # type: ignore[arg-type]
        reg.add("sess_1", "call_b", f2)  # type: ignore[arg-type]
        assert reg.in_flight_tool_call_ids("sess_1") == {"call_a", "call_b"}

    def test_in_flight_tool_call_ids_excludes_done(self) -> None:
        reg = TaskRegistry()
        f1 = asyncio.get_event_loop().create_future()
        f2 = asyncio.get_event_loop().create_future()
        f2.set_result(None)  # mark as done
        reg.add("sess_1", "call_a", f1)  # type: ignore[arg-type]
        reg.add("sess_1", "call_b", f2)  # type: ignore[arg-type]
        assert reg.in_flight_tool_call_ids("sess_1") == {"call_a"}

    def test_all_in_flight_tool_call_ids(self) -> None:
        reg = TaskRegistry()
        f1 = asyncio.get_event_loop().create_future()
        f2 = asyncio.get_event_loop().create_future()
        f3 = asyncio.get_event_loop().create_future()
        f3.set_result(None)  # done — should be excluded
        reg.add("sess_1", "call_a", f1)  # type: ignore[arg-type]
        reg.add("sess_2", "call_b", f2)  # type: ignore[arg-type]
        reg.add("sess_2", "call_c", f3)  # type: ignore[arg-type]
        result = reg.all_in_flight_tool_call_ids()
        assert result == {"sess_1": {"call_a"}, "sess_2": {"call_b"}}

    def test_all_in_flight_excludes_sessions_with_no_active(self) -> None:
        reg = TaskRegistry()
        f1 = asyncio.get_event_loop().create_future()
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
