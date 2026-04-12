"""Unit tests for the cancel tool handler."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aios.harness import runtime
from aios.harness.task_registry import TaskRegistry
from aios.tools.cancel import CancelArgumentError, cancel_handler


async def _sleeper() -> None:
    await asyncio.sleep(3600)


@pytest.fixture
def task_reg() -> Any:
    """Install a TaskRegistry on the runtime module, restore after."""
    previous = runtime.task_registry
    reg = TaskRegistry()
    runtime.task_registry = reg
    try:
        yield reg
    finally:
        runtime.task_registry = previous


class TestCancelSpecificTask:
    async def test_cancel_existing_task(self, task_reg: TaskRegistry) -> None:
        task = asyncio.create_task(_sleeper())
        task_reg.add("sess_01TEST", "call_abc", task)
        result = await cancel_handler("sess_01TEST", {"tool_call_id": "call_abc"})
        assert result["cancelled"] is True
        assert result["tool_call_id"] == "call_abc"
        await asyncio.sleep(0)  # let the task process CancelledError
        assert task.cancelled()

    async def test_cancel_missing_task(self, task_reg: TaskRegistry) -> None:
        result = await cancel_handler("sess_01TEST", {"tool_call_id": "call_nope"})
        assert result["cancelled"] is False

    async def test_non_string_tool_call_id_raises(self, task_reg: TaskRegistry) -> None:
        with pytest.raises(CancelArgumentError):
            await cancel_handler("sess_01TEST", {"tool_call_id": 42})


class TestCancelAllTasks:
    async def test_cancel_all(self, task_reg: TaskRegistry) -> None:
        t1 = asyncio.create_task(_sleeper())
        t2 = asyncio.create_task(_sleeper())
        task_reg.add("sess_01TEST", "call_a", t1)
        task_reg.add("sess_01TEST", "call_b", t2)
        result = await cancel_handler("sess_01TEST", {})
        assert result["cancelled"] is True
        assert result["count"] == 2

    async def test_cancel_all_empty(self, task_reg: TaskRegistry) -> None:
        result = await cancel_handler("sess_01TEST", {})
        assert result["cancelled"] is True
        assert result["count"] == 0
