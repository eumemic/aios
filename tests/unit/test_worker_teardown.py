"""Unit coverage for worker background-task teardown draining."""

from __future__ import annotations

import asyncio
import logging

import pytest

from aios.harness.worker import _cancel_and_drain


async def test_cancel_and_drain_cancels_running_task() -> None:
    cancelled = asyncio.Event()

    async def wait_forever() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(wait_forever(), name="healthy-task")
    await asyncio.sleep(0)

    await _cancel_and_drain(task)

    assert task.cancelled()
    assert cancelled.is_set()


async def test_cancel_and_drain_logs_pre_dead_task_without_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def boom() -> None:
        raise RuntimeError("died before teardown")

    task = asyncio.create_task(boom(), name="pre-dead-task")
    await asyncio.sleep(0)

    with caplog.at_level(logging.ERROR):
        await _cancel_and_drain(task)

    assert any("worker.teardown_task_error" in record.getMessage() for record in caplog.records)


async def test_pre_dead_task_does_not_stop_later_drains() -> None:
    second_cancelled = asyncio.Event()

    async def boom() -> None:
        raise RuntimeError("first task died")

    async def wait_forever() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            second_cancelled.set()

    first = asyncio.create_task(boom(), name="first-pre-dead")
    second = asyncio.create_task(wait_forever(), name="second-running")
    await asyncio.sleep(0)

    await _cancel_and_drain(first)
    await _cancel_and_drain(second)

    assert second.cancelled()
    assert second_cancelled.is_set()
