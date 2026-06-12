"""Unit coverage for worker fail-stop supervision helpers."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from aios.harness import worker as worker_mod


async def _wait_until_done_callbacks_run() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


async def test_supervisor_records_raising_task_and_unlinks_heartbeat(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    heartbeat = tmp_path / "alive"
    heartbeat.touch()
    latch = asyncio.Event()
    fatal: worker_mod._SupervisedTaskFailure = {"exception": None}

    async def boom() -> None:
        raise RuntimeError("worker background task exploded")

    task = asyncio.create_task(boom(), name="fake-supervised")
    with patch.object(worker_mod, "_HEARTBEAT_FILE", heartbeat), caplog.at_level(logging.ERROR):
        worker_mod._supervise(task, latch=latch, fatal=fatal)
        with pytest.raises(RuntimeError):
            await task
        await _wait_until_done_callbacks_run()

    assert latch.is_set()
    assert isinstance(fatal["exception"], RuntimeError)
    assert not heartbeat.exists()
    assert any("worker.supervised_task_died" in record.getMessage() for record in caplog.records)


async def test_supervisor_ignores_cancelled_task(tmp_path: Path) -> None:
    heartbeat = tmp_path / "alive"
    heartbeat.touch()
    latch = asyncio.Event()
    fatal: worker_mod._SupervisedTaskFailure = {"exception": None}

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(wait_forever(), name="fake-cancelled")
    with patch.object(worker_mod, "_HEARTBEAT_FILE", heartbeat):
        worker_mod._supervise(task, latch=latch, fatal=fatal)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _wait_until_done_callbacks_run()

    assert not latch.is_set()
    assert fatal["exception"] is None
    assert heartbeat.exists()


async def test_supervisor_ignores_when_latch_already_set(tmp_path: Path) -> None:
    heartbeat = tmp_path / "alive"
    heartbeat.touch()
    latch = asyncio.Event()
    latch.set()
    fatal: worker_mod._SupervisedTaskFailure = {"exception": None}

    async def boom() -> None:
        raise RuntimeError("already shutting down")

    task = asyncio.create_task(boom(), name="fake-latched")
    with patch.object(worker_mod, "_HEARTBEAT_FILE", heartbeat):
        worker_mod._supervise(task, latch=latch, fatal=fatal)
        with pytest.raises(RuntimeError):
            await task
        await _wait_until_done_callbacks_run()

    assert fatal["exception"] is None
    assert heartbeat.exists()


async def test_supervisor_treats_clean_return_as_death(tmp_path: Path) -> None:
    heartbeat = tmp_path / "alive"
    heartbeat.touch()
    latch = asyncio.Event()
    fatal: worker_mod._SupervisedTaskFailure = {"exception": None}

    async def returns_cleanly() -> None:
        return None

    task = asyncio.create_task(returns_cleanly(), name="fake-returned")
    with patch.object(worker_mod, "_HEARTBEAT_FILE", heartbeat):
        worker_mod._supervise(task, latch=latch, fatal=fatal)
        await task
        await _wait_until_done_callbacks_run()

    assert latch.is_set()
    assert isinstance(fatal["exception"], RuntimeError)
    assert "fake-returned" in str(fatal["exception"])
    assert not heartbeat.exists()
