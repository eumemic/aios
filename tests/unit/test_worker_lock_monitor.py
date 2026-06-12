"""Unit tests for advisory-lock connection loss handling."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from aios.harness import worker as worker_mod


async def test_advisory_lock_loss_trips_fail_stop_latch(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    heartbeat = tmp_path / "alive"
    heartbeat.touch()
    latch = asyncio.Event()
    fatal: worker_mod._SupervisedTaskFailure = {"exception": None}
    callback = worker_mod._make_advisory_lock_termination_listener(latch=latch, fatal=fatal)

    with patch.object(worker_mod, "_HEARTBEAT_FILE", heartbeat), caplog.at_level(logging.ERROR):
        callback(object())

    assert latch.is_set()
    assert isinstance(fatal["exception"], RuntimeError)
    assert "singleton advisory lock connection lost" in str(fatal["exception"])
    assert not heartbeat.exists()
    assert any("worker.advisory_lock_lost" in record.getMessage() for record in caplog.records)


async def test_advisory_lock_loss_listener_is_silent_when_latch_set(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    heartbeat = tmp_path / "alive"
    heartbeat.touch()
    latch = asyncio.Event()
    latch.set()
    fatal: worker_mod._SupervisedTaskFailure = {"exception": None}
    callback = worker_mod._make_advisory_lock_termination_listener(latch=latch, fatal=fatal)

    with patch.object(worker_mod, "_HEARTBEAT_FILE", heartbeat), caplog.at_level(logging.ERROR):
        callback(object())

    assert fatal["exception"] is None
    assert heartbeat.exists()
    assert not any("worker.advisory_lock_lost" in record.getMessage() for record in caplog.records)
