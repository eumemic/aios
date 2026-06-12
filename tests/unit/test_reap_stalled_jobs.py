"""Unit tests for the boot-time stalled procrastinate job reaper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pytest

from aios.harness.sweep import reap_stalled_jobs


@dataclass
class _FakeJob:
    id: int | None


class _FakeJobManager:
    def __init__(self, jobs: list[_FakeJob]) -> None:
        self.jobs = jobs
        self.stalled_thresholds: list[int] = []
        self.finished: list[dict[str, Any]] = []

    async def get_stalled_jobs(self, *, seconds_since_heartbeat: int) -> list[_FakeJob]:
        self.stalled_thresholds.append(seconds_since_heartbeat)
        return self.jobs

    async def finish_job_by_id_async(self, **kwargs: Any) -> None:
        self.finished.append(kwargs)


async def test_reaps_every_stalled_job_unconditionally(caplog: pytest.LogCaptureFixture) -> None:
    manager = _FakeJobManager([_FakeJob(id=101), _FakeJob(id=102)])

    with caplog.at_level(logging.WARNING):
        count = await reap_stalled_jobs(manager)

    assert count == 2
    assert manager.stalled_thresholds == [0]
    assert [call["job_id"] for call in manager.finished] == [101, 102]
    assert {call["status"].name for call in manager.finished} == {"FAILED"}
    assert all(call["delete_job"] is False for call in manager.finished)
    assert any("sweep.reaped_stalled_jobs" in record.getMessage() for record in caplog.records)


async def test_returns_zero_without_warning_when_no_jobs(caplog: pytest.LogCaptureFixture) -> None:
    manager = _FakeJobManager([])

    with caplog.at_level(logging.WARNING):
        count = await reap_stalled_jobs(manager)

    assert count == 0
    assert manager.stalled_thresholds == [0]
    assert manager.finished == []
    assert not any("sweep.reaped_stalled_jobs" in record.getMessage() for record in caplog.records)
