"""Regression coverage for ``defer_retry_wake`` (issue #80)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from procrastinate import App
from procrastinate.testing import InMemoryConnector

from aios.harness.wake import defer_retry_wake


@pytest.fixture
async def in_memory_app() -> AsyncIterator[App]:
    from aios.harness.procrastinate_app import app

    with app.replace_connector(InMemoryConnector()) as patched:
        yield patched


class TestDeferRetryWake:
    async def test_schedules_job_delay_seconds_in_future(self, in_memory_app: App) -> None:
        before = datetime.now(UTC)
        await defer_retry_wake("sess_x", delay_seconds=5)
        after = datetime.now(UTC)

        (job,) = in_memory_app.connector.jobs.values()
        assert job["scheduled_at"] is not None
        assert before + timedelta(seconds=5) <= job["scheduled_at"] <= after + timedelta(seconds=5)

    async def test_does_not_leak_schedule_in_into_task_args(self, in_memory_app: App) -> None:
        await defer_retry_wake("sess_x", delay_seconds=5)

        (job,) = in_memory_app.connector.jobs.values()
        assert job["args"] == {"session_id": "sess_x", "cause": "reschedule"}

    async def test_targets_wake_session_task(self, in_memory_app: App) -> None:
        await defer_retry_wake("sess_x", delay_seconds=5)

        (job,) = in_memory_app.connector.jobs.values()
        assert job["task_name"] == "harness.wake_session"

    async def test_swallows_already_enqueued(self, in_memory_app: App) -> None:
        await defer_retry_wake("sess_x", delay_seconds=5)
        await defer_retry_wake("sess_x", delay_seconds=5)
        assert len(in_memory_app.connector.jobs) == 1
