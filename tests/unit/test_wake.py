"""Unit tests for ``defer_wake`` / ``defer_retry_wake``.

Regression coverage for issue #80: ``defer_retry_wake`` was passing
``schedule_in`` as a task kwarg to ``defer_async`` instead of as a
configuration directive to ``configure_task``.  That had two failure
modes: the job's ``args`` dict leaked a ``schedule_in`` key that made
the worker raise ``TypeError: wake_session() got an unexpected keyword
argument 'schedule_in'`` at execute time, and the job wasn't actually
delayed — ``scheduled_at`` stayed ``None`` so the retry ran immediately
anyway.  Both behaviors must be pinned.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from procrastinate.testing import InMemoryConnector


@pytest.fixture
async def in_memory_app() -> AsyncIterator[Any]:
    """Yield the real ``harness.wake_session`` app with an in-memory connector.

    Uses ``App.replace_connector`` so the registered task (and its
    ``queueing_lock`` etc.) stay intact — the connector swap is scoped
    to the test.
    """
    from aios.harness.procrastinate_app import app

    with app.replace_connector(InMemoryConnector()) as patched:
        yield patched


class TestDeferRetryWake:
    async def test_schedules_job_delay_seconds_in_future(self, in_memory_app: Any) -> None:
        """The job lands with ``scheduled_at`` ≈ now + delay_seconds."""
        from aios.harness.wake import defer_retry_wake

        before = datetime.now(UTC)
        await defer_retry_wake("sess_x", delay_seconds=5)
        after = datetime.now(UTC)

        assert len(in_memory_app.connector.jobs) == 1
        (job,) = in_memory_app.connector.jobs.values()

        # Pin the delay: ``scheduled_at`` must be in the future by roughly
        # delay_seconds. The buggy form left schedule_at=None → the job
        # runs immediately, which means this bound fails.
        assert job["scheduled_at"] is not None
        assert before + timedelta(seconds=5) <= job["scheduled_at"] <= after + timedelta(seconds=5)

    async def test_does_not_leak_schedule_in_into_task_args(self, in_memory_app: Any) -> None:
        """Task kwargs must be exactly ``session_id`` and ``cause`` — nothing else.

        The buggy form stuffed ``schedule_in`` into the job's ``args``
        dict, and the worker raised ``TypeError`` when it tried to
        invoke ``wake_session(session_id=..., cause=..., schedule_in=...)``.
        """
        from aios.harness.wake import defer_retry_wake

        await defer_retry_wake("sess_x", delay_seconds=5)

        (job,) = in_memory_app.connector.jobs.values()
        assert job["args"] == {"session_id": "sess_x", "cause": "reschedule"}

    async def test_targets_wake_session_task(self, in_memory_app: Any) -> None:
        from aios.harness.wake import defer_retry_wake

        await defer_retry_wake("sess_x", delay_seconds=5)

        (job,) = in_memory_app.connector.jobs.values()
        assert job["task_name"] == "harness.wake_session"

    async def test_swallows_already_enqueued(self, in_memory_app: Any) -> None:
        """``AlreadyEnqueued`` must not propagate — the existing queued wake
        runs, sees the same error, and defers its own retry.
        """
        from aios.harness.wake import defer_retry_wake

        await defer_retry_wake("sess_x", delay_seconds=5)
        await defer_retry_wake("sess_x", delay_seconds=5)
        assert len(in_memory_app.connector.jobs) == 1
