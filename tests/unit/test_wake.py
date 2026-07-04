"""Regression coverage for retry-cause ``defer_wake`` (issue #80)."""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from procrastinate import App
from procrastinate.testing import InMemoryConnector

from aios.jobs.app import defer_run_wake, defer_wake


@pytest.fixture(autouse=True)
def mock_append_event() -> Generator[AsyncMock]:
    """Patch ``sessions_service.append_event`` so ``wake_deferred`` span
    emission doesn't require a real DB connection (issue #131)."""
    mock = AsyncMock()
    with patch("aios.jobs.app.queries.append_event", mock):
        yield mock


class TestDeferRescheduleWake:
    async def test_schedules_job_delay_seconds_in_future(self, in_memory_app: App) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        before = datetime.now(UTC)
        await defer_wake(
            MagicMock(), "sess_x", cause="reschedule", delay_seconds=5, account_id=account_id
        )
        after = datetime.now(UTC)

        connector = in_memory_app.connector
        assert isinstance(connector, InMemoryConnector)
        (job,) = connector.jobs.values()
        assert job["scheduled_at"] is not None
        assert before + timedelta(seconds=5) <= job["scheduled_at"] <= after + timedelta(seconds=5)

    async def test_does_not_leak_schedule_in_into_task_args(self, in_memory_app: App) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        await defer_wake(
            MagicMock(), "sess_x", cause="reschedule", delay_seconds=5, account_id=account_id
        )

        connector = in_memory_app.connector
        assert isinstance(connector, InMemoryConnector)
        (job,) = connector.jobs.values()
        assert job["args"] == {"session_id": "sess_x", "cause": "reschedule"}

    async def test_targets_wake_session_task(self, in_memory_app: App) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        await defer_wake(
            MagicMock(), "sess_x", cause="reschedule", delay_seconds=5, account_id=account_id
        )

        connector = in_memory_app.connector
        assert isinstance(connector, InMemoryConnector)
        (job,) = connector.jobs.values()
        assert job["task_name"] == "harness.wake_session"

    async def test_swallows_already_enqueued(self, in_memory_app: App) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        await defer_wake(
            MagicMock(), "sess_x", cause="reschedule", delay_seconds=5, account_id=account_id
        )
        await defer_wake(
            MagicMock(), "sess_x", cause="reschedule", delay_seconds=5, account_id=account_id
        )
        connector = in_memory_app.connector
        assert isinstance(connector, InMemoryConnector)
        assert len(connector.jobs) == 1

    async def test_emits_wake_deferred_span_event(
        self, in_memory_app: App, mock_append_event: AsyncMock
    ) -> None:
        """Every deferral emits a ``wake_deferred`` span carrying cause and delay."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        pool = MagicMock()
        await defer_wake(pool, "sess_x", cause="reschedule", delay_seconds=5, account_id=account_id)

        mock_append_event.assert_awaited_once_with(
            ANY,  # conn acquired from the pool
            account_id=ANY,
            session_id="sess_x",
            kind="span",
            data={"event": "wake_deferred", "cause": "reschedule", "delay_seconds": 5},
        )


@pytest.fixture
def batch_window(monkeypatch: pytest.MonkeyPatch) -> Generator[float]:
    """Set ``workflow_wake_batch_seconds`` to 2.0 for the test, clearing the
    settings cache both ways (it is lru-cached; without the clears the override
    silently tests the default — and would leak into later tests)."""
    from aios.config import get_settings

    monkeypatch.setenv("AIOS_WORKFLOW_WAKE_BATCH_SECONDS", "2.0")
    get_settings.cache_clear()
    yield 2.0
    get_settings.cache_clear()


class TestDeferRunWakeBatching:
    """#780 — the coalescing window, dark by default (setting 0.0)."""

    async def test_batch_schedules_inside_the_window(
        self, in_memory_app: App, batch_window: float
    ) -> None:
        before = datetime.now(UTC)
        await defer_run_wake("run_b", batch=True)
        after = datetime.now(UTC)
        connector = in_memory_app.connector
        assert isinstance(connector, InMemoryConnector)
        (job,) = connector.jobs.values()
        assert job["task_name"] == "harness.wake_workflow"
        assert job["scheduled_at"] is not None
        window = timedelta(seconds=batch_window)
        assert before + window <= job["scheduled_at"] <= after + window

    async def test_batch_with_setting_zero_is_immediate(self, in_memory_app: App) -> None:
        # Dark by default: batch=True with the setting at 0 behaves exactly like today.
        await defer_run_wake("run_b", batch=True)
        connector = in_memory_app.connector
        assert isinstance(connector, InMemoryConnector)
        (job,) = connector.jobs.values()
        assert job["scheduled_at"] is None

    async def test_unbatched_is_immediate_even_with_window_set(
        self, in_memory_app: App, batch_window: float
    ) -> None:
        await defer_run_wake("run_b")
        connector = in_memory_app.connector
        assert isinstance(connector, InMemoryConnector)
        (job,) = connector.jobs.values()
        assert job["scheduled_at"] is None

    async def test_pending_batched_wake_absorbs_further_defers(
        self, in_memory_app: App, batch_window: float
    ) -> None:
        """The scheduled job holds the queueing_lock for the whole window: a burst
        of batched completions AND an otherwise-immediate defer (a gate resume)
        all collapse into the one pending wake. Sound because every wake source
        commits its signal/response BEFORE deferring — the delayed step harvests
        them; the cost is latency bounded by the window."""
        await defer_run_wake("run_b", batch=True)
        await defer_run_wake("run_b", batch=True)  # burst: coalesced
        await defer_run_wake("run_b")  # "immediate" defer: absorbed by the pending wake
        connector = in_memory_app.connector
        assert isinstance(connector, InMemoryConnector)
        assert len(connector.jobs) == 1
