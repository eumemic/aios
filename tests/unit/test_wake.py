"""Regression coverage for retry-cause ``defer_wake`` (issue #80)."""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from procrastinate import App
from procrastinate.testing import InMemoryConnector

from aios.services.wake import defer_wake


@pytest.fixture(autouse=True)
def mock_append_event() -> Generator[AsyncMock]:
    """Patch ``sessions_service.append_event`` so ``wake_deferred`` span
    emission doesn't require a real DB connection (issue #131)."""
    mock = AsyncMock()
    with patch("aios.services.wake.sessions_service.append_event", mock):
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
            pool,
            "sess_x",
            "span",
            {"event": "wake_deferred", "cause": "reschedule", "delay_seconds": 5},
            account_id=ANY,
        )
