"""Unit tests for the ``cause="scheduled"`` marker-append in ``run_session_step``.

When the ``schedule_wake`` tool's delayed wake fires, the step must
materialize a synthetic user-role event before the sweep guard runs —
otherwise the sweep sees no unreacted messages and early-outs, and the
agent never actually gets a step at T+delay.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aios.harness.loop import run_session_step


class TestScheduledWakeMarker:
    async def test_marker_appended_before_sweep_for_scheduled_wake(self) -> None:
        append_event = AsyncMock()
        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
            patch(
                "aios.harness.loop.runtime.require_task_registry",
                return_value=MagicMock(),
            ),
            patch(
                "aios.harness.loop.find_sessions_needing_inference",
                AsyncMock(return_value=set()),
            ),
            patch(
                "aios.harness.loop.sessions_service.append_event",
                append_event,
            ),
        ):
            await run_session_step("sess_x", cause="scheduled", wake_reason="ping home")

        assert append_event.await_count == 1
        call = append_event.await_args
        assert call is not None
        args, _ = call.args, call.kwargs
        assert args[2] == "message"
        assert args[3] == {
            "role": "user",
            "content": "[Your scheduled wake fired. Reason: ping home]",
        }

    async def test_no_marker_for_non_scheduled_causes(self) -> None:
        append_event = AsyncMock()
        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
            patch(
                "aios.harness.loop.runtime.require_task_registry",
                return_value=MagicMock(),
            ),
            patch(
                "aios.harness.loop.find_sessions_needing_inference",
                AsyncMock(return_value=set()),
            ),
            patch(
                "aios.harness.loop.sessions_service.append_event",
                append_event,
            ),
        ):
            await run_session_step("sess_x", cause="message", wake_reason=None)
            await run_session_step("sess_x", cause="reschedule", wake_reason=None)
            await run_session_step("sess_x", cause="tool_result", wake_reason=None)

        append_event.assert_not_awaited()

    async def test_scheduled_cause_without_reason_is_noop(self) -> None:
        """Defensive: ``cause="scheduled"`` with no reason attached doesn't inject noise."""
        append_event = AsyncMock()
        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
            patch(
                "aios.harness.loop.runtime.require_task_registry",
                return_value=MagicMock(),
            ),
            patch(
                "aios.harness.loop.find_sessions_needing_inference",
                AsyncMock(return_value=set()),
            ),
            patch(
                "aios.harness.loop.sessions_service.append_event",
                append_event,
            ),
        ):
            await run_session_step("sess_x", cause="scheduled", wake_reason=None)

        append_event.assert_not_awaited()


class TestDeferWakeExtension:
    async def test_defer_wake_with_delay_schedules_in_future(self, monkeypatch: Any) -> None:
        """``defer_wake`` with ``delay_seconds`` routes through ``configure_task``'s
        ``schedule_in`` and carries ``wake_reason`` as a task kwarg.
        """
        from datetime import UTC, datetime, timedelta

        from procrastinate import App
        from procrastinate.testing import InMemoryConnector

        from aios.harness.procrastinate_app import app
        from aios.harness.wake import defer_wake

        patched: App
        with app.replace_connector(InMemoryConnector()) as patched:
            before = datetime.now(UTC)
            await defer_wake(
                "sess_x",
                cause="scheduled",
                delay_seconds=42,
                wake_reason="ring",
            )
            after = datetime.now(UTC)

            (job,) = patched.connector.jobs.values()
            assert job["args"] == {
                "session_id": "sess_x",
                "cause": "scheduled",
                "wake_reason": "ring",
            }
            assert job["scheduled_at"] is not None
            assert (
                before + timedelta(seconds=42)
                <= job["scheduled_at"]
                <= after + timedelta(seconds=42)
            )

    async def test_defer_wake_without_delay_is_immediate(self) -> None:
        from procrastinate.testing import InMemoryConnector

        from aios.harness.procrastinate_app import app
        from aios.harness.wake import defer_wake

        with app.replace_connector(InMemoryConnector()) as patched:
            await defer_wake("sess_x", cause="message")

            (job,) = patched.connector.jobs.values()
            assert job["scheduled_at"] is None
            assert "wake_reason" not in job["args"]
