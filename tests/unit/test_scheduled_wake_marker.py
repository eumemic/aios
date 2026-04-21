"""Unit tests for the ``cause="scheduled"`` marker-append in ``run_session_step``.

When the ``schedule_wake`` tool's delayed wake fires, the step must
materialize a synthetic user-role event before the sweep guard runs —
otherwise the sweep sees no unreacted messages and early-outs, and the
agent never actually gets a step at T+delay.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aios.harness.loop import run_session_step


def _message_appends(append_event: AsyncMock) -> list[tuple[Any, ...]]:
    """Extract positional args of ``append_event`` calls with ``kind == 'message'``.

    ``run_session_step`` now wraps its body in ``step_start``/``step_end``
    span appends (issue #131), so a raw ``await_count`` check is no longer
    the right assertion.  Filter to the ``message`` kind to focus on the
    scheduled-wake marker specifically.
    """
    return [call.args for call in append_event.await_args_list if call.args[2] == "message"]


class TestScheduledWakeMarker:
    async def test_marker_appended_before_sweep_for_scheduled_wake(self) -> None:
        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_x"))
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

        messages = _message_appends(append_event)
        assert len(messages) == 1
        args = messages[0]
        assert args[2] == "message"
        assert args[3] == {
            "role": "user",
            "content": "[Your scheduled wake fired. Reason: ping home]",
        }

    async def test_no_marker_for_non_scheduled_causes(self) -> None:
        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_x"))
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

        assert _message_appends(append_event) == []

    async def test_scheduled_cause_without_reason_is_noop(self) -> None:
        """Defensive: ``cause="scheduled"`` with no reason attached doesn't inject noise."""
        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_x"))
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

        assert _message_appends(append_event) == []


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

        monkeypatch.setattr("aios.harness.wake.sessions_service.append_event", AsyncMock())

        patched: App
        with app.replace_connector(InMemoryConnector()) as patched:
            before = datetime.now(UTC)
            await defer_wake(
                MagicMock(),
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

    async def test_defer_wake_without_delay_is_immediate(self, monkeypatch: Any) -> None:
        from procrastinate.testing import InMemoryConnector

        from aios.harness.procrastinate_app import app
        from aios.harness.wake import defer_wake

        monkeypatch.setattr("aios.harness.wake.sessions_service.append_event", AsyncMock())

        with app.replace_connector(InMemoryConnector()) as patched:
            await defer_wake(MagicMock(), "sess_x", cause="message")

            (job,) = patched.connector.jobs.values()
            assert job["scheduled_at"] is None
            assert "wake_reason" not in job["args"]
