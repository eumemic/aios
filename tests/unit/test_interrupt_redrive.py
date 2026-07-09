"""Unit coverage for :func:`aios.harness.worker._redrive_interrupts_for_tracked_sessions`.

#1756 hole 1: an interrupt landing while the pg-notify listener is
disconnected/reconnecting is otherwise LOST — the durable ``interrupt``
event the ``/interrupt`` endpoint writes is never re-read by anything on the
harness side. This redrive, run on every LISTEN (re)connect, closes that
hole by re-checking the durable marker for every session this worker still
has step/tool work tracked for.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.worker import _redrive_interrupts_for_tracked_sessions


def _log() -> MagicMock:
    return MagicMock()


class TestRedriveInterruptsForTrackedSessions:
    async def test_no_tracked_sessions_is_a_noop(self) -> None:
        pool = MagicMock()
        registry = MagicMock(spec=InflightToolRegistry)
        registry.tracked_session_ids.return_value = set()
        log = _log()

        with patch(
            "aios.services.sessions.find_latest_interrupt_seq", AsyncMock()
        ) as find_seq_mock:
            await _redrive_interrupts_for_tracked_sessions(pool, registry, log=log)

        find_seq_mock.assert_not_awaited()
        registry.cancel_step.assert_not_called()
        registry.cancel_session.assert_not_called()

    async def test_session_with_no_interrupt_is_left_alone(self) -> None:
        pool = MagicMock()
        registry = MagicMock(spec=InflightToolRegistry)
        registry.tracked_session_ids.return_value = {"sess_x"}
        log = _log()

        with (
            patch(
                "aios.services.sessions.load_session_account_id",
                AsyncMock(return_value="acc_test"),
            ),
            patch(
                "aios.services.sessions.find_latest_interrupt_seq",
                AsyncMock(return_value=None),
            ),
        ):
            await _redrive_interrupts_for_tracked_sessions(pool, registry, log=log)

        registry.cancel_step.assert_not_called()
        registry.cancel_session.assert_not_called()

    async def test_session_with_interrupt_redrives_seq_bounded_cancel(self) -> None:
        pool = MagicMock()
        registry = MagicMock(spec=InflightToolRegistry)
        registry.tracked_session_ids.return_value = {"sess_x"}
        registry.cancel_step.return_value = True
        registry.cancel_session.return_value = 2
        log = _log()

        with (
            patch(
                "aios.services.sessions.load_session_account_id",
                AsyncMock(return_value="acc_test"),
            ),
            patch(
                "aios.services.sessions.find_latest_interrupt_seq",
                AsyncMock(return_value=42),
            ),
        ):
            await _redrive_interrupts_for_tracked_sessions(pool, registry, log=log)

        registry.cancel_step.assert_called_once_with("sess_x", min_start_seq=42)
        registry.cancel_session.assert_called_once_with("sess_x", min_start_seq=42)

    async def test_multiple_tracked_sessions_each_redriven_independently(self) -> None:
        pool = MagicMock()
        registry = MagicMock(spec=InflightToolRegistry)
        registry.tracked_session_ids.return_value = {"sess_a", "sess_b"}
        registry.cancel_step.return_value = False
        registry.cancel_session.return_value = 0
        log = _log()

        async def fake_find_seq(_pool: object, session_id: str, *, account_id: str) -> int | None:
            return {"sess_a": 5, "sess_b": None}[session_id]

        with (
            patch(
                "aios.services.sessions.load_session_account_id",
                AsyncMock(return_value="acc_test"),
            ),
            patch(
                "aios.services.sessions.find_latest_interrupt_seq",
                AsyncMock(side_effect=fake_find_seq),
            ),
        ):
            await _redrive_interrupts_for_tracked_sessions(pool, registry, log=log)

        # sess_a has an interrupt -> redriven; sess_b has none -> skipped.
        cancel_step_sids = {call.args[0] for call in registry.cancel_step.call_args_list}
        assert cancel_step_sids == {"sess_a"}

    async def test_one_session_failure_does_not_abort_the_batch(self) -> None:
        """Per-session isolation: a NotFoundError (session gone) or transient
        DB error on one tracked session must not skip the rest."""
        pool = MagicMock()
        registry = MagicMock(spec=InflightToolRegistry)
        registry.tracked_session_ids.return_value = {"sess_bad", "sess_good"}
        registry.cancel_step.return_value = True
        registry.cancel_session.return_value = 1
        log = _log()

        async def fake_load_account_id(_pool: object, session_id: str) -> str:
            if session_id == "sess_bad":
                raise RuntimeError("simulated transient DB error")
            return "acc_test"

        with (
            patch(
                "aios.services.sessions.load_session_account_id",
                AsyncMock(side_effect=fake_load_account_id),
            ),
            patch(
                "aios.services.sessions.find_latest_interrupt_seq",
                AsyncMock(return_value=7),
            ),
        ):
            await _redrive_interrupts_for_tracked_sessions(pool, registry, log=log)

        registry.cancel_step.assert_called_once_with("sess_good", min_start_seq=7)
        log.exception.assert_called_once()
