"""Unit tests for transient-model-error retry logic in ``run_session_step``.

Covers three layers:

1. Pure backoff table lookup (``_retry_delay_for_attempt``).
2. The consecutive-rescheduling counter (``_count_consecutive_rescheduling``)
   that drives the attempt index.
3. The exception handler's state-machine — retry with the right delay on
   early attempts, give up and re-raise once the budget is exhausted.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRetryDelayForAttempt:
    """Pure lookup into the backoff table."""

    def test_retry_delay_attempt_0_returns_2s(self) -> None:
        from aios.harness.loop import _retry_delay_for_attempt

        assert _retry_delay_for_attempt(0) == 2

    def test_retry_delay_attempt_1_returns_8s(self) -> None:
        from aios.harness.loop import _retry_delay_for_attempt

        assert _retry_delay_for_attempt(1) == 8

    def test_retry_delay_attempt_2_returns_30s(self) -> None:
        from aios.harness.loop import _retry_delay_for_attempt

        assert _retry_delay_for_attempt(2) == 30

    def test_retry_delay_attempt_3_returns_120s(self) -> None:
        from aios.harness.loop import _retry_delay_for_attempt

        assert _retry_delay_for_attempt(3) == 120

    def test_retry_delay_attempt_4_returns_none(self) -> None:
        """After 4 consecutive reschedules, the budget is exhausted."""
        from aios.harness.loop import _retry_delay_for_attempt

        assert _retry_delay_for_attempt(4) is None


class TestCountConsecutiveRescheduling:
    """Streak counter that resets on any non-rescheduling lifecycle event."""

    async def test_count_consecutive_rescheduling_empty_log_returns_zero(self) -> None:
        from aios.harness.loop import _count_consecutive_rescheduling

        pool = MagicMock()
        with patch(
            "aios.harness.loop.sessions_service.read_events",
            AsyncMock(return_value=[]),
        ):
            assert await _count_consecutive_rescheduling(pool, "sess_x") == 0

    async def test_count_consecutive_rescheduling_resets_on_non_rescheduling_tail(
        self,
    ) -> None:
        """A successful turn between failures wipes the retry budget.

        Regression guard: a long-ago rescheduling streak followed by a
        clean turn_ended must not count toward the current failure's
        attempt number.
        """
        from aios.harness.loop import _count_consecutive_rescheduling

        events = [
            SimpleNamespace(data={"event": "turn_ended", "stop_reason": "rescheduling"}),
            SimpleNamespace(data={"event": "turn_ended", "stop_reason": "rescheduling"}),
            SimpleNamespace(data={"event": "turn_ended", "stop_reason": "end_turn"}),
            SimpleNamespace(data={"event": "turn_ended", "stop_reason": "rescheduling"}),
        ]
        pool = MagicMock()
        with patch(
            "aios.harness.loop.sessions_service.read_events",
            AsyncMock(return_value=events),
        ):
            # Only the trailing single rescheduling counts.
            assert await _count_consecutive_rescheduling(pool, "sess_x") == 1


# ─── exception handler tests ──────────────────────────────────────────────────


@pytest.fixture
def mock_step_dependencies() -> Any:
    """Mock everything ``run_session_step`` touches before the model call.

    The exception handler is deep inside ``run_session_step``; to reach
    it we have to let the function walk past all the setup work: sweep
    early-out, session/agent loading, MCP discovery, context build,
    span append, and the model call (which we make raise).
    """
    session = SimpleNamespace(
        id="sess_x",
        agent_id="agt_x",
        agent_version=None,
        focal_channel=None,
    )
    agent = SimpleNamespace(
        model="openrouter/x",
        tools=[],
        mcp_servers=[],
        skills=[],
        system="sys",
        window_min=1000,
        window_max=10000,
    )
    start_event = SimpleNamespace(id="ev_start")

    with (
        patch(
            "aios.harness.loop.runtime.require_pool",
            return_value=MagicMock(),
        ),
        patch(
            "aios.harness.loop.runtime.require_task_registry",
            return_value=MagicMock(),
        ),
        patch(
            "aios.harness.loop.find_sessions_needing_inference",
            AsyncMock(return_value={"sess_x"}),
        ),
        patch(
            "aios.harness.loop.sessions_service.get_session",
            AsyncMock(return_value=session),
        ),
        patch(
            "aios.harness.loop.agents_service.get_agent",
            AsyncMock(return_value=agent),
        ),
        # Channels helpers are imported lazily inside run_session_step —
        # patch them at their source module rather than on loop.
        patch(
            "aios.harness.channels.list_bindings_and_connections",
            AsyncMock(return_value=([], [])),
        ),
        patch(
            "aios.harness.channels.augment_with_connector_instructions",
            return_value="sys",
        ),
        patch(
            "aios.harness.channels.augment_with_focal_paradigm",
            return_value="sys",
        ),
        patch(
            "aios.harness.channels.build_channels_tail_block",
            return_value=None,
        ),
        patch(
            "aios.harness.skills.augment_system_prompt",
            return_value="sys",
        ),
        patch(
            "aios.services.skills.resolve_skill_refs",
            AsyncMock(return_value=[]),
        ),
        patch(
            "aios.harness.loop.sessions_service.read_windowed_events",
            AsyncMock(return_value=[]),
        ),
        patch(
            "aios.harness.loop._dispatch_confirmed_tools",
            AsyncMock(return_value=[]),
        ),
        patch(
            "aios.harness.loop.to_openai_tools",
            return_value=[],
        ),
        patch(
            "aios.harness.loop.build_messages",
            return_value=SimpleNamespace(messages=[], reacting_to=0),
        ),
        patch(
            "aios.harness.loop.sessions_service.set_session_status",
            AsyncMock(),
        ) as set_status,
        patch(
            "aios.harness.loop.sessions_service.append_event",
            AsyncMock(return_value=start_event),
        ) as append_event,
        patch(
            "aios.harness.loop.stream_litellm",
            AsyncMock(side_effect=RuntimeError("provider boom")),
        ),
        patch(
            "aios.harness.loop.defer_retry_wake",
            AsyncMock(),
        ) as defer_retry,
    ):
        yield SimpleNamespace(
            set_status=set_status,
            append_event=append_event,
            defer_retry=defer_retry,
        )


class TestRunSessionStepOnModelError:
    """End-to-end behavior of the exception handler's state machine."""

    async def test_first_attempt_defers_retry_with_2s(self, mock_step_dependencies: Any) -> None:
        """On the first transient failure, schedule a retry at 2 seconds."""
        from aios.harness.loop import run_session_step

        with patch(
            "aios.harness.loop._count_consecutive_rescheduling",
            AsyncMock(return_value=0),
        ):
            await run_session_step("sess_x")

        mock_step_dependencies.defer_retry.assert_awaited_once_with("sess_x", delay_seconds=2)
        # Status transitions to rescheduling, not idle/error.
        status_calls = [call.args[2] for call in mock_step_dependencies.set_status.call_args_list]
        assert "rescheduling" in status_calls
        assert "idle" not in status_calls

    async def test_exhausted_budget_raises_and_sets_idle_error(
        self, mock_step_dependencies: Any
    ) -> None:
        """After 4 consecutive reschedules, give up: idle/error + re-raise."""
        from aios.harness.loop import run_session_step

        with (
            patch(
                "aios.harness.loop._count_consecutive_rescheduling",
                AsyncMock(return_value=4),
            ),
            pytest.raises(RuntimeError, match="provider boom"),
        ):
            await run_session_step("sess_x")

        mock_step_dependencies.defer_retry.assert_not_awaited()
        # Final status is idle with error stop_reason.
        idle_call = next(
            call
            for call in mock_step_dependencies.set_status.call_args_list
            if call.args[2] == "idle"
        )
        assert idle_call.kwargs["stop_reason"] == {"type": "error"}
