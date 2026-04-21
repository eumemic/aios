"""Unit tests for wake-latency instrumentation (issue #131).

Covers:

- ``wake_deferred`` span event emitted inside ``defer_wake`` on every
  deferral, including coalesced calls (``AlreadyEnqueued``).
- ``step_start``/``step_end`` span pair bracketing ``run_session_step``
  on all exit paths: sweep early-out, end-turn, model-error reschedule.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from procrastinate import App
from procrastinate.testing import InMemoryConnector


@pytest.fixture
async def in_memory_app() -> AsyncIterator[App]:
    from aios.harness.procrastinate_app import app

    with app.replace_connector(InMemoryConnector()) as patched:
        yield patched


class TestE2EConftestMockSignatures:
    """The e2e conftest installs no-op mocks in place of ``defer_wake`` and
    ``defer_retry_wake``.  If the mocks' signatures drift from the real
    functions', every e2e test crashes with ``TypeError`` at the first
    deferral.  This is exactly what happened when PR #138 added ``pool`` as
    the first positional arg but forgot to update the conftest — catch
    future drift with a signature-equality assertion."""

    def test_noop_defer_wake_matches_real_defer_wake(self) -> None:
        import inspect

        from aios.harness.wake import defer_wake
        from tests.e2e.conftest import _noop_defer_wake

        real_params = list(inspect.signature(defer_wake).parameters.keys())
        mock_params = list(inspect.signature(_noop_defer_wake).parameters.keys())
        assert real_params == mock_params

    def test_noop_defer_retry_wake_matches_real(self) -> None:
        import inspect

        from aios.harness.wake import defer_retry_wake
        from tests.e2e.conftest import _noop_defer_retry_wake

        real_params = list(inspect.signature(defer_retry_wake).parameters.keys())
        mock_params = list(inspect.signature(_noop_defer_retry_wake).parameters.keys())
        assert real_params == mock_params


class TestWakeDeferredEvent:
    async def test_defer_wake_emits_span_with_cause(self, in_memory_app: App) -> None:
        from aios.harness.wake import defer_wake

        mock_append = AsyncMock()
        pool = MagicMock()
        with patch("aios.harness.wake.sessions_service.append_event", mock_append):
            await defer_wake(pool, "sess_x", cause="message")

        mock_append.assert_awaited_once_with(
            pool,
            "sess_x",
            "span",
            {"event": "wake_deferred", "cause": "message"},
        )

    async def test_defer_wake_span_carries_delay_when_scheduled(self, in_memory_app: App) -> None:
        from aios.harness.wake import defer_wake

        mock_append = AsyncMock()
        pool = MagicMock()
        with patch("aios.harness.wake.sessions_service.append_event", mock_append):
            await defer_wake(pool, "sess_x", cause="scheduled", delay_seconds=30, wake_reason="r")

        mock_append.assert_awaited_once_with(
            pool,
            "sess_x",
            "span",
            {"event": "wake_deferred", "cause": "scheduled", "delay_seconds": 30},
        )

    async def test_defer_wake_emits_span_even_when_coalesced(self, in_memory_app: App) -> None:
        """N deferrals must all emit ``wake_deferred``, even if procrastinate
        coalesces them — the profiler observes coalescing as N deferred → 1 step."""
        from aios.harness.wake import defer_wake

        mock_append = AsyncMock()
        pool = MagicMock()
        with patch("aios.harness.wake.sessions_service.append_event", mock_append):
            await defer_wake(pool, "sess_x", cause="message")
            await defer_wake(pool, "sess_x", cause="sweep")
            await defer_wake(pool, "sess_x", cause="tool_confirmation")

        assert mock_append.await_count == 3
        causes = [call.args[3]["cause"] for call in mock_append.await_args_list]
        assert causes == ["message", "sweep", "tool_confirmation"]
        # Procrastinate coalesced 2/3 but the third cause still wrote its span.
        assert len(in_memory_app.connector.jobs) == 1

    async def test_defer_retry_wake_emits_reschedule_span(self, in_memory_app: App) -> None:
        from aios.harness.wake import defer_retry_wake

        mock_append = AsyncMock()
        pool = MagicMock()
        with patch("aios.harness.wake.sessions_service.append_event", mock_append):
            await defer_retry_wake(pool, "sess_x", delay_seconds=2)

        mock_append.assert_awaited_once_with(
            pool,
            "sess_x",
            "span",
            {"event": "wake_deferred", "cause": "reschedule", "delay_seconds": 2},
        )


# ─── step_start / step_end span pair ─────────────────────────────────────────


def _span_event_names(append_event: AsyncMock) -> list[str]:
    """Extract the ``event`` field of every span append, in order."""
    return [
        call.args[3]["event"] for call in append_event.await_args_list if call.args[2] == "span"
    ]


@pytest.fixture
def mock_runtime() -> Iterator[None]:
    """Pool/registry globals aren't set outside a worker_main context."""
    with (
        patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
        patch(
            "aios.harness.loop.runtime.require_task_registry",
            return_value=MagicMock(),
        ),
    ):
        yield


class TestStepStartEndSpans:
    async def test_sweep_early_out_still_emits_pair(self, mock_runtime: None) -> None:
        """Early-out from the sweep guard is a "wasted wake" — must still
        emit ``step_start``/``step_end`` so the profiler sees the cost."""
        from aios.harness.loop import run_session_step

        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_step"))
        with (
            patch(
                "aios.harness.loop.find_sessions_needing_inference",
                AsyncMock(return_value=set()),
            ),
            patch(
                "aios.harness.loop.sessions_service.append_event",
                append_event,
            ),
        ):
            await run_session_step("sess_x", cause="message")

        # Entry-site sweep spans wrap the guard check even on the
        # wasted-wake path, so the profiler can see the SQL cost.
        assert _span_event_names(append_event) == [
            "step_start",
            "sweep_start",
            "sweep_end",
            "step_end",
        ]
        # step_end must reference step_start by id
        start_id = append_event.await_args_list[0].args[3]
        assert start_id == {"event": "step_start", "cause": "message"}
        end = append_event.await_args_list[-1].args[3]
        assert end == {"event": "step_end", "step_start_id": "ev_step"}

    async def test_reschedule_defers_wake_after_step_end(self) -> None:
        """On the model-error retry path, ``defer_retry_wake`` must fire AFTER
        ``step_end`` so its ``wake_deferred`` lands in step N+1's temporal
        window, not step N's.  Under the "all wake_deferred since previous
        step_end" pairing rule, the reverse ordering would make the
        reschedule invisible to the next step's queue-latency calculation —
        the one path where delay is a known quantity (the backoff)."""
        from aios.harness.loop import run_session_step

        session = SimpleNamespace(
            id="sess_x", agent_id="agt_x", agent_version=None, focal_channel=None
        )
        agent = SimpleNamespace(
            model="openrouter/x",
            tools=[],
            mcp_servers=[],
            skills=[],
            system="sys",
            litellm_extra={},
            window_min=1000,
            window_max=10000,
        )
        start_event = SimpleNamespace(id="ev_step")

        manager = MagicMock()
        append_event = AsyncMock(return_value=start_event)
        defer_retry = AsyncMock()
        manager.attach_mock(append_event, "append_event")
        manager.attach_mock(defer_retry, "defer_retry")

        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
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
            patch(
                "aios.harness.channels.list_bindings_and_connections",
                AsyncMock(return_value=([], [])),
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
                "aios.harness.loop.compose_step_context",
                AsyncMock(
                    return_value=SimpleNamespace(
                        model="openrouter/x",
                        messages=[],
                        tools=[],
                        reacting_to=0,
                        skill_versions=[],
                    )
                ),
            ),
            patch("aios.harness.loop.sessions_service.set_session_status", AsyncMock()),
            patch("aios.harness.loop.sessions_service.append_event", append_event),
            patch(
                "aios.harness.loop.stream_litellm",
                AsyncMock(side_effect=RuntimeError("provider boom")),
            ),
            patch("aios.harness.loop.defer_retry_wake", defer_retry),
            patch(
                "aios.harness.loop._count_consecutive_rescheduling",
                AsyncMock(return_value=0),
            ),
        ):
            await run_session_step("sess_x")

        span_names = _span_event_names(append_event)
        assert span_names[0] == "step_start"
        assert span_names[-1] == "step_end"
        assert "context_build_start" in span_names
        assert "model_request_end" in span_names

        call_names = [c[0] for c in manager.mock_calls]
        last_append = max(
            i
            for i, (name, args, _kw) in enumerate(manager.mock_calls)
            if name == "append_event" and args[2] == "span" and args[3].get("event") == "step_end"
        )
        first_defer = call_names.index("defer_retry")
        assert first_defer > last_append, (
            f"defer_retry_wake must be called after step_end; "
            f"got step_end at {last_append}, defer_retry at {first_defer}"
        )

    async def test_happy_path_span_ordering(self) -> None:
        """Regression fence: on a clean end-turn, spans nest in the expected order."""
        from aios.harness.loop import run_session_step

        session = SimpleNamespace(
            id="sess_x", agent_id="agt_x", agent_version=None, focal_channel=None
        )
        agent = SimpleNamespace(
            model="openrouter/x",
            tools=[],
            mcp_servers=[],
            skills=[],
            system="sys",
            litellm_extra={},
            window_min=1000,
            window_max=10000,
        )
        start_event = SimpleNamespace(id="ev_step")

        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
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
            patch(
                "aios.harness.channels.list_bindings_and_connections",
                AsyncMock(return_value=([], [])),
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
                "aios.harness.loop.compose_step_context",
                AsyncMock(
                    return_value=SimpleNamespace(
                        model="openrouter/x",
                        messages=[],
                        tools=[],
                        reacting_to=0,
                        skill_versions=[],
                    )
                ),
            ),
            patch("aios.harness.loop.sessions_service.set_session_status", AsyncMock()),
            patch(
                "aios.harness.loop.sessions_service.append_event",
                AsyncMock(return_value=start_event),
            ) as append_event,
            patch(
                "aios.harness.loop.call_litellm",
                AsyncMock(return_value=({"role": "assistant", "content": "ok"}, {}, 0.0)),
            ),
            patch(
                "aios.harness.loop.stream_litellm",
                AsyncMock(return_value=({"role": "assistant", "content": "ok"}, {}, 0.0)),
            ),
            patch(
                "aios.harness.loop.sessions_service.increment_usage",
                AsyncMock(),
            ),
            patch(
                "aios.db.sse_lock.has_subscriber",
                AsyncMock(return_value=False),
            ),
        ):
            await run_session_step("sess_x")

        assert _span_event_names(append_event) == [
            "step_start",
            "sweep_start",
            "sweep_end",
            "context_build_start",
            "context_build_end",
            "model_request_start",
            "model_request_end",
            "step_end",
        ]

    async def test_step_end_emitted_when_body_raises(self) -> None:
        """If the body raises past the retry budget, ``step_end`` still fires."""
        from aios.harness.loop import run_session_step

        session = SimpleNamespace(
            id="sess_x", agent_id="agt_x", agent_version=None, focal_channel=None
        )
        agent = SimpleNamespace(
            model="openrouter/x",
            tools=[],
            mcp_servers=[],
            skills=[],
            system="sys",
            litellm_extra={},
            window_min=1000,
            window_max=10000,
        )
        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_step"))

        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
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
            patch(
                "aios.harness.channels.list_bindings_and_connections",
                AsyncMock(return_value=([], [])),
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
                "aios.harness.loop.compose_step_context",
                AsyncMock(
                    return_value=SimpleNamespace(
                        model="openrouter/x",
                        messages=[],
                        tools=[],
                        reacting_to=0,
                        skill_versions=[],
                    )
                ),
            ),
            patch("aios.harness.loop.sessions_service.set_session_status", AsyncMock()),
            patch("aios.harness.loop.sessions_service.append_event", append_event),
            patch(
                "aios.harness.loop.stream_litellm",
                AsyncMock(side_effect=RuntimeError("provider boom")),
            ),
            patch("aios.harness.loop.defer_retry_wake", AsyncMock()),
            patch(
                "aios.harness.loop._count_consecutive_rescheduling",
                AsyncMock(return_value=4),  # budget exhausted — re-raises
            ),
            pytest.raises(RuntimeError, match="provider boom"),
        ):
            await run_session_step("sess_x")

        span_names = _span_event_names(append_event)
        assert span_names[0] == "step_start"
        assert span_names[-1] == "step_end"
