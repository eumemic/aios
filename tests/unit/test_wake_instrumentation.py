"""Unit tests for wake-latency instrumentation (issue #131).

Covers:

- ``wake_deferred`` span event emitted inside ``defer_wake`` on every
  deferral, including coalesced calls (``AlreadyEnqueued``).
- ``step_start``/``step_end`` span pair bracketing ``run_session_step``
  on all exit paths: sweep early-out, end-turn, model-error reschedule.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from procrastinate import App
from procrastinate.testing import InMemoryConnector

from aios.harness.completion import LlmResponse
from aios.harness.window import WindowedEvents
from aios.services.sessions import AssistantAppendResult


def _llm_response(
    message: dict[str, object],
    *,
    usage: dict[str, int] | None = None,
    cost: float | None = 0.0,
    finish_reason: str | None = None,
) -> LlmResponse:
    """Build an ``LlmResponse`` the way ``call_litellm``/``stream_litellm`` do."""
    return LlmResponse.from_message(
        dict(message),
        usage=usage or {},
        cost=cost,
        finish_reason=finish_reason,
    )


class TestE2EConftestMockSignatures:
    """The e2e conftest installs a no-op mock in place of ``defer_wake``.
    If the mock's signature drifts from the real function's, every e2e
    test crashes with ``TypeError`` at the first deferral.  This is
    exactly what happened when PR #138 added ``pool`` as the first
    positional arg but forgot to update the conftest — catch future
    drift with a signature-equality assertion."""

    def test_noop_defer_wake_matches_real_defer_wake(self) -> None:
        import inspect

        from aios.services.wake import defer_wake
        from tests.e2e.conftest import _noop_defer_wake

        real_params = list(inspect.signature(defer_wake).parameters.keys())
        mock_params = list(inspect.signature(_noop_defer_wake).parameters.keys())
        assert real_params == mock_params


class TestWakeDeferredEvent:
    async def test_defer_wake_emits_span_with_cause(self, in_memory_app: App) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services.wake import defer_wake

        mock_append = AsyncMock()
        pool = MagicMock()
        with patch("aios.services.wake.sessions_service.append_event", mock_append):
            await defer_wake(pool, "sess_x", cause="message", account_id=account_id)

        mock_append.assert_awaited_once_with(
            pool, "sess_x", "span", {"event": "wake_deferred", "cause": "message"}, account_id=ANY
        )

    async def test_defer_wake_span_carries_delay_when_delayed(self, in_memory_app: App) -> None:
        """When defer_wake is called with a delay_seconds (the harness
        retry-backoff path), the span event records the delay so the
        profiler can observe queue latency on the retry path."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services.wake import defer_wake

        mock_append = AsyncMock()
        pool = MagicMock()
        with patch("aios.services.wake.sessions_service.append_event", mock_append):
            await defer_wake(
                pool,
                "sess_x",
                cause="reschedule",
                delay_seconds=30,
                account_id=account_id,
            )

        mock_append.assert_awaited_once_with(
            pool,
            "sess_x",
            "span",
            {"event": "wake_deferred", "cause": "reschedule", "delay_seconds": 30},
            account_id=ANY,
        )

    async def test_defer_wake_emits_span_even_when_coalesced(self, in_memory_app: App) -> None:
        """N deferrals must all emit ``wake_deferred``, even if procrastinate
        coalesces them — the profiler observes coalescing as N deferred → 1 step."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services.wake import defer_wake

        mock_append = AsyncMock()
        pool = MagicMock()
        with patch("aios.services.wake.sessions_service.append_event", mock_append):
            await defer_wake(pool, "sess_x", cause="message", account_id=account_id)
            await defer_wake(pool, "sess_x", cause="sweep", account_id=account_id)
            await defer_wake(pool, "sess_x", cause="tool_confirmation", account_id=account_id)

        assert mock_append.await_count == 3
        causes = [call.args[3]["cause"] for call in mock_append.await_args_list]
        assert causes == ["message", "sweep", "tool_confirmation"]
        # Procrastinate coalesced 2/3 but the third cause still wrote its span.
        assert isinstance(in_memory_app.connector, InMemoryConnector)
        assert len(in_memory_app.connector.jobs) == 1

    async def test_defer_wake_with_reschedule_cause_emits_reschedule_span(
        self, in_memory_app: App
    ) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services.wake import defer_wake

        mock_append = AsyncMock()
        pool = MagicMock()
        with patch("aios.services.wake.sessions_service.append_event", mock_append):
            await defer_wake(
                pool, "sess_x", cause="reschedule", delay_seconds=2, account_id=account_id
            )

        mock_append.assert_awaited_once_with(
            pool,
            "sess_x",
            "span",
            {"event": "wake_deferred", "cause": "reschedule", "delay_seconds": 2},
            account_id=ANY,
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
            "aios.harness.loop.runtime.require_inflight_tool_registry",
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
        """On the model-error retry path, the reschedule ``defer_wake`` must
        fire AFTER ``step_end`` so its ``wake_deferred`` lands in step
        N+1's temporal window, not step N's.  Under the "all
        wake_deferred since previous step_end" pairing rule, the reverse
        ordering would make the reschedule invisible to the next step's
        queue-latency calculation — the one path where delay is a known
        quantity (the backoff)."""
        from aios.harness.loop import run_session_step

        session = SimpleNamespace(
            id="sess_x",
            agent_id="agt_x",
            agent_version=None,
            focal_channel=None,
            origin="foreground",
            parent_run_id=None,
            archive_when_idle=False,
        )
        agent = SimpleNamespace(
            model="openrouter/x",
            tools=[],
            mcp_servers=[],
            http_servers=[],
            skills=[],
            system="sys",
            litellm_extra={},
            window_min=1000,
            window_max=10000,
        )
        start_event = SimpleNamespace(id="ev_step")

        manager = MagicMock()
        append_event = AsyncMock(return_value=start_event)
        defer_wake_mock = AsyncMock()
        manager.attach_mock(append_event, "append_event")
        manager.attach_mock(defer_wake_mock, "defer_wake")

        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
            patch(
                "aios.harness.loop.runtime.require_inflight_tool_registry",
                return_value=MagicMock(),
            ),
            patch(
                "aios.harness.loop.find_sessions_needing_inference",
                AsyncMock(return_value={"sess_x"}),
            ),
            patch(
                "aios.harness.loop.sessions_service.get_session_basic",
                AsyncMock(return_value=session),
            ),
            patch(
                "aios.harness.loop.agents_service.load_for_session",
                AsyncMock(return_value=agent),
            ),
            patch(
                "aios.services.channels.list_session_channels",
                AsyncMock(return_value=[]),
            ),
            patch(
                "aios.harness.loop.sessions_service.read_windowed_events",
                AsyncMock(return_value=WindowedEvents(events=[], omission=None)),
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
            patch("aios.harness.loop.sessions_service.set_session_stop_reason", AsyncMock()),
            patch("aios.harness.loop.sessions_service.append_event", append_event),
            patch(
                "aios.harness.loop.stream_litellm",
                AsyncMock(side_effect=RuntimeError("provider boom")),
            ),
            patch("aios.harness.loop.defer_wake", defer_wake_mock),
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
        first_defer = call_names.index("defer_wake")
        assert first_defer > last_append, (
            f"reschedule defer_wake must be called after step_end; "
            f"got step_end at {last_append}, defer_wake at {first_defer}"
        )
        defer_wake_mock.assert_awaited_once_with(
            ANY, "sess_x", cause="reschedule", delay_seconds=2, account_id=ANY
        )

    async def test_happy_path_span_ordering(self) -> None:
        """Regression fence: on a clean end-turn, spans nest in the expected order."""
        from aios.harness.loop import run_session_step

        session = SimpleNamespace(
            id="sess_x",
            agent_id="agt_x",
            agent_version=None,
            focal_channel=None,
            origin="foreground",
            parent_run_id=None,
            archive_when_idle=False,
        )
        agent = SimpleNamespace(
            model="openrouter/x",
            tools=[],
            mcp_servers=[],
            http_servers=[],
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
                "aios.harness.loop.runtime.require_inflight_tool_registry",
                return_value=MagicMock(),
            ),
            patch(
                "aios.harness.loop.find_sessions_needing_inference",
                AsyncMock(return_value={"sess_x"}),
            ),
            patch(
                "aios.harness.loop.sessions_service.get_session_basic",
                AsyncMock(return_value=session),
            ),
            patch(
                "aios.harness.loop.agents_service.load_for_session",
                AsyncMock(return_value=agent),
            ),
            patch(
                "aios.services.channels.list_session_channels",
                AsyncMock(return_value=[]),
            ),
            patch(
                "aios.harness.loop.sessions_service.read_windowed_events",
                AsyncMock(return_value=WindowedEvents(events=[], omission=None)),
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
            patch("aios.harness.loop.sessions_service.set_session_stop_reason", AsyncMock()),
            patch(
                "aios.harness.loop.sessions_service.append_event",
                AsyncMock(return_value=start_event),
            ) as append_event,
            patch(
                "aios.harness.loop.sessions_service.append_assistant_and_guard_quiescence",
                AsyncMock(return_value=AssistantAppendResult(False, None, None)),
            ),
            patch(
                "aios.harness.loop.call_litellm",
                AsyncMock(return_value=_llm_response({"role": "assistant", "content": "ok"})),
            ),
            patch(
                "aios.harness.loop.stream_litellm",
                AsyncMock(return_value=_llm_response({"role": "assistant", "content": "ok"})),
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
            "compute_prelude_start",
            "compute_prelude_end",
            "read_window_start",
            "read_window_end",
            "context_build_start",
            "context_build_end",
            "model_request_start",
            "model_request_end",
            "step_end",
        ]

    async def test_archive_when_idle_reclaims_after_step_end(self) -> None:
        """Regression fence: an ``archive_when_idle`` session is reclaimed as the LAST
        session write of the step — strictly AFTER ``step_end``. If the reclaim ran inside
        the step body (before the ``step_end`` append in run_session_step's finally), the
        archive would fence that append on ``archived_at IS NULL`` and the step would crash."""
        from aios.harness.loop import run_session_step

        session = SimpleNamespace(
            id="sess_x",
            agent_id="agt_x",
            agent_version=None,
            focal_channel=None,
            origin="background",
            parent_run_id="run_x",
            archive_when_idle=True,
        )
        agent = SimpleNamespace(
            model="openrouter/x",
            tools=[],
            mcp_servers=[],
            http_servers=[],
            skills=[],
            system="sys",
            litellm_extra={},
            window_min=1000,
            window_max=10000,
        )
        start_event = SimpleNamespace(id="ev_step")

        manager = MagicMock()
        append_event = AsyncMock(return_value=start_event)
        reclaim = AsyncMock(return_value=True)
        manager.attach_mock(append_event, "append_event")
        manager.attach_mock(reclaim, "reclaim")

        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
            patch(
                "aios.harness.loop.runtime.require_inflight_tool_registry", return_value=MagicMock()
            ),
            patch(
                "aios.harness.loop.find_sessions_needing_inference",
                AsyncMock(return_value={"sess_x"}),
            ),
            patch(
                "aios.harness.loop.sessions_service.get_session_basic",
                AsyncMock(return_value=session),
            ),
            patch(
                "aios.harness.loop.agents_service.load_for_session", AsyncMock(return_value=agent)
            ),
            patch("aios.services.channels.list_session_channels", AsyncMock(return_value=[])),
            patch(
                "aios.harness.loop.sessions_service.read_windowed_events",
                AsyncMock(return_value=WindowedEvents(events=[], omission=None)),
            ),
            patch("aios.harness.loop._dispatch_confirmed_tools", AsyncMock(return_value=[])),
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
            patch("aios.harness.loop.sessions_service.set_session_stop_reason", AsyncMock()),
            patch("aios.harness.loop.sessions_service.append_event", append_event),
            patch(
                "aios.harness.loop.sessions_service.append_assistant_and_guard_quiescence",
                AsyncMock(return_value=AssistantAppendResult(False, None, None)),
            ),
            patch("aios.harness.loop.sessions_service.reclaim_session_if_idle", reclaim),
            patch(
                "aios.harness.loop.stream_litellm",
                AsyncMock(return_value=_llm_response({"role": "assistant", "content": "ok"})),
            ),
            patch("aios.harness.loop.sessions_service.increment_usage", AsyncMock()),
            patch("aios.db.sse_lock.has_subscriber", AsyncMock(return_value=False)),
        ):
            await run_session_step("sess_x")

        # Reclaim is awaited exactly once, against the live pool + session.
        reclaim.assert_awaited_once_with(ANY, "sess_x", account_id=ANY)
        # …and strictly AFTER the step_end span append (the bug: reclaim-before-step_end).
        last_step_end = max(
            i
            for i, (name, args, _kw) in enumerate(manager.mock_calls)
            if name == "append_event" and args[2] == "span" and args[3].get("event") == "step_end"
        )
        reclaim_idx = next(
            i for i, (name, *_rest) in enumerate(manager.mock_calls) if name == "reclaim"
        )
        assert reclaim_idx > last_step_end, (
            f"reclaim must run after step_end; step_end at {last_step_end}, reclaim at {reclaim_idx}"
        )

    async def test_archive_when_idle_reclaim_follows_the_nudge_wake(self) -> None:
        """When the quiescence guard nudges (re-activating the session), the reclaim still
        runs AFTER the ``request_nudge`` defer_wake — which appends a ``wake_deferred`` span.
        If reclaim ran first it would archive the row and that span append would crash; here
        the reclaim no-ops (the nudged session is active) but the ORDERING is the fence."""
        from aios.harness.loop import run_session_step

        session = SimpleNamespace(
            id="sess_x",
            agent_id="agt_x",
            agent_version=None,
            focal_channel=None,
            origin="background",
            parent_run_id="run_x",
            archive_when_idle=True,
        )
        agent = SimpleNamespace(
            model="openrouter/x",
            tools=[],
            mcp_servers=[],
            http_servers=[],
            skills=[],
            system="sys",
            litellm_extra={},
            window_min=1000,
            window_max=10000,
        )
        start_event = SimpleNamespace(id="ev_step")

        manager = MagicMock()
        defer_wake_mock = AsyncMock()
        reclaim = AsyncMock(return_value=False)  # nudged → active → reclaim no-ops
        manager.attach_mock(defer_wake_mock, "defer_wake")
        manager.attach_mock(reclaim, "reclaim")

        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
            patch(
                "aios.harness.loop.runtime.require_inflight_tool_registry", return_value=MagicMock()
            ),
            patch(
                "aios.harness.loop.find_sessions_needing_inference",
                AsyncMock(return_value={"sess_x"}),
            ),
            patch(
                "aios.harness.loop.sessions_service.get_session_basic",
                AsyncMock(return_value=session),
            ),
            patch(
                "aios.harness.loop.agents_service.load_for_session", AsyncMock(return_value=agent)
            ),
            patch("aios.services.channels.list_session_channels", AsyncMock(return_value=[])),
            patch(
                "aios.harness.loop.sessions_service.read_windowed_events",
                AsyncMock(return_value=WindowedEvents(events=[], omission=None)),
            ),
            patch("aios.harness.loop._dispatch_confirmed_tools", AsyncMock(return_value=[])),
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
            patch("aios.harness.loop.sessions_service.set_session_stop_reason", AsyncMock()),
            patch(
                "aios.harness.loop.sessions_service.append_event",
                AsyncMock(return_value=start_event),
            ),
            patch(
                "aios.harness.loop.sessions_service.append_assistant_and_guard_quiescence",
                AsyncMock(return_value=AssistantAppendResult(True, None, None)),  # guard nudged
            ),
            patch("aios.harness.loop.sessions_service.reclaim_session_if_idle", reclaim),
            patch("aios.harness.loop.defer_wake", defer_wake_mock),
            patch(
                "aios.harness.loop.stream_litellm",
                AsyncMock(return_value=_llm_response({"role": "assistant", "content": "ok"})),
            ),
            patch("aios.harness.loop.sessions_service.increment_usage", AsyncMock()),
            patch("aios.db.sse_lock.has_subscriber", AsyncMock(return_value=False)),
        ):
            await run_session_step("sess_x")

        defer_wake_mock.assert_awaited_once_with(
            ANY, "sess_x", cause="request_nudge", account_id=ANY
        )
        reclaim.assert_awaited_once_with(ANY, "sess_x", account_id=ANY)
        nudge_idx = next(
            i for i, (name, *_r) in enumerate(manager.mock_calls) if name == "defer_wake"
        )
        reclaim_idx = next(
            i for i, (name, *_r) in enumerate(manager.mock_calls) if name == "reclaim"
        )
        assert reclaim_idx > nudge_idx, (
            f"reclaim must run after the nudge defer_wake; nudge at {nudge_idx}, reclaim at {reclaim_idx}"
        )

    async def test_step_end_emitted_when_budget_exhausted(self) -> None:
        """If the model-call budget is exhausted, ``step_end`` still fires."""
        from aios.harness.loop import run_session_step

        session = SimpleNamespace(
            id="sess_x",
            agent_id="agt_x",
            agent_version=None,
            focal_channel=None,
            origin="foreground",
            parent_run_id=None,
            archive_when_idle=False,
        )
        agent = SimpleNamespace(
            model="openrouter/x",
            tools=[],
            mcp_servers=[],
            http_servers=[],
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
                "aios.harness.loop.runtime.require_inflight_tool_registry",
                return_value=MagicMock(),
            ),
            patch(
                "aios.harness.loop.find_sessions_needing_inference",
                AsyncMock(return_value={"sess_x"}),
            ),
            patch(
                "aios.harness.loop.sessions_service.get_session_basic",
                AsyncMock(return_value=session),
            ),
            patch(
                "aios.harness.loop.agents_service.load_for_session",
                AsyncMock(return_value=agent),
            ),
            patch(
                "aios.services.channels.list_session_channels",
                AsyncMock(return_value=[]),
            ),
            patch(
                "aios.harness.loop.sessions_service.read_windowed_events",
                AsyncMock(return_value=WindowedEvents(events=[], omission=None)),
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
            patch("aios.harness.loop.sessions_service.set_session_stop_reason", AsyncMock()),
            patch("aios.harness.loop.sessions_service.append_event", append_event),
            patch(
                "aios.harness.loop.stream_litellm",
                AsyncMock(side_effect=RuntimeError("provider boom")),
            ),
            patch("aios.harness.loop.defer_wake", AsyncMock()),
            # The terminal-error branch fails the errored child's open requests;
            # this non-child session would no-op in production, so mock the DB-backed
            # collaborator out to keep the span-ordering assertion pool-free.
            patch("aios.harness.loop.fail_all_open_requests", AsyncMock(return_value=0)),
            patch(
                "aios.harness.loop._count_consecutive_rescheduling",
                AsyncMock(return_value=4),  # budget exhausted — no re-raise, returns None
            ),
        ):
            await run_session_step("sess_x")  # must NOT raise

        span_names = _span_event_names(append_event)
        assert span_names[0] == "step_start"
        assert span_names[-1] == "step_end"


@contextlib.asynccontextmanager
async def _harness_with_guard(
    guard_result: AssistantAppendResult,
) -> AsyncIterator[tuple[AsyncMock, AsyncMock, MagicMock]]:
    """Drive ``run_session_step`` past a clean (tool-call-free) model turn with the
    quiescence guard mocked to ``guard_result``. Yields ``(defer_wake,
    defer_run_wake, manager)`` — the manager records ``append_event`` and
    ``defer_wake`` calls in order so a test can assert the post-guard wiring AND
    that the wakes fire after ``step_end`` (loop.py: nudge -> defer_wake(
    request_nudge); autoerror -> defer_run_wake)."""
    session = SimpleNamespace(
        id="sess_x",
        agent_id="agt_x",
        agent_version=None,
        focal_channel=None,
        origin="background",
        parent_run_id="run_x",
        archive_when_idle=False,
    )
    agent = SimpleNamespace(
        model="openrouter/x",
        tools=[],
        mcp_servers=[],
        http_servers=[],
        skills=[],
        system="sys",
        litellm_extra={},
        window_min=1000,
        window_max=10000,
    )
    manager = MagicMock()
    append_event = AsyncMock(return_value=SimpleNamespace(id="ev"))
    defer_wake = AsyncMock()
    defer_run_wake = AsyncMock()
    manager.attach_mock(append_event, "append_event")
    manager.attach_mock(defer_wake, "defer_wake")
    with (
        patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
        patch("aios.harness.loop.runtime.require_inflight_tool_registry", return_value=MagicMock()),
        patch(
            "aios.harness.loop.find_sessions_needing_inference", AsyncMock(return_value={"sess_x"})
        ),
        patch(
            "aios.harness.loop.sessions_service.get_session_basic", AsyncMock(return_value=session)
        ),
        patch("aios.harness.loop.agents_service.load_for_session", AsyncMock(return_value=agent)),
        patch("aios.services.channels.list_session_channels", AsyncMock(return_value=[])),
        patch(
            "aios.harness.loop.sessions_service.read_windowed_events",
            AsyncMock(return_value=WindowedEvents(events=[], omission=None)),
        ),
        patch("aios.harness.loop._dispatch_confirmed_tools", AsyncMock(return_value=[])),
        patch(
            "aios.harness.loop.compose_step_context",
            AsyncMock(
                return_value=SimpleNamespace(
                    model="openrouter/x", messages=[], tools=[], reacting_to=0, skill_versions=[]
                )
            ),
        ),
        patch("aios.harness.loop.sessions_service.set_session_stop_reason", AsyncMock()),
        patch("aios.harness.loop.sessions_service.append_event", append_event),
        patch(
            "aios.harness.loop.sessions_service.append_assistant_and_guard_quiescence",
            AsyncMock(return_value=guard_result),
        ),
        patch(
            "aios.harness.loop.stream_litellm",
            AsyncMock(return_value=_llm_response({"role": "assistant", "content": "ok"})),
        ),
        patch("aios.harness.loop.sessions_service.increment_usage", AsyncMock()),
        patch("aios.db.sse_lock.has_subscriber", AsyncMock(return_value=False)),
        patch("aios.harness.loop.defer_wake", defer_wake),
        patch("aios.harness.loop.defer_run_wake", defer_run_wake),
    ):
        yield defer_wake, defer_run_wake, manager


class TestRequestTotalityWiring:
    """The loop's post-guard wiring (loop.py) — covers the True branches the rest of
    the suite leaves untested (every other loop-driving test mocks the guard to the
    all-False no-op)."""

    async def test_nudge_defers_a_session_wake_after_step_end(self) -> None:
        """guard says nudged -> the harness wakes the session (so the model gets
        another turn to answer), AFTER step_end so its wake_deferred lands in the
        next step's window (#132), and does NOT wake a caller run."""
        from aios.harness.loop import run_session_step

        async with _harness_with_guard(AssistantAppendResult(True, None, None)) as (
            defer_wake,
            defer_run_wake,
            manager,
        ):
            await run_session_step("sess_x")
        defer_wake.assert_awaited_once_with(ANY, "sess_x", account_id=ANY, cause="request_nudge")
        defer_run_wake.assert_not_awaited()
        # Ordering fence (mirrors the reschedule path): the nudge wake fires after
        # the step_end span append, never inside the body.
        last_step_end = max(
            i
            for i, (name, args, _kw) in enumerate(manager.mock_calls)
            if name == "append_event" and args[2] == "span" and args[3].get("event") == "step_end"
        )
        first_defer = [name for name, _a, _kw in manager.mock_calls].index("defer_wake")
        assert first_defer > last_step_end

    async def test_autoerror_defers_the_caller_run_wake(self) -> None:
        """guard says autoerrored -> the harness wakes the caller run to harvest the
        no_return response, and does NOT inject a session (nudge) wake."""
        from aios.harness.loop import run_session_step

        async with _harness_with_guard(AssistantAppendResult(False, "run_x", None)) as (
            defer_wake,
            defer_run_wake,
            _mgr,
        ):
            await run_session_step("sess_x")
        defer_run_wake.assert_awaited_once_with("run_x", batch=True)
        defer_wake.assert_not_awaited()
