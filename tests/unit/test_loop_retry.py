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
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import httpx
import litellm.exceptions as litellm_exceptions
import pytest

from aios.harness.completion import ModelCallDeadlineError
from aios.harness.loop import (
    _count_consecutive_rescheduling,
    _is_terminal_model_error,
    _provider_error_detail,
    _retry_delay_for_attempt,
    run_session_step,
)
from aios.harness.window import WindowedEvents


def _make_litellm_error(cls: type[Exception]) -> Exception:
    """Construct a litellm exception instance, supplying per-class required args.

    Some 4xx/5xx classes (PermissionDeniedError, UnprocessableEntityError,
    ServiceUnavailableError) require an httpx ``response``; others don't. This
    builds a valid instance of each so the tests exercise real litellm types
    through the predicate / handler rather than stand-ins.
    """
    request = httpx.Request("POST", "https://example.test/v1")
    response = httpx.Response(400, request=request)
    kwargs: dict[str, Any] = {"message": "boom", "model": "x", "llm_provider": "y"}
    if cls in (
        litellm_exceptions.PermissionDeniedError,
        litellm_exceptions.UnprocessableEntityError,
        litellm_exceptions.ServiceUnavailableError,
    ):
        kwargs["response"] = response
    return cls(**kwargs)


_TERMINAL_ERROR_CLASSES = [
    litellm_exceptions.BadRequestError,
    litellm_exceptions.AuthenticationError,
    litellm_exceptions.ContextWindowExceededError,
    litellm_exceptions.ContentPolicyViolationError,
    litellm_exceptions.PermissionDeniedError,
    litellm_exceptions.NotFoundError,
    litellm_exceptions.UnprocessableEntityError,
]

_TRANSIENT_ERROR_CLASSES = [
    litellm_exceptions.RateLimitError,
    litellm_exceptions.APIConnectionError,
    litellm_exceptions.InternalServerError,
    litellm_exceptions.ServiceUnavailableError,
    litellm_exceptions.Timeout,
]


class TestRetryDelayForAttempt:
    def test_retry_delay_attempt_0_returns_2s(self) -> None:
        assert _retry_delay_for_attempt(0) == 2

    def test_retry_delay_attempt_1_returns_8s(self) -> None:
        assert _retry_delay_for_attempt(1) == 8

    def test_retry_delay_attempt_2_returns_30s(self) -> None:
        assert _retry_delay_for_attempt(2) == 30

    def test_retry_delay_attempt_3_returns_120s(self) -> None:
        assert _retry_delay_for_attempt(3) == 120

    def test_retry_delay_attempt_4_returns_none(self) -> None:
        assert _retry_delay_for_attempt(4) is None


class TestCountConsecutiveRescheduling:
    async def test_count_consecutive_rescheduling_empty_log_returns_zero(self) -> None:
        pool = MagicMock()
        with patch(
            "aios.harness.loop.sessions_service.read_events",
            AsyncMock(return_value=[]),
        ):
            assert (
                await _count_consecutive_rescheduling(pool, "sess_x", account_id="acc_test_stub")
                == 0
            )

    async def test_count_consecutive_rescheduling_resets_on_non_rescheduling_tail(
        self,
    ) -> None:
        """Regression: a clean turn_ended breaks the streak even if reschedulings preceded it."""
        # Lifecycle log (oldest → newest):
        #   resched, resched, end_turn, resched
        # The counter reads newest-first, so it sees resched (count=1) then
        # end_turn which breaks the streak. Only the trailing single counts.
        events_newest_first = [
            SimpleNamespace(data={"event": "turn_ended", "stop_reason": "rescheduling"}),
            SimpleNamespace(data={"event": "turn_ended", "stop_reason": "end_turn"}),
            SimpleNamespace(data={"event": "turn_ended", "stop_reason": "rescheduling"}),
            SimpleNamespace(data={"event": "turn_ended", "stop_reason": "rescheduling"}),
        ]
        pool = MagicMock()
        with patch(
            "aios.harness.loop.sessions_service.read_events",
            AsyncMock(return_value=events_newest_first),
        ):
            assert (
                await _count_consecutive_rescheduling(pool, "sess_x", account_id="acc_test_stub")
                == 1
            )

    async def test_count_consecutive_rescheduling_reads_bounded_newest_first_tail(
        self,
    ) -> None:
        """The default ASC + LIMIT 200 scan drops the recent tail on long
        sessions (the bulk of lifecycle events are early ``turn_ended``).
        Guard the call shape: newest-first and a bound small enough that a
        long session's ancient rows can't crowd out recent reschedulings.
        """
        mock_read = AsyncMock(
            return_value=[
                SimpleNamespace(data={"event": "turn_ended", "stop_reason": "rescheduling"})
            ]
            * 5
        )
        pool = MagicMock()
        with patch("aios.harness.loop.sessions_service.read_events", mock_read):
            assert (
                await _count_consecutive_rescheduling(pool, "sess_x", account_id="acc_test_stub")
                == 5
            )
        kwargs = mock_read.call_args.kwargs
        assert kwargs["newest_first"] is True
        assert kwargs["limit"] <= 10


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
        origin="foreground",
        parent_run_id=None,
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
        preempt_policy="wait",
    )
    start_event = SimpleNamespace(id="ev_start")

    with (
        patch(
            "aios.harness.loop.runtime.require_pool",
            return_value=MagicMock(),
        ),
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
        # Channels helpers are imported lazily inside run_session_step —
        # patch them at their source module rather than on loop.
        patch(
            "aios.services.channels.list_session_channels",
            AsyncMock(return_value=[]),
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
            AsyncMock(return_value=WindowedEvents(events=[], omission=None)),
        ),
        patch(
            "aios.harness.loop._dispatch_confirmed_tools",
            AsyncMock(return_value=[]),
        ),
        patch(
            "aios.harness.loop.accounts_service.get_account_spend_state",
            AsyncMock(return_value=(0, None)),
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
        patch(
            "aios.harness.loop.sessions_service.set_session_stop_reason",
            AsyncMock(),
        ) as set_stop_reason,
        patch(
            "aios.harness.loop.sessions_service.append_event",
            AsyncMock(return_value=start_event),
        ) as append_event,
        patch(
            "aios.harness.loop.stream_litellm",
            AsyncMock(side_effect=RuntimeError("provider boom")),
        ) as stream_litellm_mock,
        patch(
            "aios.harness.loop.sessions_service.increment_usage",
            AsyncMock(),
        ) as increment_usage_mock,
        patch(
            "aios.harness.loop.defer_wake",
            AsyncMock(),
        ) as defer_wake_mock,
        # The terminal-error branch errors the errored child's open requests on its
        # behalf. The session here is a non-child, so the real call would no-op;
        # mock it to keep this unit focused on the retry state machine and assert
        # whether the hook fired.
        patch(
            "aios.harness.loop.fail_all_open_requests",
            AsyncMock(return_value=0),
        ) as fail_all_open_requests_mock,
    ):
        yield SimpleNamespace(
            set_stop_reason=set_stop_reason,
            append_event=append_event,
            defer_wake=defer_wake_mock,
            fail_all_open_requests=fail_all_open_requests_mock,
            stream_litellm=stream_litellm_mock,
            increment_usage=increment_usage_mock,
        )


class TestRunSessionStepOnModelError:
    """End-to-end behavior of the exception handler's state machine."""

    async def test_first_attempt_defers_retry_with_2s(self, mock_step_dependencies: Any) -> None:
        with patch(
            "aios.harness.loop._count_consecutive_rescheduling",
            AsyncMock(return_value=0),
        ):
            await run_session_step("sess_x")

        mock_step_dependencies.defer_wake.assert_awaited_once_with(
            ANY, "sess_x", cause="reschedule", delay_seconds=2, account_id=ANY
        )
        # ``set_session_stop_reason(pool, session_id, stop_reason)`` — status is
        # derived now, so the rescheduling state is recorded only via stop_reason
        # (and the turn_ended/rescheduling lifecycle event).
        stop_reasons = [
            call.args[2] for call in mock_step_dependencies.set_stop_reason.call_args_list
        ]
        assert {"type": "rescheduling"} in stop_reasons
        assert {"type": "end_turn"} not in stop_reasons
        # The retry branch never reaches the terminal landing pad, so it must not
        # fail the session's requests — a reschedulable error is not terminal.
        mock_step_dependencies.fail_all_open_requests.assert_not_awaited()

    async def test_exhausted_budget_records_error_stop_reason_without_raising(
        self, mock_step_dependencies: Any
    ) -> None:
        """Budget exhaustion parks the session in the derived ``errored`` state
        (#353) — recorded as ``stop_reason={"type": "error"}`` plus the
        ``turn_ended``/``error`` lifecycle event the sweep keys off.

        The inner model-call handler returns ``None`` instead of re-raising, so
        ``run_session_step`` completes cleanly; the outer ``harness_error``
        handler never fires for model-call errors, preventing an extra
        ``_apply_retry_or_failure`` call that would re-record ``rescheduling``.
        """
        with patch(
            "aios.harness.loop._count_consecutive_rescheduling",
            AsyncMock(return_value=4),
        ):
            await run_session_step("sess_x")  # must NOT raise

        mock_step_dependencies.defer_wake.assert_not_awaited()
        stop_reasons = [
            call.args[2] for call in mock_step_dependencies.set_stop_reason.call_args_list
        ]
        assert {"type": "error"} in stop_reasons
        assert {"type": "end_turn"} not in stop_reasons
        # The terminal landing pad fails the session's open requests on its behalf:
        # a workflow child's owed requests get a monotonic child_errored response so
        # the invoking runs resolve instead of hanging (a no-op for non-children).
        mock_step_dependencies.fail_all_open_requests.assert_awaited_once_with(
            ANY, "sess_x", account_id=ANY, error={"kind": "child_errored"}
        )

    async def test_streaming_deadline_records_usage_and_parks_without_retry(
        self, mock_step_dependencies: Any
    ) -> None:
        usage = {
            "input_tokens": 11,
            "output_tokens": 22,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 4,
        }
        mock_step_dependencies.stream_litellm.side_effect = ModelCallDeadlineError(
            "deadline",
            usage=usage,
            cost_usd=1.25,
            chunks_seen=2,
        )

        await run_session_step("sess_x")

        mock_step_dependencies.defer_wake.assert_not_awaited()
        mock_step_dependencies.increment_usage.assert_awaited_once_with(
            ANY,
            "sess_x",
            input_tokens=11,
            output_tokens=22,
            cache_read_input_tokens=3,
            cache_creation_input_tokens=4,
            cost_microusd=1_250_000,
            account_id=ANY,
        )
        mock_step_dependencies.fail_all_open_requests.assert_awaited_once_with(
            ANY, "sess_x", account_id=ANY, error={"kind": "model_call_deadline"}
        )
        stop_reasons = [
            call.args[2] for call in mock_step_dependencies.set_stop_reason.call_args_list
        ]
        assert any(
            reason.get("type") == "error"
            and "partial token usage was recorded" in reason.get("message", "")
            for reason in stop_reasons
        )
        span_payloads = [
            call.args[3]
            for call in mock_step_dependencies.append_event.call_args_list
            if call.args[2] == "span"
        ]
        assert {
            "event": "model_request_end",
            "model_request_start_id": "ev_start",
            "is_error": True,
            "model_usage": usage,
            "cost_usd": 1.25,
            "provider_error": {
                "exception_class": "ModelCallDeadlineError",
                "http_status": None,
                "message": "deadline",
            },
        } in span_payloads
        lifecycle_payloads = [
            call.args[3]
            for call in mock_step_dependencies.append_event.call_args_list
            if call.args[2] == "lifecycle"
        ]
        assert any(payload.get("stop_reason") == "error" for payload in lifecycle_payloads)
        assert not any(
            payload.get("stop_reason") == "rescheduling" for payload in lifecycle_payloads
        )

    async def test_zero_chunk_deadline_uses_retry_path(self, mock_step_dependencies: Any) -> None:
        mock_step_dependencies.stream_litellm.side_effect = ModelCallDeadlineError(
            "deadline",
            usage={},
            cost_usd=None,
            chunks_seen=0,
        )

        with patch(
            "aios.harness.loop._count_consecutive_rescheduling",
            AsyncMock(return_value=0),
        ):
            await run_session_step("sess_x")

        mock_step_dependencies.defer_wake.assert_awaited_once_with(
            ANY, "sess_x", cause="reschedule", delay_seconds=2, account_id=ANY
        )
        mock_step_dependencies.increment_usage.assert_not_awaited()
        mock_step_dependencies.fail_all_open_requests.assert_not_awaited()
        span_payloads = [
            call.args[3]
            for call in mock_step_dependencies.append_event.call_args_list
            if call.args[2] == "span"
        ]
        assert {
            "event": "model_request_end",
            "model_request_start_id": "ev_start",
            "is_error": True,
            "model_usage": {},
            "cost_usd": None,
            "provider_error": {
                "exception_class": "ModelCallDeadlineError",
                "http_status": None,
                "message": "deadline",
            },
        } in span_payloads


class TestIsTerminalModelError:
    """Pure-predicate table test for ``_is_terminal_model_error``."""

    @pytest.mark.parametrize("cls", _TERMINAL_ERROR_CLASSES)
    def test_terminal_classes_are_terminal(self, cls: type[Exception]) -> None:
        assert _is_terminal_model_error(_make_litellm_error(cls)) is True

    @pytest.mark.parametrize("cls", _TRANSIENT_ERROR_CLASSES)
    def test_transient_classes_are_not_terminal(self, cls: type[Exception]) -> None:
        assert _is_terminal_model_error(_make_litellm_error(cls)) is False

    def test_bare_runtime_error_is_not_terminal(self) -> None:
        # Pins the fixture's default ``RuntimeError("provider boom")`` path as
        # still-transient (falls through to the backoff ladder).
        assert _is_terminal_model_error(RuntimeError("provider boom")) is False


class TestRunSessionStepOnTerminalModelError:
    """The terminal-first branch inside the generic model-call ``except``."""

    @pytest.mark.parametrize("cls", _TERMINAL_ERROR_CLASSES)
    async def test_terminal_class_latches_errored_without_retry(
        self, mock_step_dependencies: Any, cls: type[Exception]
    ) -> None:
        mock_step_dependencies.stream_litellm.side_effect = _make_litellm_error(cls)

        # Crucially WITHOUT patching ``_count_consecutive_rescheduling`` high:
        # the ladder must be bypassed regardless of attempt count.
        await run_session_step("sess_x")  # must NOT raise

        mock_step_dependencies.defer_wake.assert_not_awaited()

        recorded_reasons = [
            call.args[2] for call in mock_step_dependencies.set_stop_reason.call_args_list
        ]
        # Subset check: the terminal latch passes ``stop_message``, so the
        # recorded reason is ``{"type": "error", "message": ...}``.
        assert any({"type": "error"}.items() <= recorded.items() for recorded in recorded_reasons)

        mock_step_dependencies.fail_all_open_requests.assert_awaited_once_with(
            ANY, "sess_x", account_id=ANY, error={"kind": "model_terminal_error"}
        )

    @pytest.mark.parametrize("cls", _TRANSIENT_ERROR_CLASSES)
    async def test_transient_class_keeps_backoff_ladder(
        self, mock_step_dependencies: Any, cls: type[Exception]
    ) -> None:
        mock_step_dependencies.stream_litellm.side_effect = _make_litellm_error(cls)

        with patch(
            "aios.harness.loop._count_consecutive_rescheduling",
            AsyncMock(return_value=0),
        ):
            await run_session_step("sess_x")

        mock_step_dependencies.defer_wake.assert_awaited_once_with(
            ANY, "sess_x", cause="reschedule", delay_seconds=2, account_id=ANY
        )
        mock_step_dependencies.fail_all_open_requests.assert_not_awaited()


class TestProviderErrorDetail:
    """Pure extraction of the durable diagnostic triple (#1442).

    The errored-turn span must carry the provider failure detail (exception
    class + HTTP status + message) so a ``child_errored`` latch is diagnosable
    straight from Postgres — not by reconstructing the cause from span timings.
    """

    def test_litellm_rate_limit_carries_429_status_and_message(self) -> None:
        request = httpx.Request("POST", "https://example.test/v1")
        response = httpx.Response(429, request=request)
        exc = litellm_exceptions.RateLimitError(
            message="provider is rate limiting", model="x", llm_provider="y", response=response
        )

        detail = _provider_error_detail(exc)

        assert detail["exception_class"] == "RateLimitError"
        assert detail["http_status"] == 429
        assert "rate limiting" in detail["message"]

    def test_internal_server_error_carries_5xx_status(self) -> None:
        exc = litellm_exceptions.InternalServerError(
            message="overloaded_error 529", model="x", llm_provider="y"
        )

        detail = _provider_error_detail(exc)

        assert detail["exception_class"] == "InternalServerError"
        # litellm maps InternalServerError to a 5xx status code.
        assert detail["http_status"] == 500
        assert "529" in detail["message"]

    def test_non_http_exception_has_null_status(self) -> None:
        detail = _provider_error_detail(RuntimeError("provider boom"))

        assert detail["exception_class"] == "RuntimeError"
        assert detail["http_status"] is None
        assert detail["message"] == "provider boom"

    def test_deadline_error_has_null_status_and_keeps_message(self) -> None:
        exc = ModelCallDeadlineError("deadline exceeded", usage={}, cost_usd=None, chunks_seen=0)

        detail = _provider_error_detail(exc)

        assert detail["exception_class"] == "ModelCallDeadlineError"
        assert detail["http_status"] is None
        assert detail["message"] == "deadline exceeded"

    def test_oversized_message_is_truncated(self) -> None:
        detail = _provider_error_detail(RuntimeError("x" * 5000))

        # Bounded so a pathological multi-KB provider body can't bloat the span.
        assert len(detail["message"]) <= 2001
        assert detail["message"].endswith("…")

    def test_non_int_status_code_is_dropped(self) -> None:
        class WeirdError(Exception):
            status_code = "not-a-number"

        detail = _provider_error_detail(WeirdError("boom"))

        assert detail["http_status"] is None


def _error_span_provider_errors(mock_deps: Any) -> list[dict[str, Any]]:
    """All ``provider_error`` payloads stamped on errored model-request spans."""
    return [
        call.args[3]["provider_error"]
        for call in mock_deps.append_event.call_args_list
        if call.args[2] == "span"
        and call.args[3].get("event") == "model_request_end"
        and call.args[3].get("is_error") is True
        and "provider_error" in call.args[3]
    ]


class TestErroredSpanCarriesProviderError:
    """End-to-end: the persisted errored span carries the provider detail (#1442)."""

    async def test_budget_exhausted_child_errored_span_has_provider_error(
        self, mock_step_dependencies: Any
    ) -> None:
        request = httpx.Request("POST", "https://example.test/v1")
        response = httpx.Response(429, request=request)
        mock_step_dependencies.stream_litellm.side_effect = litellm_exceptions.RateLimitError(
            message="rate limited", model="x", llm_provider="y", response=response
        )

        with patch(
            "aios.harness.loop._count_consecutive_rescheduling",
            AsyncMock(return_value=4),  # budget spent → child_errored latch
        ):
            await run_session_step("sess_x")

        provider_errors = _error_span_provider_errors(mock_step_dependencies)
        assert provider_errors, "errored span must persist provider_error"
        detail = provider_errors[-1]
        assert detail["exception_class"] == "RateLimitError"
        assert detail["http_status"] == 429
        assert "rate limited" in detail["message"]

    async def test_retry_span_also_carries_provider_error(
        self, mock_step_dependencies: Any
    ) -> None:
        mock_step_dependencies.stream_litellm.side_effect = RuntimeError("provider boom")

        with patch(
            "aios.harness.loop._count_consecutive_rescheduling",
            AsyncMock(return_value=0),  # still retrying
        ):
            await run_session_step("sess_x")

        provider_errors = _error_span_provider_errors(mock_step_dependencies)
        assert provider_errors
        assert provider_errors[-1] == {
            "exception_class": "RuntimeError",
            "http_status": None,
            "message": "provider boom",
        }

    async def test_terminal_error_span_carries_provider_error(
        self, mock_step_dependencies: Any
    ) -> None:
        mock_step_dependencies.stream_litellm.side_effect = _make_litellm_error(
            litellm_exceptions.AuthenticationError
        )

        await run_session_step("sess_x")

        provider_errors = _error_span_provider_errors(mock_step_dependencies)
        assert provider_errors
        assert provider_errors[-1]["exception_class"] == "AuthenticationError"
        assert provider_errors[-1]["http_status"] == 401
