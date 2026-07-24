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
    _CONTEXT_OVERFLOW_SHRINK_BASE,
    _RETRY_BACKOFF_SECONDS,
    _STREAK_TRANSPARENT_LIFECYCLE_EVENTS,
    _apply_context_overflow_retry,
    _count_consecutive_context_overflow,
    _count_consecutive_rescheduling,
    _is_terminal_model_error,
    _provider_error_detail,
    _retry_delay_for_attempt,
    run_session_step,
)
from aios.harness.window import WindowedEvents

# These tests deliberately exercise downstream provider retry semantics.


@pytest.fixture(autouse=True)
def _legacy_inference_policy(legacy_env: None) -> None:
    """This suite intentionally reaches model behavior beyond credential admission."""


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
        ) as read_windowed_events_mock,
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
            read_windowed_events=read_windowed_events_mock,
            session=session,
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


class TestRunSessionStepOnContextOverflow:
    """Overflow is non-retryable-as-is: each retry shrinks, and the ladder is
    bounded so a persistently-oversized request lands terminal instead of
    looping verbatim (the 2026-07-09 Ultron/sol outage class)."""

    async def test_first_overflow_shrinks_to_80pct(self, mock_step_dependencies: Any) -> None:
        mock_step_dependencies.stream_litellm.side_effect = _make_litellm_error(
            litellm_exceptions.ContextWindowExceededError
        )

        with patch(
            "aios.harness.loop._count_consecutive_context_overflow",
            AsyncMock(return_value=0),
        ):
            await run_session_step("sess_x")

        mock_step_dependencies.defer_wake.assert_awaited_once_with(
            ANY, "sess_x", cause="reschedule", delay_seconds=2, account_id=ANY
        )
        # The stop_reason stamps the shrink factor the next build reads, so the
        # retry request is strictly smaller — never the identical payload.
        assert any(
            call.args[2]
            == {"type": "rescheduling", "context_overflow": True, "context_shrink_factor": 0.8}
            for call in mock_step_dependencies.set_stop_reason.call_args_list
        )
        mock_step_dependencies.fail_all_open_requests.assert_not_awaited()

    @pytest.mark.parametrize(
        "error_class,message",
        [
            (
                litellm_exceptions.BadGatewayError,
                "Your input exceeds the context window of this model",
            ),
            (litellm_exceptions.ServiceUnavailableError, "input exceeds context window"),
        ],
    )
    async def test_gateway_wrapped_overflow_uses_adaptive_recovery(
        self,
        mock_step_dependencies: Any,
        error_class: Any,
        message: str,
    ) -> None:
        request = httpx.Request("POST", "https://example.test/v1")
        response = httpx.Response(502, request=request)
        mock_step_dependencies.stream_litellm.side_effect = error_class(
            message=message,
            model="x",
            llm_provider="gateway",
            response=response,
        )

        with patch(
            "aios.harness.loop._count_consecutive_context_overflow",
            AsyncMock(return_value=0),
        ):
            await run_session_step("sess_x")

        mock_step_dependencies.defer_wake.assert_awaited_once_with(
            ANY, "sess_x", cause="reschedule", delay_seconds=2, account_id=ANY
        )
        lifecycle_payloads = [
            call.args[3] for call in mock_step_dependencies.append_event.call_args_list
        ]
        assert {
            "event": "adaptive_context_truncation",
            "axis": "tokens",
            "shrink_factor": 0.8,
        } in lifecycle_payloads
        mock_step_dependencies.fail_all_open_requests.assert_not_awaited()

    async def test_adaptive_retry_drops_configured_window_band(
        self, mock_step_dependencies: Any
    ) -> None:
        mock_step_dependencies.session.stop_reason = {
            "type": "rescheduling",
            "context_overflow": True,
            "context_shrink_factor": 0.8,
        }
        mock_step_dependencies.stream_litellm.side_effect = _make_litellm_error(
            litellm_exceptions.ContextWindowExceededError
        )

        with patch(
            "aios.harness.loop._count_consecutive_context_overflow",
            AsyncMock(return_value=1),
        ):
            await run_session_step("sess_x")

        read_call = mock_step_dependencies.read_windowed_events.await_args
        assert read_call.kwargs["window_min"] == 0
        assert read_call.kwargs["window_max"] == 8_000

    async def test_repeated_overflow_shrinks_progressively_and_is_not_verbatim(
        self, mock_step_dependencies: Any
    ) -> None:
        """Regression: a SECOND overflow must tighten further (0.64) on a longer
        backoff, not re-fire the same 80% request forever."""
        mock_step_dependencies.stream_litellm.side_effect = _make_litellm_error(
            litellm_exceptions.ContextWindowExceededError
        )

        with patch(
            "aios.harness.loop._count_consecutive_context_overflow",
            AsyncMock(return_value=1),  # one prior overflow already at 0.8
        ):
            await run_session_step("sess_x")

        mock_step_dependencies.defer_wake.assert_awaited_once_with(
            ANY, "sess_x", cause="reschedule", delay_seconds=8, account_id=ANY
        )
        assert any(
            call.args[2]
            == {"type": "rescheduling", "context_overflow": True, "context_shrink_factor": 0.64}
            for call in mock_step_dependencies.set_stop_reason.call_args_list
        )
        mock_step_dependencies.fail_all_open_requests.assert_not_awaited()

    async def test_overflow_budget_exhausted_lands_terminal(
        self, mock_step_dependencies: Any
    ) -> None:
        """Once the backoff ladder is spent, overflow stops retrying and latches
        the session errored — the unbounded identical-retry loop is gone."""
        mock_step_dependencies.stream_litellm.side_effect = _make_litellm_error(
            litellm_exceptions.ContextWindowExceededError
        )

        with patch(
            "aios.harness.loop._count_consecutive_context_overflow",
            AsyncMock(return_value=len(_RETRY_BACKOFF_SECONDS)),  # budget spent
        ):
            await run_session_step("sess_x")  # must NOT raise

        mock_step_dependencies.defer_wake.assert_not_awaited()
        mock_step_dependencies.fail_all_open_requests.assert_awaited_once_with(
            ANY, "sess_x", account_id=ANY, error={"kind": "context_overflow"}
        )
        stop_reasons = [
            call.args[2] for call in mock_step_dependencies.set_stop_reason.call_args_list
        ]
        assert {"type": "error"} in stop_reasons
        # No further rescheduling stop_reason recorded on the terminal attempt.
        assert not any(reason.get("context_overflow") is True for reason in stop_reasons)


class TestCountConsecutiveContextOverflow:
    async def test_counts_only_trailing_context_overflow_streak(self) -> None:
        # Newest-first: overflow, overflow, end_turn, overflow. The end_turn
        # breaks the streak, so only the two trailing overflows count.
        events_newest_first = [
            SimpleNamespace(data={"event": "turn_ended", "stop_reason": "context_overflow"}),
            SimpleNamespace(data={"event": "turn_ended", "stop_reason": "context_overflow"}),
            SimpleNamespace(data={"event": "turn_ended", "stop_reason": "end_turn"}),
            SimpleNamespace(data={"event": "turn_ended", "stop_reason": "context_overflow"}),
        ]
        pool = MagicMock()
        with patch(
            "aios.harness.loop.sessions_service.read_events",
            AsyncMock(return_value=events_newest_first),
        ):
            assert (
                await _count_consecutive_context_overflow(
                    pool, "sess_x", account_id="acc_test_stub"
                )
                == 2
            )

    async def test_rescheduling_tail_does_not_count_as_overflow(self) -> None:
        """A plain transient rescheduling is a DIFFERENT streak — it must not be
        mistaken for an overflow retry (which would over-shrink)."""
        events_newest_first = [
            SimpleNamespace(data={"event": "turn_ended", "stop_reason": "rescheduling"}),
        ]
        pool = MagicMock()
        with patch(
            "aios.harness.loop.sessions_service.read_events",
            AsyncMock(return_value=events_newest_first),
        ):
            assert (
                await _count_consecutive_context_overflow(
                    pool, "sess_x", account_id="acc_test_stub"
                )
                == 0
            )


def _turn_ended(stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(data={"event": "turn_ended", "stop_reason": stop_reason})


def _adaptive_truncation(shrink_factor: float) -> SimpleNamespace:
    """The diagnostic breadcrumb ``_apply_context_overflow_retry`` writes
    immediately BEFORE each overflow ``turn_ended``."""
    return SimpleNamespace(
        data={
            "event": "adaptive_context_truncation",
            "axis": "tokens",
            "shrink_factor": shrink_factor,
        }
    )


def _overflow_log_newest_first(attempts: int) -> list[SimpleNamespace]:
    """Realistic newest-first lifecycle tail after ``attempts`` overflow retries.

    Each retry writes ``adaptive_context_truncation`` then ``turn_ended``, so
    newest-first the pairs come back (turn_ended, adaptive), interleaved.
    """
    events: list[SimpleNamespace] = []
    for n in range(attempts, 0, -1):
        events.append(_turn_ended("context_overflow"))
        events.append(_adaptive_truncation(round(_CONTEXT_OVERFLOW_SHRINK_BASE**n, 4)))
    return events


class TestOverflowStreakSurvivesDiagnosticEvents:
    """Regression for the #2010 review block: the ``adaptive_context_truncation``
    breadcrumb sits between consecutive overflow ``turn_ended`` events. If the
    streak counter treats it as a streak breaker, every attempt past the second
    reads 1 — the shrink factor sticks at 0.64 and the backoff never reaches the
    end of the ladder, so an oversized prompt retries FOREVER instead of latching
    errored. These tests drive the real counter (no mock) over a realistic log.
    """

    @pytest.mark.parametrize("attempts", [1, 2, 3, 4])
    async def test_counter_ignores_interleaved_diagnostics(self, attempts: int) -> None:
        pool = MagicMock()
        with patch(
            "aios.harness.loop.sessions_service.read_events",
            AsyncMock(return_value=_overflow_log_newest_first(attempts)),
        ):
            assert (
                await _count_consecutive_context_overflow(
                    pool, "sess_x", account_id="acc_test_stub"
                )
                == attempts
            )

    async def test_read_limit_covers_a_full_interleaved_streak(self) -> None:
        """The read window must be wide enough to hold the whole ladder PLUS its
        interleaved diagnostics; otherwise a live streak is silently truncated
        and the ladder stalls just as surely as a miscount would."""
        pool = MagicMock()
        read_events = AsyncMock(return_value=_overflow_log_newest_first(4))
        with patch("aios.harness.loop.sessions_service.read_events", read_events):
            await _count_consecutive_context_overflow(pool, "sess_x", account_id="acc_test_stub")
        await_args = read_events.await_args
        assert await_args is not None
        limit = await_args.kwargs["limit"]
        # Full ladder + the terminal attempt, each with its diagnostic breadcrumb.
        assert limit >= (len(_RETRY_BACKOFF_SECONDS) + 1) * 2

    async def test_real_turn_boundary_still_breaks_the_streak(self) -> None:
        """The fix must skip ONLY the diagnostic event — a genuine intervening
        turn outcome still resets the ladder (overflow after real progress is a
        fresh incident, not attempt N+1)."""
        events_newest_first = [
            _turn_ended("context_overflow"),
            _adaptive_truncation(0.8),
            _turn_ended("end_turn"),
            _adaptive_truncation(0.64),
            _turn_ended("context_overflow"),
        ]
        pool = MagicMock()
        with patch(
            "aios.harness.loop.sessions_service.read_events",
            AsyncMock(return_value=events_newest_first),
        ):
            assert (
                await _count_consecutive_context_overflow(
                    pool, "sess_x", account_id="acc_test_stub"
                )
                == 1
            )

    async def test_rescheduling_streak_also_survives_diagnostics(self) -> None:
        """The transparency lives in the shared helper, so the sibling
        transient-error counter gets the same treatment."""
        events_newest_first = [
            _turn_ended("rescheduling"),
            _adaptive_truncation(0.8),
            _turn_ended("rescheduling"),
        ]
        pool = MagicMock()
        with patch(
            "aios.harness.loop.sessions_service.read_events",
            AsyncMock(return_value=events_newest_first),
        ):
            assert (
                await _count_consecutive_rescheduling(pool, "sess_x", account_id="acc_test_stub")
                == 2
            )

    async def test_diagnostic_event_is_declared_streak_transparent(self) -> None:
        assert "adaptive_context_truncation" in _STREAK_TRANSPARENT_LIFECYCLE_EVENTS


class TestOverflowLadderProgressesToExhaustion:
    """End-to-end over the ladder: walk the real counter and the real applier
    together across every attempt and assert the shrink factor keeps tightening,
    the backoff keeps lengthening, and the session finally latches errored."""

    async def test_ladder_tightens_each_attempt_then_latches_errored(self) -> None:
        pool = MagicMock()
        appended: list[dict[str, Any]] = []
        stop_reasons: list[dict[str, Any]] = []

        async def _append_event(_pool: Any, _sid: str, _kind: str, data: Any, **_kw: Any) -> None:
            appended.append(data)

        async def _set_stop_reason(_pool: Any, _sid: str, data: Any, **_kw: Any) -> None:
            stop_reasons.append(data)

        # The log grows exactly as production writes it: every applier call
        # appends its diagnostic + turn_ended, and the counter reads it back.
        async def _read_events(*_a: Any, limit: int = 200, **_kw: Any) -> list[SimpleNamespace]:
            tail = [SimpleNamespace(data=d) for d in appended]
            return list(reversed(tail))[:limit]

        latched: list[str] = []

        async def _latch(_pool: Any, _sid: str, *, error_kind: str, **_kw: Any) -> None:
            latched.append(error_kind)

        delays: list[float | None] = []
        with (
            patch("aios.harness.loop.sessions_service.read_events", _read_events),
            patch("aios.harness.loop.sessions_service.append_event", _append_event),
            patch("aios.harness.loop.sessions_service.set_session_stop_reason", _set_stop_reason),
            patch("aios.harness.loop._latch_errored_turn", _latch),
        ):
            # One call per rung of the ladder, plus one past the end.
            for _ in range(len(_RETRY_BACKOFF_SECONDS) + 1):
                delays.append(
                    await _apply_context_overflow_retry(pool, "sess_x", account_id="acc_test_stub")
                )

        # Every rung retried on its own (strictly increasing) backoff...
        assert delays[:-1] == _RETRY_BACKOFF_SECONDS
        # ...and the budget is spent exactly once, landing terminal.
        assert delays[-1] is None
        assert latched == ["context_overflow"]

        # The shrink factor tightened on EVERY attempt — never stuck at 0.64.
        shrinks = [
            d["shrink_factor"] for d in appended if d.get("event") == "adaptive_context_truncation"
        ]
        assert shrinks == [
            round(_CONTEXT_OVERFLOW_SHRINK_BASE ** (n + 1), 4)
            for n in range(len(_RETRY_BACKOFF_SECONDS))
        ]
        assert shrinks == sorted(shrinks, reverse=True)
        assert len(set(shrinks)) == len(shrinks)

        # The stop_reason handed to the next build tracked the same ladder.
        overflow_stops = [s for s in stop_reasons if s.get("context_overflow") is True]
        assert [s["context_shrink_factor"] for s in overflow_stops] == shrinks
