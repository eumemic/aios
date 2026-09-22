"""A ``finish_reason == "length"`` turn is surfaced, not reported as clean (#2451).

A completion that hit its output ceiling is a **truncated** turn: the assistant
text stops mid-sentence, or — when extended thinking consumed the whole budget
before any text was emitted — is empty. Before this change the harness recorded
it exactly like a clean ``stop``: full cost billed, no warning, no marker. The
only way to detect the class was noticing that ``output_tokens`` happened to
equal the cap exactly.

This is deliberately **observability, not a latch**. Unlike a ``content_filter``
refusal (which bricks the turn — see ``loop.REFUSAL_FINISH_REASON``), a truncated
turn's content is real, partial work; discarding it or erroring the session would
throw away paid-for output and could strand a session mid-tool-call. So the turn
still persists and dispatches as before, and the truncation is made *queryable*:
a ``step.model_output_truncated`` warning plus an ``output_truncated`` flag on the
``model_request_end`` span.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness.completion import LlmResponse
from aios.harness.loop import TRUNCATED_FINISH_REASON, _run_session_step_body
from aios.harness.window import WindowedEvents
from aios.models.model_providers import ProviderAuth

_SESSION = SimpleNamespace(
    id="sess_x",
    agent_id="agt_x",
    agent_version=None,
    focal_channel=None,
    origin="foreground",
    parent_run_id=None,
    archive_when_idle=False,
    token_baseline_v=1,
)
_AGENT = SimpleNamespace(
    model="anthropic/claude-opus-4-1",
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
_STEP_CTX = SimpleNamespace(
    messages=[{"role": "user", "content": "hi"}],
    tools=[],
    skill_versions=[],
    reacting_to=0,
    reminders_written=(),
    reminders_skipped=0,
)


def _enter_base_patches(stack: ExitStack) -> None:
    auth = ProviderAuth(api_key="sk-resolved", api_base=None, owner_account_id="acc_x")
    for target, mock in [
        ("aios.harness.loop.find_sessions_needing_inference", AsyncMock(return_value={"sess_x"})),
        ("aios.harness.loop.sessions_service.get_session_basic", AsyncMock(return_value=_SESSION)),
        ("aios.harness.loop.agents_service.load_for_session", AsyncMock(return_value=_AGENT)),
        ("aios.services.channels.list_session_channels", AsyncMock(return_value=[])),
        ("aios.harness.loop.refresh_session_mount_state", AsyncMock(return_value=[])),
        ("aios.harness.loop.compute_step_prelude", AsyncMock(return_value=SimpleNamespace())),
        (
            "aios.harness.loop.sessions_service.read_windowed_events",
            AsyncMock(return_value=WindowedEvents(events=[], omission=None)),
        ),
        ("aios.harness.loop._dispatch_confirmed_tools", AsyncMock(return_value=[])),
        ("aios.harness.loop.compose_step_context", AsyncMock(return_value=_STEP_CTX)),
        (
            "aios.harness.loop.model_providers_service.resolve_provider_auth_or_conflict",
            AsyncMock(return_value=(auth, None)),
        ),
        ("aios.harness.loop.has_subscriber", AsyncMock(return_value=False)),
        ("aios.harness.loop.sessions_service.increment_usage", AsyncMock(return_value=0)),
    ]:
        stack.enter_context(patch(target, mock))
    stack.enter_context(patch("aios.harness.loop.prelude_overhead_local", return_value=0))


async def _run_step_with_finish_reason(
    finish_reason: str | None, *, content: str = "partial answ"
) -> tuple[list[Any], list[Any]]:
    """Drive one step whose model call returns ``finish_reason``.

    Returns ``(span payloads, warning log calls)``.
    """
    pool = MagicMock()
    inflight_tool_registry = MagicMock()
    inflight_tool_registry.in_flight_tool_call_ids.return_value = set()
    append_event = AsyncMock(return_value=SimpleNamespace(id="ev", seq=1))
    warnings: list[Any] = []

    with ExitStack() as stack:
        _enter_base_patches(stack)
        stack.enter_context(patch("aios.harness.loop.sessions_service.append_event", append_event))
        stack.enter_context(
            patch(
                "aios.harness.loop.call_litellm",
                AsyncMock(
                    return_value=LlmResponse.from_message(
                        {"role": "assistant", "content": content},
                        usage={"input_tokens": 10, "output_tokens": 4096},
                        cost=0.001,
                        finish_reason=finish_reason,
                    )
                ),
            )
        )
        real_warning = __import__("aios.harness.loop", fromlist=["log"]).log.warning

        def spy_warning(event: str, **kw: Any) -> Any:
            warnings.append((event, kw))
            return real_warning(event, **kw)

        stack.enter_context(patch("aios.harness.loop.log.warning", spy_warning))
        # Stop right after the model response is recorded — nothing past it matters.
        stack.enter_context(
            patch(
                "aios.harness.loop.sessions_service.append_assistant_and_guard_quiescence",
                AsyncMock(side_effect=RuntimeError("stop after model call")),
            )
        )
        with pytest.raises(RuntimeError, match="stop after model call"):
            await _run_session_step_body(
                pool, inflight_tool_registry, "sess_x", cause="message", account_id="acc_x"
            )

    spans = [c.args[3] for c in append_event.call_args_list if c.args[2] == "span"]
    return spans, warnings


def _model_request_end(spans: list[Any]) -> dict[str, Any]:
    ends = [s for s in spans if s.get("event") == "model_request_end"]
    assert len(ends) == 1, f"expected exactly one model_request_end, got {len(ends)}"
    return dict(ends[0])


class TestOutputTruncationSurfaced:
    async def test_length_finish_reason_is_logged_and_stamped(self) -> None:
        """REQUIRED TEST 4 — a truncated completion is surfaced distinctly.

        Discriminating: before the fix the ``model_request_end`` span carried no
        ``output_truncated`` key at all and no warning was emitted, so both
        assertions failed.
        """
        spans, warnings = await _run_step_with_finish_reason(TRUNCATED_FINISH_REASON)

        end = _model_request_end(spans)
        assert end["output_truncated"] is True
        assert end["finish_reason"] == "length"
        assert end["is_error"] is False  # observability, not a latch

        truncation_warnings = [
            kw for event, kw in warnings if event == "step.model_output_truncated"
        ]
        assert len(truncation_warnings) == 1
        assert truncation_warnings[0]["finish_reason"] == "length"
        assert truncation_warnings[0]["output_tokens"] == 4096

    async def test_empty_truncated_turn_is_flagged_as_empty(self) -> None:
        """The catastrophic shape: thinking ate the whole budget, content is EMPTY.

        This is the turn that looked like a clean ``end_turn`` while delivering
        nothing at all, so the log must say so explicitly.
        """
        _spans, warnings = await _run_step_with_finish_reason(TRUNCATED_FINISH_REASON, content="")

        truncation_warnings = [
            kw for event, kw in warnings if event == "step.model_output_truncated"
        ]
        assert len(truncation_warnings) == 1
        assert truncation_warnings[0]["empty_content"] is True

    async def test_clean_stop_is_not_flagged(self) -> None:
        """The discriminator in the other direction: a clean turn must stay clean.

        A flag that fires on every turn would be as useless as no flag.
        """
        spans, warnings = await _run_step_with_finish_reason("stop")

        end = _model_request_end(spans)
        assert end["output_truncated"] is False
        assert not [event for event, _kw in warnings if event == "step.model_output_truncated"]

    async def test_tool_calls_finish_reason_is_not_flagged(self) -> None:
        spans, warnings = await _run_step_with_finish_reason("tool_calls")

        assert _model_request_end(spans)["output_truncated"] is False
        assert not [event for event, _kw in warnings if event == "step.model_output_truncated"]

    async def test_absent_finish_reason_is_not_flagged(self) -> None:
        """Providers that omit ``finish_reason`` must not be reported as truncated."""
        spans, _warnings = await _run_step_with_finish_reason(None)

        end = _model_request_end(spans)
        assert end["output_truncated"] is False
        assert end["finish_reason"] is None
