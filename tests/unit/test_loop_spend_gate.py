from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from aios.harness.loop import (
    _crossed_spend_warning_threshold,
    _limit_to_microusd,
    _run_session_step_body,
    _StepResult,
)
from aios.harness.window import WindowedEvents


def test_limit_to_microusd_allows_none_zero_and_rounding() -> None:
    assert _limit_to_microusd(None) is None
    assert _limit_to_microusd(0) == 0
    assert _limit_to_microusd(1.2345674) == 1_234_567


def test_spend_warning_crossing_fires_once() -> None:
    limit = 1_000_000
    assert _crossed_spend_warning_threshold(800_000, 10_000, limit) is True
    assert _crossed_spend_warning_threshold(810_000, 10_000, limit) is False
    assert _crossed_spend_warning_threshold(790_000, 10_000, limit) is False
    assert _crossed_spend_warning_threshold(800_000, 0, limit) is False
    assert _crossed_spend_warning_threshold(800_000, 10_000, None) is False


async def test_spend_gate_trips_before_context_build() -> None:
    pool = MagicMock()
    task_registry = MagicMock()
    task_registry.in_flight_tool_call_ids.return_value = set()
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
    )
    append_event = AsyncMock(return_value=SimpleNamespace(id="ev"))
    with (
        patch(
            "aios.harness.loop.find_sessions_needing_inference", AsyncMock(return_value={"sess_x"})
        ),
        patch(
            "aios.harness.loop.sessions_service.get_session_basic", AsyncMock(return_value=session)
        ),
        patch("aios.harness.loop.agents_service.load_for_session", AsyncMock(return_value=agent)),
        patch("aios.services.channels.list_session_channels", AsyncMock(return_value=[])),
        patch("aios.harness.loop.refresh_session_mount_state", AsyncMock(return_value=[])),
        patch("aios.harness.loop.compute_step_prelude", AsyncMock(return_value=SimpleNamespace())),
        patch("aios.harness.loop.prelude_overhead_local", return_value=0),
        patch(
            "aios.harness.loop.sessions_service.read_windowed_events",
            AsyncMock(return_value=WindowedEvents(events=[], omission=None)),
        ),
        patch("aios.harness.loop._dispatch_confirmed_tools", AsyncMock(return_value=[])),
        patch(
            "aios.harness.loop.accounts_service.get_account_spend_state",
            AsyncMock(return_value=(1_000_000, 1.0)),
        ),
        patch("aios.harness.loop.sessions_service.append_event", append_event),
        patch("aios.harness.loop.fail_all_open_requests", AsyncMock()) as fail_open,
        patch(
            "aios.harness.loop.sessions_service.set_session_stop_reason", AsyncMock()
        ) as set_stop,
        patch("aios.harness.loop.compose_step_context", AsyncMock()) as compose,
    ):
        result = await _run_session_step_body(
            pool, task_registry, "sess_x", cause="message", account_id="acc_x"
        )

    assert result == _StepResult()
    compose.assert_not_awaited()
    fail_open.assert_awaited_once_with(
        ANY, "sess_x", account_id="acc_x", error={"kind": "spend_cap_exceeded"}
    )
    assert set_stop.await_args is not None
    stop_reason = set_stop.await_args.args[2]
    assert stop_reason["type"] == "error"
    span_payloads = [c.args[3] for c in append_event.call_args_list if c.args[2] == "span"]
    assert any(p.get("event") == "spend_cap_exceeded" for p in span_payloads)


async def test_usage_charged_only_after_assistant_persists() -> None:
    """Spend must be charged AFTER the assistant message durably persists.

    increment_usage previously committed BEFORE append_assistant_and_guard_quiescence;
    if the persist raises (a DB error caught upstream → retry that re-calls the
    model), the pre-persist charge double-bills the account for one response.
    With the charge after the persist, a failed persist bills nothing and the
    retry charges exactly once (fail-safe: under-, never over-charge)."""
    pool = MagicMock()
    task_registry = MagicMock()
    task_registry.in_flight_tool_call_ids.return_value = set()
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
    )
    step_ctx = SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}], tools=[], skill_versions=[], reacting_to=0
    )
    increment = AsyncMock(return_value=5)
    # The persist fails — models the soft path (DB error caught upstream → retry).
    append_assistant = AsyncMock(side_effect=RuntimeError("persist failed"))
    with (
        patch(
            "aios.harness.loop.find_sessions_needing_inference", AsyncMock(return_value={"sess_x"})
        ),
        patch(
            "aios.harness.loop.sessions_service.get_session_basic", AsyncMock(return_value=session)
        ),
        patch("aios.harness.loop.agents_service.load_for_session", AsyncMock(return_value=agent)),
        patch("aios.services.channels.list_session_channels", AsyncMock(return_value=[])),
        patch("aios.harness.loop.refresh_session_mount_state", AsyncMock(return_value=[])),
        patch("aios.harness.loop.compute_step_prelude", AsyncMock(return_value=SimpleNamespace())),
        patch("aios.harness.loop.prelude_overhead_local", return_value=0),
        patch(
            "aios.harness.loop.sessions_service.read_windowed_events",
            AsyncMock(return_value=WindowedEvents(events=[], omission=None)),
        ),
        patch("aios.harness.loop._dispatch_confirmed_tools", AsyncMock(return_value=[])),
        patch(
            "aios.harness.loop.accounts_service.get_account_spend_state",
            AsyncMock(return_value=(0, 100.0)),  # well under the cap → gate passes
        ),
        patch(
            "aios.harness.loop.sessions_service.append_event",
            AsyncMock(return_value=SimpleNamespace(id="ev")),
        ),
        patch("aios.harness.loop.compose_step_context", AsyncMock(return_value=step_ctx)),
        patch("aios.harness.loop.has_subscriber", AsyncMock(return_value=False)),
        patch(
            "aios.harness.loop.call_litellm",
            AsyncMock(
                return_value=(
                    {"role": "assistant", "content": "hi"},
                    {"input_tokens": 10, "output_tokens": 5},
                    0.001,
                    "stop",
                )
            ),
        ),
        patch("aios.harness.loop.sessions_service.increment_usage", increment),
        patch(
            "aios.harness.loop.sessions_service.append_assistant_and_guard_quiescence",
            append_assistant,
        ),
        pytest.raises(RuntimeError, match="persist failed"),
    ):
        await _run_session_step_body(
            pool, task_registry, "sess_x", cause="message", account_id="acc_x"
        )

    append_assistant.assert_awaited_once()
    # The charge comes AFTER the (failed) persist, so this attempt billed nothing.
    increment.assert_not_awaited()
