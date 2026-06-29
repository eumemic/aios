from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from aios.harness.completion import LlmResponse
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
    inflight_tool_registry = MagicMock()
    inflight_tool_registry.in_flight_tool_call_ids.return_value = set()
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
            "aios.harness.loop.accounts_service.get_account_subtree_spend_state",
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
            pool, inflight_tool_registry, "sess_x", cause="message", account_id="acc_x"
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


async def test_preflight_gate_trips_on_subtree_rollup() -> None:
    """The pre-flight gate refuses to start when the ROLLED-UP subtree envelope
    breaches the effective limit — even though the account's own flat meter
    would be under the cap on its own (the descendant-cumulative case, #1279).

    The harness reads `get_account_subtree_spend_state` (P1's rollup), not the
    flat per-account meter, for the admission decision. To prove the gate keys
    on the SUBTREE value, the flat collaborator is wired well under the cap and
    must be ignored: only the rollup (1_500_000 µ$ ≥ a 1.0 USD limit) trips it.
    """
    pool = MagicMock()
    inflight_tool_registry = MagicMock()
    inflight_tool_registry.in_flight_tool_call_ids.return_value = set()
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
    # Flat per-account meter is well under the cap; only the subtree rollup trips.
    flat_state = AsyncMock(return_value=(100_000, 1.0))
    subtree_state = AsyncMock(return_value=(1_500_000, 1.0))
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
        patch("aios.harness.loop.accounts_service.get_account_spend_state", flat_state),
        patch("aios.harness.loop.accounts_service.get_account_subtree_spend_state", subtree_state),
        patch("aios.harness.loop.sessions_service.append_event", append_event),
        patch("aios.harness.loop.fail_all_open_requests", AsyncMock()) as fail_open,
        patch(
            "aios.harness.loop.sessions_service.set_session_stop_reason", AsyncMock()
        ) as set_stop,
        patch("aios.harness.loop.compose_step_context", AsyncMock()) as compose,
    ):
        result = await _run_session_step_body(
            pool, inflight_tool_registry, "sess_x", cause="message", account_id="acc_x"
        )

    assert result == _StepResult()
    # No dispatch — the gate fired BEFORE context build / the model call.
    compose.assert_not_awaited()
    # Admission keyed on the rollup, not the flat meter.
    subtree_state.assert_awaited_once_with(pool, "acc_x")
    fail_open.assert_awaited_once_with(
        ANY, "sess_x", account_id="acc_x", error={"kind": "spend_cap_exceeded"}
    )
    assert set_stop.await_args is not None
    stop_reason = set_stop.await_args.args[2]
    assert stop_reason["type"] == "error"
    span_payloads = [c.args[3] for c in append_event.call_args_list if c.args[2] == "span"]
    cap_spans = [p for p in span_payloads if p.get("event") == "spend_cap_exceeded"]
    assert cap_spans, "expected a spend_cap_exceeded span"
    # The span records the rolled-up subtree spend that breached the envelope.
    assert cap_spans[0]["spent_microusd"] == 1_500_000


async def test_preflight_gate_admits_when_subtree_under_limit() -> None:
    """A subtree rollup under the effective limit admits the step (dispatch
    proceeds past the gate). The flat meter being at/over its own row would not
    matter — the rollup is the envelope — but here both are under the cap.
    """
    pool = MagicMock()
    inflight_tool_registry = MagicMock()
    inflight_tool_registry.in_flight_tool_call_ids.return_value = set()
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
    compose = AsyncMock(side_effect=RuntimeError("stop after admission"))
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
            "aios.harness.loop.accounts_service.get_account_subtree_spend_state",
            AsyncMock(return_value=(500_000, 1.0)),  # subtree under the 1.0 USD cap
        ),
        patch(
            "aios.harness.loop.sessions_service.append_event",
            AsyncMock(return_value=SimpleNamespace(id="ev")),
        ),
        patch("aios.harness.loop.compose_step_context", compose),
        pytest.raises(RuntimeError, match="stop after admission"),
    ):
        await _run_session_step_body(
            pool, inflight_tool_registry, "sess_x", cause="message", account_id="acc_x"
        )

    # The gate ADMITTED the step — execution reached context build / dispatch.
    compose.assert_awaited_once()


async def test_usage_charged_only_after_assistant_persists() -> None:
    """Spend must be charged AFTER the assistant message durably persists.

    increment_usage previously committed BEFORE append_assistant_and_guard_quiescence;
    if the persist raises (a DB error caught upstream → retry that re-calls the
    model), the pre-persist charge double-bills the account for one response.
    With the charge after the persist, a failed persist bills nothing and the
    retry charges exactly once (fail-safe: under-, never over-charge)."""
    pool = MagicMock()
    inflight_tool_registry = MagicMock()
    inflight_tool_registry.in_flight_tool_call_ids.return_value = set()
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
            "aios.harness.loop.accounts_service.get_account_subtree_spend_state",
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
                return_value=LlmResponse.from_message(
                    {"role": "assistant", "content": "hi"},
                    usage={"input_tokens": 10, "output_tokens": 5},
                    cost=0.001,
                    finish_reason="stop",
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
            pool, inflight_tool_registry, "sess_x", cause="message", account_id="acc_x"
        )

    append_assistant.assert_awaited_once()
    # The charge comes AFTER the (failed) persist, so this attempt billed nothing.
    increment.assert_not_awaited()
