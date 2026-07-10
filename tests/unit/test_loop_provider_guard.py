"""The provider-auth conflict guard in the inline-model-call arm of `_run_session_step_body`.

Mirrors `test_loop_spend_gate.py`'s shape: drive `_run_session_step_body` far
enough to reach the guard (past the spend gate and context build), then assert
either the latch fires (conflict) or the resolved auth reaches `call_litellm`
(clean path). The conftest-level `_unit_provider_auth_ungated` fixture stubs
both collaborators to a clean pass for every OTHER unit test; these tests
override them at the `aios.harness.loop.model_providers_service` call site.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from aios.harness.completion import LlmResponse
from aios.harness.loop import _run_session_step_body, _StepResult
from aios.harness.window import WindowedEvents
from aios.models.model_providers import ProviderAuth

_SESSION = SimpleNamespace(
    id="sess_x",
    agent_id="agt_x",
    agent_version=None,
    focal_channel=None,
    origin="foreground",
    parent_run_id=None,
)
_AGENT = SimpleNamespace(
    model="openrouter/x",
    tools=[],
    mcp_servers=[],
    http_servers=[],
    skills=[],
    system="sys",
    litellm_extra={"api_base": "https://evil.example"},
    window_min=1000,
    window_max=10000,
    preempt_policy="wait",
)
_STEP_CTX = SimpleNamespace(
    messages=[{"role": "user", "content": "hi"}], tools=[], skill_versions=[], reacting_to=0
)


def _enter_base_patches(
    stack: ExitStack, *, resolved: ProviderAuth | None, conflict: str | None
) -> None:
    """Everything needed to drive `_run_session_step_body` past the spend gate
    and context build, up to (and including) the provider-auth guard.
    """
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
            AsyncMock(return_value=(resolved, conflict)),
        ),
    ]:
        stack.enter_context(patch(target, mock))
    stack.enter_context(patch("aios.harness.loop.prelude_overhead_local", return_value=0))


async def test_conflict_latches_errored_before_model_call() -> None:
    pool = MagicMock()
    inflight_tool_registry = MagicMock()
    inflight_tool_registry.in_flight_tool_call_ids.return_value = set()
    append_event = AsyncMock(return_value=SimpleNamespace(id="ev"))
    with ExitStack() as stack:
        _enter_base_patches(stack, resolved=None, conflict="conflict message")
        stack.enter_context(patch("aios.harness.loop.sessions_service.append_event", append_event))
        fail_open = stack.enter_context(
            patch("aios.harness.loop.fail_all_open_requests", AsyncMock())
        )
        set_stop = stack.enter_context(
            patch("aios.harness.loop.sessions_service.set_session_stop_reason", AsyncMock())
        )
        call_litellm = stack.enter_context(patch("aios.harness.loop.call_litellm", AsyncMock()))
        stream_litellm = stack.enter_context(patch("aios.harness.loop.stream_litellm", AsyncMock()))
        result = await _run_session_step_body(
            pool, inflight_tool_registry, "sess_x", cause="message", account_id="acc_x"
        )

    assert result == _StepResult()  # no retry_delay — deterministic config error, not retried
    call_litellm.assert_not_awaited()
    stream_litellm.assert_not_awaited()
    fail_open.assert_awaited_once_with(
        ANY, "sess_x", account_id="acc_x", error={"kind": "provider_auth_conflict"}
    )
    assert set_stop.await_args is not None
    stop_reason = set_stop.await_args.args[2]
    assert stop_reason == {"type": "error", "message": "conflict message"}
    span_payloads = [c.args[3] for c in append_event.call_args_list if c.args[2] == "span"]
    assert any(p.get("event") == "provider_auth_conflict" for p in span_payloads)


async def test_clean_path_forwards_resolved_auth_to_call_litellm() -> None:
    pool = MagicMock()
    inflight_tool_registry = MagicMock()
    inflight_tool_registry.in_flight_tool_call_ids.return_value = set()
    auth = ProviderAuth(api_key="sk-resolved", api_base=None, owner_account_id="acc_x")
    with ExitStack() as stack:
        _enter_base_patches(stack, resolved=auth, conflict=None)
        stack.enter_context(
            patch(
                "aios.harness.loop.sessions_service.append_event",
                AsyncMock(return_value=SimpleNamespace(id="ev")),
            )
        )
        stack.enter_context(
            patch("aios.harness.loop.has_subscriber", AsyncMock(return_value=False))
        )
        call_litellm = stack.enter_context(
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
            )
        )
        # Stop right after the model call — nothing past it matters here.
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

    assert call_litellm.await_args is not None
    assert call_litellm.await_args.kwargs["auth"] is auth
