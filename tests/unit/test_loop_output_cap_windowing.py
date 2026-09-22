"""``loop`` windows against the SAME cap the wire will carry (#2451 / #2453).

The table in ``test_output_cap_table`` recomposes windowing from
``resolve_output_cap``; this pins that the step body actually does so, so the
loop cannot drift back to a second, independently-derived reservation.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness import completion
from aios.harness.loop import _run_session_step_body
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


def _agent(litellm_extra: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        model="anthropic/claude-opus-4-1",
        tools=[],
        mcp_servers=[],
        http_servers=[],
        skills=[],
        system="sys",
        litellm_extra=litellm_extra,
        window_min=1000,
        window_max=10000,
        preempt_policy="wait",
    )


class _Stop(Exception):
    pass


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"max_output_tokens": 1111, "max_tokens": 2222, "max_completion_tokens": 3333},
        {"max_tokens": 0, "max_completion_tokens": 3333},
        {"max_output_tokens": "x"},
    ],
    ids=["none", "all-three", "invalid-then-valid", "invalid-only"],
)
async def test_step_windows_with_the_resolved_wire_cap(extra: dict[str, Any]) -> None:
    captured: dict[str, Any] = {}

    def spy(**kwargs: Any) -> int:
        captured.update(kwargs)
        raise _Stop

    auth = ProviderAuth(api_key="sk", api_base=None, owner_account_id="acc_x")
    with ExitStack() as stack:
        for target, mock in [
            (
                "aios.harness.loop.find_sessions_needing_inference",
                AsyncMock(return_value={"sess_x"}),
            ),
            (
                "aios.harness.loop.sessions_service.get_session_basic",
                AsyncMock(return_value=_SESSION),
            ),
            (
                "aios.harness.loop.agents_service.load_for_session",
                AsyncMock(return_value=_agent(extra)),
            ),
            ("aios.services.channels.list_session_channels", AsyncMock(return_value=[])),
            ("aios.harness.loop.refresh_session_mount_state", AsyncMock(return_value=[])),
            ("aios.harness.loop.compute_step_prelude", AsyncMock(return_value=SimpleNamespace())),
            (
                "aios.harness.loop.model_providers_service.resolve_provider_auth_or_conflict",
                AsyncMock(return_value=(auth, None)),
            ),
            (
                "aios.harness.loop.sessions_service.append_event",
                AsyncMock(return_value=SimpleNamespace(id="ev", seq=1)),
            ),
        ]:
            stack.enter_context(patch(target, mock))
        stack.enter_context(patch("aios.harness.loop.effective_window_max", spy))
        registry = MagicMock()
        registry.in_flight_tool_call_ids.return_value = set()
        with pytest.raises(_Stop):
            await _run_session_step_body(
                MagicMock(), registry, "sess_x", cause="message", account_id="acc_x"
            )

    cap = completion.resolve_output_cap("anthropic/claude-opus-4-1", extra)
    wire = completion._build_litellm_kwargs(
        model="anthropic/claude-opus-4-1",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        auth=None,
        extra=dict(extra),
        session_id="s",
        stream=False,
    )
    assert captured["output_reserve"] == cap.value == wire["max_tokens"]
    assert captured["context_limit"] == cap.context_limit
