"""The builtin tool-surface default-deny at its REAL enforcement site (#1683).

Issue #1683's boundary, stated precisely: **on the session model path, a
model-emitted ``tool_use`` naming a builtin that is not in the executing
session's clamped tool surface must never reach its handler** — the builtin
analog of the default-deny MCP tools already get via ``mcp_tool_suppressed``.
The reported exposure was a low-trust child session seeded ``tools=[]`` still
executing same-account metadata readers (``trigger_list``, ``list_runs``,
``list_account_triggers``).

That enforcement EXISTS — it is the offered-set partition in
``_run_session_step_body`` (``loop.py``, the ``offered_names`` / ``offered_calls``
/ ``unoffered_calls`` split), landed later and for a different stated reason
(#1773 defect 2, commit a876da88). ``step_ctx.tools`` is built by
``compose_step_context`` from ``to_openai_tools(agent.tools)``, and for a
workflow-spawned child ``agent.tools`` IS the frozen clamped surface
(``services/agents.py::load_for_session`` overlays it) — so the offered set is
the projection of the clamped surface, and anything outside it is denied.

Nothing pinned that partition, though. ``test_tool_dispatch.py``'s coverage
calls ``reject_unoffered_tool_calls`` DIRECTLY, so it exercises the rejection
message but never the decision that classifies a call as unoffered: delete the
partition in ``loop.py`` and those tests stay green while the boundary is gone.
These two tests close exactly that hole, driving the real
``_run_session_step_body`` end to end:

* the DENY direction — an ungranted builtin on an empty surface is routed to
  rejection and never to ``launch_tool_calls``; and
* the POSITIVE CONTROL — a granted builtin on the same path still dispatches,
  so the guard cannot be satisfied by a build that blocks everything.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aios.harness.completion import LlmResponse
from aios.harness.loop import _run_session_step_body
from aios.harness.window import WindowedEvents
from aios.models.agents import ToolSpec
from aios.models.model_providers import ProviderAuth

_AUTH = ProviderAuth(api_key="sk-test", api_base=None, owner_account_id="acc_x")

_SESSION = SimpleNamespace(
    id="sess_x",
    agent_id="agt_x",
    agent_version=None,
    focal_channel=None,
    origin="foreground",
    parent_run_id="run_parent",  # a workflow-spawned child: born clamped
    archive_when_idle=False,
)


def _agent(tools: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(
        model="openrouter/x",
        tools=tools,
        mcp_servers=[],
        http_servers=[],
        skills=[],
        system="sys",
        litellm_extra={},
        window_min=1000,
        window_max=10000,
        preempt_policy="wait",
    )


def _openai_tool(name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def _drive(
    stack: ExitStack,
    *,
    offered: list[str],
    called: str,
    agent_tools: list[Any],
) -> tuple[MagicMock, MagicMock]:
    """Patch ``_run_session_step_body`` up to the tool-dispatch tail.

    ``offered`` is the FROZEN tool array the step sends with this inference (the
    projection of the session's clamped surface); ``called`` is the tool name the
    model emits. Returns the ``(reject, launch)`` spies the partition feeds.
    """
    step_ctx = SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}],
        tools=[_openai_tool(n) for n in offered],
        skill_versions=[],
        reacting_to=0,
        reminders_written=(),
        reminders_skipped=0,
    )
    assistant_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "tc_1", "function": {"name": called, "arguments": "{}"}}],
    }
    guard_result = SimpleNamespace(
        nudged=False,
        autoerror_caller_run_id=None,
        autoerror_caller_session_ids=[],
        assistant_focal_at_arrival=None,
    )
    for target, mock in [
        ("aios.harness.loop.find_sessions_needing_inference", AsyncMock(return_value={"sess_x"})),
        ("aios.harness.loop.sessions_service.get_session_basic", AsyncMock(return_value=_SESSION)),
        (
            "aios.harness.loop.agents_service.load_for_session",
            AsyncMock(return_value=_agent(agent_tools)),
        ),
        ("aios.services.channels.list_session_channels", AsyncMock(return_value=[])),
        ("aios.harness.loop.refresh_session_mount_state", AsyncMock(return_value=[])),
        ("aios.harness.loop.compute_step_prelude", AsyncMock(return_value=SimpleNamespace())),
        (
            "aios.harness.loop.sessions_service.read_windowed_events",
            AsyncMock(return_value=WindowedEvents(events=[], omission=None)),
        ),
        ("aios.harness.loop._dispatch_confirmed_tools", AsyncMock(return_value=[])),
        ("aios.harness.loop.compose_step_context", AsyncMock(return_value=step_ctx)),
        (
            "aios.harness.loop.sessions_service.append_event",
            AsyncMock(return_value=SimpleNamespace(id="ev")),
        ),
        ("aios.harness.loop.has_subscriber", AsyncMock(return_value=False)),
        (
            # The unit env runs the ``account_only`` credential policy, so an
            # unresolved provider auth latches the step BEFORE the dispatch tail.
            # Resolve it cleanly — this suite is about the tool-surface boundary.
            "aios.harness.loop.model_providers_service.resolve_provider_auth_or_conflict",
            AsyncMock(return_value=(_AUTH, None)),
        ),
        (
            "aios.harness.loop.call_litellm",
            AsyncMock(
                return_value=LlmResponse.from_message(
                    assistant_msg,
                    usage={"input_tokens": 10, "output_tokens": 5},
                    cost=0.001,
                    finish_reason="tool_calls",
                )
            ),
        ),
        (
            "aios.harness.loop.sessions_service.append_assistant_and_guard_quiescence",
            AsyncMock(return_value=guard_result),
        ),
        ("aios.harness.loop.sessions_service.increment_usage", AsyncMock(return_value=0)),
        ("aios.harness.loop.sessions_service.set_session_stop_reason", AsyncMock()),
    ]:
        stack.enter_context(patch(target, mock))
    stack.enter_context(patch("aios.harness.loop.prelude_overhead_local", return_value=0))
    reject = stack.enter_context(patch("aios.harness.loop.reject_unoffered_tool_calls"))
    launch = stack.enter_context(patch("aios.harness.loop.launch_tool_calls"))
    return reject, launch


async def _run() -> None:
    pool = MagicMock()
    inflight_tool_registry = MagicMock()
    inflight_tool_registry.in_flight_tool_call_ids.return_value = set()
    await _run_session_step_body(
        pool, inflight_tool_registry, "sess_x", cause="message", account_id="acc_x"
    )


async def test_ungranted_builtin_is_denied_on_an_empty_clamped_surface() -> None:
    """#1683: the reported exposure, at the enforcement site.

    A child session whose clamped surface offered NO tools emits a call to the
    registered metadata reader ``trigger_list``. The partition must route it to
    rejection and it must never reach ``launch_tool_calls`` (hence never
    ``invoke_builtin``, never the handler, never Postgres).
    """
    with ExitStack() as stack:
        reject, launch = _drive(stack, offered=[], called="trigger_list", agent_tools=[])
        await _run()

    launch.assert_not_called()
    reject.assert_called_once()
    denied = reject.call_args.args[2]
    assert [c["function"]["name"] for c in denied] == ["trigger_list"]
    assert reject.call_args.kwargs["offered_names"] == []


async def test_granted_builtin_still_dispatches_positive_control() -> None:
    """Positive control: the guard must not be satisfied by blocking everything.

    Same path, same tool — but now ``bash`` IS in the frozen offered surface, so
    it must reach ``launch_tool_calls`` and NOT the rejection path.
    """
    with ExitStack() as stack:
        reject, launch = _drive(
            stack,
            offered=["bash"],
            called="bash",
            agent_tools=[ToolSpec(type="bash")],
        )
        await _run()

    reject.assert_not_called()
    launch.assert_called_once()
    dispatched = launch.call_args.args[2]
    assert [c["function"]["name"] for c in dispatched] == ["bash"]
