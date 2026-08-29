"""E2E: the MCP auto-review checker on the real step function (jarbot#229).

Real Postgres, real ``run_session_step``, scripted agent AND checker models
(``harness.script_checker`` — the checker FIFO keys on
``settings.auto_review_model``). MCP discovery and execution are stubbed at
``aios.mcp.client`` so no server is contacted; everything between — the
partition bucket, the background review task, the confirm/marker writes, the
awaiting view, the confirmed cold-dispatch, and the stranded-review sweep —
is the production code path.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

from aios.config import get_settings
from aios.harness.auto_review import AUTO_REVIEW_SOURCE, CHECKER_UNAVAILABLE_REASON
from aios.models.agents import (
    McpPermissionPolicy,
    McpServerSpec,
    McpToolConfig,
    McpToolsetConfig,
    ToolSpec,
)
from aios.services import agents as agents_service
from aios.services import environments as env_svc
from aios.services import sessions as sess_svc
from tests.e2e.harness import Harness, assistant, tool_call

ACCOUNT_ID = "acc_test_stub"
SERVER = "srv"
TOOL = f"mcp__{SERVER}__do_thing"

_DISCOVERED = [
    {
        "type": "function",
        "function": {
            "name": TOOL,
            "description": "does the thing",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": True},
        },
    }
]


async def _fake_discover(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], str | None]:
    return list(_DISCOVERED), None


async def _fake_auth(*args: Any, **kwargs: Any) -> tuple[str | None, dict[str, str]]:
    return None, {}


async def _fake_call_mcp_tool(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {"content": "did the thing"}


def _mcp_patches() -> list[Any]:
    return [
        mock.patch("aios.mcp.client.discover_mcp_tools", _fake_discover),
        mock.patch("aios.mcp.client.resolve_auth_for_mcp_mount", _fake_auth),
        mock.patch("aios.mcp.client.call_mcp_tool", _fake_call_mcp_tool),
    ]


async def _start_auto_review_session(
    harness: Harness,
    *,
    configs: list[McpToolConfig] | None = None,
    description: str | None = "Accountant. Keeps the books and answers mail.",
) -> str:
    """Agent with one auto_review-defaulted MCP toolset + a session asking for the tool."""
    from aios.ids import make_id

    agent = await agents_service.create_agent(
        harness._pool,
        name=f"auto-review-{make_id('agent')[-8:]}",
        model="fake/test",
        system="You are a test assistant.",
        tools=[
            ToolSpec(
                type="mcp_toolset",
                mcp_server_name=SERVER,
                default_config=McpToolsetConfig(
                    permission_policy=McpPermissionPolicy(type="auto_review")
                ),
                configs=configs,
            )
        ],
        mcp_servers=[McpServerSpec(name=SERVER, url="https://mcp-fake.invalid")],
        description=description,
        metadata={},
        window_min=50_000,
        window_max=150_000,
        account_id=ACCOUNT_ID,
    )
    if harness._env_id is None:
        env = await env_svc.create_environment(
            harness._pool, name=f"test-env-{make_id('env')[-8:]}", account_id=ACCOUNT_ID
        )
        harness._env_id = env.id
    session = await sess_svc.create_session(
        harness._pool,
        agent_id=agent.id,
        environment_id=harness._env_id,
        title="auto-review-e2e",
        metadata={},
        account_id=ACCOUNT_ID,
    )
    await sess_svc.append_user_message(
        harness._pool, session.id, "do the thing with srv", account_id=ACCOUNT_ID
    )
    return session.id


def _events_of(events: list[Any], kind: str, event: str) -> list[Any]:
    return [e for e in events if e.kind == kind and e.data.get("event") == event]


async def test_allow_executes_out_of_band(harness: Harness) -> None:
    """allow → verdict logged, confirmed with source, cold-dispatched, tool runs."""
    harness.script_model([assistant(tool_calls=[tool_call(TOOL, {}, call_id="call_ar1")])])
    harness.script_checker([{"verdict": "allow", "reason": "user asked for exactly this"}])

    with _patch_all(_mcp_patches()):
        session_id = await _start_auto_review_session(harness)
        await harness.run_step(session_id)

        # The turn ended immediately — the review resolves out-of-band.
        s = await harness.session(session_id)
        assert s.stop_reason == {"type": "end_turn"}

        await harness.wait_for_tools(session_id)  # drains the review: task too

        events = await harness.all_events(session_id)
        verdicts = [
            e for e in events if e.kind == "span" and e.data.get("event") == "mcp_auto_review"
        ]
        assert len(verdicts) == 1
        assert verdicts[0].data["verdict"] == "allow"
        assert verdicts[0].data["tool_call_id"] == "call_ar1"
        assert verdicts[0].data["model"] == get_settings().auto_review_model
        confirmed = _events_of(events, "lifecycle", "tool_confirmed")
        assert len(confirmed) == 1
        assert confirmed[0].data["source"] == AUTO_REVIEW_SOURCE
        assert confirmed[0].data["result"] == "allow"
        assert _events_of(events, "lifecycle", "tool_requested") == []  # no card, ever

        # The checker saw fenced args, the scope, and the user's line — and the
        # agent's scripted FIFO was untouched by the checker call.
        assert len(harness.checker_calls) == 1
        checker_user_msg = harness.checker_calls[0]["messages"][1]["content"]
        assert "<<<ARGS" in checker_user_msg
        assert "Accountant. Keeps the books" in checker_user_msg
        assert "do the thing with srv" in checker_user_msg

        # The confirmed call cold-dispatches at the top of the next step
        # (``_dispatch_confirmed_tools`` — no inference burned) and executes.
        await harness.run_step(session_id)
        await harness.wait_for_tools(session_id)
        events = await harness.all_events(session_id)
        results = [
            e
            for e in events
            if e.kind == "message"
            and e.data.get("role") == "tool"
            and e.data.get("tool_call_id") == "call_ar1"
        ]
        assert len(results) == 1
        assert "did the thing" in results[0].data["content"]


async def test_ask_holds_card_with_reason(harness: Harness) -> None:
    """ask → tool_requested marker with the one-line reason; awaiting surfaces it."""
    harness.script_model([assistant(tool_calls=[tool_call(TOOL, {}, call_id="call_ar2")])])
    harness.script_checker([{"verdict": "ask", "reason": "recipient the user never mentioned"}])

    with _patch_all(_mcp_patches()):
        session_id = await _start_auto_review_session(harness)
        await harness.run_step(session_id)
        await harness.wait_for_tools(session_id)

        events = await harness.all_events(session_id)
        requested = _events_of(events, "lifecycle", "tool_requested")
        assert len(requested) == 1
        assert requested[0].data["tool_call_id"] == "call_ar2"
        assert requested[0].data["kind"] == "mcp"
        assert requested[0].data["reason"] == "recipient the user never mentioned"
        assert requested[0].data["source"] == AUTO_REVIEW_SOURCE
        assert _events_of(events, "lifecycle", "tool_confirmed") == []

        s = await harness.session(session_id)
        assert s.status == "active"
        assert {a.tool_call_id for a in s.awaiting} == {"call_ar2"}
        assert s.awaiting[0].kind == "mcp"


async def test_checker_junk_fails_closed(harness: Harness) -> None:
    """Junk output twice (one retry) → ask with the checker-unavailable copy."""
    harness.script_model([assistant(tool_calls=[tool_call(TOOL, {}, call_id="call_ar3")])])
    harness.script_checker(["not json at all", "still not json"])

    with _patch_all(_mcp_patches()):
        session_id = await _start_auto_review_session(harness)
        await harness.run_step(session_id)
        await harness.wait_for_tools(session_id)

        events = await harness.all_events(session_id)
        requested = _events_of(events, "lifecycle", "tool_requested")
        assert len(requested) == 1
        assert requested[0].data["reason"] == CHECKER_UNAVAILABLE_REASON
        assert len(harness.checker_calls) == 2  # exactly one retry


async def test_always_allow_config_skips_checker(harness: Harness) -> None:
    """A per-tool always_allow under an auto_review default never reaches luna."""
    harness.script_model(
        [
            assistant(tool_calls=[tool_call(TOOL, {}, call_id="call_ar4")]),
            assistant("done"),
        ]
    )
    harness.script_checker([])  # any checker call would fail the scripted FIFO

    with _patch_all(_mcp_patches()):
        session_id = await _start_auto_review_session(
            harness,
            configs=[
                McpToolConfig(
                    name="do_thing",
                    permission_policy=McpPermissionPolicy(type="always_allow"),
                )
            ],
        )
        await harness.run_until_idle(session_id)

        assert harness.checker_calls == []
        events = await harness.all_events(session_id)
        assert [
            e for e in events if e.kind == "span" and e.data.get("event") == "mcp_auto_review"
        ] == []
        assert _events_of(events, "lifecycle", "tool_requested") == []
        results = [
            e
            for e in events
            if e.kind == "message"
            and e.data.get("role") == "tool"
            and e.data.get("tool_call_id") == "call_ar4"
        ]
        assert len(results) == 1 and "did the thing" in results[0].data["content"]


async def test_stranded_review_is_fail_closed_by_sweep(harness: Harness) -> None:
    """A review task killed with the worker → no card until the sweep holds one."""
    harness.script_model([assistant(tool_calls=[tool_call(TOOL, {}, call_id="call_ar5")])])
    harness.script_checker([{"verdict": "allow", "reason": "never lands"}])

    with _patch_all(_mcp_patches()):
        session_id = await _start_auto_review_session(harness)
        await harness.run_step(session_id)
        # Kill the worker mid-review: the review task dies appending nothing.
        await harness.simulate_sigkill(session_id)

        # Under review with no marker: NOT awaiting — no premature card.
        s = await harness.session(session_id)
        assert s.awaiting == []

        # The sweep fail-closes it once past the stranded bound.
        stranded_now = get_settings().model_copy(update={"auto_review_stranded_after_s": 0.0})
        with mock.patch("aios.harness.sweep.get_settings", lambda: stranded_now):
            await harness.run_ghost_repair(session_id)

        events = await harness.all_events(session_id)
        requested = _events_of(events, "lifecycle", "tool_requested")
        assert len(requested) == 1
        assert requested[0].data["reason"] == CHECKER_UNAVAILABLE_REASON
        assert requested[0].data["source"] == AUTO_REVIEW_SOURCE
        # No fabricated tool result — the call is held for the user, not errored.
        assert [
            e
            for e in events
            if e.kind == "message"
            and e.data.get("role") == "tool"
            and e.data.get("tool_call_id") == "call_ar5"
        ] == []
        s = await harness.session(session_id)
        assert {a.tool_call_id for a in s.awaiting} == {"call_ar5"}


class _patch_all:
    def __init__(self, patches: list[Any]) -> None:
        self._patches = patches

    def __enter__(self) -> None:
        for p in self._patches:
            p.start()

    def __exit__(self, *exc: Any) -> None:
        for p in self._patches:
            p.stop()
