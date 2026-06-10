"""run_tools: surface gating + routing for a workflow run's ``tool()`` calls.

Pure in-memory — the run is a stand-in carrying only the snapshotted surface the
dispatcher reads (``tools``/``http_servers``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from aios.models.agents import HttpPermissionPolicy, HttpRouteSpec, HttpServerSpec, ToolSpec
from aios.workflows import run_tools
from aios.workflows.run_tools import invoke_run_tool
from aios.workflows.wf_script_host import tool


def _run(
    *, tools: list[ToolSpec] | None = None, http_servers: list[HttpServerSpec] | None = None
) -> Any:
    return SimpleNamespace(
        id="wfr_1", account_id="acc_t", tools=tools or [], http_servers=http_servers or []
    )


async def test_undeclared_tool_is_recoverable_error() -> None:
    # web_search is a run tool, but the workflow didn't declare it → recoverable value.
    run = _run(tools=[])
    out = await invoke_run_tool(
        run=run, account_id="acc_t", tool_name="web_search", tool_input={"query": "x"}
    )
    assert "error" in out and "declared" in out["error"]


async def test_non_run_tool_rejected() -> None:
    # bash is a sandbox tool — not callable from a run, even if (somehow) declared.
    run = _run(tools=[ToolSpec(type="bash")])
    out = await invoke_run_tool(run=run, account_id="acc_t", tool_name="bash", tool_input={})
    assert "error" in out and "workflow run" in out["error"]


async def test_web_search_routed_when_declared() -> None:
    run = _run(tools=[ToolSpec(type="web_search")])
    with patch.object(
        run_tools, "web_search_handler", new=AsyncMock(return_value={"results": ["R"]})
    ) as handler:
        out = await invoke_run_tool(
            run=run, account_id="acc_t", tool_name="web_search", tool_input={"query": "x"}
        )
    assert out == {"results": ["R"]}
    handler.assert_awaited_once_with("", {"query": "x"})  # owner-agnostic: empty owner id


async def test_invalid_args_surface_a_schema_error() -> None:
    run = _run(tools=[ToolSpec(type="web_search")])
    out = await invoke_run_tool(
        run=run, account_id="acc_t", tool_name="web_search", tool_input={}
    )  # missing required "query"
    assert "error" in out


async def test_http_request_routed_to_the_run_resolver() -> None:
    """invoke_run_tool feeds the run's snapshotted http_servers + the RUN credential
    resolver into the shared http core — never an agent's surface or the session resolver."""
    run = _run(
        tools=[ToolSpec(type="http_request")],
        http_servers=[HttpServerSpec(name="api", base_url="https://x")],
    )
    captured: dict[str, Any] = {}

    async def _fake_do(*, servers: Any, arguments: Any, resolve_auth: Any) -> dict[str, Any]:
        captured["servers"] = servers
        captured["auth"] = await resolve_auth("https://x")  # exercise the injected resolver
        return {"status": 200}

    with (
        patch.object(run_tools, "_do_http_request", new=_fake_do),
        patch("aios.harness.runtime.require_pool", return_value=object()),
        patch("aios.harness.runtime.require_crypto_box", return_value=object()),
        patch.object(
            run_tools,
            "resolve_auth_for_target_url_run",
            new=AsyncMock(return_value=("vlt", {"Authorization": "Bearer t"})),
        ) as run_resolver,
    ):
        out = await invoke_run_tool(
            run=run,
            account_id="acc_t",
            tool_name="http_request",
            tool_input={"server_ref": "api", "path": "/p", "method": "GET"},
        )

    assert out == {"status": 200}
    assert captured["servers"] == run.http_servers  # the run's snapshot
    assert captured["auth"] == ("vlt", {"Authorization": "Bearer t"})
    run_resolver.assert_awaited_once()  # the run-scoped resolver, not the session one


async def test_disabled_tool_is_treated_as_undeclared() -> None:
    # A declared-but-disabled tool is invisible to the run (fail-closed), like a session.
    run = _run(tools=[ToolSpec(type="web_search", enabled=False)])
    out = await invoke_run_tool(
        run=run, account_id="acc_t", tool_name="web_search", tool_input={"query": "x"}
    )
    assert "error" in out and "declared" in out["error"]


async def test_always_ask_route_is_denied_from_a_run() -> None:
    # An always_ask route needs human confirmation, which a run has no channel for → denied
    # (a run must not be more privileged than a session on the same declared surface).
    route = HttpRouteSpec(
        path_pattern="/things/*", permission_policy=HttpPermissionPolicy(type="always_ask")
    )
    run = _run(
        tools=[ToolSpec(type="http_request")],
        http_servers=[HttpServerSpec(name="api", base_url="https://x", routes=[route])],
    )
    out = await invoke_run_tool(
        run=run,
        account_id="acc_t",
        tool_name="http_request",
        tool_input={"server_ref": "api", "path": "/things/1", "method": "GET"},
    )
    assert "error" in out and "always_ask" in out["error"]


def test_tool_verb_builds_capability() -> None:
    cap = tool("web_search", {"query": "x"})
    assert cap._capability_id == "tool"
    assert cap._spec == {"tool_name": "web_search", "input": {"query": "x"}}
