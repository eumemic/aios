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
        run=run,
        call_key="sha:k#0",
        account_id="acc_t",
        tool_name="web_search",
        tool_input={"query": "x"},
    )
    assert "error" in out and "declared" in out["error"]


async def test_non_run_tool_rejected() -> None:
    # read is an out-of-scope sandbox tool — not callable from a run, even if
    # (somehow) declared. (bash IS run-callable now; it routes to the run sandbox.)
    run = _run(tools=[ToolSpec(type="read")])
    out = await invoke_run_tool(
        run=run, call_key="sha:k#0", account_id="acc_t", tool_name="read", tool_input={}
    )
    assert "error" in out and "workflow run" in out["error"]


def test_gate_run_tool_strings() -> None:
    # The shared gate — its two byte-exact error strings are the contract both the
    # worker and sandbox executors depend on.
    from aios.workflows.run_tools import gate_run_tool

    run = _run(tools=[ToolSpec(type="bash")])
    # read is not run-callable at all.
    assert gate_run_tool(run, "read") == {
        "error": "tool 'read' is not callable from a workflow run"
    }
    # bash is run-callable but only if declared — undeclared here.
    undeclared = _run(tools=[])
    assert gate_run_tool(undeclared, "bash") == {
        "error": "tool 'bash' is not in the workflow's declared tools"
    }
    # bash declared + enabled → admitted (None).
    assert gate_run_tool(run, "bash") is None


async def test_web_search_routed_when_declared() -> None:
    run = _run(tools=[ToolSpec(type="web_search")])
    with patch.object(
        run_tools, "web_search_handler", new=AsyncMock(return_value={"results": ["R"]})
    ) as handler:
        out = await invoke_run_tool(
            run=run,
            call_key="sha:k#0",
            account_id="acc_t",
            tool_name="web_search",
            tool_input={"query": "x"},
        )
    assert out == {"results": ["R"]}
    handler.assert_awaited_once_with("", {"query": "x"})  # owner-agnostic: empty owner id


async def test_invalid_args_surface_a_schema_error() -> None:
    run = _run(tools=[ToolSpec(type="web_search")])
    out = await invoke_run_tool(
        run=run, call_key="sha:k#0", account_id="acc_t", tool_name="web_search", tool_input={}
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
            call_key="sha:k#0",
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
        run=run,
        call_key="sha:k#0",
        account_id="acc_t",
        tool_name="web_search",
        tool_input={"query": "x"},
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
        call_key="sha:k#0",
        account_id="acc_t",
        tool_name="http_request",
        tool_input={"server_ref": "api", "path": "/things/1", "method": "GET"},
    )
    assert "error" in out and "always_ask" in out["error"]


def test_tool_verb_builds_capability() -> None:
    cap = tool("web_search", {"query": "x"})
    assert cap._capability_id == "tool"
    assert cap._spec == {"tool_name": "web_search", "input": {"query": "x"}}


def _http_run() -> Any:
    return _run(
        tools=[ToolSpec(type="http_request")],
        http_servers=[HttpServerSpec(name="api", base_url="https://x")],
    )


def _capture_args_do(captured: dict[str, Any]) -> Any:
    async def _fake_do(*, servers: Any, arguments: Any, resolve_auth: Any) -> dict[str, Any]:
        captured["arguments"] = arguments
        return {"status": 200}

    return _fake_do


def _invoke_http(run: Any, *, call_key: str, tool_input: dict[str, Any]) -> Any:
    return invoke_run_tool(
        run=run,
        call_key=call_key,
        account_id="acc_t",
        tool_name="http_request",
        tool_input=tool_input,
    )


async def test_idempotency_sentinel_substituted_with_per_call_token() -> None:
    """A script opts in by writing the sentinel as the Idempotency-Key header value;
    the worker substitutes the real per-call sha256(run_id‖call_key) token before the
    request leaves the worker. The script never sees (and cannot forge) the token."""
    from aios.workflows.idempotency_key import (
        AIOS_IDEMPOTENCY_KEY_SENTINEL,
        idempotency_key,
    )

    run = _http_run()
    captured: dict[str, Any] = {}
    with (
        patch.object(run_tools, "_do_http_request", new=_capture_args_do(captured)),
        patch("aios.harness.runtime.require_pool", return_value=object()),
        patch("aios.harness.runtime.require_crypto_box", return_value=object()),
        patch.object(
            run_tools,
            "resolve_auth_for_target_url_run",
            new=AsyncMock(return_value=("vlt", {})),
        ),
    ):
        out = await _invoke_http(
            run,
            call_key="sha:abc#0",
            tool_input={
                "server_ref": "api",
                "path": "/charge",
                "method": "POST",
                "headers": {"Idempotency-Key": AIOS_IDEMPOTENCY_KEY_SENTINEL},
            },
        )

    assert out == {"status": 200}
    expected = idempotency_key("wfr_1", "sha:abc#0")
    assert captured["arguments"]["headers"]["Idempotency-Key"] == expected
    assert expected != AIOS_IDEMPOTENCY_KEY_SENTINEL  # the sentinel itself never goes out


async def test_idempotency_sentinel_is_case_insensitive_on_the_header_key() -> None:
    # HTTP header names are case-insensitive; the opt-in works under any casing.
    from aios.workflows.idempotency_key import (
        AIOS_IDEMPOTENCY_KEY_SENTINEL,
        idempotency_key,
    )

    run = _http_run()
    captured: dict[str, Any] = {}
    with (
        patch.object(run_tools, "_do_http_request", new=_capture_args_do(captured)),
        patch("aios.harness.runtime.require_pool", return_value=object()),
        patch("aios.harness.runtime.require_crypto_box", return_value=object()),
        patch.object(
            run_tools,
            "resolve_auth_for_target_url_run",
            new=AsyncMock(return_value=("vlt", {})),
        ),
    ):
        await _invoke_http(
            run,
            call_key="sha:abc#0",
            tool_input={
                "server_ref": "api",
                "path": "/charge",
                "method": "POST",
                "headers": {"idempotency-key": AIOS_IDEMPOTENCY_KEY_SENTINEL},
            },
        )

    assert captured["arguments"]["headers"]["idempotency-key"] == idempotency_key(
        "wfr_1", "sha:abc#0"
    )


async def test_author_supplied_idempotency_key_is_not_clobbered() -> None:
    # A literal author value (NOT the sentinel) is the author's own choice — left intact.
    run = _http_run()
    captured: dict[str, Any] = {}
    with (
        patch.object(run_tools, "_do_http_request", new=_capture_args_do(captured)),
        patch("aios.harness.runtime.require_pool", return_value=object()),
        patch("aios.harness.runtime.require_crypto_box", return_value=object()),
        patch.object(
            run_tools,
            "resolve_auth_for_target_url_run",
            new=AsyncMock(return_value=("vlt", {})),
        ),
    ):
        await _invoke_http(
            run,
            call_key="sha:abc#0",
            tool_input={
                "server_ref": "api",
                "path": "/charge",
                "method": "POST",
                "headers": {"Idempotency-Key": "my-own-key"},
            },
        )

    assert captured["arguments"]["headers"]["Idempotency-Key"] == "my-own-key"


async def test_no_idempotency_header_means_no_injection() -> None:
    # Opt-in: a call that doesn't ask for a token gets no token (at-least-once stays).
    run = _http_run()
    captured: dict[str, Any] = {}
    with (
        patch.object(run_tools, "_do_http_request", new=_capture_args_do(captured)),
        patch("aios.harness.runtime.require_pool", return_value=object()),
        patch("aios.harness.runtime.require_crypto_box", return_value=object()),
        patch.object(
            run_tools,
            "resolve_auth_for_target_url_run",
            new=AsyncMock(return_value=("vlt", {})),
        ),
    ):
        await _invoke_http(
            run,
            call_key="sha:abc#0",
            tool_input={"server_ref": "api", "path": "/p", "method": "GET"},
        )

    headers = captured["arguments"].get("headers") or {}
    assert not any(k.lower() == "idempotency-key" for k in headers)


async def test_sentinel_substitution_does_not_mutate_caller_input() -> None:
    # The worker must not write the real token back into the script-visible tool_input
    # (replay/journaling sees the verbatim author args, exactly like the bash preamble).
    from aios.workflows.idempotency_key import AIOS_IDEMPOTENCY_KEY_SENTINEL

    run = _http_run()
    captured: dict[str, Any] = {}
    headers = {"Idempotency-Key": AIOS_IDEMPOTENCY_KEY_SENTINEL}
    tool_input: dict[str, Any] = {
        "server_ref": "api",
        "path": "/charge",
        "method": "POST",
        "headers": headers,
    }
    with (
        patch.object(run_tools, "_do_http_request", new=_capture_args_do(captured)),
        patch("aios.harness.runtime.require_pool", return_value=object()),
        patch("aios.harness.runtime.require_crypto_box", return_value=object()),
        patch.object(
            run_tools,
            "resolve_auth_for_target_url_run",
            new=AsyncMock(return_value=("vlt", {})),
        ),
    ):
        await _invoke_http(run, call_key="sha:abc#0", tool_input=tool_input)

    # The original author dict still carries the sentinel, not the substituted token.
    assert headers["Idempotency-Key"] == AIOS_IDEMPOTENCY_KEY_SENTINEL
