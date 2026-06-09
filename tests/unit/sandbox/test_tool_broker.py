"""Unit tests for :class:`aios.sandbox.tool_broker.ToolBroker`.

Boots a real broker (so the actual Starlette + uvicorn routing is
exercised) and stubs the upstream MCP calls + the session→agent
lookup. The MCP TCP path was previously covered by ``test_mcp_proxy.py``;
that file's body tested an obsolete route shape and was dropped at the
rename. These tests cover the generalised CLI surface — discovery,
built-in invoke, MCP invoke, and the v1 gate (transport ∈ {cli, both}
∧ permission == always_allow ∧ type != custom).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from aios.errors import CryptoDecryptError, ForbiddenError
from aios.models.agents import (
    McpPermissionPolicy,
    McpServerSpec,
    McpToolConfig,
    McpToolsetConfig,
    ToolSpec,
)
from aios.sandbox.tool_broker import ToolBroker
from aios.tools.registry import ToolDefinition, ToolResult, registry

# ── shared fixtures + helpers ─────────────────────────────────────────────────


def _agent(
    *,
    tools: list[ToolSpec] | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(tools=tools or [], mcp_servers=mcp_servers or [])


def _server(name: str = "tav", url: str = "https://tav.example/mcp") -> McpServerSpec:
    return McpServerSpec(name=name, url=url)


def _toolset(
    server_name: str = "tav",
    *,
    enabled: bool = True,
    default_policy: str | None = "always_allow",
    default_transport: str | None = None,
    configs: list[McpToolConfig] | None = None,
) -> ToolSpec:
    pol = McpPermissionPolicy(type=default_policy) if default_policy else None  # type: ignore[arg-type]
    return ToolSpec(
        type="mcp_toolset",
        enabled=enabled,
        mcp_server_name=server_name,
        default_config=McpToolsetConfig(
            permission_policy=pol,
            transport=default_transport,  # type: ignore[arg-type]
        ),
        configs=configs,
    )


@pytest.fixture
async def broker() -> AsyncIterator[ToolBroker]:
    b = ToolBroker()
    await b.start()
    try:
        yield b
    finally:
        await b.stop()


@pytest.fixture
def hijack_tool() -> AsyncIterator[
    Callable[[str, Callable[[str, dict[str, Any]], Awaitable[Any]]], None]
]:
    """Yield a function that overrides a real built-in's handler.

    ``ToolSpec.type`` is constrained to the ``BuiltinToolType`` literal,
    so the tests can't introduce arbitrary new names; instead a real
    built-in's registry entry is swapped for a test stub, then restored
    in cleanup. ``web_search`` is the default victim — chosen because
    it has no live network behaviour the tests would accidentally
    trigger if a path mis-routed.
    """
    saved: dict[str, ToolDefinition] = {}

    def hijack(
        name: str,
        handler: Callable[[str, dict[str, Any]], Awaitable[Any]],
    ) -> None:
        # Save the *original* on first hijack only — re-hijacking the same
        # name within a single test would otherwise capture the prior stub
        # and restore that on teardown instead of the true original.
        if name not in saved:
            saved[name] = registry.get(name)
        registry._tools[name] = ToolDefinition(
            name=name,
            description=f"hijacked {name}",
            parameters_schema={"type": "object", "additionalProperties": True},
            handler=handler,
            transport="both",
        )

    yield hijack
    for n, original in saved.items():
        registry._tools[n] = original


def _patch_agent(agent: Any) -> Any:
    """Patch ``ToolBroker._load_agent`` to return ``agent`` synchronously."""

    async def fake(_self: Any, _session_id: str) -> Any:
        return agent

    return patch.object(ToolBroker, "_load_agent", fake)


def _url(broker: ToolBroker, secret: str, *path: str) -> str:
    return f"http://127.0.0.1:{broker.port}/v1/{secret}/" + "/".join(path)


# ── secret / discovery ────────────────────────────────────────────────────────


class TestSecretAuth:
    async def test_unknown_secret_returns_403(self, broker: ToolBroker) -> None:
        async with httpx.AsyncClient() as c:
            r = await c.get(_url(broker, "bogus", "tools"))
        assert r.status_code == 403
        assert "unknown or expired" in r.json()["error"]

    async def test_registered_secret_resolves(self, broker: ToolBroker) -> None:
        broker.register_session("sess_X", "s")
        with _patch_agent(_agent()):
            async with httpx.AsyncClient() as c:
                r = await c.get(_url(broker, "s", "tools"))
        assert r.status_code == 200
        assert r.json() == {"builtins": [], "servers": []}

    async def test_unregister_drops_secret(self, broker: ToolBroker) -> None:
        broker.register_session("sess_X", "s")
        broker.unregister_session("sess_X")
        async with httpx.AsyncClient() as c:
            r = await c.get(_url(broker, "s", "tools"))
        assert r.status_code == 403


# ── built-in surface filtering ────────────────────────────────────────────────


class TestBuiltinSurface:
    async def test_lists_cli_reachable_builtin(self, broker: ToolBroker) -> None:
        broker.register_session("sess_X", "s")
        agent = _agent(tools=[ToolSpec(type="web_fetch")])
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.get(_url(broker, "s", "tools"))
        names = [t["name"] for t in r.json()["builtins"]]
        assert names == ["web_fetch"]

    async def test_hides_agent_tool_only(self, broker: ToolBroker) -> None:
        """``bash`` defaults to ``agent_tool`` in the registry."""
        broker.register_session("sess_X", "s")
        agent = _agent(tools=[ToolSpec(type="bash"), ToolSpec(type="web_fetch")])
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.get(_url(broker, "s", "tools"))
        names = [t["name"] for t in r.json()["builtins"]]
        assert names == ["web_fetch"]

    async def test_hides_disabled(self, broker: ToolBroker) -> None:
        broker.register_session("sess_X", "s")
        agent = _agent(tools=[ToolSpec(type="web_fetch", enabled=False)])
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.get(_url(broker, "s", "tools"))
        assert r.json()["builtins"] == []

    async def test_hides_always_ask(self, broker: ToolBroker) -> None:
        """``always_ask`` needs the deferred sync-confirmation bridge."""
        broker.register_session("sess_X", "s")
        agent = _agent(tools=[ToolSpec(type="web_fetch", permission="always_ask")])
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.get(_url(broker, "s", "tools"))
        assert r.json()["builtins"] == []

    async def test_hides_custom(self, broker: ToolBroker) -> None:
        broker.register_session("sess_X", "s")
        agent = _agent(
            tools=[
                ToolSpec(
                    type="custom",
                    name="my_tool",
                    description="x",
                    input_schema={"type": "object"},
                ),
                ToolSpec(type="web_fetch"),
            ]
        )
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.get(_url(broker, "s", "tools"))
        names = [t["name"] for t in r.json()["builtins"]]
        assert names == ["web_fetch"]

    async def test_agent_override_disables_cli(self, broker: ToolBroker) -> None:
        """Per-agent ``transport="agent_tool"`` narrows a default-``both``
        built-in off the CLI surface (preserves the bidirectional override
        semantics decided in #635)."""
        broker.register_session("sess_X", "s")
        agent = _agent(tools=[ToolSpec(type="web_fetch", transport="agent_tool")])
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.get(_url(broker, "s", "tools"))
        assert r.json()["builtins"] == []


# ── built-in invoke ───────────────────────────────────────────────────────────


class TestBuiltinInvoke:
    async def test_happy_path_dict(
        self,
        broker: ToolBroker,
        hijack_tool: Callable[[str, Callable[[str, dict[str, Any]], Awaitable[Any]]], None],
    ) -> None:
        async def handler(_session_id: str, args: dict[str, Any]) -> dict[str, Any]:
            return {"echo": args}

        hijack_tool("web_search", handler)
        broker.register_session("sess_X", "s")
        agent = _agent(tools=[ToolSpec(type="web_search")])
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.post(
                    _url(broker, "s", "builtins", "web_search"),
                    json={"arguments": {"x": 1}},
                )
        assert r.status_code == 200
        assert r.json() == {"content": '{"echo": {"x": 1}}'}

    async def test_tool_result_string_content(
        self,
        broker: ToolBroker,
        hijack_tool: Callable[[str, Callable[[str, dict[str, Any]], Awaitable[Any]]], None],
    ) -> None:
        async def handler(_session_id: str, _args: dict[str, Any]) -> ToolResult:
            return ToolResult(content="hello")

        hijack_tool("web_search", handler)
        broker.register_session("sess_X", "s")
        agent = _agent(tools=[ToolSpec(type="web_search")])
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.post(
                    _url(broker, "s", "builtins", "web_search"), json={"arguments": {}}
                )
        assert r.status_code == 200
        assert r.json() == {"content": "hello"}

    async def test_tool_result_is_error(
        self,
        broker: ToolBroker,
        hijack_tool: Callable[[str, Callable[[str, dict[str, Any]], Awaitable[Any]]], None],
    ) -> None:
        async def handler(_session_id: str, _args: dict[str, Any]) -> ToolResult:
            return ToolResult(content="boom", is_error=True)

        hijack_tool("web_search", handler)
        broker.register_session("sess_X", "s")
        agent = _agent(tools=[ToolSpec(type="web_search")])
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.post(
                    _url(broker, "s", "builtins", "web_search"), json={"arguments": {}}
                )
        assert r.status_code == 200
        assert r.json() == {"error": "boom", "code": "tool_error"}

    async def test_handler_exception_500(
        self,
        broker: ToolBroker,
        hijack_tool: Callable[[str, Callable[[str, dict[str, Any]], Awaitable[Any]]], None],
    ) -> None:
        async def handler(_session_id: str, _args: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("kaboom")

        hijack_tool("web_search", handler)
        broker.register_session("sess_X", "s")
        agent = _agent(tools=[ToolSpec(type="web_search")])
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.post(
                    _url(broker, "s", "builtins", "web_search"), json={"arguments": {}}
                )
        assert r.status_code == 500
        body = r.json()
        assert body["code"] == "internal_error"
        assert "RuntimeError" in body["error"]

    async def test_typed_client_error_uses_real_status_and_code(
        self,
        broker: ToolBroker,
        hijack_tool: Callable[[str, Callable[[str, dict[str, Any]], Awaitable[Any]]], None],
    ) -> None:
        """A 4xx AiosError surfaces with its real status + ``error_type`` code (not 500),
        matching the model dispatch path's clean-refusal treatment."""

        async def handler(_session_id: str, _args: dict[str, Any]) -> dict[str, Any]:
            raise ForbiddenError("denied", detail={"x": 1})

        hijack_tool("web_search", handler)
        broker.register_session("sess_X", "s")
        agent = _agent(tools=[ToolSpec(type="web_search")])
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.post(
                    _url(broker, "s", "builtins", "web_search"), json={"arguments": {}}
                )
        assert r.status_code == 403
        assert r.json() == {"error": 'denied ({"x": 1})', "code": "forbidden"}

    async def test_typed_server_error_keeps_500_but_typed_code(
        self,
        broker: ToolBroker,
        hijack_tool: Callable[[str, Callable[[str, dict[str, Any]], Awaitable[Any]]], None],
    ) -> None:
        """A 5xx AiosError stays 500 but carries its typed code, not the opaque
        ``internal_error`` reserved for genuinely untyped exceptions."""

        async def handler(_session_id: str, _args: dict[str, Any]) -> dict[str, Any]:
            raise CryptoDecryptError("boom")

        hijack_tool("web_search", handler)
        broker.register_session("sess_X", "s")
        agent = _agent(tools=[ToolSpec(type="web_search")])
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.post(
                    _url(broker, "s", "builtins", "web_search"), json={"arguments": {}}
                )
        assert r.status_code == 500
        assert r.json() == {"error": "boom", "code": "crypto_decrypt_error"}

    async def test_bad_json_400(self, broker: ToolBroker) -> None:
        broker.register_session("sess_X", "s")
        async with httpx.AsyncClient() as c:
            r = await c.post(
                _url(broker, "s", "builtins", "web_fetch"),
                content=b"not json",
                headers={"Content-Type": "application/json"},
            )
        assert r.status_code == 400
        assert r.json()["error"] == "request body must be JSON"

    async def test_missing_arguments_400(self, broker: ToolBroker) -> None:
        broker.register_session("sess_X", "s")
        async with httpx.AsyncClient() as c:
            r = await c.post(_url(broker, "s", "builtins", "web_fetch"), json={})
        assert r.status_code == 400
        assert "arguments" in r.json()["error"]

    async def test_undeclared_builtin_404(self, broker: ToolBroker) -> None:
        broker.register_session("sess_X", "s")
        agent = _agent()  # no tools
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.post(
                    _url(broker, "s", "builtins", "web_fetch"),
                    json={"arguments": {}},
                )
        assert r.status_code == 404
        assert "not declared" in r.json()["error"]

    async def test_agent_tool_only_403(self, broker: ToolBroker) -> None:
        """``bash`` defaults ``agent_tool`` in the registry — refused on CLI."""
        broker.register_session("sess_X", "s")
        agent = _agent(tools=[ToolSpec(type="bash")])
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.post(
                    _url(broker, "s", "builtins", "bash"),
                    json={"arguments": {"command": "echo hi"}},
                )
        assert r.status_code == 403
        assert "not CLI-reachable" in r.json()["error"]

    async def test_always_ask_403(self, broker: ToolBroker) -> None:
        broker.register_session("sess_X", "s")
        agent = _agent(tools=[ToolSpec(type="web_fetch", permission="always_ask")])
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.post(
                    _url(broker, "s", "builtins", "web_fetch"),
                    json={"arguments": {"url": "https://x"}},
                )
        assert r.status_code == 403
        assert "always_ask" in r.json()["error"]


# ── MCP gate ──────────────────────────────────────────────────────────────────


def _fake_discover(
    server_name: str,
) -> Callable[..., Awaitable[tuple[list[dict[str, Any]], str | None]]]:
    """Build a stand-in for ``discover_mcp_tools`` that yields a fixed
    pair of tools (``echo`` and ``send``) for the given server.
    """

    async def fake(*_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], str | None]:
        return (
            [
                {
                    "type": "function",
                    "function": {
                        "name": f"mcp__{server_name}__echo",
                        "description": "echo",
                        "parameters": {"type": "object"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": f"mcp__{server_name}__send",
                        "description": "send a thing",
                        "parameters": {"type": "object"},
                    },
                },
            ],
            None,
        )

    return fake


class TestMcpGate:
    async def test_lists_methods_filtering_by_per_tool_transport(self, broker: ToolBroker) -> None:
        """A per-tool ``transport="agent_tool"`` override drops the tool
        from the CLI listing even when the server-wide default is
        ``both`` (registry default for MCP)."""
        broker.register_session("sess_X", "s")
        agent = _agent(
            tools=[
                _toolset(
                    "tav",
                    configs=[McpToolConfig(name="send", transport="agent_tool")],
                )
            ],
            mcp_servers=[_server("tav")],
        )
        with (
            _patch_agent(agent),
            patch(
                "aios.sandbox.tool_broker.discover_mcp_tools",
                side_effect=_fake_discover("tav"),
            ),
            patch.object(
                ToolBroker,
                "_load_auth_for",
                lambda _self, _sid, _url: asyncio.sleep(0, result=(None, {})),
            ),
        ):
            async with httpx.AsyncClient() as c:
                r = await c.get(_url(broker, "s", "mcp", "tav"))
        names = [t["name"] for t in r.json()["tools"]]
        assert names == ["echo"]

    async def test_invoke_refused_for_per_tool_agent_tool(self, broker: ToolBroker) -> None:
        broker.register_session("sess_X", "s")
        agent = _agent(
            tools=[
                _toolset(
                    "tav",
                    configs=[McpToolConfig(name="send", transport="agent_tool")],
                )
            ],
            mcp_servers=[_server("tav")],
        )
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.post(
                    _url(broker, "s", "mcp", "tav", "send"),
                    json={"arguments": {}},
                )
        assert r.status_code == 403
        assert "not CLI-reachable" in r.json()["error"]

    async def test_invoke_refused_for_per_tool_always_ask(self, broker: ToolBroker) -> None:
        broker.register_session("sess_X", "s")
        agent = _agent(
            tools=[
                _toolset(
                    "tav",
                    configs=[
                        McpToolConfig(
                            name="echo",
                            permission_policy=McpPermissionPolicy(type="always_ask"),
                        )
                    ],
                )
            ],
            mcp_servers=[_server("tav")],
        )
        with _patch_agent(agent):
            async with httpx.AsyncClient() as c:
                r = await c.post(
                    _url(broker, "s", "mcp", "tav", "echo"),
                    json={"arguments": {}},
                )
        assert r.status_code == 403
        assert "always_ask" in r.json()["error"]
