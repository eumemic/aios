"""Unit tests for :class:`aios.sandbox.mcp_proxy.McpBroker`.

Boots a real broker (so we exercise the actual starlette + uvicorn
routing) and stubs the upstream MCP calls + the session→agent lookup.
Verifies the v1 authorization model — unknown secret, server not
visible, tool disabled, tool requires confirmation — alongside the
happy-path listing / --help / invocation flow.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import uvicorn

from aios.models.agents import (
    McpPermissionPolicy,
    McpServerSpec,
    McpToolConfig,
    McpToolsetConfig,
    ToolSpec,
)
from aios.sandbox.mcp_proxy import McpBroker


def _agent(*, mcp_servers: list[McpServerSpec], tools: list[ToolSpec]) -> SimpleNamespace:
    return SimpleNamespace(mcp_servers=mcp_servers, tools=tools)


def _server(name: str = "tav", url: str = "https://tavily.example/mcp") -> McpServerSpec:
    return McpServerSpec(name=name, url=url)


def _toolset(
    server_name: str = "tav",
    *,
    enabled: bool = True,
    default_policy: str | None = "always_allow",
    configs: list[McpToolConfig] | None = None,
) -> ToolSpec:
    return ToolSpec(
        type="mcp_toolset",
        enabled=enabled,
        mcp_server_name=server_name,
        default_config=McpToolsetConfig(
            enabled=True,
            permission_policy=(
                McpPermissionPolicy(type=default_policy) if default_policy else None
            ),
        ),
        configs=configs,
    )


def _tool_dict(
    server: str, tool: str, *, description: str = "", parameters: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Replicate the shape that ``discover_mcp_tools`` returns — namespaced
    OpenAI-format function tools."""
    return {
        "type": "function",
        "function": {
            "name": f"mcp__{server}__{tool}",
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }


@pytest.fixture
async def broker() -> AsyncIterator[McpBroker]:
    """Boot a real broker for the test; tear it down on cleanup."""
    b = McpBroker()
    await b.start()
    try:
        yield b
    finally:
        await b.stop()


@pytest.fixture
def base_url(broker: McpBroker) -> str:
    return f"http://127.0.0.1:{broker.port}"


@pytest.fixture(autouse=True)
def _mock_crypto_and_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """The broker reaches into the runtime crypto box only to call
    ``resolve_auth_for_target_url``. Both can be stubbed wholesale — tests
    don't care about the actual headers."""
    monkeypatch.setattr("aios.harness.runtime.require_crypto_box", lambda: object())
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: object())

    async def _stub_auth(*_args: Any, **_kwargs: Any) -> tuple[str | None, dict[str, str]]:
        return None, {}

    monkeypatch.setattr("aios.sandbox.mcp_proxy.resolve_auth_for_target_url", _stub_auth)


class TestSessionResolution:
    async def test_unknown_secret_is_403(self, base_url: str) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/v1/bogus/servers")
        assert resp.status_code == 403
        assert "unknown" in resp.json()["error"].lower()

    async def test_unregister_makes_secret_unknown(
        self, broker: McpBroker, base_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broker.register_session("sess_X", "secret123")
        monkeypatch.setattr(
            broker, "_load_agent", _async_returning(_agent(mcp_servers=[], tools=[]))
        )
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/v1/secret123/servers")
            assert resp.status_code == 200

            broker.unregister_session("sess_X")
            resp2 = await client.get(f"{base_url}/v1/secret123/servers")
            assert resp2.status_code == 403


class TestListServers:
    async def test_lists_only_enabled_toolset_servers(
        self, broker: McpBroker, base_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(
            mcp_servers=[
                _server("tav", "https://tav.example/mcp"),
                _server("off", "https://off.example/mcp"),
                _server("noset", "https://noset.example/mcp"),
            ],
            tools=[
                _toolset("tav", enabled=True),
                _toolset("off", enabled=False),
                # noset is in mcp_servers but has no mcp_toolset entry
            ],
        )
        broker.register_session("sess_X", "s")
        monkeypatch.setattr(broker, "_load_agent", _async_returning(agent))

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/v1/s/servers")
        assert resp.status_code == 200
        names = {s["name"] for s in resp.json()["servers"]}
        assert names == {"tav"}


class TestListTools:
    async def test_filters_to_always_allow(
        self, broker: McpBroker, base_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(
            mcp_servers=[_server()],
            tools=[
                _toolset(
                    "tav",
                    default_policy="always_allow",
                    configs=[
                        McpToolConfig(
                            name="confirm_me",
                            permission_policy=McpPermissionPolicy(type="always_ask"),
                        ),
                        McpToolConfig(name="disabled_tool", enabled=False),
                    ],
                )
            ],
        )
        broker.register_session("sess_X", "s")
        monkeypatch.setattr(broker, "_load_agent", _async_returning(agent))

        async def _discover(
            _url: str, _vault_id: str | None, _headers: dict[str, str], name: str
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [
                _tool_dict(name, "web_search", description="Search the web"),
                _tool_dict(name, "confirm_me"),
                _tool_dict(name, "disabled_tool"),
            ], None

        monkeypatch.setattr("aios.sandbox.mcp_proxy.discover_mcp_tools", _discover)

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/v1/s/servers/tav/tools")
        assert resp.status_code == 200
        names = {t["name"] for t in resp.json()["tools"]}
        # Only web_search resolves to always_allow; the other two are filtered.
        assert names == {"web_search"}

    async def test_unknown_server_is_404(
        self, broker: McpBroker, base_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broker.register_session("sess_X", "s")
        monkeypatch.setattr(
            broker, "_load_agent", _async_returning(_agent(mcp_servers=[], tools=[]))
        )
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/v1/s/servers/nope/tools")
        assert resp.status_code == 404

    async def test_discovery_failure_returns_502(
        self, broker: McpBroker, base_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the model runs ``mcp list-tools <server>`` and the upstream
        server is unreachable, the broker must surface the transport
        failure as a 502 so the sandboxed model sees that its conscious
        action failed — not a 200 with an empty tools list, which would
        be indistinguishable from a server that genuinely has no tools.
        """
        agent = _agent(
            mcp_servers=[_server()], tools=[_toolset("tav", default_policy="always_allow")]
        )
        broker.register_session("sess_X", "s")
        monkeypatch.setattr(broker, "_load_agent", _async_returning(agent))

        async def _discover(*_args: object, **_kwargs: object) -> object:
            raise ConnectionError("simulated upstream down")

        monkeypatch.setattr("aios.sandbox.mcp_proxy.discover_mcp_tools", _discover)

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/v1/s/servers/tav/tools")
        assert resp.status_code == 502
        assert resp.json().get("code") == "transport_error"


class TestToolHelp:
    async def test_returns_schema_for_always_allow_tool(
        self, broker: McpBroker, base_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(
            mcp_servers=[_server()],
            tools=[_toolset("tav", default_policy="always_allow")],
        )
        broker.register_session("sess_X", "s")
        monkeypatch.setattr(broker, "_load_agent", _async_returning(agent))

        async def _discover(
            _url: str, _vault_id: str | None, _headers: dict[str, str], name: str
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [
                _tool_dict(
                    name,
                    "web_search",
                    description="Search the web",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                )
            ], None

        monkeypatch.setattr("aios.sandbox.mcp_proxy.discover_mcp_tools", _discover)

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/v1/s/servers/tav/tools/web_search")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "web_search"
        assert body["description"] == "Search the web"
        assert body["input_schema"]["required"] == ["query"]

    async def test_always_ask_tool_is_403(
        self, broker: McpBroker, base_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(
            mcp_servers=[_server()],
            tools=[_toolset("tav", default_policy="always_ask")],
        )
        broker.register_session("sess_X", "s")
        monkeypatch.setattr(broker, "_load_agent", _async_returning(agent))

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/v1/s/servers/tav/tools/web_search")
        assert resp.status_code == 403
        assert "model-tool" in resp.json()["error"]

    async def test_disabled_per_tool_config_is_403(
        self, broker: McpBroker, base_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(
            mcp_servers=[_server()],
            tools=[
                _toolset(
                    "tav",
                    default_policy="always_allow",
                    configs=[McpToolConfig(name="dangerous", enabled=False)],
                )
            ],
        )
        broker.register_session("sess_X", "s")
        monkeypatch.setattr(broker, "_load_agent", _async_returning(agent))

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/v1/s/servers/tav/tools/dangerous")
        assert resp.status_code == 403
        assert "disabled" in resp.json()["error"]

    async def test_discovery_failure_returns_502(
        self, broker: McpBroker, base_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``mcp tool-help <server> <tool>`` is a conscious model action;
        a discovery transport failure must surface as 502, not silently
        masquerade as 'tool not found'."""
        agent = _agent(
            mcp_servers=[_server()], tools=[_toolset("tav", default_policy="always_allow")]
        )
        broker.register_session("sess_X", "s")
        monkeypatch.setattr(broker, "_load_agent", _async_returning(agent))

        async def _discover(*_args: object, **_kwargs: object) -> object:
            raise ConnectionError("simulated upstream down")

        monkeypatch.setattr("aios.sandbox.mcp_proxy.discover_mcp_tools", _discover)

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/v1/s/servers/tav/tools/web_search")
        assert resp.status_code == 502
        assert resp.json().get("code") == "transport_error"


class TestInvoke:
    async def test_happy_path_returns_envelope(
        self, broker: McpBroker, base_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(
            mcp_servers=[_server()],
            tools=[_toolset("tav", default_policy="always_allow")],
        )
        broker.register_session("sess_X", "s")
        monkeypatch.setattr(broker, "_load_agent", _async_returning(agent))

        captured: dict[str, Any] = {}

        async def _call(
            _url: str,
            _vault_id: str | None,
            _headers: dict[str, str],
            tool: str,
            args: dict[str, Any],
        ) -> dict[str, Any]:
            captured["tool"] = tool
            captured["args"] = args
            return {"content": "match found"}

        monkeypatch.setattr("aios.sandbox.mcp_proxy.call_mcp_tool", _call)

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/v1/s/servers/tav/tools/web_search",
                json={"arguments": {"query": "aios"}},
            )
        assert resp.status_code == 200
        assert resp.json() == {"content": "match found"}
        assert captured == {"tool": "web_search", "args": {"query": "aios"}}

    async def test_always_ask_tool_is_403(
        self, broker: McpBroker, base_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(
            mcp_servers=[_server()],
            tools=[_toolset("tav", default_policy="always_ask")],
        )
        broker.register_session("sess_X", "s")
        monkeypatch.setattr(broker, "_load_agent", _async_returning(agent))

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/v1/s/servers/tav/tools/web_search",
                json={"arguments": {}},
            )
        assert resp.status_code == 403

    async def test_missing_arguments_is_400(
        self, broker: McpBroker, base_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broker.register_session("sess_X", "s")
        monkeypatch.setattr(
            broker, "_load_agent", _async_returning(_agent(mcp_servers=[], tools=[]))
        )
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/v1/s/servers/tav/tools/web_search",
                json={"no_arguments_key": "here"},
            )
        assert resp.status_code == 400


def _async_returning(value: Any) -> Any:
    """Return an async callable that always returns ``value``.

    Used to mock ``McpBroker._load_agent`` so tests don't go through the
    DB. The broker calls ``await self._load_agent(session_id)``; the
    callable here ignores its argument and returns the canned agent.
    """

    async def _impl(*_args: Any, **_kwargs: Any) -> Any:
        return value

    return _impl


class TestStartBindFailureCleanup:
    async def test_bind_timeout_releases_serve_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # uvicorn.Server.serve that never starts — bind never completes.
        async def _hang(_self: uvicorn.Server, *_a: object, **_k: object) -> None:
            await asyncio.sleep(60)

        monkeypatch.setattr(uvicorn.Server, "serve", _hang)
        monkeypatch.setattr("aios.sandbox.mcp_proxy._BIND_TIMEOUT_S", 0.05)

        b = McpBroker()
        with pytest.raises(RuntimeError, match="mcp broker failed to bind"):
            await b.start()

        assert b._serve_task is not None
        assert b._serve_task.done()
