"""Contract tests for the HTTP/JSON envelope between ``bin/mcp`` and
:class:`aios.sandbox.mcp_proxy.McpBroker`. Pins request/response shape,
exit codes, and secret validation. Runs over loopback — cross-container
reachability is covered by
:mod:`tests.e2e.test_sandbox_broker_reachability`."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aios.models.agents import (
    McpPermissionPolicy,
    McpServerSpec,
    McpToolsetConfig,
    ToolSpec,
)
from aios.sandbox.mcp_proxy import McpBroker

# bin/mcp lives at the repo root; resolve relative to this test file so
# the path is stable regardless of pytest's cwd.
_MCP_BINARY = Path(__file__).resolve().parents[3] / "bin" / "mcp"


async def _run_cli(
    *args: str, broker: McpBroker, secret: str = "s"
) -> subprocess.CompletedProcess[str]:
    """Run ``bin/mcp`` and return the completed process.

    Dispatches to a thread so the broker (running in this same event
    loop) can keep serving requests while the CLI waits on HTTP.
    """
    return await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(_MCP_BINARY), *args],
        env={
            "MCP_BROKER_URL": f"http://127.0.0.1:{broker.port}",
            "MCP_BROKER_SECRET": secret,
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.fixture
async def broker() -> AsyncIterator[McpBroker]:
    b = McpBroker()
    await b.start()
    try:
        yield b
    finally:
        await b.stop()


@pytest.fixture(autouse=True)
def _mock_crypto_and_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aios.harness.runtime.require_crypto_box", lambda: object())
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: object())

    async def _stub_auth(*_args: Any, **_kwargs: Any) -> tuple[str | None, dict[str, str]]:
        return None, {}

    monkeypatch.setattr("aios.sandbox.mcp_proxy.resolve_auth_for_target_url", _stub_auth)


def _agent_with_one_tool() -> SimpleNamespace:
    return SimpleNamespace(
        mcp_servers=[McpServerSpec(name="tav", url="https://tav.example/mcp")],
        tools=[
            ToolSpec(
                type="mcp_toolset",
                enabled=True,
                mcp_server_name="tav",
                default_config=McpToolsetConfig(
                    enabled=True,
                    permission_policy=McpPermissionPolicy(type="always_allow"),
                ),
            )
        ],
    )


class TestMcpCli:
    async def test_list_servers_prints_server_name(
        self, broker: McpBroker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broker.register_session("sess_X", "s")
        agent = _agent_with_one_tool()

        async def _load(*_a: Any, **_k: Any) -> Any:
            return agent

        monkeypatch.setattr(broker, "_load_agent", _load)

        result = await _run_cli(broker=broker)
        assert result.returncode == 0
        assert "tav" in result.stdout

    async def test_list_tools_prints_tool_name(
        self, broker: McpBroker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broker.register_session("sess_X", "s")
        agent = _agent_with_one_tool()

        async def _load(*_a: Any, **_k: Any) -> Any:
            return agent

        monkeypatch.setattr(broker, "_load_agent", _load)

        async def _discover(
            _url: str, _vault_id: str | None, _h: dict[str, str], name: str
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": f"mcp__{name}__web_search",
                        "description": "Search the web",
                        "parameters": {"type": "object"},
                    },
                }
            ], None

        monkeypatch.setattr("aios.sandbox.mcp_proxy.discover_mcp_tools", _discover)

        result = await _run_cli("tav", broker=broker)
        assert result.returncode == 0
        assert "web_search" in result.stdout
        assert "Search the web" in result.stdout

    async def test_invoke_prints_content_and_exits_zero(
        self, broker: McpBroker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broker.register_session("sess_X", "s")
        agent = _agent_with_one_tool()

        async def _load(*_a: Any, **_k: Any) -> Any:
            return agent

        monkeypatch.setattr(broker, "_load_agent", _load)

        async def _call(*_a: Any, **_k: Any) -> dict[str, Any]:
            return {"content": "the result"}

        monkeypatch.setattr("aios.sandbox.mcp_proxy.call_mcp_tool", _call)

        result = await _run_cli("tav", "web_search", "{}", broker=broker)
        assert result.returncode == 0
        assert result.stdout.strip() == "the result"

    async def test_invoke_no_args_sends_empty_object(
        self, broker: McpBroker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``mcp <server> <tool>`` with no positional JSON sends ``{}``."""
        broker.register_session("sess_X", "s")
        agent = _agent_with_one_tool()

        async def _load(*_a: Any, **_k: Any) -> Any:
            return agent

        monkeypatch.setattr(broker, "_load_agent", _load)

        captured: dict[str, Any] = {}

        async def _call(
            _url: str, _vault_id: str | None, _h: dict[str, str], _tool: str, args: dict[str, Any]
        ) -> dict[str, Any]:
            captured["args"] = args
            return {"content": "ok"}

        monkeypatch.setattr("aios.sandbox.mcp_proxy.call_mcp_tool", _call)

        result = await _run_cli("tav", "web_search", broker=broker)
        assert result.returncode == 0
        assert captured["args"] == {}

    async def test_invoke_unknown_secret_exits_nonzero(self, broker: McpBroker) -> None:
        result = await _run_cli("tav", "web_search", "{}", broker=broker, secret="wrong")
        assert result.returncode != 0
        assert "mcp:" in result.stderr.lower()

    async def test_invoke_full_envelope(
        self, broker: McpBroker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broker.register_session("sess_X", "s")
        agent = _agent_with_one_tool()

        async def _load(*_a: Any, **_k: Any) -> Any:
            return agent

        monkeypatch.setattr(broker, "_load_agent", _load)

        async def _call(*_a: Any, **_k: Any) -> dict[str, Any]:
            return {"content": "ok"}

        monkeypatch.setattr("aios.sandbox.mcp_proxy.call_mcp_tool", _call)

        result = await _run_cli("tav", "web_search", "{}", "--full", broker=broker)
        assert result.returncode == 0
        # --full prints the JSON envelope, not the bare content.
        assert json.loads(result.stdout) == {"content": "ok"}
