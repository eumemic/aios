"""Unit tests for discover_session_mcp_tools.

After the connector redesign (#200), discovery only walks the agent's
declared ``mcp_servers`` filtered by enabled ``mcp_toolset`` entries —
connector tools come through a parallel registry in PR2/PR3.  The MCP
SDK and auth lookup are both mocked; only the orchestration is under
test here.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.models.agents import AgentBinding, McpServerSpec, StepSurface, ToolSpec


@pytest.fixture(autouse=True)
def _mock_crypto_box() -> Any:
    """Bypass the runtime crypto-box requirement — discovery doesn't
    actually decrypt anything when resolve_auth_for_target_url is mocked.
    """
    with patch("aios.harness.loop.runtime.require_crypto_box") as m:
        m.return_value = object()
        yield m


def _agent(
    mcp_servers: list[McpServerSpec] | None = None,
    tools: list[ToolSpec] | None = None,
) -> StepSurface:
    return StepSurface(
        model="test/dummy",
        system="sys",
        tools=tools or [],
        skills=[],
        mcp_servers=mcp_servers or [],
        http_servers=[],
        litellm_extra={},
        window_min=1000,
        window_max=100000,
        binding=AgentBinding(agent_id="agt_1", version=3),
    )


class TestDiscoverSessionMcpTools:
    async def test_no_sources_returns_empty(self) -> None:
        from aios.harness.loop import discover_session_mcp_tools

        tools, instructions = await discover_session_mcp_tools(
            pool=AsyncMock(),
            session_id="sess_x",
            agent=_agent(),
            account_id="acc_test_stub",
        )
        assert tools == []
        assert instructions == {}

    async def test_only_enabled_server_discovered(self) -> None:
        """Only enabled mcp_toolset entries produce discovery; disabled
        ones are skipped.
        """
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            mcp_servers=[
                McpServerSpec(name="gh", url="https://mcp.github"),
                McpServerSpec(name="off", url="https://mcp.off"),
            ],
            tools=[
                ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="gh"),
                ToolSpec(type="mcp_toolset", enabled=False, mcp_server_name="off"),
            ],
        )

        async def _discover(
            url: str,
            _vault_id: str | None,
            _headers: dict[str, str],
            name: str,
            **_kwargs: Any,
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [{"name": f"mcp__{name}__t", "url": url}], None

        with (
            patch("aios.mcp.client.resolve_auth_for_target_url", new_callable=AsyncMock) as resolve,
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
        ):
            resolve.return_value = (None, {})
            tools, _instructions = await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_x",
                agent=agent,
                account_id="acc_test_stub",
            )
        names = {t["name"] for t in tools}
        assert names == {"mcp__gh__t"}

    async def test_auth_resolved_per_url(self) -> None:
        """Each URL resolves auth independently — goes through
        resolve_auth_for_target_url once per server, not once per batch.
        """
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            mcp_servers=[
                McpServerSpec(name="gh", url="https://mcp.github"),
                McpServerSpec(name="ln", url="https://mcp.linear"),
            ],
            tools=[
                ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="gh"),
                ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="ln"),
            ],
        )

        seen_urls: list[str] = []

        async def _fake_resolve(
            _pool: Any, _cb: Any, _sid: str, url: str, **kwargs: Any
        ) -> tuple[str | None, dict[str, str]]:
            seen_urls.append(url)
            return None, {"Authorization": f"Bearer token-for-{url}"}

        async def _discover(
            url: str,
            _vault_id: str | None,
            headers: dict[str, str],
            name: str,
            **_kwargs: Any,
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [{"name": f"mcp__{name}__t", "auth": headers["Authorization"]}], None

        with (
            patch("aios.mcp.client.resolve_auth_for_target_url", side_effect=_fake_resolve),
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
        ):
            tools, _instructions = await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_x",
                agent=agent,
                account_id="acc_test_stub",
            )
        assert sorted(seen_urls) == ["https://mcp.github", "https://mcp.linear"]
        auths = {t["auth"] for t in tools}
        assert auths == {
            "Bearer token-for-https://mcp.github",
            "Bearer token-for-https://mcp.linear",
        }

    async def test_instructions_keyed_by_server_name(self) -> None:
        """Each server's ``InitializeResult.instructions`` flows into the
        returned dict under its server_name key.  Servers that supply no
        instructions are omitted.
        """
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            mcp_servers=[
                McpServerSpec(name="gh", url="https://mcp.github"),
                McpServerSpec(name="ln", url="https://mcp.linear"),
            ],
            tools=[
                ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="gh"),
                ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="ln"),
            ],
        )

        async def _discover(
            url: str,
            _vault_id: str | None,
            _headers: dict[str, str],
            name: str,
            **_kwargs: Any,
        ) -> tuple[list[dict[str, Any]], str | None]:
            if name == "gh":
                return [], None
            return [], "## linear\n\nbe brief"

        with (
            patch("aios.mcp.client.resolve_auth_for_target_url", new_callable=AsyncMock) as resolve,
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
        ):
            resolve.return_value = (None, {})
            _tools, instructions = await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_x",
                agent=agent,
                account_id="acc_test_stub",
            )
        # 'gh' returned None → omitted; 'ln' returned prose → present.
        assert instructions == {"ln": "## linear\n\nbe brief"}

    async def test_empty_string_instructions_omitted(self) -> None:
        """An empty string is treated identically to ``None`` — no
        affordance section should be rendered for a server that returned
        ``""``.
        """
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            mcp_servers=[McpServerSpec(name="gh", url="https://mcp.github")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="gh")],
        )

        async def _discover(
            _url: str,
            _vault_id: str | None,
            _headers: dict[str, str],
            _name: str,
            **_kwargs: Any,
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [], ""

        with (
            patch("aios.mcp.client.resolve_auth_for_target_url", new_callable=AsyncMock) as resolve,
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
        ):
            resolve.return_value = (None, {})
            _tools, instructions = await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_x",
                agent=agent,
                account_id="acc_test_stub",
            )
        assert instructions == {}

    async def test_failed_server_filtered_from_tools_and_instructions(self) -> None:
        """When one server's discovery raises, the others still succeed:
        partial-success preserved, the failed server contributes neither
        tools nor an instructions entry. The discovery failure happens in
        the step prelude — a process the model didn't consciously initiate —
        so it's logged at WARN for ops, not surfaced as a model-visible
        event. The model sees a consistent view: only servers that are
        actually usable appear in its tools list and instructions block.
        """
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            mcp_servers=[
                McpServerSpec(name="gh", url="https://mcp.github"),
                McpServerSpec(name="broken", url="https://mcp.broken"),
                McpServerSpec(name="ln", url="https://mcp.linear"),
            ],
            tools=[
                ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="gh"),
                ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="broken"),
                ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="ln"),
            ],
        )

        async def _discover(
            _url: str,
            _vault_id: str | None,
            _headers: dict[str, str],
            name: str,
            **_kwargs: Any,
        ) -> tuple[list[dict[str, Any]], str | None]:
            if name == "broken":
                raise ConnectionError("simulated discovery failure")
            return [{"name": f"mcp__{name}__t"}], f"{name}-instructions"

        with (
            patch("aios.mcp.client.resolve_auth_for_target_url", new_callable=AsyncMock) as resolve,
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
        ):
            resolve.return_value = (None, {})
            tools, instructions = await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_x",
                agent=agent,
                account_id="acc_test_stub",
            )

        tool_names = {t["name"] for t in tools}
        assert tool_names == {"mcp__gh__t", "mcp__ln__t"}, (
            "Failed server's tools must be omitted; healthy servers must still appear"
        )
        assert set(instructions) == {"gh", "ln"}, (
            "Failed server must NOT appear in the instructions dict — the system "
            "prompt would otherwise list a server the model can't actually use"
        )

    async def test_unhealthy_server_is_skipped_via_circuit_breaker(self) -> None:
        """#1391: a server whose discovery recently timed out is in backoff on
        the pool; the prelude skips its (uncached) discovery fast — it never
        calls ``discover_mcp_tools`` for it — while a healthy server proceeds.
        The agent runs degraded on the healthy server's tools, not stalled.
        """
        from aios.harness import runtime
        from aios.harness.loop import discover_session_mcp_tools
        from aios.mcp.client import _headers_key
        from aios.mcp.pool import McpSessionPool

        agent = _agent(
            mcp_servers=[
                McpServerSpec(name="slow", url="https://mcp.slow"),
                McpServerSpec(name="ok", url="https://mcp.ok"),
            ],
            tools=[
                ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="slow"),
                ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="ok"),
            ],
        )

        pool = McpSessionPool()
        # Mark the slow server's transport key unhealthy (as a prior discovery
        # timeout would). vault_id is None here (resolve mocked to no-cred).
        pool.mark_unhealthy("https://mcp.slow", None, _headers_key(None), backoff_s=60.0)

        discovered: list[str] = []

        async def _discover(
            _url: str,
            _vault_id: str | None,
            _headers: dict[str, str],
            name: str,
            **_kwargs: Any,
        ) -> tuple[list[dict[str, Any]], str | None]:
            discovered.append(name)
            return [{"name": f"mcp__{name}__t"}], None

        prior = runtime.mcp_session_pool
        runtime.mcp_session_pool = pool
        try:
            with (
                patch(
                    "aios.mcp.client.resolve_auth_for_target_url", new_callable=AsyncMock
                ) as resolve,
                patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
            ):
                resolve.return_value = (None, {})
                tools, _instructions = await discover_session_mcp_tools(
                    pool=AsyncMock(),
                    session_id="sess_x",
                    agent=agent,
                    account_id="acc_test_stub",
                )
        finally:
            runtime.mcp_session_pool = prior

        # The slow server was short-circuited (never discovered); only 'ok' ran.
        assert discovered == ["ok"]
        assert {t["name"] for t in tools} == {"mcp__ok__t"}

    async def test_group_raising_server_omitted_healthy_returned(self) -> None:
        """#1698 (b): a server whose discovery raises a bare
        ``BaseExceptionGroup`` (non-Exception leaf) is omitted from the returned
        tools/instructions; the healthy servers' tools are still returned. The
        group must not abort the whole prelude."""
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            mcp_servers=[
                McpServerSpec(name="gh", url="https://mcp.github"),
                McpServerSpec(name="grp", url="https://mcp.grp"),
            ],
            tools=[
                ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="gh"),
                ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="grp"),
            ],
        )

        async def _discover(
            _url: str,
            _vault_id: str | None,
            _headers: dict[str, str],
            name: str,
            **_kwargs: Any,
        ) -> tuple[list[dict[str, Any]], str | None]:
            if name == "grp":
                raise BaseExceptionGroup(
                    "unhandled errors in a TaskGroup",
                    [ConnectionError("401"), asyncio.CancelledError()],
                )
            return [{"name": f"mcp__{name}__t"}], f"{name}-instructions"

        with (
            patch("aios.mcp.client.resolve_auth_for_target_url", new_callable=AsyncMock) as resolve,
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
        ):
            resolve.return_value = (None, {})
            tools, instructions = await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_x",
                agent=agent,
                account_id="acc_test_stub",
            )

        assert {t["name"] for t in tools} == {"mcp__gh__t"}, (
            "group-raising server omitted; healthy server's tools returned"
        )
        assert set(instructions) == {"gh"}

    async def test_down_transition_emits_one_mcp_server_unavailable_event(self) -> None:
        """#1698 (e): a breaker DOWN transition (drained from the pool) emits
        exactly one durable ``mcp_server_unavailable`` span event, keyed by
        server name + url, with ``is_error: false``."""
        from unittest.mock import MagicMock

        from aios.harness import runtime
        from aios.harness.loop import discover_session_mcp_tools
        from aios.mcp.pool import McpSessionPool

        agent = _agent(
            mcp_servers=[McpServerSpec(name="ultron", url="https://mcp.down")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="ultron")],
        )

        pool = McpSessionPool()
        # Simulate the breaker having recorded a DOWN edge for this url.
        pool._pending_degraded_events.append("https://mcp.down")

        async def _discover(*_a: Any, **_k: Any) -> tuple[list[dict[str, Any]], str | None]:
            return [], None

        emitted: list[dict[str, Any]] = []

        async def _append_event(_pool: Any, _sid: str, _kind: str, data: Any, **_k: Any) -> Any:
            emitted.append(data)
            return MagicMock()

        prior = runtime.mcp_session_pool
        runtime.mcp_session_pool = pool
        try:
            with (
                patch(
                    "aios.mcp.client.resolve_auth_for_target_url", new_callable=AsyncMock
                ) as resolve,
                patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
                patch("aios.harness.loop.sessions_service.append_event", side_effect=_append_event),
            ):
                resolve.return_value = (None, {})
                await discover_session_mcp_tools(
                    pool=AsyncMock(),
                    session_id="sess_x",
                    agent=agent,
                    account_id="acc_test_stub",
                )
        finally:
            runtime.mcp_session_pool = prior

        unavailable = [e for e in emitted if e.get("event") == "mcp_server_unavailable"]
        assert len(unavailable) == 1
        ev = unavailable[0]
        assert ev["server"] == "ultron"
        assert ev["url"] == "https://mcp.down"
        assert ev["is_error"] is False
        # Deduped: the edge queue is drained, so a re-run emits nothing.
        assert pool.drain_degraded_events() == []
