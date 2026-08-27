"""Unit tests for #2270: auto-allow readOnlyHint-annotated MCP tools.

Covers three layers:
1. ``make_function_tool`` / ``mcp_read_only_hint_for`` — the annotation
   flows through the internal tool-dict envelope and is strippable before
   the provider call.
2. ``auto_allow_readonly_tools`` — the pure composer that unions
   ``configs[]`` entries in, union-by-name, never touching
   ``default_config`` or an already-configured tool.
3. ``discover_session_mcp_tools`` — the persistence wiring: a plain latest
   agent binding gets an ``update_agent`` call when something changed; a
   frozen/version-pinned/generic-child binding never does.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import Tool, ToolAnnotations

from aios.mcp.schema import (
    make_function_tool,
    mcp_read_only_hint_for,
    sanitize_tools_for_provider,
)
from aios.models.agents import (
    AgentBinding,
    GenericChildBinding,
    McpPermissionPolicy,
    McpServerSpec,
    McpToolConfig,
    McpToolsetConfig,
    StepSurface,
    ToolSpec,
    auto_allow_readonly_tools,
)


def _tool(name: str, *, read_only: bool | None) -> Tool:
    annotations = ToolAnnotations(readOnlyHint=read_only) if read_only is not None else None
    return Tool(name=name, inputSchema={"type": "object"}, annotations=annotations)


class TestAnnotationEnvelope:
    def test_readonly_hint_true_survives_into_envelope(self) -> None:
        envelope = make_function_tool(
            "mcp__gh__get_me",
            _tool("get_me", read_only=True),
            origin_server="gh",
            origin_tool="get_me",
        )
        assert mcp_read_only_hint_for("mcp__gh__get_me", [envelope]) is True

    def test_readonly_hint_false_does_not_auto_allow(self) -> None:
        envelope = make_function_tool(
            "mcp__gh__delete_repo",
            _tool("delete_repo", read_only=False),
            origin_server="gh",
            origin_tool="delete_repo",
        )
        assert mcp_read_only_hint_for("mcp__gh__delete_repo", [envelope]) is False

    def test_no_annotations_at_all(self) -> None:
        envelope = make_function_tool(
            "mcp__gh__mystery",
            _tool("mystery", read_only=None),
            origin_server="gh",
            origin_tool="mystery",
        )
        assert mcp_read_only_hint_for("mcp__gh__mystery", [envelope]) is False

    def test_unknown_qualified_name_returns_false(self) -> None:
        envelope = make_function_tool(
            "mcp__gh__get_me",
            _tool("get_me", read_only=True),
            origin_server="gh",
            origin_tool="get_me",
        )
        assert mcp_read_only_hint_for("mcp__gh__other", [envelope]) is False

    def test_annotations_key_stripped_before_provider_call(self) -> None:
        envelope = make_function_tool(
            "mcp__gh__get_me",
            _tool("get_me", read_only=True),
            origin_server="gh",
            origin_tool="get_me",
        )
        cleaned = sanitize_tools_for_provider([envelope])
        assert "_mcp_annotations" not in cleaned[0]
        assert "_mcp_origin" not in cleaned[0]
        # Original envelope (the internal copy discovery/composer read) is untouched.
        assert "_mcp_annotations" in envelope


def _discovered(server: str, name: str, *, read_only: bool) -> dict[str, Any]:
    return make_function_tool(
        f"mcp__{server}__{name}",
        _tool(name, read_only=read_only),
        origin_server=server,
        origin_tool=name,
    )


def _mcp_toolset(server: str, *, configs: list[McpToolConfig] | None = None) -> ToolSpec:
    return ToolSpec(
        type="mcp_toolset",
        mcp_server_name=server,
        default_config=McpToolsetConfig(
            enabled=True, permission_policy=McpPermissionPolicy(type="always_ask")
        ),
        configs=configs,
    )


class TestAutoAllowReadonlyTools:
    def test_adds_config_for_readonly_tool_on_trusted_server(self) -> None:
        tools = [_mcp_toolset("github")]
        discovered = [_discovered("github", "get_me", read_only=True)]
        new_tools, changed = auto_allow_readonly_tools(
            tools, discovered, trusted_servers=["github"]
        )
        assert changed is True
        cfg = new_tools[0].configs
        assert cfg is not None and len(cfg) == 1
        assert cfg[0].name == "get_me"
        assert cfg[0].permission_policy is not None
        assert cfg[0].permission_policy.type == "always_allow"
        # default_config is untouched.
        assert new_tools[0].default_config is not None
        assert new_tools[0].default_config.permission_policy is not None
        assert new_tools[0].default_config.permission_policy.type == "always_ask"

    def test_does_not_touch_untrusted_server(self) -> None:
        tools = [_mcp_toolset("random_mcp")]
        discovered = [_discovered("random_mcp", "get_me", read_only=True)]
        new_tools, changed = auto_allow_readonly_tools(
            tools, discovered, trusted_servers=["github"]
        )
        assert changed is False
        assert new_tools is tools

    def test_write_tool_not_hinted_readonly_is_untouched(self) -> None:
        tools = [_mcp_toolset("github")]
        discovered = [_discovered("github", "delete_repo", read_only=False)]
        new_tools, changed = auto_allow_readonly_tools(
            tools, discovered, trusted_servers=["github"]
        )
        assert changed is False
        assert new_tools[0].configs is None

    def test_never_overwrites_existing_config_even_if_tightened(self) -> None:
        existing = McpToolConfig(
            name="get_me", permission_policy=McpPermissionPolicy(type="always_ask")
        )
        tools = [_mcp_toolset("github", configs=[existing])]
        discovered = [_discovered("github", "get_me", read_only=True)]
        new_tools, changed = auto_allow_readonly_tools(
            tools, discovered, trusted_servers=["github"]
        )
        assert changed is False
        assert new_tools[0].configs == [existing]

    def test_empty_trusted_servers_is_noop(self) -> None:
        tools = [_mcp_toolset("github")]
        discovered = [_discovered("github", "get_me", read_only=True)]
        new_tools, changed = auto_allow_readonly_tools(tools, discovered, trusted_servers=[])
        assert changed is False
        assert new_tools is tools

    def test_empty_discovered_is_noop(self) -> None:
        tools = [_mcp_toolset("github")]
        new_tools, changed = auto_allow_readonly_tools(tools, [], trusted_servers=["github"])
        assert changed is False
        assert new_tools is tools

    def test_non_mcp_toolset_tools_untouched(self) -> None:
        tools = [ToolSpec(type="bash"), _mcp_toolset("github")]
        discovered = [_discovered("github", "get_me", read_only=True)]
        new_tools, changed = auto_allow_readonly_tools(
            tools, discovered, trusted_servers=["github"]
        )
        assert changed is True
        assert new_tools[0].type == "bash"
        assert new_tools[1].configs is not None and len(new_tools[1].configs) == 1

    def test_multiple_readonly_tools_all_added_in_one_pass(self) -> None:
        tools = [_mcp_toolset("github")]
        discovered = [
            _discovered("github", "get_me", read_only=True),
            _discovered("github", "list_repos", read_only=True),
            _discovered("github", "delete_repo", read_only=False),
        ]
        new_tools, changed = auto_allow_readonly_tools(
            tools, discovered, trusted_servers=["github"]
        )
        assert changed is True
        names = {c.name for c in (new_tools[0].configs or [])}
        assert names == {"get_me", "list_repos"}

    def test_wrong_server_segment_in_qualified_name_not_matched(self) -> None:
        """A discovered tool namespaced to a DIFFERENT server must never bleed
        into this toolset's configs, even if both are trusted."""
        tools = [_mcp_toolset("github")]
        discovered = [_discovered("notion", "notion-search", read_only=True)]
        new_tools, changed = auto_allow_readonly_tools(
            tools, discovered, trusted_servers=["github", "notion"]
        )
        assert changed is False
        assert new_tools[0].configs is None


def _agent(
    *,
    binding: AgentBinding | GenericChildBinding,
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
        preempt_policy="wait",
        binding=binding,
    )


@pytest.fixture(autouse=True)
def _mock_crypto_box() -> Any:
    with patch("aios.harness.loop.runtime.require_crypto_box") as m:
        m.return_value = object()
        yield m


@pytest.fixture(autouse=True)
def _trusted_github(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from aios.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("AIOS_AUTO_ALLOW_READONLY_MCP_SERVERS", '["github"]')
    yield
    get_settings.cache_clear()


class TestDiscoverSessionMcpToolsPersistsAutoAllow:
    async def test_persists_for_plain_latest_agent_binding(self) -> None:
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            binding=AgentBinding(agent_id="agt_1", version=3),
            mcp_servers=[McpServerSpec(name="github", url="https://mcp.github")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="github")],
        )

        async def _discover(
            _url: str, _vault_id: str | None, _headers: dict[str, str], name: str, **_kw: Any
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [_discovered(name, "get_me", read_only=True)], None

        with (
            patch("aios.mcp.client.resolve_auth_for_mcp_mount", new_callable=AsyncMock) as resolve,
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
            patch("aios.services.agents.update_agent", new_callable=AsyncMock) as update_agent,
        ):
            resolve.return_value = (None, {})
            await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_x",
                agent=agent,
                account_id="acc_test_stub",
            )

        update_agent.assert_awaited_once()
        _, kwargs = update_agent.call_args
        assert kwargs["expected_version"] == 3
        new_tools = kwargs["tools"]
        assert new_tools[0].configs is not None
        assert new_tools[0].configs[0].name == "get_me"

    async def test_frozen_or_pinned_agent_binding_never_persists(self) -> None:
        """A frozen workflow-child overlay or a version-pinned session still
        produces an ``AgentBinding`` (#794) — the caller's ``persist_auto_allow=False``
        is what actually prevents writing an attenuated/stale tool list back
        onto the LIVE agent row."""
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            binding=AgentBinding(agent_id="agt_1", version=3),
            mcp_servers=[McpServerSpec(name="github", url="https://mcp.github")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="github")],
        )

        async def _discover(
            _url: str, _vault_id: str | None, _headers: dict[str, str], name: str, **_kw: Any
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [_discovered(name, "get_me", read_only=True)], None

        with (
            patch("aios.mcp.client.resolve_auth_for_mcp_mount", new_callable=AsyncMock) as resolve,
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
            patch("aios.services.agents.update_agent", new_callable=AsyncMock) as update_agent,
        ):
            resolve.return_value = (None, {})
            await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_frozen",
                agent=agent,
                account_id="acc_test_stub",
                persist_auto_allow=False,
            )

        update_agent.assert_not_awaited()

    async def test_generic_child_binding_never_persists(self) -> None:
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            binding=GenericChildBinding(session_id="sess_child"),
            mcp_servers=[McpServerSpec(name="github", url="https://mcp.github")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="github")],
        )

        async def _discover(
            _url: str, _vault_id: str | None, _headers: dict[str, str], name: str, **_kw: Any
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [_discovered(name, "get_me", read_only=True)], None

        with (
            patch("aios.mcp.client.resolve_auth_for_mcp_mount", new_callable=AsyncMock) as resolve,
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
            patch("aios.services.agents.update_agent", new_callable=AsyncMock) as update_agent,
        ):
            resolve.return_value = (None, {})
            await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_child",
                agent=agent,
                account_id="acc_test_stub",
            )

        update_agent.assert_not_awaited()

    async def test_no_change_skips_persistence(self) -> None:
        """A tool with no readOnlyHint contributes nothing to auto-allow, so
        update_agent is never called — no-op discovery must not pay a write."""
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            binding=AgentBinding(agent_id="agt_1", version=3),
            mcp_servers=[McpServerSpec(name="github", url="https://mcp.github")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="github")],
        )

        async def _discover(
            _url: str, _vault_id: str | None, _headers: dict[str, str], name: str, **_kw: Any
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [_discovered(name, "delete_repo", read_only=False)], None

        with (
            patch("aios.mcp.client.resolve_auth_for_mcp_mount", new_callable=AsyncMock) as resolve,
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
            patch("aios.services.agents.update_agent", new_callable=AsyncMock) as update_agent,
        ):
            resolve.return_value = (None, {})
            await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_x",
                agent=agent,
                account_id="acc_test_stub",
            )

        update_agent.assert_not_awaited()

    async def test_empty_trusted_allowlist_skips_persistence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aios.config import get_settings
        from aios.harness.loop import discover_session_mcp_tools

        monkeypatch.setenv("AIOS_AUTO_ALLOW_READONLY_MCP_SERVERS", "[]")
        get_settings.cache_clear()

        agent = _agent(
            binding=AgentBinding(agent_id="agt_1", version=3),
            mcp_servers=[McpServerSpec(name="github", url="https://mcp.github")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="github")],
        )

        async def _discover(
            _url: str, _vault_id: str | None, _headers: dict[str, str], name: str, **_kw: Any
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [_discovered(name, "get_me", read_only=True)], None

        try:
            with (
                patch(
                    "aios.mcp.client.resolve_auth_for_mcp_mount", new_callable=AsyncMock
                ) as resolve,
                patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
                patch("aios.services.agents.update_agent", new_callable=AsyncMock) as update_agent,
            ):
                resolve.return_value = (None, {})
                await discover_session_mcp_tools(
                    pool=AsyncMock(),
                    session_id="sess_x",
                    agent=agent,
                    account_id="acc_test_stub",
                )
            update_agent.assert_not_awaited()
        finally:
            get_settings.cache_clear()

    async def test_update_agent_failure_is_swallowed(self) -> None:
        """A lost optimistic-concurrency race (or any update_agent error) must
        not fail the step — discovery already has its tools for THIS turn."""
        from aios.errors import ConflictError
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            binding=AgentBinding(agent_id="agt_1", version=3),
            mcp_servers=[McpServerSpec(name="github", url="https://mcp.github")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="github")],
        )

        async def _discover(
            _url: str, _vault_id: str | None, _headers: dict[str, str], name: str, **_kw: Any
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [_discovered(name, "get_me", read_only=True)], None

        with (
            patch("aios.mcp.client.resolve_auth_for_mcp_mount", new_callable=AsyncMock) as resolve,
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
            patch(
                "aios.services.agents.update_agent",
                new_callable=AsyncMock,
                side_effect=ConflictError("version mismatch"),
            ),
        ):
            resolve.return_value = (None, {})
            tools, _instructions = await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_x",
                agent=agent,
                account_id="acc_test_stub",
            )
        # The step's own tool list is unaffected by the persistence failure.
        assert tools[0]["function"]["name"] == "mcp__github__get_me"


class TestComputeStepPreludeWiresPersistAutoAllow:
    """``compute_step_prelude`` must compute ``persist_auto_allow`` from the
    SESSION (not the surface alone) — ``StepSurface``/``AgentBinding`` can't
    tell a frozen workflow-child overlay or a version-pinned session apart
    from a plain live-latest agent session (#794/#2270 interaction)."""

    async def test_plain_live_session_persists(self) -> None:
        from unittest import mock

        from aios.harness.step_context import compute_step_prelude

        agent = _agent(
            binding=AgentBinding(agent_id="agt_1", version=3),
            mcp_servers=[McpServerSpec(name="github", url="https://mcp.github")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="github")],
        )
        session = mock.Mock(surface_frozen=False, agent_version=None, parent_run_id=None)

        with (
            patch(
                "aios.harness.loop.discover_session_mcp_tools", new_callable=AsyncMock
            ) as discover,
            patch("aios.db.queries.get_open_obligations", new=AsyncMock(return_value=[])),
        ):
            discover.return_value = ([], {})

            class _StubConn:
                async def __aenter__(self) -> _StubConn:
                    return self

                async def __aexit__(self, *exc: object) -> None:
                    return None

            class _StubPool:
                def acquire(self) -> _StubConn:
                    return _StubConn()

            await compute_step_prelude(
                _StubPool(),
                "sess_x",
                account_id="acc_test_stub",
                session=session,
                agent=agent,
                channels=[],
                memory_store_echoes=[],
            )

        discover.assert_awaited_once()
        assert discover.call_args.kwargs["persist_auto_allow"] is True

    async def test_frozen_session_does_not_persist(self) -> None:
        from unittest import mock

        from aios.harness.step_context import compute_step_prelude

        agent = _agent(
            binding=AgentBinding(agent_id="agt_1", version=3),
            mcp_servers=[McpServerSpec(name="github", url="https://mcp.github")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="github")],
        )
        session = mock.Mock(surface_frozen=True, agent_version=None, parent_run_id="run_1")

        with (
            patch(
                "aios.harness.loop.discover_session_mcp_tools", new_callable=AsyncMock
            ) as discover,
            patch("aios.db.queries.get_open_obligations", new=AsyncMock(return_value=[])),
        ):
            discover.return_value = ([], {})

            class _StubConn:
                async def __aenter__(self) -> _StubConn:
                    return self

                async def __aexit__(self, *exc: object) -> None:
                    return None

            class _StubPool:
                def acquire(self) -> _StubConn:
                    return _StubConn()

            await compute_step_prelude(
                _StubPool(),
                "sess_x",
                account_id="acc_test_stub",
                session=session,
                agent=agent,
                channels=[],
                memory_store_echoes=[],
            )

        discover.assert_awaited_once()
        assert discover.call_args.kwargs["persist_auto_allow"] is False

    async def test_version_pinned_session_does_not_persist(self) -> None:
        from unittest import mock

        from aios.harness.step_context import compute_step_prelude

        agent = _agent(
            binding=AgentBinding(agent_id="agt_1", version=2),
            mcp_servers=[McpServerSpec(name="github", url="https://mcp.github")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="github")],
        )
        session = mock.Mock(surface_frozen=False, agent_version=2, parent_run_id=None)

        with (
            patch(
                "aios.harness.loop.discover_session_mcp_tools", new_callable=AsyncMock
            ) as discover,
            patch("aios.db.queries.get_open_obligations", new=AsyncMock(return_value=[])),
        ):
            discover.return_value = ([], {})

            class _StubConn:
                async def __aenter__(self) -> _StubConn:
                    return self

                async def __aexit__(self, *exc: object) -> None:
                    return None

            class _StubPool:
                def acquire(self) -> _StubConn:
                    return _StubConn()

            await compute_step_prelude(
                _StubPool(),
                "sess_x",
                account_id="acc_test_stub",
                session=session,
                agent=agent,
                channels=[],
                memory_store_echoes=[],
            )

        discover.assert_awaited_once()
        assert discover.call_args.kwargs["persist_auto_allow"] is False
