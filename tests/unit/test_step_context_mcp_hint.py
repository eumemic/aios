"""Tests for the ``tool`` CLI system-prompt hint.

The hint is rendered into the system prompt whenever the agent has at
least one ``always_allow`` MCP toolset entry. Emitting it for an agent
whose toolset has only ``always_ask`` policies would lie to the model
(every CLI call would 403), so the predicate has to walk both
``default_config`` and per-tool ``configs`` overrides.

Wording assertions (issue #675) pin the binary name and INVOKE form so
the prose can't drift from the parser again — the original bug was a
``--json`` flag mentioned in the help text that the CLI never accepted.
"""

from __future__ import annotations

from aios.harness.step_context import _MCP_CLI_HINT, _has_always_allow_mcp_tool
from aios.models.agents import (
    McpPermissionPolicy,
    McpToolConfig,
    McpToolsetConfig,
    ToolSpec,
)


def _ts(
    *,
    enabled: bool = True,
    default_policy: str | None = "always_allow",
    configs: list[McpToolConfig] | None = None,
) -> ToolSpec:
    return ToolSpec(
        type="mcp_toolset",
        enabled=enabled,
        mcp_server_name="s",
        default_config=McpToolsetConfig(
            enabled=True,
            permission_policy=(
                McpPermissionPolicy(type=default_policy) if default_policy else None
            ),
        ),
        configs=configs,
    )


class TestHasAlwaysAllowMcpTool:
    def test_empty_tools(self) -> None:
        assert _has_always_allow_mcp_tool([]) is False

    def test_only_builtin_tool(self) -> None:
        assert _has_always_allow_mcp_tool([ToolSpec(type="bash")]) is False

    def test_disabled_toolset_does_not_count(self) -> None:
        assert _has_always_allow_mcp_tool([_ts(enabled=False)]) is False

    def test_default_always_allow(self) -> None:
        assert _has_always_allow_mcp_tool([_ts(default_policy="always_allow")]) is True

    def test_default_always_ask(self) -> None:
        assert _has_always_allow_mcp_tool([_ts(default_policy="always_ask")]) is False

    def test_per_tool_override_to_always_allow(self) -> None:
        spec = _ts(
            default_policy="always_ask",
            configs=[
                McpToolConfig(name="x", permission_policy=McpPermissionPolicy(type="always_allow"))
            ],
        )
        assert _has_always_allow_mcp_tool([spec]) is True

    def test_per_tool_override_disabled(self) -> None:
        # Disabled per-tool overrides shouldn't count even if their policy
        # is always_allow.
        spec = _ts(
            default_policy="always_ask",
            configs=[
                McpToolConfig(
                    name="x",
                    enabled=False,
                    permission_policy=McpPermissionPolicy(type="always_allow"),
                )
            ],
        )
        assert _has_always_allow_mcp_tool([spec]) is False


class TestMcpCliHintWording:
    """Pinned wording for the in-sandbox CLI hint (issue #675).

    The hint went stale twice already: ``--json`` was dropped from the
    binary in commit f8aefa3 and the binary was renamed ``mcp`` →
    ``tool`` in commit 57a0747, but this prose wasn't updated either
    time. These assertions lock the binary name and INVOKE form so the
    same drift can't recur silently.
    """

    def test_hint_names_tool_binary_not_legacy_mcp(self) -> None:
        # The binary is invoked as ``tool`` post-rename.
        assert "`tool`" in _MCP_CLI_HINT
        # And the legacy ``mcp`` binary name must not reappear as a
        # callable. Match the backticked form to avoid false positives
        # on prose like "MCP tools" / "MCP server".
        assert "`mcp`" not in _MCP_CLI_HINT

    def test_hint_shows_positional_invoke_form_without_json_flag(self) -> None:
        # The original #675 symptom: prose advertising ``--json`` for a
        # flag the parser never accepted. The positional form is the
        # only thing the binary supports.
        assert "tool <server> <method> '{...}'" in _MCP_CLI_HINT
        assert "--json" not in _MCP_CLI_HINT
