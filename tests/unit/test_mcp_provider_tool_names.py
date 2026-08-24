"""Provider-facing MCP tool names and envelope sanitization."""

from __future__ import annotations

import re

from mcp.types import Tool

from aios.harness.completion import _build_litellm_kwargs
from aios.mcp.schema import (
    PROVIDER_TOOL_NAME_MAX,
    make_function_tool,
    mcp_origin_for,
    qualify_mcp_tool_name,
    sanitize_tools_for_provider,
)

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class TestQualifyMcpToolName:
    def test_short_legal_name_is_unchanged(self) -> None:
        advertised, server, tool = qualify_mcp_tool_name("x", "search_posts")
        assert advertised == "mcp__x__search_posts"
        assert (server, tool) == ("x", "search_posts")
        assert len(advertised) <= PROVIDER_TOOL_NAME_MAX

    def test_dots_and_slashes_become_underscores(self) -> None:
        advertised, server, tool = qualify_mcp_tool_name("api.x.com", "users/lookup")
        assert advertised == "mcp__api_x_com__users_lookup"
        assert (server, tool) == ("api.x.com", "users/lookup")

    def test_long_custom_server_plus_tool_fits_64(self) -> None:
        server = "custom_api_x_com_QFZ6ZWJR"
        tool = "get_user_bookmark_folder_items_by_id"
        advertised, origin_server, origin_tool = qualify_mcp_tool_name(server, tool)
        assert len(advertised) <= PROVIDER_TOOL_NAME_MAX
        assert _NAME_RE.match(advertised)
        assert (origin_server, origin_tool) == (server, tool)
        assert len(f"mcp__{server}__{tool}") > PROVIDER_TOOL_NAME_MAX

    def test_huge_server_name_still_fits(self) -> None:
        server = "s" * 80
        advertised, origin_server, origin_tool = qualify_mcp_tool_name(server, "t")
        assert len(advertised) <= PROVIDER_TOOL_NAME_MAX
        assert _NAME_RE.match(advertised)
        assert (origin_server, origin_tool) == (server, "t")


class TestSanitizeToolsForProvider:
    def test_clean_list_is_same_object(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "mcp__x__search",
                    "description": "",
                    "parameters": {"type": "object"},
                    "strict": False,
                },
            }
        ]
        assert sanitize_tools_for_provider(tools) is tools

    def test_strips_output_schema_and_origin(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "mcp__x__search",
                    "description": "",
                    "parameters": {"type": "object"},
                    "strict": False,
                    "outputSchema": {"type": "object", "properties": {"n": {"type": "number"}}},
                },
                "_mcp_origin": {"server": "x", "tool": "search"},
            }
        ]
        cleaned = sanitize_tools_for_provider(tools)
        assert cleaned is not tools
        assert "outputSchema" not in cleaned[0]["function"]
        assert "_mcp_origin" not in cleaned[0]
        assert cleaned[0]["function"]["name"] == "mcp__x__search"
        assert "outputSchema" in tools[0]["function"]
        assert tools[0]["_mcp_origin"]["tool"] == "search"

    def test_origin_lookup(self) -> None:
        advertised, server, tool = qualify_mcp_tool_name(
            "custom_api_x_com_QFZ6ZWJR", "get_user_bookmark_folder_items_by_id"
        )
        env = make_function_tool(
            advertised,
            Tool(name=tool, description="", inputSchema={"type": "object"}),
            origin_server=server,
            origin_tool=tool,
        )
        assert mcp_origin_for(advertised, [env]) == (server, tool)
        assert mcp_origin_for("mcp__other__x", [env]) is None

    def test_build_kwargs_strips_output_schema(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "mcp__x__search",
                    "description": "",
                    "parameters": {"type": "object"},
                    "strict": False,
                    "outputSchema": {"type": "object"},
                },
            }
        ]
        kwargs = _build_litellm_kwargs(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            auth=None,
            extra=None,
            session_id=None,
            stream=False,
        )
        assert "outputSchema" not in kwargs["tools"][0]["function"]
        assert "outputSchema" in tools[0]["function"]
