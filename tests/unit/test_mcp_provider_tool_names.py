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
    uniquify_advertised_tool_names,
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
        assert tools[0]["_mcp_origin"] == {"server": "x", "tool": "search"}

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


def _fn_envelope(name: str, *, server: str, tool: str) -> dict:
    return make_function_tool(
        name,
        Tool(name=tool, description="", inputSchema={"type": "object"}),
        origin_server=server,
        origin_tool=tool,
    )


class TestUniquifyAdvertisedToolNames:
    def test_already_unique_list_is_same_object(self) -> None:
        t = Tool(name="search_posts", description="", inputSchema={"type": "object"})
        catalog, s1, t1 = qualify_mcp_tool_name("x", "search_posts")
        custom, s2, t2 = qualify_mcp_tool_name("custom_api_x_com_QFZ6ZWJR", "search_posts")
        tools = [
            make_function_tool(catalog, t, origin_server=s1, origin_tool=t1),
            make_function_tool(custom, t, origin_server=s2, origin_tool=t2),
        ]
        assert catalog != custom
        assert uniquify_advertised_tool_names(tools) is tools

    def test_catalog_x_and_leftover_custom_collision(self) -> None:
        """Production mute: both mounts advertised the same name after sanitize."""
        collided = "mcp__x__search_posts"
        tools = [
            _fn_envelope(collided, server="x", tool="search_posts"),
            _fn_envelope(collided, server="custom_api_x_com_QFZ6ZWJR", tool="search_posts"),
        ]
        out = uniquify_advertised_tool_names(tools)
        names = [item["function"]["name"] for item in out]
        assert names[0] == collided
        assert names[1] != names[0]
        assert len(names) == len(set(names))
        assert all(_NAME_RE.match(name) and len(name) <= PROVIDER_TOOL_NAME_MAX for name in names)
        assert mcp_origin_for(names[0], out) == ("x", "search_posts")
        assert mcp_origin_for(names[1], out) == ("custom_api_x_com_QFZ6ZWJR", "search_posts")
        cleaned = sanitize_tools_for_provider(out)
        assert [item["function"]["name"] for item in cleaned] == names

    def test_sanitized_server_segments_that_qualify_identically(self) -> None:
        t = Tool(name="search_posts", description="", inputSchema={"type": "object"})
        a, s1, t1 = qualify_mcp_tool_name("api.x.com", "search_posts")
        b, s2, t2 = qualify_mcp_tool_name("api_x_com", "search_posts")
        assert a == b == "mcp__api_x_com__search_posts"
        tools = [
            make_function_tool(a, t, origin_server=s1, origin_tool=t1),
            make_function_tool(b, t, origin_server=s2, origin_tool=t2),
        ]
        out = uniquify_advertised_tool_names(tools)
        names = [item["function"]["name"] for item in out]
        assert names[0] == a
        assert names[1] != names[0]
        assert all(_NAME_RE.match(name) and len(name) <= PROVIDER_TOOL_NAME_MAX for name in names)
        assert mcp_origin_for(names[0], out) == (s1, t1)
        assert mcp_origin_for(names[1], out) == (s2, t2)

    def test_sixty_four_char_cap_collision(self) -> None:
        collided = "n" * PROVIDER_TOOL_NAME_MAX
        tools = [
            _fn_envelope(collided, server="server_one", tool="long_tool"),
            _fn_envelope(collided, server="server_two", tool="long_tool"),
        ]
        out = uniquify_advertised_tool_names(tools)
        names = [item["function"]["name"] for item in out]
        assert names[0] == collided
        assert names[1] != names[0]
        assert all(_NAME_RE.match(name) and len(name) <= PROVIDER_TOOL_NAME_MAX for name in names)
        assert mcp_origin_for(names[0], out) == ("server_one", "long_tool")
        assert mcp_origin_for(names[1], out) == ("server_two", "long_tool")

    def test_sanitize_uniquifies_even_without_prior_origin_rewrite(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "mcp__x__search_posts",
                    "description": "",
                    "parameters": {"type": "object"},
                    "strict": False,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mcp__x__search_posts",
                    "description": "",
                    "parameters": {"type": "object"},
                    "strict": False,
                },
            },
        ]
        cleaned = sanitize_tools_for_provider(tools)
        names = [item["function"]["name"] for item in cleaned]
        assert names[0] == "mcp__x__search_posts"
        assert names[1] != names[0]
        assert all(_NAME_RE.match(name) and len(name) <= PROVIDER_TOOL_NAME_MAX for name in names)
        assert cleaned is not tools

    def test_build_kwargs_never_sends_duplicate_names(self) -> None:
        collided = "mcp__x__search_posts"
        tools = [
            _fn_envelope(collided, server="x", tool="search_posts"),
            _fn_envelope(collided, server="custom_api_x_com_QFZ6ZWJR", tool="search_posts"),
        ]
        unique = uniquify_advertised_tool_names(tools)
        kwargs = _build_litellm_kwargs(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=unique,
            auth=None,
            extra=None,
            session_id=None,
            stream=False,
        )
        names = [item["function"]["name"] for item in kwargs["tools"]]
        assert len(names) == len(set(names))
        assert all(_NAME_RE.match(name) for name in names)
