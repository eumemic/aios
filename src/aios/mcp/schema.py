"""MCP tool schemas — sanitize inputSchemas, build OpenAI function-tool envelopes."""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any

from mcp.types import Tool

# Anthropic (and OpenAI) function-tool names: ``^[a-zA-Z0-9_-]{1,64}$``.
# MCP SEP-986 is more permissive (``.`` / ``/``) and does not enforce 64; the
# provider 400s. Qualified names are ``mcp__<server>__<tool>``.
PROVIDER_TOOL_NAME_MAX = 64
_PROVIDER_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_UNSAFE_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]+")

# Envelope key (sibling of ``function``, never inside it) mapping an advertised
# name back to the MCP server + raw tool. Stripped before the provider call.
MCP_ORIGIN_KEY = "_mcp_origin"


def sanitize_mcp_schema(node: Any) -> Any:
    """Drop the ``type`` keyword next to ``anyOf``/``oneOf`` (the union carries the real shape).

    Only the JSON Schema ``type`` **keyword** — whose value is a type name
    (``"string"``) or a list of them (``["string", "null"]``) — is redundant beside
    a union. A property literally **named** ``type`` (its value is a sub-schema
    ``dict``) inside a ``properties`` map is a parameter, not a keyword, and must be
    preserved even when a sibling property is named ``anyOf``/``oneOf``; aios
    sanitizes untrusted third-party tool schemas, so dropping it would silently
    corrupt a valid tool and make the model call it wrong.
    """
    if isinstance(node, dict):
        has_union = "anyOf" in node or "oneOf" in node
        drop_type = has_union and isinstance(node.get("type"), (str, list))
        return {
            key: sanitize_mcp_schema(value)
            for key, value in node.items()
            if not (drop_type and key == "type")
        }
    if isinstance(node, list):
        return [sanitize_mcp_schema(item) for item in node]
    return node


def _sanitize_name_segment(raw: str) -> str:
    cleaned = _UNSAFE_NAME_CHARS.sub("_", raw).strip("_")
    return cleaned or "x"


def qualify_mcp_tool_name(server_name: str, tool_name: str) -> tuple[str, str, str]:
    """Fit ``mcp__<server>__<tool>`` to the provider name regex.

    Returns ``(advertised, origin_server, origin_tool)``. Dispatch must call the
    origin tool on the origin server; the advertised name is what the model sees.
    """
    server = _sanitize_name_segment(server_name)
    tool = _sanitize_name_segment(tool_name)
    advertised = f"mcp__{server}__{tool}"
    if len(advertised) <= PROVIDER_TOOL_NAME_MAX and _PROVIDER_TOOL_NAME_RE.match(advertised):
        return advertised, server_name, tool_name
    digest = hashlib.sha1(f"{server_name}\0{tool_name}".encode()).hexdigest()
    prefix = f"mcp__{server}__"
    # ``<truncated>_<hash6>`` must fit after the prefix.
    room = PROVIDER_TOOL_NAME_MAX - len(prefix) - 7
    if room >= 1:
        fitted = f"{prefix}{tool[:room]}_{digest[:6]}"
        return fitted, server_name, tool_name
    # Server segment itself leaves no room for a tool — hash the server too.
    server_h = hashlib.sha1(server_name.encode()).hexdigest()[:8]
    prefix = f"mcp__{server_h}__"
    room = max(PROVIDER_TOOL_NAME_MAX - len(prefix) - 7, 1)
    fitted = f"{prefix}{tool[:room]}_{digest[:6]}"
    return fitted[:PROVIDER_TOOL_NAME_MAX], server_name, tool_name


def make_function_tool(
    qualified_name: str,
    tool: Tool,
    *,
    origin_server: str | None = None,
    origin_tool: str | None = None,
) -> dict[str, Any]:
    """Build the envelope, applying :func:`sanitize_mcp_schema` to ``tool.inputSchema``.

    When the tool declares an ``outputSchema`` (MCP 2025-06-18 structured
    output), keep it on the *internal* envelope so result-shaping and tests
    can see it (#1493). :func:`sanitize_tools_for_provider` strips it before
    the LiteLLM call — Anthropic rejects the unknown field with a 400
    (``drop_params`` is forced off, so LiteLLM will not silently drop it).
    """
    function: dict[str, Any] = {
        "name": qualified_name,
        "description": tool.description or "",
        "parameters": sanitize_mcp_schema(tool.inputSchema),
        # Arbitrary MCP schemas are not guaranteed to fit providers' narrower
        # strict-function subset. Explicit false also prevents provider-side
        # auto-promotion from turning optional MCP properties into required ones.
        "strict": False,
    }
    output_schema = getattr(tool, "outputSchema", None)
    if output_schema is not None:
        function["outputSchema"] = sanitize_mcp_schema(output_schema)
    envelope: dict[str, Any] = {
        "type": "function",
        "function": function,
    }
    if origin_server and origin_tool:
        envelope[MCP_ORIGIN_KEY] = {"server": origin_server, "tool": origin_tool}
    return envelope


def _tool_needs_provider_sanitize(tool: dict[str, Any]) -> bool:
    if MCP_ORIGIN_KEY in tool:
        return True
    function = tool.get("function")
    if not isinstance(function, dict):
        return False
    if "outputSchema" in function:
        return True
    name = function.get("name")
    return isinstance(name, str) and _PROVIDER_TOOL_NAME_RE.match(name) is None


def sanitize_tools_for_provider(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy-on-write: drop fields / names Anthropic (and OpenAI) 400 on.

    Returns the input list object when nothing needs changing so existing
    identity assertions on a clean toolset keep holding.
    """
    if not any(_tool_needs_provider_sanitize(tool) for tool in tools):
        return tools
    cleaned: list[dict[str, Any]] = []
    for tool in tools:
        item = copy.deepcopy(tool)
        item.pop(MCP_ORIGIN_KEY, None)
        function = item.get("function")
        if isinstance(function, dict):
            function.pop("outputSchema", None)
            name = function.get("name")
            if isinstance(name, str) and _PROVIDER_TOOL_NAME_RE.match(name) is None:
                function["name"] = _sanitize_name_segment(name)[:PROVIDER_TOOL_NAME_MAX]
        cleaned.append(item)
    return cleaned


def mcp_origin_for(
    qualified_name: str, tools: list[dict[str, Any]] | None
) -> tuple[str, str] | None:
    """Look up the raw ``(server, tool)`` for an advertised name, if stored."""
    for tool in tools or []:
        function = tool.get("function") or {}
        if function.get("name") != qualified_name:
            continue
        origin = tool.get(MCP_ORIGIN_KEY)
        if not isinstance(origin, dict):
            return None
        server = origin.get("server")
        name = origin.get("tool")
        if isinstance(server, str) and server and isinstance(name, str) and name:
            return server, name
        return None
    return None
