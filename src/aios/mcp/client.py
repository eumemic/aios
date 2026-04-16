"""MCP client — tool discovery, invocation, and credential resolution.

Each function opens a fresh connection to the MCP server via streamable
HTTP transport. There is no connection caching or pooling — this matches
the per-step model where each ``wake_session`` job is a single
procrastinate invocation.

Tool names are namespaced as ``mcp__<server_name>__<tool_name>`` to
avoid collisions with built-in and custom tools. The model sees these
as regular function-calling tools.
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any

import asyncpg
import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.logging import get_logger

log = get_logger("aios.mcp.client")

MAX_TOOLS_PER_SERVER = 128


async def _open_session(url: str, headers: dict[str, str], stack: AsyncExitStack) -> ClientSession:
    """Open a fully initialized MCP session, registering all contexts on *stack*."""
    http_client = await stack.enter_async_context(httpx.AsyncClient(headers=headers))
    read_stream, write_stream, _ = await stack.enter_async_context(
        streamable_http_client(url, http_client=http_client)
    )
    session: ClientSession = await stack.enter_async_context(
        ClientSession(read_stream, write_stream)
    )
    await session.initialize()
    return session


def _headers_from_credential(
    crypto_box: CryptoBox,
    blob: Any,
    auth_type: str,
) -> dict[str, str]:
    payload = json.loads(crypto_box.decrypt(blob))

    if auth_type == "static_bearer":
        token = payload.get("token", "")
    elif auth_type == "mcp_oauth":
        token = payload.get("access_token", "")
    else:
        log.warning("mcp.unknown_auth_type", auth_type=auth_type)
        return {}

    if not token:
        return {}

    return {"Authorization": f"Bearer {token}"}


async def resolve_auth_for_url(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    session_id: str,
    mcp_server_url: str,
) -> dict[str, str]:
    """Resolve MCP auth headers for ``mcp_server_url``.

    Connection-owned URLs resolve through the connection's vault;
    other URLs fall back to the session's bound vaults.  A connection
    that owns the URL but has no matching credential returns ``{}``
    rather than falling back — ownership decides the source, not
    whether the lookup hits (prevents a misconfigured connection from
    silently leaking a tenant-level credential).
    """
    async with pool.acquire() as conn:
        connection_vault_id = await queries.get_connection_vault_for_url(conn, mcp_server_url)
        if connection_vault_id is not None:
            result = await queries.resolve_vault_credential(
                conn, vault_id=connection_vault_id, mcp_server_url=mcp_server_url
            )
        else:
            result = await queries.resolve_mcp_credential(conn, session_id, mcp_server_url)
    if result is None:
        return {}
    blob, auth_type = result
    return _headers_from_credential(crypto_box, blob, auth_type)


async def discover_mcp_tools(
    url: str,
    server_name: str,
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    """Connect to an MCP server and discover available tools.

    Returns a list of OpenAI-format tool dicts with namespaced names
    (``mcp__<server_name>__<tool_name>``). On any error, logs a warning
    and returns an empty list — the model simply doesn't see those tools.
    """
    try:
        async with AsyncExitStack() as stack:
            session = await _open_session(url, headers, stack)
            result = await session.list_tools()

        if len(result.tools) > MAX_TOOLS_PER_SERVER:
            log.warning(
                "mcp.tools_truncated",
                server_name=server_name,
                total=len(result.tools),
                limit=MAX_TOOLS_PER_SERVER,
            )

        tools: list[dict[str, Any]] = []
        for tool in result.tools[:MAX_TOOLS_PER_SERVER]:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f"mcp__{server_name}__{tool.name}",
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
            )
        return tools

    except Exception:
        log.warning(
            "mcp.discovery_failed",
            server_name=server_name,
            url=url,
            exc_info=True,
        )
        return []


async def call_mcp_tool(
    url: str,
    headers: dict[str, str],
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Connect to an MCP server and invoke a tool.

    ``tool_name`` is the raw MCP tool name (without the ``mcp__`` prefix).
    Returns a result dict with either ``content`` (success) or ``error``
    (failure).
    """
    try:
        async with AsyncExitStack() as stack:
            session = await _open_session(url, headers, stack)
            result = await session.call_tool(tool_name, arguments)

        # Concatenate text content from the result.
        parts: list[str] = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(f"[{item.type} content]")

        content = "\n".join(parts) if parts else ""

        if result.isError:
            return {"error": content}
        return {"content": content}

    except Exception as err:
        log.warning(
            "mcp.call_failed",
            url=url,
            tool_name=tool_name,
            exc_info=True,
        )
        return {"error": f"MCP server error: {type(err).__name__}: {err}"}
