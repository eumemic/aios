"""MCP client — tool discovery, invocation, and credential resolution.

In worker context the module uses the worker-scoped
:class:`~aios.mcp.pool.McpSessionPool` (stashed on
:mod:`aios.harness.runtime`) to reuse persistent ``ClientSession``
instances across calls. Outside a worker context (tests, API process)
the pool is ``None`` and a fresh connection is opened per call — the
original behaviour.

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
from mcp.types import InitializeResult

from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.logging import get_logger
from aios.services.vaults import is_expiring, refresh_credential

log = get_logger("aios.mcp.client")

MAX_TOOLS_PER_SERVER = 128


async def _open_session(
    url: str, headers: dict[str, str], stack: AsyncExitStack
) -> tuple[ClientSession, InitializeResult]:
    """Open a fully initialized MCP session, registering all contexts on *stack*.

    Returns the session along with the ``InitializeResult`` so callers can
    read server-supplied metadata (notably ``instructions``).
    """
    http_client = await stack.enter_async_context(httpx.AsyncClient(headers=headers))
    read_stream, write_stream, _ = await stack.enter_async_context(
        streamable_http_client(url, http_client=http_client)
    )
    session: ClientSession = await stack.enter_async_context(
        ClientSession(read_stream, write_stream)
    )
    init_result = await session.initialize()
    return session, init_result


def _token_from_payload(payload: dict[str, Any], auth_type: str) -> str:
    if auth_type == "static_bearer":
        return str(payload.get("token", ""))
    if auth_type == "mcp_oauth":
        return str(payload.get("access_token", ""))
    log.warning("mcp.unknown_auth_type", auth_type=auth_type)
    return ""


async def resolve_auth_for_url(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    session_id: str,
    mcp_server_url: str,
) -> dict[str, str]:
    """Resolve MCP auth headers for ``mcp_server_url``.

    Agent-declared MCP servers resolve through the session's bound vaults
    (``session_vaults``). The agent chooses a tool name; the worker resolves
    the server URL from agent config and injects the matching vault credential.

    For ``mcp_oauth`` credentials whose ``expires_at`` falls within the
    refresh skew window, the access token is transparently refreshed
    (row-locked via ``refresh_credential`` to serialize concurrent
    refreshes of the same row) before returning.  OAuth refresh failures
    bubble up as :class:`OAuthRefreshError`; there is no silent fallback
    to the stale token.
    """
    async with pool.acquire() as conn:
        session_result = await queries.resolve_mcp_credential(conn, session_id, mcp_server_url)
        if session_result is None:
            return {}
        blob, auth_type, vault_id = session_result

        if auth_type == "mcp_oauth":
            payload = json.loads(crypto_box.decrypt(blob))
            if is_expiring(payload):
                await refresh_credential(
                    crypto_box, conn, vault_id=vault_id, mcp_server_url=mcp_server_url
                )
                refreshed = await queries.resolve_vault_credential(
                    conn, vault_id=vault_id, mcp_server_url=mcp_server_url
                )
                if refreshed is None:
                    return {}
                blob = refreshed[0]
                payload = json.loads(crypto_box.decrypt(blob))
            token = str(payload.get("access_token", ""))
        else:
            payload = json.loads(crypto_box.decrypt(blob))
            token = _token_from_payload(payload, auth_type)

    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


async def discover_mcp_tools(
    url: str,
    server_name: str,
    headers: dict[str, str],
) -> tuple[list[dict[str, Any]], str | None]:
    """Connect to an MCP server and discover available tools.

    Returns a ``(tools, instructions)`` pair:

    * ``tools`` — OpenAI-format tool dicts with namespaced names
      (``mcp__<server_name>__<tool_name>``).
    * ``instructions`` — the server's ``InitializeResult.instructions``
      string (per the MCP spec), or ``None`` if the server didn't supply
      any. Used by the harness to compose per-connector affordance prose
      into the system prompt.

    On any error, logs a warning and returns ``([], None)`` — the model
    simply doesn't see those tools (or that connector's instructions).
    """
    try:
        from aios.harness import runtime

        _pool = runtime.mcp_session_pool

        if _pool is not None:
            # Pool path: reuse cached session; on failure evict + retry once.
            try:
                session, init_result = await _pool.get_or_connect(url, headers)
                result = await session.list_tools()
            except Exception:
                _pool.evict(url, headers)
                session, init_result = await _pool.get_or_connect(url, headers)
                result = await session.list_tools()
        else:
            # Fallback: fresh connection per call (API process, tests).
            async with AsyncExitStack() as stack:
                session, init_result = await _open_session(url, headers, stack)
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
        return tools, init_result.instructions

    except Exception:
        log.warning(
            "mcp.discovery_failed",
            server_name=server_name,
            url=url,
            exc_info=True,
        )
        return [], None


async def call_mcp_tool(
    url: str,
    headers: dict[str, str],
    tool_name: str,
    arguments: dict[str, Any],
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Connect to an MCP server and invoke a tool.

    ``tool_name`` is the raw MCP tool name (without the ``mcp__`` prefix).
    ``meta`` is an optional per-request metadata dict forwarded as the
    JSON-RPC request's ``_meta`` field — used by the focal-channel
    redesign to pass ``aios.focal_channel_path`` to channel-aware MCP
    servers without stuffing it into arguments.  Returns a result
    dict with either ``content`` (success) or ``error`` (failure).
    """
    try:
        from aios.harness import runtime

        _pool = runtime.mcp_session_pool

        if _pool is not None:
            try:
                session, _ = await _pool.get_or_connect(url, headers)
                result = await session.call_tool(tool_name, arguments, meta=meta)
            except Exception:
                _pool.evict(url, headers)
                session, _ = await _pool.get_or_connect(url, headers)
                result = await session.call_tool(tool_name, arguments, meta=meta)
        else:
            async with AsyncExitStack() as stack:
                session, _ = await _open_session(url, headers, stack)
                result = await session.call_tool(tool_name, arguments, meta=meta)

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
