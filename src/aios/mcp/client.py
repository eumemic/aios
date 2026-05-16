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

import asyncio
import base64
import json
from contextlib import AsyncExitStack
from typing import Any, assert_never

import asyncpg
import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import InitializeResult

from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.logging import get_logger
from aios.mcp.schema import make_function_tool
from aios.models.vaults import AuthType
from aios.services.vaults import is_expiring, refresh_credential

log = get_logger("aios.mcp.client")

MAX_TOOLS_PER_SERVER = 128

# Per-call bound for ``session.call_tool``. The harness needs every external
# await to have a finite ceiling so the worker can't hang on a misbehaving
# MCP server. Mirrors the same intent as the LiteLLM call timeouts in
# ``harness/completion.py``.
_TOOL_CALL_TIMEOUT_S = 120.0

# httpx client bounds for MCP transport. ``read`` is the longest leg —
# tool calls that do real work (DB lookups, external APIs) commonly take
# tens of seconds. Connect/write/pool are tight because they're network
# fast paths.
_MCP_HTTPX_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)


async def _open_session(
    url: str, headers: dict[str, str], stack: AsyncExitStack
) -> tuple[ClientSession, InitializeResult]:
    """Open a fully initialized MCP session, registering all contexts on *stack*.

    Returns the session along with the ``InitializeResult`` so callers can
    read server-supplied metadata (notably ``instructions``).
    """
    http_client = await stack.enter_async_context(
        httpx.AsyncClient(headers=headers, timeout=_MCP_HTTPX_TIMEOUT)
    )
    read_stream, write_stream, _ = await stack.enter_async_context(
        streamable_http_client(url, http_client=http_client)
    )
    session: ClientSession = await stack.enter_async_context(
        ClientSession(read_stream, write_stream)
    )
    init_result = await session.initialize()
    return session, init_result


def _auth_headers_from_payload(payload: dict[str, Any], auth_type: AuthType) -> dict[str, str]:
    """Render the auth header(s) the broker writes for ``auth_type``.

    Empty dict means "no headers to send" — caller treats the request
    as unauthenticated.  ``bearer_header`` and ``oauth2_refresh``
    produce ``Authorization: Bearer <token>``; ``basic`` produces
    ``Authorization: Basic <base64(user:pass)>``; ``custom_header``
    writes the operator-configured header name (e.g.
    ``X-Browser-Use-API-Key``) with the credential value.
    """
    if auth_type == "bearer_header":
        token = str(payload.get("token", ""))
        return {"Authorization": f"Bearer {token}"} if token else {}
    if auth_type == "oauth2_refresh":
        token = str(payload.get("access_token", ""))
        return {"Authorization": f"Bearer {token}"} if token else {}
    if auth_type == "basic":
        username = str(payload.get("username", ""))
        password = str(payload.get("password", ""))
        if not username and not password:
            return {}
        encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
        return {"Authorization": f"Basic {encoded}"}
    if auth_type == "custom_header":
        header_name = str(payload.get("header_name", ""))
        header_value = str(payload.get("header_value", ""))
        if not header_name or not header_value:
            return {}
        return {header_name: header_value}
    assert_never(auth_type)


async def resolve_auth_for_target_url(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    session_id: str,
    target_url: str,
    *,
    account_id: str,
) -> tuple[str | None, dict[str, str]]:
    """Resolve auth identity + headers for ``target_url`` via the
    session's bound vaults.

    Returns ``(vault_id, headers)`` on the success path, or
    ``(None, {})`` whenever there are no auth headers to send (no
    credential bound, or a credential whose secret material is empty
    / a refresh re-read missed). ``vault_id`` is the row id of the
    ``vault_credentials`` entry — stable across OAuth token rotation
    (``refresh_credential`` updates the row in place), so callers feed
    it to the pool as a stable cache key (see #459).

    For ``oauth2_refresh`` credentials whose ``expires_at`` falls within
    the refresh skew window, the access token is transparently refreshed
    (row-locked via ``refresh_credential`` to serialize concurrent
    refreshes of the same row) before returning.  OAuth refresh failures
    bubble up as :class:`OAuthRefreshError`; there is no silent fallback
    to the stale token.
    """
    async with pool.acquire() as conn:
        session_result = await queries.resolve_session_credential(
            conn, session_id, target_url, account_id=account_id
        )
        if session_result is None:
            return None, {}
        blob, auth_type, vault_id = session_result
        subkey = crypto_box.derive_account_subkey(account_id)

        if auth_type == "oauth2_refresh":
            payload = json.loads(subkey.decrypt(blob))
            if is_expiring(payload):
                await refresh_credential(
                    crypto_box,
                    conn,
                    vault_id=vault_id,
                    target_url=target_url,
                    account_id=account_id,
                )
                refreshed = await queries.resolve_vault_credential(
                    conn, vault_id=vault_id, target_url=target_url, account_id=account_id
                )
                if refreshed is None:
                    return None, {}
                blob = refreshed[0]
                payload = json.loads(subkey.decrypt(blob))
        else:
            payload = json.loads(subkey.decrypt(blob))

    headers = _auth_headers_from_payload(payload, auth_type)
    if not headers:
        return None, {}
    return vault_id, headers


async def discover_mcp_tools(
    url: str,
    vault_id: str | None,
    headers: dict[str, str],
    server_name: str,
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
                session, init_result = await _pool.get_or_connect(url, vault_id, headers)
                result = await session.list_tools()
            except Exception:
                _pool.evict(url, vault_id)
                session, init_result = await _pool.get_or_connect(url, vault_id, headers)
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

        tools: list[dict[str, Any]] = [
            make_function_tool(f"mcp__{server_name}__{tool.name}", tool)
            for tool in result.tools[:MAX_TOOLS_PER_SERVER]
        ]
        return tools, init_result.instructions

    except Exception:
        log.warning(
            "mcp.discovery_failed",
            server_name=server_name,
            url=url,
            exc_info=True,
        )
        return [], None


def shape_call_result(result: Any) -> dict[str, Any]:
    """Project an MCP ``CallToolResult`` onto aios's ``{"content"|"error": str}`` envelope.

    Tool-level errors carry ``code="tool_error"`` so the API router can
    distinguish them from transport / not-ready / circuit-open
    failures (mapped to different HTTP statuses) without substring
    matching on the human-readable message.
    """
    parts: list[str] = []
    for item in result.content:
        if hasattr(item, "text"):
            parts.append(item.text)
        else:
            parts.append(f"[{item.type} content]")
    content = "\n".join(parts) if parts else ""
    if result.isError:
        return {"error": content, "code": "tool_error"}
    return {"content": content}


async def call_mcp_tool(
    url: str,
    vault_id: str | None,
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
    redesign to pass ``aios.focal_channel_path`` to connection-provided
    MCP servers without stuffing it into arguments.  Returns a result
    dict with either ``content`` (success) or ``error`` (failure).
    """
    try:
        from aios.harness import runtime

        _pool = runtime.mcp_session_pool

        if _pool is not None:
            try:
                session, _ = await _pool.get_or_connect(url, vault_id, headers)
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments, meta=meta),
                    timeout=_TOOL_CALL_TIMEOUT_S,
                )
            except TimeoutError:
                # Don't retry on timeout: the wait_for fires while we
                # wait for the response, but the request may already
                # have reached the server and been processed. A retry
                # would duplicate the side effect — e.g. ``signal_send``
                # / ``telegram_send`` delivering the same message twice.
                # Evict so the next caller doesn't reuse the same
                # session; surface the error so the model can decide
                # whether to retry.
                _pool.evict(url, vault_id)
                raise
            except Exception:
                _pool.evict(url, vault_id)
                session, _ = await _pool.get_or_connect(url, vault_id, headers)
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments, meta=meta),
                    timeout=_TOOL_CALL_TIMEOUT_S,
                )
        else:
            async with AsyncExitStack() as stack:
                session, _ = await _open_session(url, headers, stack)
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments, meta=meta),
                    timeout=_TOOL_CALL_TIMEOUT_S,
                )

        return shape_call_result(result)

    except Exception as err:
        log.warning(
            "mcp.call_failed",
            url=url,
            tool_name=tool_name,
            exc_info=True,
        )
        return {
            "error": f"MCP server error: {type(err).__name__}: {err}",
            "code": "transport_error",
        }
