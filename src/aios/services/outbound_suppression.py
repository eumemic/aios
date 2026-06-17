"""Outbound-suppression primitive (#710).

When a session runs with ``outbound_suppression == "on"``, side-effecting
outbound tool calls are intercepted at the broker boundary: instead of leaving
the broker, a *write* returns a synthesized success and the intent is recorded
as a ``tool_call_suppressed`` audit event on the session log. Reads pass
through against the session's real credentials.

This module is the single source of the suppression policy and the audit-event
shape, shared by:

* the HTTP broker (:mod:`aios.tools.http_request`) — both the model-tool path
  and, transitively, the sandbox ``tool http_request`` / bash path, since both
  funnel through the same ``http_request_handler``; and
* the MCP broker — both the model-tool path
  (:mod:`aios.harness.tool_dispatch`) and the sandbox CLI broker
  (:mod:`aios.sandbox.tool_broker`).

Connector tools (``signal_send`` etc.) are deliberately NOT routed here — a
test session's account boundary isolates them (see the migration cutover
playbook). Bash is opaque to aios and is not suppressed directly; its external
effects compose through these brokers.

Classification policy:

* HTTP — the HTTP method is the default classifier (``GET`` passes, the
  side-effecting verbs are suppressed); a per-route ``suppress`` override
  handles the GETs-with-side-effects / reads-over-POST cases. See
  :func:`aios.models.agents.http_route_suppressed`.
* MCP — default-deny (suppress everything); an operator opts known-safe reads
  in via ``McpToolConfig.read_allow``. See
  :func:`aios.models.agents.mcp_tool_suppressed`.

The agent is NOT told a call was suppressed — synthesized responses look like
real successes, because behavior validation requires the agent to act as it
would in production.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.logging import get_logger
from aios.services import sessions as sessions_service

log = get_logger("aios.services.outbound_suppression")

# The audit event lives on the session log as a ``span``-kind event with this
# discriminator under ``data.event`` — searchable via ``search_events`` for
# post-test / post-cutover review. Reusing the existing ``span`` kind (rather
# than minting a new top-level ``EventKind``) keeps the event model stable; the
# discriminator names the new logical kind the migration doc asked for.
SUPPRESSED_EVENT: str = "tool_call_suppressed"


def http_synthesized_response(status: int) -> dict[str, Any]:
    """The synthesized success an HTTP write observes under suppression.

    Mirrors the real ``http_request`` return shape (``status``/``headers``/
    ``body``) so the agent can't tell it apart from a real success. The body is
    empty (``""``) — enough for v1; a per-route synthetic-response template is
    explicitly deferred until a real test hits the limitation (e.g. a write that
    returns the created entity's id).
    """
    return {"status": status, "headers": {}, "body": ""}


def mcp_synthesized_result() -> dict[str, Any]:
    """The synthesized success an MCP write observes under suppression.

    Mirrors the MCP-tool success envelope (``call_mcp_tool`` returns a dict;
    a real success carries ``content``, an error carries ``error``). An empty
    ``content`` reads as a clean, side-effect-free success.
    """
    return {"content": ""}


async def record_http_suppression(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
    server_ref: str,
    base_url: str,
    method: str,
    path: str,
    body: str | None,
    synthesized_status: int,
) -> None:
    """Append the ``tool_call_suppressed`` audit span for a suppressed HTTP write.

    Records the un-suppressed request (server_ref + base_url + method + path +
    body) plus ``would_have_returned: null`` so a reviewer sees exactly what the
    agent attempted and that nothing came back from upstream. No auth header is
    recorded — the agent never supplies one (the worker injects the vault
    secret only on a *real* dispatch, which never happens here), so the secret
    cannot leak into the log.
    """
    await _append_suppressed_event(
        pool,
        session_id,
        account_id=account_id,
        data={
            "tool": "http_request",
            "server_ref": server_ref,
            "base_url": base_url,
            "method": method,
            "path": path,
            "body": body,
            "would_have_returned": None,
            "synthesized_status": synthesized_status,
        },
    )


async def record_mcp_suppression(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    """Append the ``tool_call_suppressed`` audit span for a suppressed MCP call.

    Records the server + tool + arguments and ``would_have_returned: null``.
    MCP arguments are the agent's own input (no vault secret is ever placed
    there — credentials are resolved into transport headers, never the JSON-RPC
    params), so they're safe to log verbatim, same as authenticated request
    logging.
    """
    await _append_suppressed_event(
        pool,
        session_id,
        account_id=account_id,
        data={
            "tool": f"mcp__{server_name}__{tool_name}",
            "server_name": server_name,
            "tool_name": tool_name,
            "arguments": arguments,
            "would_have_returned": None,
        },
    )


async def _append_suppressed_event(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
    data: dict[str, Any],
) -> None:
    payload = {"event": SUPPRESSED_EVENT, **data}
    await sessions_service.append_event(pool, session_id, "span", payload, account_id=account_id)
    log.info(
        "outbound_suppression.suppressed",
        session_id=session_id,
        tool=data.get("tool"),
    )
