"""The list_related_sessions tool — enumerate chat sessions on your account.

A flat listing of the chat sessions routed through the caller account's
connection(s): each ``(chat_id, session_id, created_at)`` row recorded in
``chat_sessions``, account-scoped to the caller. There is NO ACL or subset
semantics — every chat-session row under the caller's account is returned.

An optional ``connection_id`` narrows the listing to a single connection,
which must belong to the caller's account; a cross-account (or unknown) id
raises ``NotFoundError`` via the account-scoped ``get_connection`` guard.
"""

from __future__ import annotations

from typing import Any

from aios.harness import runtime
from aios.services import sessions as sessions_service
from aios.tools.registry import registry

LIST_RELATED_SESSIONS_DESCRIPTION = (
    "List the chat sessions routed through your account's connection(s). "
    "Returns a flat array of {chat_id, session_id, created_at} rows — one per "
    "chat that a connector has bound to a session on your account. This is a "
    "plain listing with no access-control or subset semantics: every "
    "chat-session binding on your account is returned. Pass an optional "
    "connection_id to restrict the listing to a single connection (it must "
    "belong to your account)."
)

LIST_RELATED_SESSIONS_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "connection_id": {
            "type": "string",
            "minLength": 1,
            "description": "Optional: restrict to one connection (must belong to your account).",
        },
    },
    "required": [],
    "additionalProperties": False,
}


async def list_related_sessions_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    from aios.db import queries

    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    connection_id = arguments.get("connection_id")

    rows_out: list[dict[str, Any]] = []
    async with pool.acquire() as conn:
        if isinstance(connection_id, str) and connection_id:
            # Account-scoped guard: raises NotFoundError on a cross-account
            # (or unknown) connection id before any listing happens.
            await queries.get_connection(conn, connection_id, account_id=account_id)
            conns = [connection_id]
        else:
            connections = await queries.list_connections(conn, account_id=account_id)
            conns = [c.id for c in connections]
        for cid in conns:
            rows = await queries.list_chat_sessions_for_connection(conn, cid, account_id=account_id)
            for chat_id, sess_id, created_at in rows:
                rows_out.append(
                    {
                        "chat_id": chat_id,
                        "session_id": sess_id,
                        "created_at": created_at.isoformat(),
                    }
                )
    return {"sessions": rows_out}


def _register() -> None:
    registry.register(
        name="list_related_sessions",
        description=LIST_RELATED_SESSIONS_DESCRIPTION,
        parameters_schema=LIST_RELATED_SESSIONS_PARAMETERS_SCHEMA,
        handler=list_related_sessions_handler,
        transport="agent_tool",
    )


_register()
