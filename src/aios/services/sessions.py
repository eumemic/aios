"""Business logic for sessions and their event log.

Phase 1 sessions own a workspace directory on the host but no Docker
container yet — Phase 3 wires the sandbox in. The session creation flow
allocates the workspace path under ``settings.workspace_root`` and persists
it on the row so future workers can re-mount the same volume.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.db import queries
from aios.models.events import Event, EventKind
from aios.models.sessions import Session, SessionStatus


def _workspace_path_for(session_id: str) -> str:
    """Compute the host-side workspace directory for ``session_id``."""
    return str(get_settings().workspace_root / session_id)


async def create_session(
    pool: asyncpg.Pool[Any],
    *,
    agent_id: str,
    environment_id: str,
    agent_version: int | None = None,
    title: str | None,
    metadata: dict[str, Any],
) -> Session:
    """Create a session row and return it.

    ``agent_version=None`` means "latest" — the session will always use
    whatever version of the agent is current at step time.
    """
    async with pool.acquire() as conn:
        from aios.ids import SESSION, make_id

        new_id = make_id(SESSION)
        workspace_path = str(get_settings().workspace_root / new_id)

        import json

        try:
            row = await conn.fetchrow(
                """
                INSERT INTO sessions (
                    id, agent_id, environment_id, agent_version, title, metadata,
                    status, workspace_volume_path
                )
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, 'idle', $7)
                RETURNING *
                """,
                new_id,
                agent_id,
                environment_id,
                agent_version,
                title,
                json.dumps(metadata),
                workspace_path,
            )
        except asyncpg.ForeignKeyViolationError as exc:
            from aios.errors import NotFoundError

            raise NotFoundError(
                "agent or environment not found",
                detail={"agent_id": agent_id, "environment_id": environment_id},
            ) from exc
        assert row is not None
        return queries._row_to_session(row)


async def get_session(pool: asyncpg.Pool[Any], session_id: str) -> Session:
    async with pool.acquire() as conn:
        return await queries.get_session(conn, session_id)


async def list_sessions(
    pool: asyncpg.Pool[Any],
    *,
    agent_id: str | None = None,
    status: SessionStatus | None = None,
    limit: int = 50,
    after: str | None = None,
) -> list[Session]:
    async with pool.acquire() as conn:
        return await queries.list_sessions(
            conn, agent_id=agent_id, status=status, limit=limit, after=after
        )


async def append_user_message(pool: asyncpg.Pool[Any], session_id: str, content: str) -> Event:
    """Append a `role: user` message event to the session log."""
    async with pool.acquire() as conn:
        return await queries.append_event(
            conn,
            session_id=session_id,
            kind="message",
            data={"role": "user", "content": content},
        )


async def append_event(
    pool: asyncpg.Pool[Any],
    session_id: str,
    kind: EventKind,
    data: dict[str, Any],
) -> Event:
    """Append an arbitrary event. Used by the harness loop."""
    async with pool.acquire() as conn:
        return await queries.append_event(conn, session_id=session_id, kind=kind, data=data)


async def read_events(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    after_seq: int = 0,
    kind: EventKind | None = None,
    limit: int = 200,
) -> list[Event]:
    async with pool.acquire() as conn:
        return await queries.read_events(
            conn, session_id, after_seq=after_seq, kind=kind, limit=limit
        )


async def read_message_events(pool: asyncpg.Pool[Any], session_id: str) -> list[Event]:
    async with pool.acquire() as conn:
        return await queries.read_message_events(conn, session_id)


async def set_session_status(
    pool: asyncpg.Pool[Any],
    session_id: str,
    status: SessionStatus,
    stop_reason: dict[str, Any] | None = None,
) -> None:
    async with pool.acquire() as conn:
        await queries.set_session_status(conn, session_id, status, stop_reason)


async def archive_session(pool: asyncpg.Pool[Any], session_id: str) -> Session:
    async with pool.acquire() as conn:
        return await queries.archive_session(conn, session_id)


async def delete_session(pool: asyncpg.Pool[Any], session_id: str) -> None:
    async with pool.acquire() as conn:
        await queries.delete_session(conn, session_id)


async def update_session(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    agent_id: str | None = None,
    agent_version: int | None = queries._UNSET,
    title: str | None = queries._UNSET,
    metadata: dict[str, Any] | None = None,
) -> Session:
    async with pool.acquire() as conn:
        return await queries.update_session(
            conn,
            session_id,
            agent_id=agent_id,
            agent_version=agent_version,
            title=title,
            metadata=metadata,
        )


# ─── tool confirmations ────────────────────────────────────────────────────


def _find_tool_call(events: list[Event], tool_call_id: str) -> dict[str, Any] | None:
    """Find a tool call dict by its id in the session's message events.

    Scans assistant messages in reverse order and returns the raw tool_call
    dict matching ``tool_call_id``, or ``None`` if not found.
    """
    for e in reversed(events):
        if e.kind != "message" or e.data.get("role") != "assistant":
            continue
        for tc in e.data.get("tool_calls") or []:
            if tc.get("id") == tool_call_id:
                result: dict[str, Any] = tc
                return result
    return None


async def confirm_tool_allow(
    pool: asyncpg.Pool[Any],
    session_id: str,
    tool_call_id: str,
) -> Event:
    """Record an allow confirmation for an ``always_ask`` tool call.

    Appends a lifecycle event. The worker's step function will see this
    and dispatch the tool call.
    """
    return await append_event(
        pool,
        session_id,
        "lifecycle",
        {"event": "tool_confirmed", "tool_call_id": tool_call_id, "result": "allow"},
    )


async def confirm_tool_deny(
    pool: asyncpg.Pool[Any],
    session_id: str,
    tool_call_id: str,
    deny_message: str,
) -> Event:
    """Deny an ``always_ask`` tool call.

    Appends a tool-role error event that the model will see in its next
    context window. The deny message is formatted to match Anthropic's
    ``"Permission to use <tool> has been rejected."`` pattern.
    """
    # Find the tool name from the event log for the error message.
    events = await read_message_events(pool, session_id)
    tc = _find_tool_call(events, tool_call_id)
    tool_name = ((tc.get("function") or {}).get("name", "unknown")) if tc else "unknown"

    content = json.dumps(
        {
            "error": (
                f"Permission to use {tool_name} has been rejected. "
                f"Rejection message: {deny_message}"
            )
        },
        ensure_ascii=False,
    )
    return await append_event(
        pool,
        session_id,
        "message",
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": content,
            "is_error": True,
        },
    )
