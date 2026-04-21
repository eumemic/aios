"""Business logic for sessions and their event log.

Session creation persists the workspace volume path (caller-supplied or
defaulting to ``settings.workspace_root / session_id``) and optional
per-session env vars on the row so workers can mount the correct volume
and inject environment variables at container provisioning time.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from aios.db import queries
from aios.errors import PayloadTooLargeError
from aios.models.events import Event, EventKind
from aios.models.sessions import MAX_USER_MESSAGE_CHARS, Session, SessionStatus


async def create_session(
    pool: asyncpg.Pool[Any],
    *,
    agent_id: str,
    environment_id: str,
    agent_version: int | None = None,
    title: str | None,
    metadata: dict[str, Any],
    vault_ids: list[str] | None = None,
    workspace_path: str | None = None,
    env: dict[str, str] | None = None,
) -> Session:
    """Create a session row and return it.

    ``agent_version=None`` means "latest" — the session will always use
    whatever version of the agent is current at step time.
    """
    async with pool.acquire() as conn:
        session = await queries.insert_session(
            conn,
            agent_id=agent_id,
            environment_id=environment_id,
            agent_version=agent_version,
            title=title,
            metadata=metadata,
            workspace_path=workspace_path,
            env=env,
        )
        if vault_ids:
            await queries.set_session_vaults(conn, session.id, vault_ids)
            session = session.model_copy(update={"vault_ids": vault_ids})
        return session


async def get_session(pool: asyncpg.Pool[Any], session_id: str) -> Session:
    async with pool.acquire() as conn:
        session = await queries.get_session(conn, session_id)
        vault_ids = await queries.get_session_vault_ids(conn, session_id)
        return session.model_copy(update={"vault_ids": vault_ids})


async def list_sessions(
    pool: asyncpg.Pool[Any],
    *,
    agent_id: str | None = None,
    status: SessionStatus | None = None,
    limit: int = 50,
    after: str | None = None,
) -> list[Session]:
    async with pool.acquire() as conn:
        sessions = await queries.list_sessions(
            conn, agent_id=agent_id, status=status, limit=limit, after=after
        )
        if sessions:
            vault_map = await queries.batch_get_session_vault_ids(conn, [s.id for s in sessions])
            sessions = [
                s.model_copy(update={"vault_ids": vault_map.get(s.id, [])}) for s in sessions
            ]
        return sessions


async def append_user_message(
    pool: asyncpg.Pool[Any],
    session_id: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> Event:
    """Append a `role: user` message event to the session log.

    When the inbound path stamps ``metadata["channel"]`` (the connector's
    full channel address), we lift it into the event's ``orig_channel``
    column so the context builder and unread-derivation helpers can key
    off it directly — without re-parsing a JSONB blob on every read.
    """
    if len(content) > MAX_USER_MESSAGE_CHARS:
        raise PayloadTooLargeError(
            f"user message exceeds {MAX_USER_MESSAGE_CHARS:,} characters "
            f"(got {len(content):,}); split into multiple messages",
            detail={"max_chars": MAX_USER_MESSAGE_CHARS, "got_chars": len(content)},
        )
    data: dict[str, Any] = {"role": "user", "content": content}
    if metadata:
        data["metadata"] = metadata
    orig_channel: str | None = None
    if metadata is not None:
        channel = metadata.get("channel")
        if isinstance(channel, str):
            orig_channel = channel
    async with pool.acquire() as conn:
        event = await queries.append_event(
            conn,
            session_id=session_id,
            kind="message",
            data=data,
            orig_channel=orig_channel,
        )
        # Flip idle → pending so polling orchestrators can distinguish
        # "queued but not started" from "turn finished." Other states
        # (running / rescheduling / terminated) are left alone — the
        # worker owns the running status, and changing rescheduling
        # would lose the retry-in-progress signal.  Narrow scope by
        # design: the tool-result and tool-confirmation paths have the
        # same race but are deferred; an orchestrator resolving those
        # still has to combine status polling with event-cursor
        # tracking.  See issue #39.
        await conn.execute(
            "UPDATE sessions SET status = 'pending', updated_at = now() "
            "WHERE id = $1 AND status = 'idle'",
            session_id,
        )
        return event


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


async def read_windowed_events(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    window_min: int,
    window_max: int,
) -> list[Event]:
    async with pool.acquire() as conn:
        return await queries.read_windowed_events(
            conn, session_id, window_min=window_min, window_max=window_max
        )


async def set_session_status(
    pool: asyncpg.Pool[Any],
    session_id: str,
    status: SessionStatus,
    stop_reason: dict[str, Any] | None = None,
) -> None:
    async with pool.acquire() as conn:
        await queries.set_session_status(conn, session_id, status, stop_reason)


async def increment_usage(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> None:
    """Atomically add token counts to a session's cumulative usage."""
    async with pool.acquire() as conn:
        await queries.increment_session_usage(
            conn,
            session_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
        )


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
    vault_ids: list[str] | None = None,
) -> Session:
    async with pool.acquire() as conn:
        session = await queries.update_session(
            conn,
            session_id,
            agent_id=agent_id,
            agent_version=agent_version,
            title=title,
            metadata=metadata,
        )
        if vault_ids is not None:
            await queries.set_session_vaults(conn, session_id, vault_ids)
        vids = await queries.get_session_vault_ids(conn, session_id)
        return session.model_copy(update={"vault_ids": vids})


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
