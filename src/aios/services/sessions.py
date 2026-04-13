"""Business logic for sessions and their event log.

Phase 1 sessions own a workspace directory on the host but no Docker
container yet — Phase 3 wires the sandbox in. The session creation flow
allocates the workspace path under ``settings.workspace_root`` and persists
it on the row so future workers can re-mount the same volume.
"""

from __future__ import annotations

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
