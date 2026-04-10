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
    title: str | None,
    metadata: dict[str, Any],
) -> Session:
    """Create a session row and return it.

    Note: Phase 1 doesn't actually mkdir the workspace path — that happens
    in Phase 3 when the Docker pool is wired up. The path is stored on the
    row so the directory creation is deterministic later.
    """
    async with pool.acquire() as conn:
        # We need the session id before we know the workspace path. Insert
        # with a placeholder, then UPDATE — except we want everything in one
        # statement, so we generate the id ahead of time.
        from aios.ids import SESSION, make_id

        new_id = make_id(SESSION)
        workspace_path = str(get_settings().workspace_root / new_id)

        # We bypass queries.insert_session because we need to provide the id
        # ourselves; replicate the SQL directly.
        import json

        try:
            row = await conn.fetchrow(
                """
                INSERT INTO sessions (
                    id, agent_id, environment_id, title, metadata,
                    status, workspace_volume_path
                )
                VALUES ($1, $2, $3, $4, $5::jsonb, 'idle', $6)
                RETURNING *
                """,
                new_id,
                agent_id,
                environment_id,
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
    stop_reason: str | None = None,
) -> None:
    async with pool.acquire() as conn:
        await queries.set_session_status(conn, session_id, status, stop_reason)
