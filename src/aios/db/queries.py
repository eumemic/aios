"""All database queries for aios v1, written as raw SQL against asyncpg.

Each query is an async function that takes an asyncpg connection (or
acquires one from the pool) and returns either a pydantic model or a list
of them. JSONB columns are decoded as Python dicts; bytea as bytes.

The :func:`append_event` function is the only query that needs careful
serialization: it acquires a row lock on the parent session, increments
``last_event_seq``, inserts the new event with that seq, then issues
``pg_notify`` so SSE subscribers are nudged. The row lock guarantees gapless
seqs even when the API and the harness are appending concurrently.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from aios.crypto.vault import EncryptedBlob
from aios.errors import ConflictError, NotFoundError
from aios.ids import AGENT, CREDENTIAL, ENVIRONMENT, EVENT, SESSION, make_id
from aios.models.agents import Agent, ToolSpec
from aios.models.credentials import Credential
from aios.models.environments import Environment
from aios.models.events import Event, EventKind
from aios.models.sessions import Session, SessionStatus

# ─── credentials ──────────────────────────────────────────────────────────────


def _row_to_credential(row: asyncpg.Record) -> Credential:
    return Credential(
        id=row["id"],
        name=row["name"],
        provider=row["provider"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def insert_credential(
    conn: asyncpg.Connection[Any],
    *,
    name: str,
    provider: str,
    blob: EncryptedBlob,
) -> Credential:
    new_id = make_id(CREDENTIAL)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO credentials (id, name, provider, ciphertext, nonce)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
            """,
            new_id,
            name,
            provider,
            blob.ciphertext,
            blob.nonce,
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"a credential named {name!r} already exists",
            detail={"name": name},
        ) from exc
    assert row is not None
    return _row_to_credential(row)


async def get_credential(conn: asyncpg.Connection[Any], cred_id: str) -> Credential:
    row = await conn.fetchrow("SELECT * FROM credentials WHERE id = $1", cred_id)
    if row is None:
        raise NotFoundError(f"credential {cred_id} not found", detail={"id": cred_id})
    return _row_to_credential(row)


async def get_credential_blob(conn: asyncpg.Connection[Any], cred_id: str) -> EncryptedBlob:
    """Fetch a credential's encrypted blob for decryption inside the harness."""
    row = await conn.fetchrow(
        "SELECT ciphertext, nonce FROM credentials WHERE id = $1 AND archived_at IS NULL",
        cred_id,
    )
    if row is None:
        raise NotFoundError(f"credential {cred_id} not found or archived")
    return EncryptedBlob(ciphertext=bytes(row["ciphertext"]), nonce=bytes(row["nonce"]))


async def list_credentials(
    conn: asyncpg.Connection[Any], *, limit: int = 50, after: str | None = None
) -> list[Credential]:
    if after is None:
        rows = await conn.fetch(
            "SELECT * FROM credentials WHERE archived_at IS NULL ORDER BY id DESC LIMIT $1",
            limit,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM credentials WHERE archived_at IS NULL AND id < $1 "
            "ORDER BY id DESC LIMIT $2",
            after,
            limit,
        )
    return [_row_to_credential(r) for r in rows]


async def archive_credential(conn: asyncpg.Connection[Any], cred_id: str) -> None:
    result = await conn.execute(
        "UPDATE credentials SET archived_at = now() WHERE id = $1 AND archived_at IS NULL",
        cred_id,
    )
    if result == "UPDATE 0":
        raise NotFoundError(f"credential {cred_id} not found or already archived")


# ─── environments ─────────────────────────────────────────────────────────────


def _row_to_environment(row: asyncpg.Record) -> Environment:
    return Environment(
        id=row["id"],
        name=row["name"],
        created_at=row["created_at"],
        archived_at=row["archived_at"],
    )


async def insert_environment(conn: asyncpg.Connection[Any], *, name: str) -> Environment:
    new_id = make_id(ENVIRONMENT)
    try:
        row = await conn.fetchrow(
            "INSERT INTO environments (id, name) VALUES ($1, $2) RETURNING *",
            new_id,
            name,
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"an environment named {name!r} already exists",
            detail={"name": name},
        ) from exc
    assert row is not None
    return _row_to_environment(row)


async def get_environment(conn: asyncpg.Connection[Any], env_id: str) -> Environment:
    row = await conn.fetchrow("SELECT * FROM environments WHERE id = $1", env_id)
    if row is None:
        raise NotFoundError(f"environment {env_id} not found", detail={"id": env_id})
    return _row_to_environment(row)


async def list_environments(
    conn: asyncpg.Connection[Any], *, limit: int = 50, after: str | None = None
) -> list[Environment]:
    if after is None:
        rows = await conn.fetch(
            "SELECT * FROM environments WHERE archived_at IS NULL ORDER BY id DESC LIMIT $1",
            limit,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM environments WHERE archived_at IS NULL AND id < $1 "
            "ORDER BY id DESC LIMIT $2",
            after,
            limit,
        )
    return [_row_to_environment(r) for r in rows]


async def archive_environment(conn: asyncpg.Connection[Any], env_id: str) -> None:
    result = await conn.execute(
        "UPDATE environments SET archived_at = now() WHERE id = $1 AND archived_at IS NULL",
        env_id,
    )
    if result == "UPDATE 0":
        raise NotFoundError(f"environment {env_id} not found or already archived")


# ─── agents ───────────────────────────────────────────────────────────────────


def _row_to_agent(row: asyncpg.Record) -> Agent:
    raw_tools = row["tools"]
    tools_data = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
    raw_metadata = row["metadata"]
    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
    return Agent(
        id=row["id"],
        name=row["name"],
        model=row["model"],
        system=row["system"],
        tools=[ToolSpec.model_validate(t) for t in tools_data],
        credential_id=row["credential_id"],
        description=row["description"],
        metadata=metadata,
        window_min=row["window_min"],
        window_max=row["window_max"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def insert_agent(
    conn: asyncpg.Connection[Any],
    *,
    name: str,
    model: str,
    system: str,
    tools: list[ToolSpec],
    credential_id: str | None,
    description: str | None,
    metadata: dict[str, Any],
    window_min: int,
    window_max: int,
) -> Agent:
    new_id = make_id(AGENT)
    tools_json = json.dumps([t.model_dump() for t in tools])
    metadata_json = json.dumps(metadata)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO agents (
                id, name, model, system, tools, credential_id,
                description, metadata, window_min, window_max
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8::jsonb, $9, $10)
            RETURNING *
            """,
            new_id,
            name,
            model,
            system,
            tools_json,
            credential_id,
            description,
            metadata_json,
            window_min,
            window_max,
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"an agent named {name!r} already exists",
            detail={"name": name},
        ) from exc
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            f"credential {credential_id} not found",
            detail={"credential_id": credential_id},
        ) from exc
    assert row is not None
    return _row_to_agent(row)


async def get_agent(conn: asyncpg.Connection[Any], agent_id: str) -> Agent:
    row = await conn.fetchrow("SELECT * FROM agents WHERE id = $1", agent_id)
    if row is None:
        raise NotFoundError(f"agent {agent_id} not found", detail={"id": agent_id})
    return _row_to_agent(row)


async def list_agents(
    conn: asyncpg.Connection[Any], *, limit: int = 50, after: str | None = None
) -> list[Agent]:
    if after is None:
        rows = await conn.fetch(
            "SELECT * FROM agents WHERE archived_at IS NULL ORDER BY id DESC LIMIT $1",
            limit,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM agents WHERE archived_at IS NULL AND id < $1 ORDER BY id DESC LIMIT $2",
            after,
            limit,
        )
    return [_row_to_agent(r) for r in rows]


async def archive_agent(conn: asyncpg.Connection[Any], agent_id: str) -> None:
    result = await conn.execute(
        "UPDATE agents SET archived_at = now() WHERE id = $1 AND archived_at IS NULL",
        agent_id,
    )
    if result == "UPDATE 0":
        raise NotFoundError(f"agent {agent_id} not found or already archived")


# ─── sessions ─────────────────────────────────────────────────────────────────


def _row_to_session(row: asyncpg.Record) -> Session:
    raw_metadata = row["metadata"]
    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
    return Session(
        id=row["id"],
        agent_id=row["agent_id"],
        environment_id=row["environment_id"],
        title=row["title"],
        metadata=metadata,
        status=row["status"],
        stop_reason=row["stop_reason"],
        last_event_seq=row["last_event_seq"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def insert_session(
    conn: asyncpg.Connection[Any],
    *,
    agent_id: str,
    environment_id: str,
    title: str | None,
    metadata: dict[str, Any],
    workspace_volume_path: str,
) -> Session:
    new_id = make_id(SESSION)
    metadata_json = json.dumps(metadata)
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
            metadata_json,
            workspace_volume_path,
        )
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            "agent or environment not found",
            detail={"agent_id": agent_id, "environment_id": environment_id},
        ) from exc
    assert row is not None
    return _row_to_session(row)


async def get_session(conn: asyncpg.Connection[Any], session_id: str) -> Session:
    row = await conn.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
    if row is None:
        raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})
    return _row_to_session(row)


async def list_sessions(
    conn: asyncpg.Connection[Any],
    *,
    agent_id: str | None = None,
    status: SessionStatus | None = None,
    limit: int = 50,
    after: str | None = None,
) -> list[Session]:
    clauses: list[str] = ["archived_at IS NULL"]
    args: list[Any] = []
    if agent_id is not None:
        args.append(agent_id)
        clauses.append(f"agent_id = ${len(args)}")
    if status is not None:
        args.append(status)
        clauses.append(f"status = ${len(args)}")
    if after is not None:
        args.append(after)
        clauses.append(f"id < ${len(args)}")
    args.append(limit)
    sql = (
        f"SELECT * FROM sessions WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT ${len(args)}"
    )
    rows = await conn.fetch(sql, *args)
    return [_row_to_session(r) for r in rows]


async def set_session_status(
    conn: asyncpg.Connection[Any],
    session_id: str,
    status: SessionStatus,
    stop_reason: str | None = None,
) -> None:
    await conn.execute(
        "UPDATE sessions SET status = $1, stop_reason = $2, updated_at = now() WHERE id = $3",
        status,
        stop_reason,
        session_id,
    )


# ─── events ───────────────────────────────────────────────────────────────────


def _row_to_event(row: asyncpg.Record) -> Event:
    raw_data = row["data"]
    data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    return Event(
        id=row["id"],
        session_id=row["session_id"],
        seq=row["seq"],
        kind=row["kind"],
        data=data,
        created_at=row["created_at"],
    )


async def append_event(
    conn: asyncpg.Connection[Any],
    *,
    session_id: str,
    kind: EventKind,
    data: dict[str, Any],
) -> Event:
    """Append an event to ``session_id`` with gapless seq allocation.

    Wraps the increment + insert in a single transaction with a row lock on
    the parent session, so concurrent appenders (the API server adding a
    user message while the harness is mid-turn) serialize correctly. Issues
    ``pg_notify`` after the insert so SSE subscribers receive the new event.
    """
    new_id = make_id(EVENT)
    data_json = json.dumps(data)

    async with conn.transaction():
        seq_row = await conn.fetchrow(
            "UPDATE sessions SET last_event_seq = last_event_seq + 1 "
            "WHERE id = $1 RETURNING last_event_seq",
            session_id,
        )
        if seq_row is None:
            raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})
        seq = seq_row["last_event_seq"]
        row = await conn.fetchrow(
            "INSERT INTO events (id, session_id, seq, kind, data) "
            "VALUES ($1, $2, $3, $4, $5::jsonb) RETURNING *",
            new_id,
            session_id,
            seq,
            kind,
            data_json,
        )
        assert row is not None

    # NOTIFY happens outside the transaction so subscribers don't see it
    # before the row is committed.
    await conn.execute(f"NOTIFY events_{session_id}, '{new_id}'")
    return _row_to_event(row)


async def read_events(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    after_seq: int = 0,
    kind: EventKind | None = None,
    limit: int = 200,
) -> list[Event]:
    if kind is None:
        rows = await conn.fetch(
            "SELECT * FROM events WHERE session_id = $1 AND seq > $2 ORDER BY seq ASC LIMIT $3",
            session_id,
            after_seq,
            limit,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM events WHERE session_id = $1 AND seq > $2 AND kind = $3 "
            "ORDER BY seq ASC LIMIT $4",
            session_id,
            after_seq,
            kind,
            limit,
        )
    return [_row_to_event(r) for r in rows]


async def read_message_events(conn: asyncpg.Connection[Any], session_id: str) -> list[Event]:
    """Read every message-kind event for a session in chronological order.

    Used by the harness to reconstruct the conversation history before
    applying the windowing function.
    """
    rows = await conn.fetch(
        "SELECT * FROM events WHERE session_id = $1 AND kind = 'message' ORDER BY seq ASC",
        session_id,
    )
    return [_row_to_event(r) for r in rows]
