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
from aios.models.agents import Agent, AgentVersion, ToolSpec
from aios.models.credentials import Credential
from aios.models.environments import Environment, EnvironmentConfig
from aios.models.events import Event, EventKind
from aios.models.sessions import Session, SessionStatus, SessionUsage

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
    raw_config = row["config"]
    config_data = json.loads(raw_config) if isinstance(raw_config, str) else raw_config
    return Environment(
        id=row["id"],
        name=row["name"],
        config=EnvironmentConfig.model_validate(config_data),
        created_at=row["created_at"],
        archived_at=row["archived_at"],
    )


async def insert_environment(
    conn: asyncpg.Connection[Any],
    *,
    name: str,
    config: EnvironmentConfig | None = None,
) -> Environment:
    new_id = make_id(ENVIRONMENT)
    config_json = json.dumps((config or EnvironmentConfig()).model_dump(exclude_none=True))
    try:
        row = await conn.fetchrow(
            "INSERT INTO environments (id, name, config) VALUES ($1, $2, $3::jsonb) RETURNING *",
            new_id,
            name,
            config_json,
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
        version=row["version"],
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


def _row_to_agent_version(row: asyncpg.Record) -> AgentVersion:
    raw_tools = row["tools"]
    tools_data = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
    return AgentVersion(
        agent_id=row["agent_id"],
        version=row["version"],
        model=row["model"],
        system=row["system"],
        tools=[ToolSpec.model_validate(t) for t in tools_data],
        credential_id=row["credential_id"],
        window_min=row["window_min"],
        window_max=row["window_max"],
        created_at=row["created_at"],
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
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO agents (
                    id, name, model, system, tools, credential_id,
                    description, metadata, window_min, window_max, version
                )
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8::jsonb, $9, $10, 1)
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
            assert row is not None
            # Snapshot version 1 into agent_versions.
            await conn.execute(
                """
                INSERT INTO agent_versions (
                    agent_id, version, model, system, tools,
                    credential_id, window_min, window_max
                )
                VALUES ($1, 1, $2, $3, $4::jsonb, $5, $6, $7)
                """,
                new_id,
                model,
                system,
                tools_json,
                credential_id,
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


async def update_agent(
    conn: asyncpg.Connection[Any],
    agent_id: str,
    *,
    expected_version: int,
    name: str | None = None,
    model: str | None = None,
    system: str | None = None,
    tools: list[ToolSpec] | None = None,
    credential_id: str | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    window_min: int | None = None,
    window_max: int | None = None,
) -> Agent:
    """Update an agent, creating a new version.

    Requires ``expected_version`` to match the current version (optimistic
    concurrency). Omitted fields are preserved. If nothing changed, the
    existing version is returned without creating a new one (no-op).
    """
    current = await get_agent(conn, agent_id)
    if current.version != expected_version:
        raise ConflictError(
            f"version mismatch: expected {expected_version}, current is {current.version}",
            detail={"expected": expected_version, "current": current.version},
        )
    if current.archived_at is not None:
        raise ConflictError(f"agent {agent_id} is archived", detail={"id": agent_id})

    # Resolve final values (omitted = preserve current).
    new_name = name if name is not None else current.name
    new_model = model if model is not None else current.model
    new_system = system if system is not None else current.system
    new_tools = tools if tools is not None else current.tools
    new_cred = credential_id if credential_id is not None else current.credential_id
    new_desc = description if description is not None else current.description
    new_meta = metadata if metadata is not None else current.metadata
    new_wmin = window_min if window_min is not None else current.window_min
    new_wmax = window_max if window_max is not None else current.window_max

    # No-op detection.
    if (
        new_name == current.name
        and new_model == current.model
        and new_system == current.system
        and new_tools == current.tools
        and new_cred == current.credential_id
        and new_desc == current.description
        and new_meta == current.metadata
        and new_wmin == current.window_min
        and new_wmax == current.window_max
    ):
        return current

    new_version = current.version + 1
    tools_json = json.dumps([t.model_dump() for t in new_tools])
    meta_json = json.dumps(new_meta)

    async with conn.transaction():
        row = await conn.fetchrow(
            """
            UPDATE agents
               SET version = $2, name = $3, model = $4, system = $5,
                   tools = $6::jsonb, credential_id = $7, description = $8,
                   metadata = $9::jsonb, window_min = $10, window_max = $11,
                   updated_at = now()
             WHERE id = $1
            RETURNING *
            """,
            agent_id,
            new_version,
            new_name,
            new_model,
            new_system,
            tools_json,
            new_cred,
            new_desc,
            meta_json,
            new_wmin,
            new_wmax,
        )
        assert row is not None
        await conn.execute(
            """
            INSERT INTO agent_versions (
                agent_id, version, model, system, tools,
                credential_id, window_min, window_max
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)
            """,
            agent_id,
            new_version,
            new_model,
            new_system,
            tools_json,
            new_cred,
            new_wmin,
            new_wmax,
        )
    return _row_to_agent(row)


async def get_agent_version(
    conn: asyncpg.Connection[Any], agent_id: str, version: int
) -> AgentVersion:
    row = await conn.fetchrow(
        "SELECT * FROM agent_versions WHERE agent_id = $1 AND version = $2",
        agent_id,
        version,
    )
    if row is None:
        raise NotFoundError(
            f"agent {agent_id} version {version} not found",
            detail={"agent_id": agent_id, "version": version},
        )
    return _row_to_agent_version(row)


async def list_agent_versions(
    conn: asyncpg.Connection[Any],
    agent_id: str,
    *,
    limit: int = 50,
    after: int | None = None,
) -> list[AgentVersion]:
    """List versions in descending order (newest first)."""
    if after is None:
        rows = await conn.fetch(
            "SELECT * FROM agent_versions WHERE agent_id = $1 ORDER BY version DESC LIMIT $2",
            agent_id,
            limit,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM agent_versions WHERE agent_id = $1 AND version < $2 "
            "ORDER BY version DESC LIMIT $3",
            agent_id,
            after,
            limit,
        )
    return [_row_to_agent_version(r) for r in rows]


# ─── sessions ─────────────────────────────────────────────────────────────────


def _row_to_session(row: asyncpg.Record) -> Session:
    raw_metadata = row["metadata"]
    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
    raw_stop = row["stop_reason"]
    stop_reason = json.loads(raw_stop) if isinstance(raw_stop, str) else raw_stop
    return Session(
        id=row["id"],
        agent_id=row["agent_id"],
        environment_id=row["environment_id"],
        agent_version=row["agent_version"],
        title=row["title"],
        metadata=metadata,
        status=row["status"],
        stop_reason=stop_reason,
        last_event_seq=row["last_event_seq"],
        usage=SessionUsage(
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cache_read_input_tokens=row["cache_read_input_tokens"],
            cache_creation_input_tokens=row["cache_creation_input_tokens"],
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def insert_session(
    conn: asyncpg.Connection[Any],
    *,
    agent_id: str,
    environment_id: str,
    agent_version: int | None,
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
    stop_reason: dict[str, Any] | None = None,
) -> None:
    stop_json = json.dumps(stop_reason) if stop_reason is not None else None
    await conn.execute(
        "UPDATE sessions SET status = $1, stop_reason = $2::jsonb, updated_at = now() WHERE id = $3",
        status,
        stop_json,
        session_id,
    )


async def increment_session_usage(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> None:
    """Atomically add token counts to a session's cumulative usage."""
    await conn.execute(
        "UPDATE sessions SET "
        "input_tokens = input_tokens + $2, "
        "output_tokens = output_tokens + $3, "
        "cache_read_input_tokens = cache_read_input_tokens + $4, "
        "cache_creation_input_tokens = cache_creation_input_tokens + $5 "
        "WHERE id = $1",
        session_id,
        input_tokens,
        output_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
    )


async def list_running_session_ids(conn: asyncpg.Connection[Any]) -> list[str]:
    """Return ids of sessions with ``status = 'running'``.

    Used by the sandbox orphan reaper at worker startup: any Docker
    container labelled ``aios.managed=true`` whose ``aios.session_id``
    is NOT in this list is a corpse from a dead worker and gets
    force-removed.
    """
    rows = await conn.fetch(
        """
        SELECT id
          FROM sessions
         WHERE status = 'running'
           AND archived_at IS NULL
        """
    )
    return [str(r["id"]) for r in rows]


_UNSET: Any = object()


async def update_session(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    agent_id: str | None = None,
    agent_version: int | None = _UNSET,
    title: str | None = _UNSET,
    metadata: dict[str, Any] | None = None,
) -> Session:
    """Partial update of a session. Omitted fields are preserved.

    ``agent_version`` and ``title`` use a sentinel to distinguish "not
    provided" from "explicitly set to None".
    """
    sets: list[str] = []
    args: list[Any] = [session_id]  # $1 = session_id

    if agent_id is not None:
        args.append(agent_id)
        sets.append(f"agent_id = ${len(args)}")
    if agent_version is not _UNSET:
        args.append(agent_version)
        sets.append(f"agent_version = ${len(args)}")
    if title is not _UNSET:
        args.append(title)
        sets.append(f"title = ${len(args)}")
    if metadata is not None:
        args.append(json.dumps(metadata))
        sets.append(f"metadata = ${len(args)}::jsonb")

    if not sets:
        return await get_session(conn, session_id)

    sets.append("updated_at = now()")
    sql = f"UPDATE sessions SET {', '.join(sets)} WHERE id = $1 RETURNING *"
    row = await conn.fetchrow(sql, *args)
    if row is None:
        raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})
    return _row_to_session(row)


async def archive_session(conn: asyncpg.Connection[Any], session_id: str) -> Session:
    row = await conn.fetchrow(
        "UPDATE sessions SET archived_at = now(), updated_at = now() "
        "WHERE id = $1 AND archived_at IS NULL RETURNING *",
        session_id,
    )
    if row is None:
        raise NotFoundError(
            f"session {session_id} not found or already archived",
            detail={"id": session_id},
        )
    return _row_to_session(row)


async def delete_session(conn: asyncpg.Connection[Any], session_id: str) -> None:
    async with conn.transaction():
        row = await conn.fetchrow(
            "SELECT status FROM sessions WHERE id = $1",
            session_id,
        )
        if row is None:
            raise NotFoundError(
                f"session {session_id} not found",
                detail={"id": session_id},
            )
        if row["status"] == "running":
            raise ConflictError(
                f"session {session_id} is running and cannot be deleted",
                detail={"id": session_id},
            )
        await conn.execute("DELETE FROM events WHERE session_id = $1", session_id)
        await conn.execute("DELETE FROM sessions WHERE id = $1", session_id)


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
    # before the row is committed. Use pg_notify (the function form) rather
    # than the literal NOTIFY statement, because Postgres case-folds unquoted
    # identifiers in NOTIFY <chan> — and our prefixed-ULID session ids
    # contain uppercase letters. asyncpg's add_listener quotes the channel,
    # preserving case, so the two would never match. pg_notify(text, text)
    # treats the channel as a string literal and preserves it byte-for-byte.
    await conn.execute("SELECT pg_notify($1, $2)", f"events_{session_id}", new_id)
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
