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
from aios.ids import AGENT, ENVIRONMENT, EVENT, SESSION, SKILL, VAULT, VAULT_CREDENTIAL, make_id
from aios.models.agents import Agent, AgentVersion, McpServerSpec, ToolSpec
from aios.models.environments import Environment, EnvironmentConfig
from aios.models.events import Event, EventKind
from aios.models.sessions import Session, SessionStatus, SessionUsage
from aios.models.skills import AgentSkillRef, Skill, SkillVersion
from aios.models.vaults import Vault, VaultCredential

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


async def update_environment(
    conn: asyncpg.Connection[Any],
    env_id: str,
    *,
    name: str | None = None,
    config: EnvironmentConfig | None = None,
) -> Environment:
    """Update an environment. Omitted fields are preserved."""
    current = await get_environment(conn, env_id)
    if current.archived_at is not None:
        raise ConflictError(f"environment {env_id} is archived", detail={"id": env_id})

    new_name = name if name is not None else current.name
    new_config = config if config is not None else current.config

    # No-op detection.
    if new_name == current.name and new_config == current.config:
        return current

    config_json = json.dumps(new_config.model_dump(exclude_none=True))
    try:
        row = await conn.fetchrow(
            "UPDATE environments SET name = $2, config = $3::jsonb WHERE id = $1 RETURNING *",
            env_id,
            new_name,
            config_json,
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"an environment named {new_name!r} already exists",
            detail={"name": new_name},
        ) from exc
    assert row is not None
    return _row_to_environment(row)


async def get_environment_config_for_session(
    conn: asyncpg.Connection[Any], session_id: str
) -> EnvironmentConfig | None:
    """Return the environment config for a session, or None if not found."""
    row = await conn.fetchrow(
        """
        SELECT e.config FROM environments e
        JOIN sessions s ON s.environment_id = e.id
        WHERE s.id = $1
        """,
        session_id,
    )
    if row is None:
        return None
    raw_config = row["config"]
    config_data = json.loads(raw_config) if isinstance(raw_config, str) else raw_config
    return EnvironmentConfig.model_validate(config_data)


# ─── agents ───────────────────────────────────────────────────────────────────


def _parse_jsonb(raw: Any) -> Any:
    return json.loads(raw) if isinstance(raw, str) else raw


def _row_to_agent(row: asyncpg.Record) -> Agent:
    tools_data = _parse_jsonb(row["tools"])
    skills_data = _parse_jsonb(row["skills"])
    mcp_data = _parse_jsonb(row.get("mcp_servers", []))
    metadata = _parse_jsonb(row["metadata"])
    return Agent(
        id=row["id"],
        version=row["version"],
        name=row["name"],
        model=row["model"],
        system=row["system"],
        tools=[ToolSpec.model_validate(t) for t in tools_data],
        skills=[AgentSkillRef.model_validate(s) for s in skills_data],
        mcp_servers=[McpServerSpec.model_validate(s) for s in (mcp_data or [])],
        description=row["description"],
        metadata=metadata,
        window_min=row["window_min"],
        window_max=row["window_max"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


def _row_to_agent_version(row: asyncpg.Record) -> AgentVersion:
    tools_data = _parse_jsonb(row["tools"])
    skills_data = _parse_jsonb(row["skills"])
    mcp_data = _parse_jsonb(row.get("mcp_servers", []))
    return AgentVersion(
        agent_id=row["agent_id"],
        version=row["version"],
        model=row["model"],
        system=row["system"],
        tools=[ToolSpec.model_validate(t) for t in tools_data],
        skills=[AgentSkillRef.model_validate(s) for s in skills_data],
        mcp_servers=[McpServerSpec.model_validate(s) for s in (mcp_data or [])],
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
    skills_json: str = "[]",
    mcp_servers: list[McpServerSpec],
    description: str | None,
    metadata: dict[str, Any],
    window_min: int,
    window_max: int,
) -> Agent:
    new_id = make_id(AGENT)
    tools_json = json.dumps([t.model_dump() for t in tools])
    mcp_json = json.dumps([s.model_dump() for s in mcp_servers])
    metadata_json = json.dumps(metadata)
    try:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO agents (
                    id, name, model, system, tools, skills, mcp_servers,
                    description, metadata, window_min, window_max, version
                )
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb,
                        $8, $9::jsonb, $10, $11, 1)
                RETURNING *
                """,
                new_id,
                name,
                model,
                system,
                tools_json,
                skills_json,
                mcp_json,
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
                    agent_id, version, model, system, tools, skills, mcp_servers,
                    window_min, window_max
                )
                VALUES ($1, 1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7, $8)
                """,
                new_id,
                model,
                system,
                tools_json,
                skills_json,
                mcp_json,
                window_min,
                window_max,
            )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"an agent named {name!r} already exists",
            detail={"name": name},
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
    skills_json: str | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    window_min: int | None = None,
    window_max: int | None = None,
) -> Agent:
    """Update an agent, creating a new version.

    Requires ``expected_version`` to match the current version (optimistic
    concurrency). Omitted fields are preserved. If nothing changed, the
    existing version is returned without creating a new one (no-op).

    ``skills_json`` is a pre-serialized JSON string (resolved by the
    service layer which snapshots concrete versions).
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
    cur_skills_json = json.dumps([s.model_dump() for s in current.skills])
    new_skills_json = skills_json if skills_json is not None else cur_skills_json
    new_mcp = mcp_servers if mcp_servers is not None else current.mcp_servers
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
        and new_skills_json == cur_skills_json
        and new_mcp == current.mcp_servers
        and new_desc == current.description
        and new_meta == current.metadata
        and new_wmin == current.window_min
        and new_wmax == current.window_max
    ):
        return current

    new_version = current.version + 1
    tools_json = json.dumps([t.model_dump() for t in new_tools])
    mcp_json = json.dumps([s.model_dump() for s in new_mcp])
    meta_json = json.dumps(new_meta)

    async with conn.transaction():
        row = await conn.fetchrow(
            """
            UPDATE agents
               SET version = $2, name = $3, model = $4, system = $5,
                   tools = $6::jsonb, skills = $7::jsonb, mcp_servers = $8::jsonb,
                   description = $9, metadata = $10::jsonb,
                   window_min = $11, window_max = $12,
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
            new_skills_json,
            mcp_json,
            new_desc,
            meta_json,
            new_wmin,
            new_wmax,
        )
        assert row is not None
        await conn.execute(
            """
            INSERT INTO agent_versions (
                agent_id, version, model, system, tools, skills, mcp_servers,
                window_min, window_max
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb, $8, $9)
            """,
            agent_id,
            new_version,
            new_model,
            new_system,
            tools_json,
            new_skills_json,
            mcp_json,
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
        await conn.execute("DELETE FROM session_vaults WHERE session_id = $1", session_id)
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
        cumulative_tokens=row["cumulative_tokens"],
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

    For message events, computes and stores ``cumulative_tokens`` — the
    running total of approximate token counts through this event.  The
    previous cumulative value is fetched inside the same transaction (under
    the session row lock), so there is no race with concurrent appenders.
    """
    from aios.harness.tokens import approx_tokens

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

        # Compute cumulative_tokens for message events.
        cum_tokens: int | None = None
        if kind == "message":
            prev = await conn.fetchval(
                "SELECT cumulative_tokens FROM events "
                "WHERE session_id = $1 AND kind = 'message' "
                "AND cumulative_tokens IS NOT NULL "
                "ORDER BY seq DESC LIMIT 1",
                session_id,
            )
            cum_tokens = (prev or 0) + approx_tokens(data)

        row = await conn.fetchrow(
            "INSERT INTO events (id, session_id, seq, kind, data, cumulative_tokens) "
            "VALUES ($1, $2, $3, $4, $5::jsonb, $6) RETURNING *",
            new_id,
            session_id,
            seq,
            kind,
            data_json,
            cum_tokens,
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

    Used by callers that need the full unwindowed log (e.g.
    ``confirm_tool_deny`` searching for a tool_call_id).
    """
    rows = await conn.fetch(
        "SELECT * FROM events WHERE session_id = $1 AND kind = 'message' ORDER BY seq ASC",
        session_id,
    )
    return [_row_to_event(r) for r in rows]


async def read_windowed_events(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    window_min: int,
    window_max: int,
) -> list[Event]:
    """Read message events for the session's trailing context window.

    Uses the ``cumulative_tokens`` column to compute the chunked-window
    snap boundary (same math as :func:`~aios.harness.window.select_window`)
    and loads only the events past that boundary.

    Falls back to :func:`read_message_events` when cumulative data is
    not available (pre-backfill sessions or rolling deploys).
    """
    # Index seek: total cumulative tokens from the latest message event.
    total = await conn.fetchval(
        "SELECT cumulative_tokens FROM events "
        "WHERE session_id = $1 AND kind = 'message' "
        "AND cumulative_tokens IS NOT NULL "
        "ORDER BY seq DESC LIMIT 1",
        session_id,
    )

    # Fallback: no cumulative data yet — load everything.
    if total is None:
        return await read_message_events(conn, session_id)

    # Everything fits in the window — no need to drop.
    if total <= window_max:
        return await read_message_events(conn, session_id)

    # Snap math (mirrors window.py:select_window lines 92-95).
    overshoot = total - window_max
    chunk = window_max - window_min
    snaps = (overshoot + chunk - 1) // chunk  # ceil division
    tokens_to_drop = snaps * chunk

    # Bounded range scan: only events past the boundary.
    rows = await conn.fetch(
        "SELECT * FROM events "
        "WHERE session_id = $1 AND kind = 'message' "
        "AND cumulative_tokens > $2 "
        "ORDER BY seq ASC",
        session_id,
        tokens_to_drop,
    )
    return [_row_to_event(r) for r in rows]


# ─── vaults ─────────────────────────────────────────────────────────────────


def _row_to_vault(row: asyncpg.Record) -> Vault:
    raw_metadata = row["metadata"]
    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
    return Vault(
        id=row["id"],
        display_name=row["display_name"],
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def insert_vault(
    conn: asyncpg.Connection[Any],
    *,
    display_name: str,
    metadata: dict[str, Any],
) -> Vault:
    new_id = make_id(VAULT)
    metadata_json = json.dumps(metadata)
    row = await conn.fetchrow(
        """
        INSERT INTO vaults (id, display_name, metadata)
        VALUES ($1, $2, $3::jsonb)
        RETURNING *
        """,
        new_id,
        display_name,
        metadata_json,
    )
    assert row is not None
    return _row_to_vault(row)


async def get_vault(conn: asyncpg.Connection[Any], vault_id: str) -> Vault:
    row = await conn.fetchrow("SELECT * FROM vaults WHERE id = $1", vault_id)
    if row is None:
        raise NotFoundError(f"vault {vault_id} not found", detail={"id": vault_id})
    return _row_to_vault(row)


async def list_vaults(
    conn: asyncpg.Connection[Any], *, limit: int = 50, after: str | None = None
) -> list[Vault]:
    if after is None:
        rows = await conn.fetch(
            "SELECT * FROM vaults WHERE archived_at IS NULL ORDER BY id DESC LIMIT $1",
            limit,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM vaults WHERE archived_at IS NULL AND id < $1 ORDER BY id DESC LIMIT $2",
            after,
            limit,
        )
    return [_row_to_vault(r) for r in rows]


async def update_vault(
    conn: asyncpg.Connection[Any],
    vault_id: str,
    *,
    display_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Vault:
    sets: list[str] = []
    args: list[Any] = [vault_id]
    if display_name is not None:
        args.append(display_name)
        sets.append(f"display_name = ${len(args)}")
    if metadata is not None:
        args.append(json.dumps(metadata))
        sets.append(f"metadata = ${len(args)}::jsonb")
    if not sets:
        return await get_vault(conn, vault_id)
    sets.append("updated_at = now()")
    sql = f"UPDATE vaults SET {', '.join(sets)} WHERE id = $1 RETURNING *"
    row = await conn.fetchrow(sql, *args)
    if row is None:
        raise NotFoundError(f"vault {vault_id} not found", detail={"id": vault_id})
    return _row_to_vault(row)


async def archive_vault(conn: asyncpg.Connection[Any], vault_id: str) -> Vault:
    row = await conn.fetchrow(
        "UPDATE vaults SET archived_at = now(), updated_at = now() "
        "WHERE id = $1 AND archived_at IS NULL RETURNING *",
        vault_id,
    )
    if row is None:
        raise NotFoundError(
            f"vault {vault_id} not found or already archived",
            detail={"id": vault_id},
        )
    return _row_to_vault(row)


async def delete_vault(conn: asyncpg.Connection[Any], vault_id: str) -> None:
    async with conn.transaction():
        await conn.execute("DELETE FROM vault_credentials WHERE vault_id = $1", vault_id)
        result = await conn.execute("DELETE FROM vaults WHERE id = $1", vault_id)
    if result == "DELETE 0":
        raise NotFoundError(f"vault {vault_id} not found", detail={"id": vault_id})


# ─── vault credentials ──────────────────────────────────────────────────────


def _row_to_vault_credential(row: asyncpg.Record) -> VaultCredential:
    raw_metadata = row["metadata"]
    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
    return VaultCredential(
        id=row["id"],
        vault_id=row["vault_id"],
        display_name=row["display_name"],
        mcp_server_url=row["mcp_server_url"],
        auth_type=row["auth_type"],
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def insert_vault_credential(
    conn: asyncpg.Connection[Any],
    *,
    vault_id: str,
    display_name: str | None,
    mcp_server_url: str,
    auth_type: str,
    blob: EncryptedBlob,
    metadata: dict[str, Any],
) -> VaultCredential:
    new_id = make_id(VAULT_CREDENTIAL)
    metadata_json = json.dumps(metadata)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO vault_credentials (
                id, vault_id, display_name, mcp_server_url,
                auth_type, ciphertext, nonce, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            RETURNING *
            """,
            new_id,
            vault_id,
            display_name,
            mcp_server_url,
            auth_type,
            blob.ciphertext,
            blob.nonce,
            metadata_json,
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"an active credential for {mcp_server_url!r} already exists in this vault",
            detail={"mcp_server_url": mcp_server_url, "vault_id": vault_id},
        ) from exc
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            f"vault {vault_id} not found",
            detail={"vault_id": vault_id},
        ) from exc
    assert row is not None
    return _row_to_vault_credential(row)


async def get_vault_credential(
    conn: asyncpg.Connection[Any], vault_id: str, credential_id: str
) -> VaultCredential:
    row = await conn.fetchrow(
        "SELECT * FROM vault_credentials WHERE id = $1 AND vault_id = $2",
        credential_id,
        vault_id,
    )
    if row is None:
        raise NotFoundError(
            f"credential {credential_id} not found in vault {vault_id}",
            detail={"id": credential_id, "vault_id": vault_id},
        )
    return _row_to_vault_credential(row)


async def get_vault_credential_blob(
    conn: asyncpg.Connection[Any], vault_id: str, credential_id: str
) -> EncryptedBlob:
    row = await conn.fetchrow(
        "SELECT ciphertext, nonce FROM vault_credentials "
        "WHERE id = $1 AND vault_id = $2 AND archived_at IS NULL",
        credential_id,
        vault_id,
    )
    if row is None:
        raise NotFoundError(
            f"credential {credential_id} not found or archived",
            detail={"id": credential_id, "vault_id": vault_id},
        )
    return EncryptedBlob(ciphertext=bytes(row["ciphertext"]), nonce=bytes(row["nonce"]))


async def list_vault_credentials(
    conn: asyncpg.Connection[Any],
    vault_id: str,
    *,
    limit: int = 50,
    after: str | None = None,
) -> list[VaultCredential]:
    if after is None:
        rows = await conn.fetch(
            "SELECT * FROM vault_credentials "
            "WHERE vault_id = $1 AND archived_at IS NULL ORDER BY id DESC LIMIT $2",
            vault_id,
            limit,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM vault_credentials "
            "WHERE vault_id = $1 AND archived_at IS NULL AND id < $2 "
            "ORDER BY id DESC LIMIT $3",
            vault_id,
            after,
            limit,
        )
    return [_row_to_vault_credential(r) for r in rows]


async def update_vault_credential(
    conn: asyncpg.Connection[Any],
    vault_id: str,
    credential_id: str,
    *,
    display_name: str | None = _UNSET,
    blob: EncryptedBlob | None = None,
    metadata: dict[str, Any] | None = None,
) -> VaultCredential:
    sets: list[str] = []
    args: list[Any] = [credential_id, vault_id]
    if display_name is not _UNSET:
        args.append(display_name)
        sets.append(f"display_name = ${len(args)}")
    if blob is not None:
        args.append(blob.ciphertext)
        sets.append(f"ciphertext = ${len(args)}")
        args.append(blob.nonce)
        sets.append(f"nonce = ${len(args)}")
    if metadata is not None:
        args.append(json.dumps(metadata))
        sets.append(f"metadata = ${len(args)}::jsonb")
    if not sets:
        return await get_vault_credential(conn, vault_id, credential_id)
    sets.append("updated_at = now()")
    sql = (
        f"UPDATE vault_credentials SET {', '.join(sets)} "
        f"WHERE id = $1 AND vault_id = $2 RETURNING *"
    )
    row = await conn.fetchrow(sql, *args)
    if row is None:
        raise NotFoundError(
            f"credential {credential_id} not found in vault {vault_id}",
            detail={"id": credential_id, "vault_id": vault_id},
        )
    return _row_to_vault_credential(row)


async def archive_vault_credential(
    conn: asyncpg.Connection[Any], vault_id: str, credential_id: str
) -> VaultCredential:
    row = await conn.fetchrow(
        "UPDATE vault_credentials SET archived_at = now(), updated_at = now() "
        "WHERE id = $1 AND vault_id = $2 AND archived_at IS NULL RETURNING *",
        credential_id,
        vault_id,
    )
    if row is None:
        raise NotFoundError(
            f"credential {credential_id} not found or already archived",
            detail={"id": credential_id, "vault_id": vault_id},
        )
    return _row_to_vault_credential(row)


async def delete_vault_credential(
    conn: asyncpg.Connection[Any], vault_id: str, credential_id: str
) -> None:
    result = await conn.execute(
        "DELETE FROM vault_credentials WHERE id = $1 AND vault_id = $2",
        credential_id,
        vault_id,
    )
    if result == "DELETE 0":
        raise NotFoundError(
            f"credential {credential_id} not found in vault {vault_id}",
            detail={"id": credential_id, "vault_id": vault_id},
        )


async def count_active_vault_credentials(conn: asyncpg.Connection[Any], vault_id: str) -> int:
    row = await conn.fetchrow(
        "SELECT count(*) AS cnt FROM vault_credentials WHERE vault_id = $1 AND archived_at IS NULL",
        vault_id,
    )
    assert row is not None
    result: int = row["cnt"]
    return result


# ─── session-vault binding ──────────────────────────────────────────────────


async def set_session_vaults(
    conn: asyncpg.Connection[Any], session_id: str, vault_ids: list[str]
) -> None:
    """Replace the session's vault bindings. Order is preserved via rank."""
    async with conn.transaction():
        await conn.execute("DELETE FROM session_vaults WHERE session_id = $1", session_id)
        for rank, vault_id in enumerate(vault_ids):
            try:
                await conn.execute(
                    "INSERT INTO session_vaults (session_id, vault_id, rank) VALUES ($1, $2, $3)",
                    session_id,
                    vault_id,
                    rank,
                )
            except asyncpg.ForeignKeyViolationError as exc:
                raise NotFoundError(
                    f"vault {vault_id} not found",
                    detail={"vault_id": vault_id},
                ) from exc


async def get_session_vault_ids(conn: asyncpg.Connection[Any], session_id: str) -> list[str]:
    rows = await conn.fetch(
        "SELECT vault_id FROM session_vaults WHERE session_id = $1 ORDER BY rank",
        session_id,
    )
    return [str(r["vault_id"]) for r in rows]


async def batch_get_session_vault_ids(
    conn: asyncpg.Connection[Any], session_ids: list[str]
) -> dict[str, list[str]]:
    """Batch-fetch vault_ids for multiple sessions. Returns a dict keyed by session_id."""
    if not session_ids:
        return {}
    rows = await conn.fetch(
        "SELECT session_id, vault_id FROM session_vaults "
        "WHERE session_id = ANY($1) ORDER BY session_id, rank",
        session_ids,
    )
    result: dict[str, list[str]] = {sid: [] for sid in session_ids}
    for r in rows:
        result[str(r["session_id"])].append(str(r["vault_id"]))
    return result


# ─── MCP credential resolution ───────────────────────────────────────────────


async def resolve_mcp_credential(
    conn: asyncpg.Connection[Any],
    session_id: str,
    mcp_server_url: str,
) -> tuple[EncryptedBlob, str] | None:
    """Find the first matching MCP credential across a session's bound vaults.

    Joins ``session_vaults`` (rank-ordered) with ``vault_credentials``
    filtered by ``mcp_server_url``. Returns ``(EncryptedBlob, auth_type)``
    for the first match, or ``None`` if no credential exists.
    """
    row = await conn.fetchrow(
        """
        SELECT vc.ciphertext, vc.nonce, vc.auth_type
          FROM session_vaults sv
          JOIN vault_credentials vc ON vc.vault_id = sv.vault_id
         WHERE sv.session_id = $1
           AND vc.mcp_server_url = $2
           AND vc.archived_at IS NULL
         ORDER BY sv.rank
         LIMIT 1
        """,
        session_id,
        mcp_server_url,
    )
    if row is None:
        return None
    return EncryptedBlob(ciphertext=row["ciphertext"], nonce=row["nonce"]), str(row["auth_type"])


# ─── skills ──────────────────────────────────────────────────────────────────


def _row_to_skill(row: asyncpg.Record) -> Skill:
    return Skill(
        id=row["id"],
        display_title=row["display_title"],
        latest_version=row["latest_version"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


def _row_to_skill_version(row: asyncpg.Record) -> SkillVersion:
    files_data = _parse_jsonb(row["files"])
    return SkillVersion(
        skill_id=row["skill_id"],
        version=row["version"],
        directory=row["directory"],
        name=row["name"],
        description=row["description"],
        files=files_data,
        created_at=row["created_at"],
    )


async def insert_skill(
    conn: asyncpg.Connection[Any],
    *,
    display_title: str,
    directory: str,
    name: str,
    description: str,
    files: dict[str, str],
) -> tuple[Skill, SkillVersion]:
    """Create a skill and its first version atomically.

    Returns ``(skill, version_1)``.
    """
    new_id = make_id(SKILL)
    files_json = json.dumps(files)
    async with conn.transaction():
        skill_row = await conn.fetchrow(
            """
            INSERT INTO skills (id, display_title, latest_version)
            VALUES ($1, $2, 1)
            RETURNING *
            """,
            new_id,
            display_title,
        )
        assert skill_row is not None
        ver_row = await conn.fetchrow(
            """
            INSERT INTO skill_versions (skill_id, version, directory, name, description, files)
            VALUES ($1, 1, $2, $3, $4, $5::jsonb)
            RETURNING *
            """,
            new_id,
            directory,
            name,
            description,
            files_json,
        )
        assert ver_row is not None
    return _row_to_skill(skill_row), _row_to_skill_version(ver_row)


async def get_skill(conn: asyncpg.Connection[Any], skill_id: str) -> Skill:
    row = await conn.fetchrow("SELECT * FROM skills WHERE id = $1", skill_id)
    if row is None:
        raise NotFoundError(f"skill {skill_id} not found", detail={"id": skill_id})
    return _row_to_skill(row)


async def list_skills(
    conn: asyncpg.Connection[Any], *, limit: int = 50, after: str | None = None
) -> list[Skill]:
    if after is None:
        rows = await conn.fetch(
            "SELECT * FROM skills WHERE archived_at IS NULL ORDER BY id DESC LIMIT $1",
            limit,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM skills WHERE archived_at IS NULL AND id < $1 ORDER BY id DESC LIMIT $2",
            after,
            limit,
        )
    return [_row_to_skill(r) for r in rows]


async def archive_skill(conn: asyncpg.Connection[Any], skill_id: str) -> None:
    result = await conn.execute(
        "UPDATE skills SET archived_at = now() WHERE id = $1 AND archived_at IS NULL",
        skill_id,
    )
    if result == "UPDATE 0":
        raise NotFoundError(f"skill {skill_id} not found or already archived")


async def insert_skill_version(
    conn: asyncpg.Connection[Any],
    *,
    skill_id: str,
    directory: str,
    name: str,
    description: str,
    files: dict[str, str],
) -> SkillVersion:
    """Create a new immutable version for an existing skill.

    Locks the skills row, increments ``latest_version``, inserts the
    version, and updates the head row's ``updated_at``.
    """
    files_json = json.dumps(files)
    async with conn.transaction():
        head = await conn.fetchrow(
            "SELECT latest_version FROM skills WHERE id = $1 FOR UPDATE",
            skill_id,
        )
        if head is None:
            raise NotFoundError(f"skill {skill_id} not found", detail={"id": skill_id})
        new_ver = head["latest_version"] + 1
        ver_row = await conn.fetchrow(
            """
            INSERT INTO skill_versions (skill_id, version, directory, name, description, files)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            RETURNING *
            """,
            skill_id,
            new_ver,
            directory,
            name,
            description,
            files_json,
        )
        assert ver_row is not None
        await conn.execute(
            "UPDATE skills SET latest_version = $2, updated_at = now() WHERE id = $1",
            skill_id,
            new_ver,
        )
    return _row_to_skill_version(ver_row)


async def get_skill_version(
    conn: asyncpg.Connection[Any], skill_id: str, version: int
) -> SkillVersion:
    row = await conn.fetchrow(
        "SELECT * FROM skill_versions WHERE skill_id = $1 AND version = $2",
        skill_id,
        version,
    )
    if row is None:
        raise NotFoundError(
            f"skill {skill_id} version {version} not found",
            detail={"skill_id": skill_id, "version": version},
        )
    return _row_to_skill_version(row)


async def get_latest_skill_version(conn: asyncpg.Connection[Any], skill_id: str) -> SkillVersion:
    """Get the latest version of a skill by joining with the head row."""
    row = await conn.fetchrow(
        """
        SELECT sv.* FROM skill_versions sv
        JOIN skills s ON s.id = sv.skill_id AND sv.version = s.latest_version
        WHERE sv.skill_id = $1
        """,
        skill_id,
    )
    if row is None:
        raise NotFoundError(f"skill {skill_id} has no versions", detail={"skill_id": skill_id})
    return _row_to_skill_version(row)


async def list_skill_versions(
    conn: asyncpg.Connection[Any],
    skill_id: str,
    *,
    limit: int = 50,
    after: int | None = None,
) -> list[SkillVersion]:
    """List versions in descending order (newest first)."""
    if after is None:
        rows = await conn.fetch(
            "SELECT * FROM skill_versions WHERE skill_id = $1 ORDER BY version DESC LIMIT $2",
            skill_id,
            limit,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM skill_versions WHERE skill_id = $1 AND version < $2 "
            "ORDER BY version DESC LIMIT $3",
            skill_id,
            after,
            limit,
        )
    return [_row_to_skill_version(r) for r in rows]


async def resolve_skill_refs(
    conn: asyncpg.Connection[Any],
    refs: list[AgentSkillRef],
) -> list[SkillVersion]:
    """Resolve a list of skill references to concrete versions.

    For each ref, if ``version`` is ``None``, resolves to the latest
    version. Otherwise fetches the pinned version. Returns versions in
    the same order as the input refs.
    """
    results: list[SkillVersion] = []
    for ref in refs:
        if ref.version is None:
            sv = await get_latest_skill_version(conn, ref.skill_id)
        else:
            sv = await get_skill_version(conn, ref.skill_id, ref.version)
        results.append(sv)
    return results
