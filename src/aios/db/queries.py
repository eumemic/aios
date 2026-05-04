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
import math
import time
from types import EllipsisType
from typing import Any, NoReturn

import asyncpg

from aios.crypto.vault import EncryptedBlob
from aios.errors import (
    ConflictError,
    MemoryPathConflictError,
    MemoryPreconditionFailedError,
    MemoryStoreArchivedError,
    NotFoundError,
)
from aios.ids import (
    AGENT,
    CONNECTION,
    ENVIRONMENT,
    EVENT,
    MEMORY,
    MEMORY_STORE,
    MEMORY_VERSION,
    SESSION,
    SESSION_TEMPLATE,
    SKILL,
    VAULT,
    VAULT_CREDENTIAL,
    make_id,
)
from aios.models.agents import Agent, AgentVersion, McpServerSpec, ToolSpec
from aios.models.connections import Connection
from aios.models.environments import Environment, EnvironmentConfig
from aios.models.events import Event, EventKind
from aios.models.memory_stores import (
    Actor,
    Memory,
    MemoryPrefix,
    MemoryStore,
    MemoryStoreResource,
    MemoryStoreResourceEcho,
    MemoryVersion,
)
from aios.models.session_templates import SessionTemplate
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
    litellm_extra = _parse_jsonb(row["litellm_extra"])
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
        litellm_extra=litellm_extra or {},
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
    litellm_extra = _parse_jsonb(row["litellm_extra"])
    return AgentVersion(
        agent_id=row["agent_id"],
        version=row["version"],
        model=row["model"],
        system=row["system"],
        tools=[ToolSpec.model_validate(t) for t in tools_data],
        skills=[AgentSkillRef.model_validate(s) for s in skills_data],
        mcp_servers=[McpServerSpec.model_validate(s) for s in (mcp_data or [])],
        litellm_extra=litellm_extra or {},
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
    litellm_extra: dict[str, Any],
    window_min: int,
    window_max: int,
) -> Agent:
    new_id = make_id(AGENT)
    tools_json = json.dumps([t.model_dump() for t in tools])
    mcp_json = json.dumps([s.model_dump() for s in mcp_servers])
    metadata_json = json.dumps(metadata)
    extra_json = json.dumps(litellm_extra)
    try:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO agents (
                    id, name, model, system, tools, skills, mcp_servers,
                    description, metadata, litellm_extra,
                    window_min, window_max, version
                )
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb,
                        $8, $9::jsonb, $10::jsonb, $11, $12, 1)
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
                extra_json,
                window_min,
                window_max,
            )
            assert row is not None
            # Snapshot version 1 into agent_versions.
            await conn.execute(
                """
                INSERT INTO agent_versions (
                    agent_id, version, model, system, tools, skills, mcp_servers,
                    litellm_extra, window_min, window_max
                )
                VALUES ($1, 1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb,
                        $7::jsonb, $8, $9)
                """,
                new_id,
                model,
                system,
                tools_json,
                skills_json,
                mcp_json,
                extra_json,
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
    conn: asyncpg.Connection[Any],
    *,
    limit: int = 50,
    after: str | None = None,
    name: str | None = None,
) -> list[Agent]:
    where = ["archived_at IS NULL"]
    args: list[Any] = []
    if name is not None:
        args.append(name)
        where.append(f"name = ${len(args)}")
    if after is not None:
        args.append(after)
        where.append(f"id < ${len(args)}")
    args.append(limit)
    sql = f"SELECT * FROM agents WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT ${len(args)}"
    rows = await conn.fetch(sql, *args)
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
    litellm_extra: dict[str, Any] | None = None,
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
    new_extra = litellm_extra if litellm_extra is not None else current.litellm_extra
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
        and new_extra == current.litellm_extra
        and new_wmin == current.window_min
        and new_wmax == current.window_max
    ):
        return current

    new_version = current.version + 1
    tools_json = json.dumps([t.model_dump() for t in new_tools])
    mcp_json = json.dumps([s.model_dump() for s in new_mcp])
    meta_json = json.dumps(new_meta)
    extra_json = json.dumps(new_extra)

    async with conn.transaction():
        row = await conn.fetchrow(
            """
            UPDATE agents
               SET version = $2, name = $3, model = $4, system = $5,
                   tools = $6::jsonb, skills = $7::jsonb, mcp_servers = $8::jsonb,
                   description = $9, metadata = $10::jsonb,
                   litellm_extra = $11::jsonb,
                   window_min = $12, window_max = $13,
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
            extra_json,
            new_wmin,
            new_wmax,
        )
        assert row is not None
        await conn.execute(
            """
            INSERT INTO agent_versions (
                agent_id, version, model, system, tools, skills, mcp_servers,
                litellm_extra, window_min, window_max
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb,
                    $8::jsonb, $9, $10)
            """,
            agent_id,
            new_version,
            new_model,
            new_system,
            tools_json,
            new_skills_json,
            mcp_json,
            extra_json,
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
        focal_channel=row["focal_channel"],
    )


async def insert_session(
    conn: asyncpg.Connection[Any],
    *,
    agent_id: str,
    environment_id: str,
    agent_version: int | None,
    title: str | None,
    metadata: dict[str, Any],
    workspace_path: str | None = None,
    env: dict[str, str] | None = None,
    spawned_from_connection_id: str | None = None,
    focal_channel: str | None = None,
) -> Session:
    """Insert a fresh session row.

    ``workspace_path`` defaults to ``settings.workspace_root / session_id``.
    Caller sets up vault bindings via :func:`set_session_vaults` after.
    Raises :class:`NotFoundError` if either the agent or environment FK
    is unsatisfied.

    ``spawned_from_connection_id`` + ``focal_channel`` are written
    atomically with the row insert so the focal-locked invariant (see
    ``switch_channel``'s rejection of mutations on per_chat sessions)
    holds from creation.
    """
    from aios.config import get_settings

    new_id = make_id(SESSION)
    if workspace_path is None:
        workspace_path = str(get_settings().workspace_root / new_id)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO sessions (
                id, agent_id, environment_id, agent_version, title, metadata,
                status, workspace_volume_path, env,
                spawned_from_connection_id, focal_channel
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, 'idle', $7, $8::jsonb, $9, $10)
            RETURNING *
            """,
            new_id,
            agent_id,
            environment_id,
            agent_version,
            title,
            json.dumps(metadata),
            workspace_path,
            json.dumps(env or {}),
            spawned_from_connection_id,
            focal_channel,
        )
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            "agent, environment, or connection not found",
            detail={
                "agent_id": agent_id,
                "environment_id": environment_id,
                "spawned_from_connection_id": spawned_from_connection_id,
            },
        ) from exc
    assert row is not None
    return _row_to_session(row)


async def get_session(conn: asyncpg.Connection[Any], session_id: str) -> Session:
    row = await conn.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
    if row is None:
        raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})
    return _row_to_session(row)


async def get_session_workspace_path(conn: asyncpg.Connection[Any], session_id: str) -> str:
    """Return the host-side workspace path stored on the session row."""
    val: str | None = await conn.fetchval(
        "SELECT workspace_volume_path FROM sessions WHERE id = $1", session_id
    )
    if val is None:
        raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})
    return val


async def get_session_focal_channel(conn: asyncpg.Connection[Any], session_id: str) -> str | None:
    """Return the session's current ``focal_channel`` (or NULL = phone down)."""
    focal: str | None = await conn.fetchval(
        "SELECT focal_channel FROM sessions WHERE id = $1",
        session_id,
    )
    return focal


async def get_session_spawn_origin(conn: asyncpg.Connection[Any], session_id: str) -> str | None:
    """Return the session's ``spawned_from_connection_id`` (or NULL).

    A non-null value indicates a per_chat-spawned session whose focal
    channel is locked at creation; ``switch_channel`` rejects mutations
    on those sessions.
    """
    val: str | None = await conn.fetchval(
        "SELECT spawned_from_connection_id FROM sessions WHERE id = $1",
        session_id,
    )
    return val


async def set_session_focal_channel(
    conn: asyncpg.Connection[Any], session_id: str, focal: str | None
) -> None:
    """Mutate the session's ``focal_channel``.  Only ``switch_channel``
    should call this — it's the single source of truth for the agent's
    focal attention.
    """
    await conn.execute(
        "UPDATE sessions SET focal_channel = $1 WHERE id = $2",
        focal,
        session_id,
    )


async def get_session_provisioning(
    conn: asyncpg.Connection[Any], session_id: str
) -> tuple[str, dict[str, str]]:
    """Return ``(workspace_volume_path, env)`` for provisioning a session's container."""
    row = await conn.fetchrow(
        "SELECT workspace_volume_path, env FROM sessions WHERE id = $1", session_id
    )
    if row is None:
        raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})
    raw_env = row["env"]
    env: dict[str, str] = json.loads(raw_env) if isinstance(raw_env, str) else raw_env
    return row["workspace_volume_path"], env


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
        orig_channel=row["orig_channel"],
        focal_channel_at_arrival=row["focal_channel_at_arrival"],
        channel=row["channel"],
    )


async def _latest_cumulative_tokens(conn: asyncpg.Connection[Any], session_id: str) -> int | None:
    """Fetch the cumulative_tokens value of the most recent message event."""
    val: int | None = await conn.fetchval(
        "SELECT cumulative_tokens FROM events "
        "WHERE session_id = $1 AND kind = 'message' "
        "AND cumulative_tokens IS NOT NULL "
        "ORDER BY seq DESC LIMIT 1",
        session_id,
    )
    return val


_MODEL_TOKEN_RATIO_MIN_SAMPLES = 5
_MODEL_TOKEN_RATIO_MIN = 0.5
_MODEL_TOKEN_RATIO_BUCKET_FLOOR = 0.001
_MODEL_TOKEN_RATIO_CACHE_TTL_SECONDS = 60.0
# Fixed per-sample stddev prior for the tokenizer ratio.  Empirically,
# observed per-span CV is ~0.5-1.5 % across the models we've measured
# (Opus 4.7: 0.75 %; Haiku 4.5: ~1 %), so 0.02 is a conservative upper
# bound.  Keeping this fixed (rather than using the observed sample
# stddev) makes the bucket width a deterministic function of ``n`` alone
# — the core property #170 / #171 require: quantization stability is a
# function of ``(n, mean)`` only, independent of the noisy observed-
# stddev estimate that wobbles at small n.
_MODEL_TOKEN_RATIO_SIGMA_PRIOR = 0.02
_model_token_ratio_cache: dict[tuple[str, float], tuple[float, float]] = {}


def _clear_model_token_ratio_cache() -> None:
    """Clear the process-local token-ratio cache for tests."""
    _model_token_ratio_cache.clear()


async def model_token_ratio(
    conn: asyncpg.Connection[Any],
    model: str,
    *,
    k_bucket: float = 2.0,
) -> float:
    """Per-model actual/local token correction.

    Treats R as a fixed tokenizer parameter estimated from noisy
    observed spans.  Returns the lifetime unweighted mean of per-span
    ``actual/local`` ratios, quantized to a prior-shaped bucket
    ``max(k_bucket * sigma_prior / sqrt(n), 0.001)``.  With very little
    data, returns ``1.0`` so newly seen models preserve the old
    model-agnostic windowing behavior until calibration is meaningful.

    ``sigma_prior`` is a fixed per-sample spread prior (see
    :data:`_MODEL_TOKEN_RATIO_SIGMA_PRIOR`).  Using the prior instead of
    the observed sample stddev is what makes the bucket width a
    deterministic function of ``n`` alone — the quantized R depends on
    ``(n, mean)`` only.

    The bucket floor (``0.001``) guards against float rounding nudging
    the drop boundary across an event at very large ``n``.  The returned
    ratio is clamped to ``0.5`` as a physical lower bound: when
    calibration data is pathological, prefer near-neutral windowing over
    dividing by a near-zero R.

    Mature calibrated ratios are cached in-process for 60 seconds.  The
    lifetime aggregate is intentionally slow-moving, and caching prevents
    every windowing call from rescanning all historical calibration spans.
    Below-threshold results are not cached, so newly accumulating models
    can activate as soon as the minimum sample count is reached.

    ``model`` is the raw mind string (``agent.model``) — NO NORMALIZATION.
    Different LiteLLM routes (``anthropic/...`` vs
    ``openrouter/anthropic/...``) hit different provider tokenizers and
    must partition separately.  The same string must appear at stamp time
    and at query time for the same step — always plumb ``agent.model`` on
    both sides.  aios sessions do not carry a model override; the session's
    active mind is always its agent's configured model.

    Scope: the aggregate pools samples across every session in this
    database.  Token counts are scalar only — no content crosses between
    sessions — but the ratio reflects the mixed workload of whatever
    traffic has accumulated.

    "actual" is the provider's ``input_tokens`` usage value, which
    LiteLLM normalizes to the OpenAI convention: ``input_tokens`` is
    **the full prompt count**, including any cached-read or
    cache-creation portion.  Do NOT sum ``cache_read_input_tokens`` or
    ``cache_creation_input_tokens`` on top — they are breakdown metrics
    within the same total, not disjoint extensions.  Output tokens are
    excluded: we're correcting the size of the context we sent, not
    what the model returned.  Uses the
    ``events_model_request_end_calibration_idx`` partial index
    (migration 0024).
    """
    if k_bucket <= 0:
        raise ValueError("k_bucket must be positive")

    cache_key = (model, k_bucket)
    now = time.monotonic()
    cached = _model_token_ratio_cache.get(cache_key)
    if cached is not None:
        expires_at, ratio = cached
        if expires_at > now:
            return ratio
        del _model_token_ratio_cache[cache_key]

    row = await conn.fetchrow(
        """
        WITH calibration AS (
            SELECT
                (data->'model_usage'->>'input_tokens')::float AS it,
                (data->>'local_tokens')::bigint                AS lt
            FROM events
            WHERE kind = 'span'
              AND data->>'event' = 'model_request_end'
              AND (data->>'is_error')::boolean = false
              AND data->>'model' = $1
              AND data ? 'local_tokens'
              AND data ? 'model'
              -- Exclude old/malformed success spans before casting.
              AND (data->'model_usage') ? 'input_tokens'
              AND (data->'model_usage'->>'input_tokens') IS NOT NULL
              AND (data->>'local_tokens')::bigint > 0
        )
        SELECT
            COUNT(*)::bigint                            AS n,
            COALESCE(AVG(it / NULLIF(lt, 0)), 0)::float AS mean_ratio
        FROM calibration
        """,
        model,
    )
    assert row is not None
    if row["n"] < _MODEL_TOKEN_RATIO_MIN_SAMPLES:
        return 1.0

    raw = float(row["mean_ratio"])
    bucket = max(
        k_bucket * _MODEL_TOKEN_RATIO_SIGMA_PRIOR / math.sqrt(float(row["n"])),
        _MODEL_TOKEN_RATIO_BUCKET_FLOOR,
    )
    quantized = round(raw / bucket) * bucket
    ratio = max(quantized, _MODEL_TOKEN_RATIO_MIN)
    _model_token_ratio_cache[cache_key] = (
        now + _MODEL_TOKEN_RATIO_CACHE_TTL_SECONDS,
        ratio,
    )
    return ratio


def _derive_tool_name(kind: str, data: dict[str, Any]) -> str | None:
    """Compute the stamped ``tool_name`` column for a new event.

    For tool-result events the name lives at ``data->>'name'``.  For
    assistant events that requested tools, the first tool_call's function
    name is promoted — multi-tool turns remain discoverable by that first
    name; the full list still lives in ``data->'tool_calls'``.  Pure
    function; paths mirror the backfill in migration 0022 so old and new
    rows stay byte-equivalent in this column.
    """
    if kind != "message":
        return None
    role = data.get("role")
    if role == "tool":
        name = data.get("name")
        return name if isinstance(name, str) else None
    if role == "assistant":
        tool_calls = data.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return None
        first = tool_calls[0]
        if not isinstance(first, dict):
            return None
        function = first.get("function")
        if not isinstance(function, dict):
            return None
        name = function.get("name")
        return name if isinstance(name, str) else None
    return None


def _derive_sender_name(kind: str, data: dict[str, Any]) -> str | None:
    """Sender name for user events carrying connector metadata; else NULL."""
    if kind != "message" or data.get("role") != "user":
        return None
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return None
    name = metadata.get("sender_name")
    return name if isinstance(name, str) else None


def _derive_is_error(kind: str, data: dict[str, Any]) -> bool | None:
    """Error flag on tool-result events; NULL when unset.

    The field is written only when truthy — successful results omit it —
    so the column is TRUE on failure and NULL otherwise.  Matches the
    existing semantics in ``src/aios/harness/tool_dispatch.py``.
    """
    if kind != "message":
        return None
    flag = data.get("is_error")
    if flag is None:
        return None
    return bool(flag)


async def _derive_event_channel(
    conn: asyncpg.Connection[Any],
    session_id: str,
    kind: str,
    data: dict[str, Any],
    orig_channel: str | None,
    focal_at_arrival: str | None,
) -> str | None:
    """Compute the derived ``channel`` for a new event, pre-insert.

    User events → ``orig_channel``.
    Assistant events → ``focal_at_arrival`` (the live focal at stamp time).
    Tool events → the parent assistant's ``focal_channel_at_arrival``,
    looked up by matching ``tool_call_id`` against prior assistant rows'
    ``data->'tool_calls'``. Returns NULL if no parent is found (shouldn't
    happen in practice — tool results only arrive for assistant-requested
    tool calls — but the recap filter tolerates NULL).

    Non-message events and message events with no identifiable role
    return NULL.
    """
    if kind != "message":
        return None
    role = data.get("role")
    if role == "user":
        return orig_channel
    if role == "assistant":
        return focal_at_arrival
    if role == "tool":
        tool_call_id = data.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return None
        # Predicates match ``events_assistant_tool_calls_idx`` (partial
        # index on (session_id, seq) for role=assistant rows that have
        # tool_calls — migration 0011) so the planner can walk it in
        # reverse-seq order and stop at the first matching parent.
        parent_focal: str | None = await conn.fetchval(
            "SELECT focal_channel_at_arrival FROM events "
            "WHERE session_id = $1 "
            "  AND kind = 'message' "
            "  AND data->>'role' = 'assistant' "
            "  AND data ? 'tool_calls' "
            "  AND data->'tool_calls' @> jsonb_build_array("
            "    jsonb_build_object('id', $2::text)) "
            "ORDER BY seq DESC LIMIT 1",
            session_id,
            tool_call_id,
        )
        return parent_focal
    return None


async def lookup_tool_name_by_call_id(
    conn: asyncpg.Connection[Any],
    session_id: str,
    tool_call_id: str,
) -> str | None:
    """Return the function name of the matching ``tool_call`` on the parent
    assistant event, or None if no parent is found.

    Used by the custom tool-result handler to stamp a ``name`` field on
    the tool-role event it appends, so ``_derive_tool_name`` populates
    the ``tool_name`` column (issue #133, migration 0022).  Mirrors the
    parent-assistant lookup in ``_derive_event_channel`` — same ``@>``
    predicate, same partial index (``events_assistant_tool_calls_idx``).
    """
    raw = await conn.fetchval(
        "SELECT data->'tool_calls' FROM events "
        "WHERE session_id = $1 "
        "  AND kind = 'message' "
        "  AND data->>'role' = 'assistant' "
        "  AND data ? 'tool_calls' "
        "  AND data->'tool_calls' @> jsonb_build_array("
        "    jsonb_build_object('id', $2::text)) "
        "ORDER BY seq DESC LIMIT 1",
        session_id,
        tool_call_id,
    )
    if raw is None:
        return None
    tool_calls = _parse_jsonb(raw)
    if not isinstance(tool_calls, list):
        return None
    for tc in tool_calls:
        if not isinstance(tc, dict) or tc.get("id") != tool_call_id:
            continue
        function = tc.get("function")
        if not isinstance(function, dict):
            return None
        name = function.get("name")
        return name if isinstance(name, str) else None
    return None


async def append_event(
    conn: asyncpg.Connection[Any],
    *,
    session_id: str,
    kind: EventKind,
    data: dict[str, Any],
    orig_channel: str | None = None,
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

    Focal-channel stamping (issue #29 redesign): the session's current
    ``focal_channel`` is read from the same UPDATE that allocates the seq
    (via its RETURNING clause) and written to ``focal_channel_at_arrival``
    on the new event row.  Pairing it with the caller-supplied
    ``orig_channel`` (stamped for user events via ``append_user_message``)
    lets the context builder render each event deterministically at arrival
    time without ever needing to re-project past events.

    Derived-channel stamping (issue #52): in the same transaction, the
    new event's ``channel`` column is computed as — for user events,
    ``orig_channel``; for assistant events, ``focal_at_arrival``; for
    tool events, the parent assistant's ``focal_channel_at_arrival``
    (looked up by matching the tool_call_id against prior assistant
    rows' ``data->'tool_calls'``).  This answers "which channel does
    this event belong to?" once and for all; downstream filters become
    a single column read.
    """
    from aios.harness.context import render_user_event
    from aios.harness.tokens import approx_tokens

    new_id = make_id(EVENT)
    data_json = json.dumps(data)

    async with conn.transaction():
        seq_row = await conn.fetchrow(
            "UPDATE sessions SET last_event_seq = last_event_seq + 1 "
            "WHERE id = $1 RETURNING last_event_seq, focal_channel",
            session_id,
        )
        if seq_row is None:
            raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})
        seq = seq_row["last_event_seq"]
        focal_at_arrival: str | None = seq_row["focal_channel"]

        # Compute cumulative_tokens for message events against the
        # as-rendered form so the column stays honest for non-focal
        # notification markers (which occupy far fewer tokens than their
        # full-content counterparts).
        cum_tokens: int | None = None
        if kind == "message":
            prev = await _latest_cumulative_tokens(conn, session_id)
            if data.get("role") == "user" and orig_channel is not None:
                rendered = render_user_event(data, orig_channel, focal_at_arrival)
                cum_tokens = (prev or 0) + approx_tokens([rendered])
            else:
                cum_tokens = (prev or 0) + approx_tokens([data])

        channel = await _derive_event_channel(
            conn, session_id, kind, data, orig_channel, focal_at_arrival
        )
        # Pure promotions: role and the three tool/user-derived columns are
        # stamped from the same JSON paths the 0022 backfill uses.  Agents
        # query these via events_search; they're absent from the Event
        # pydantic model (query-side surface only).
        role: str | None = None
        if kind == "message":
            raw_role = data.get("role")
            if isinstance(raw_role, str):
                role = raw_role
        tool_name = _derive_tool_name(kind, data)
        is_error = _derive_is_error(kind, data)
        sender_name = _derive_sender_name(kind, data)

        row = await conn.fetchrow(
            "INSERT INTO events "
            "(id, session_id, seq, kind, data, cumulative_tokens, "
            " orig_channel, focal_channel_at_arrival, channel, "
            " role, tool_name, is_error, sender_name) "
            "VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, "
            " $10, $11, $12, $13) RETURNING *",
            new_id,
            session_id,
            seq,
            kind,
            data_json,
            cum_tokens,
            orig_channel,
            focal_at_arrival,
            channel,
            role,
            tool_name,
            is_error,
            sender_name,
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
    newest_first: bool = False,
) -> list[Event]:
    order = "DESC" if newest_first else "ASC"
    if kind is None:
        rows = await conn.fetch(
            f"SELECT * FROM events WHERE session_id = $1 AND seq > $2 ORDER BY seq {order} LIMIT $3",
            session_id,
            after_seq,
            limit,
        )
    else:
        rows = await conn.fetch(
            f"SELECT * FROM events WHERE session_id = $1 AND seq > $2 AND kind = $3 "
            f"ORDER BY seq {order} LIMIT $4",
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


async def list_session_channels(conn: asyncpg.Connection[Any], session_id: str) -> list[str]:
    """Distinct channel addresses the session has interacted with, sorted.

    Derived from the event log's ``channel`` column (stamped at append
    time per :func:`_derive_event_channel`).
    """
    rows = await conn.fetch(
        """
        SELECT DISTINCT channel
          FROM events
         WHERE session_id = $1
           AND kind = 'message'
           AND channel IS NOT NULL
         ORDER BY channel
        """,
        session_id,
    )
    return [str(r["channel"]) for r in rows]


async def read_windowed_events(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    window_min: int,
    window_max: int,
    model: str,
    overhead_local: int,
) -> list[Event]:
    """Read message events for the session's trailing context window.

    Uses the ``cumulative_tokens`` column to compute the chunked-window
    snap boundary (same math as :func:`~aios.harness.window.select_window`)
    and loads only the events past that boundary.

    ``cumulative_tokens`` is stored in model-agnostic units (see
    :func:`aios.harness.tokens.approx_tokens`), so the raw value
    systematically diverges from what the provider actually counts —
    ~18 % low on Sonnet 4.6, ~34 % low on Opus 4.7.  This function
    corrects for that at read time: ``window_min`` / ``window_max`` are
    interpreted as provider tokens, ``total_effective = total_local * R``
    where ``R = model_token_ratio(model)``, and the drop boundary is
    translated back to local units for the ``cumulative_tokens`` index
    scan.  When the model has fewer than ``model_token_ratio``'s sample
    threshold, ``R`` is ``1.0`` and the math reduces to the plain
    chunked-snap algorithm.

    ``overhead_local`` is the token cost the caller will add on top of
    the returned events — system prompt plus tool schemas — in local
    (``approx_tokens``) units.  It is NOT included in
    ``cumulative_tokens``, so the windower has to subtract it from the
    effective budget up-front or the sent prompt will exceed
    ``window_max`` by the overhead amount.  Callers that don't have any
    such overhead (preview tooling, test scaffolds) pass ``0``.

    ``model`` must be the session's currently-active mind string —
    ``agent.model`` on the session's pinned agent/version.  The same
    string is what :func:`~aios.harness.loop.run_session_step` stamps on
    ``model_request_end`` spans, so stamp-side and query-side stay
    partitioned on identical keys.

    Prefix-cache invariant: the plain chunked-snap algorithm gave a
    *strict* guarantee of byte-identical prompt prefix within a snap
    chunk.  With the ratio correction this remains stable in practice
    because :func:`model_token_ratio` uses a lifetime aggregate and
    standard-error bucketing, so mature calibrations do not drift on every
    new sample.  Early calibrations are coarse by design and converge as
    the sample count grows.

    Falls back to :func:`read_message_events` (loading all events) when
    cumulative data is not available (pre-backfill sessions or rolling
    deploys) or when the entire session fits within ``window_max``.
    """
    # Index seek: total cumulative tokens from the latest message event.
    total = await _latest_cumulative_tokens(conn, session_id)

    # Fallback: no cumulative data yet — load everything.
    if total is None:
        return await read_message_events(conn, session_id)

    ratio = await model_token_ratio(conn, model)

    # Shrink the effective window by the caller's overhead contribution.
    # Apply R to overhead_local up-front so the subtraction happens in the
    # same effective (provider-token) space tokens_to_drop operates in.
    overhead_effective = round(overhead_local * ratio)
    events_window_max = window_max - overhead_effective
    events_window_min = max(0, window_min - overhead_effective)
    if events_window_max <= 0:
        raise ValueError(
            f"system+tools overhead ({overhead_effective} provider tokens) "
            f"exceeds window_max ({window_max}); no budget remains for events"
        )

    total_effective = round(total * ratio)
    if total_effective <= events_window_max:
        return await read_message_events(conn, session_id)

    from aios.harness.tokens import tokens_to_drop

    # Forward-convert local → effective with plain rounding: best-estimate
    # of the provider-token total.  Back-convert effective → local with
    # ceil: deliberately asymmetric so the post-drop remaining fits under
    # ``window_max`` even when ratio error would otherwise leave one
    # message straddling the boundary.
    drop_effective = tokens_to_drop(
        total_effective, window_min=events_window_min, window_max=events_window_max
    )
    if drop_effective == 0:
        return await read_message_events(conn, session_id)

    drop = math.ceil(drop_effective / ratio)

    # Bounded range scan: only events past the boundary.
    rows = await conn.fetch(
        "SELECT * FROM events "
        "WHERE session_id = $1 AND kind = 'message' "
        "AND cumulative_tokens > $2 "
        "ORDER BY seq ASC",
        session_id,
        drop,
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
    """Archive a vault and purge the encrypted blobs of its active credentials.

    Archive is an UPDATE, so ``ON DELETE CASCADE`` on the FK does not fire
    here — child credentials must be archived and zeroed explicitly. Both
    operations run in one transaction.
    """
    async with conn.transaction():
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
        # Purge every active child credential's secret payload at the same
        # moment we archive the parent vault.
        await conn.execute(
            "UPDATE vault_credentials "
            "SET ciphertext = ''::bytea, nonce = ''::bytea, "
            "    archived_at = now(), updated_at = now() "
            "WHERE vault_id = $1 AND archived_at IS NULL",
            vault_id,
        )
    return _row_to_vault(row)


async def delete_vault(conn: asyncpg.Connection[Any], vault_id: str) -> None:
    # Child credentials are removed by ``ON DELETE CASCADE`` (migration 0015).
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


async def lock_oauth_credential_for_refresh(
    conn: asyncpg.Connection[Any], vault_id: str, mcp_server_url: str
) -> tuple[str, EncryptedBlob] | None:
    """``SELECT FOR UPDATE`` the active credential for ``(vault_id, url)``.

    Used by the OAuth refresh path to serialize concurrent refreshes of the
    same credential. Returns ``(credential_id, EncryptedBlob)`` or ``None``
    if no active credential exists. Caller owns the surrounding transaction.
    """
    row = await conn.fetchrow(
        "SELECT id, ciphertext, nonce FROM vault_credentials "
        "WHERE vault_id = $1 AND mcp_server_url = $2 AND archived_at IS NULL "
        "FOR UPDATE",
        vault_id,
        mcp_server_url,
    )
    if row is None:
        return None
    blob = EncryptedBlob(ciphertext=bytes(row["ciphertext"]), nonce=bytes(row["nonce"]))
    return str(row["id"]), blob


async def get_vault_credential_with_blob(
    conn: asyncpg.Connection[Any], vault_id: str, credential_id: str
) -> tuple[VaultCredential, EncryptedBlob]:
    """Fetch the credential metadata and decrypted-blob inputs in one round-trip.

    Excludes archived credentials — the blob is meaningless once archived
    (and gets zeroed out at archive time).
    """
    row = await conn.fetchrow(
        "SELECT * FROM vault_credentials WHERE id = $1 AND vault_id = $2 AND archived_at IS NULL",
        credential_id,
        vault_id,
    )
    if row is None:
        raise NotFoundError(
            f"credential {credential_id} not found or archived",
            detail={"id": credential_id, "vault_id": vault_id},
        )
    cred = _row_to_vault_credential(row)
    blob = EncryptedBlob(ciphertext=bytes(row["ciphertext"]), nonce=bytes(row["nonce"]))
    return cred, blob


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
    blob: EncryptedBlob | None = None,
    display_name: str | None | EllipsisType = ...,
    metadata: dict[str, Any] | None | EllipsisType = ...,
) -> VaultCredential:
    sets: list[str] = []
    args: list[Any] = [credential_id, vault_id]
    if display_name is not ...:
        args.append(display_name)
        sets.append(f"display_name = ${len(args)}")
    if blob is not None:
        args.append(blob.ciphertext)
        sets.append(f"ciphertext = ${len(args)}")
        args.append(blob.nonce)
        sets.append(f"nonce = ${len(args)}")
    if metadata is not ...:
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
    """Archive a credential and zero out its encrypted secret payload.

    The bytes are scrubbed at archive time so a future DB dump or query
    cannot leak the secret, even though ``WHERE archived_at IS NULL``
    filters in the read path already prevent resolution.
    """
    row = await conn.fetchrow(
        "UPDATE vault_credentials "
        "SET ciphertext = ''::bytea, nonce = ''::bytea, "
        "    archived_at = now(), updated_at = now() "
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


async def resolve_vault_credential(
    conn: asyncpg.Connection[Any],
    *,
    vault_id: str,
    mcp_server_url: str,
) -> tuple[EncryptedBlob, str] | None:
    """Look up an MCP credential in a specific vault by URL — no
    ``session_vaults`` join."""
    row = await conn.fetchrow(
        """
        SELECT ciphertext, nonce, auth_type
          FROM vault_credentials
         WHERE vault_id = $1
           AND mcp_server_url = $2
           AND archived_at IS NULL
         LIMIT 1
        """,
        vault_id,
        mcp_server_url,
    )
    if row is None:
        return None
    return EncryptedBlob(ciphertext=row["ciphertext"], nonce=row["nonce"]), str(row["auth_type"])


async def resolve_mcp_credential(
    conn: asyncpg.Connection[Any],
    session_id: str,
    mcp_server_url: str,
) -> tuple[EncryptedBlob, str, str] | None:
    """Find the first matching MCP credential across a session's bound vaults.

    Joins ``session_vaults`` (rank-ordered) with ``vault_credentials``
    filtered by ``mcp_server_url``. Returns
    ``(EncryptedBlob, auth_type, vault_id)`` for the first match, or
    ``None`` if no credential exists. The ``vault_id`` is needed by the
    OAuth refresh path to scope ``SELECT … FOR UPDATE`` to a specific row.
    """
    row = await conn.fetchrow(
        """
        SELECT vc.ciphertext, vc.nonce, vc.auth_type, vc.vault_id
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
    return (
        EncryptedBlob(ciphertext=row["ciphertext"], nonce=row["nonce"]),
        str(row["auth_type"]),
        str(row["vault_id"]),
    )


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


# ─── connections ────────────────────────────────────────────────────────────
#
# Three valid shapes per ``connections_one_mode_ck``:
#
#   detached       — session_id NULL,  session_template_id NULL
#   single_session — session_id SET,   session_template_id NULL
#   per_chat       — session_id NULL,  session_template_id SET


def _row_to_connection(row: asyncpg.Record) -> Connection:
    raw_metadata = row["metadata"]
    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
    return Connection(
        id=row["id"],
        connector=row["connector"],
        account=row["account"],
        session_id=row["session_id"],
        session_template_id=row["session_template_id"],
        metadata=metadata,
        created_at=row["created_at"],
        attached_at=row["attached_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def insert_connection(
    conn: asyncpg.Connection[Any],
    *,
    connector: str,
    account: str,
    metadata: dict[str, Any],
) -> Connection:
    """Insert a detached connection, idempotent on the active uniqueness key.

    Per plan decision #5, both the explicit ``POST /v1/connections`` and
    the supervisor's auto-create-on-first-inbound path race-safely
    converge on a single row via ``INSERT ... ON CONFLICT DO NOTHING
    RETURNING``: empty RETURNING means another writer beat us, so we
    re-read the existing active row and hand it back.  The unique index
    is ``(connector, account) WHERE archived_at IS NULL`` — an archived
    row with the same pair will not collide; a fresh insert will land.

    Use ``attach_connection`` or ``configure_per_chat_connection`` to bind
    a routing mode after creation.
    """
    new_id = make_id(CONNECTION)
    row = await conn.fetchrow(
        """
        INSERT INTO connections (id, connector, account, metadata)
        VALUES ($1, $2, $3, $4::jsonb)
        ON CONFLICT (connector, account) WHERE archived_at IS NULL DO NOTHING
        RETURNING *
        """,
        new_id,
        connector,
        account,
        json.dumps(metadata),
    )
    if row is not None:
        return _row_to_connection(row)
    existing = await get_connection_for_account(conn, connector=connector, account=account)
    if existing is None:
        # CONFLICT means an active row existed at INSERT time.  If it's
        # gone now, it was archived between the two queries — fall back
        # to a fresh insert which the unique index now permits.
        row = await conn.fetchrow(
            """
            INSERT INTO connections (id, connector, account, metadata)
            VALUES ($1, $2, $3, $4::jsonb)
            RETURNING *
            """,
            new_id,
            connector,
            account,
            json.dumps(metadata),
        )
        assert row is not None
        return _row_to_connection(row)
    return existing


async def get_connection(conn: asyncpg.Connection[Any], connection_id: str) -> Connection:
    row = await conn.fetchrow("SELECT * FROM connections WHERE id = $1", connection_id)
    if row is None:
        raise NotFoundError(
            f"connection {connection_id} not found",
            detail={"id": connection_id},
        )
    return _row_to_connection(row)


async def session_authorizes_connector_account(
    conn: asyncpg.Connection[Any],
    session_id: str,
    connector: str,
    account: str,
) -> bool:
    """Permission check for outbound connector tool calls.

    True iff there's an active connection whose ``(connector, account)``
    matches AND whose ``session_id`` is this session OR whose ``id`` is
    the session's ``spawned_from_connection_id``.  Used by the
    outbound MCP dispatch to gate tool calls that take an explicit
    ``account`` argument — the supervisor will happily forward to the
    connector regardless, but the model shouldn't be able to reach
    accounts the operator hasn't bound to this session.
    """
    row = await conn.fetchrow(
        """
        SELECT 1
          FROM connections c
          LEFT JOIN sessions s ON s.id = $1
         WHERE c.connector = $2
           AND c.account = $3
           AND c.archived_at IS NULL
           AND (c.session_id = $1 OR c.id = s.spawned_from_connection_id)
         LIMIT 1
        """,
        session_id,
        connector,
        account,
    )
    return row is not None


async def get_connection_for_account(
    conn: asyncpg.Connection[Any], connector: str, account: str
) -> Connection | None:
    """Active connection for ``(connector, account)``, or ``None``."""
    row = await conn.fetchrow(
        "SELECT * FROM connections WHERE connector = $1 AND account = $2 AND archived_at IS NULL",
        connector,
        account,
    )
    if row is None:
        return None
    return _row_to_connection(row)


async def _raise_for_failed_mode_transition(
    conn: asyncpg.Connection[Any], connection_id: str
) -> NoReturn:
    """Diagnose why a guarded mode-transition UPDATE returned no row.

    Called from ``attach_connection``/``configure_per_chat_connection``
    when the WHERE clause didn't match any row.  Picks the most specific
    error to raise: NotFound > archived > already-bound.
    """
    existing = await conn.fetchrow(
        "SELECT archived_at FROM connections WHERE id = $1",
        connection_id,
    )
    if existing is None:
        raise NotFoundError(
            f"connection {connection_id} not found",
            detail={"id": connection_id},
        )
    if existing["archived_at"] is not None:
        raise ConflictError(
            f"connection {connection_id} is archived",
            detail={"id": connection_id},
        )
    raise ConflictError(
        f"connection {connection_id} is already attached or configured per_chat; "
        "detach or unconfigure first",
        detail={"id": connection_id},
    )


_MODE_PREDICATES: dict[str, str] = {
    "detached": "session_id IS NULL AND session_template_id IS NULL",
    "single_session": "session_id IS NOT NULL",
    "per_chat": "session_template_id IS NOT NULL",
}


async def list_connections(
    conn: asyncpg.Connection[Any],
    *,
    connector: str | None = None,
    session_id: str | None = None,
    mode: str | None = None,
    limit: int = 50,
    after: str | None = None,
) -> list[Connection]:
    """List active connections.  Optional filters narrow by connector type,
    attached session id, or routing mode (``detached`` / ``single_session``
    / ``per_chat``).
    """
    clauses: list[str] = ["archived_at IS NULL"]
    args: list[Any] = []
    if connector is not None:
        args.append(connector)
        clauses.append(f"connector = ${len(args)}")
    if session_id is not None:
        args.append(session_id)
        clauses.append(f"session_id = ${len(args)}")
    if mode is not None:
        clauses.append(_MODE_PREDICATES[mode])
    if after is not None:
        args.append(after)
        clauses.append(f"id < ${len(args)}")
    args.append(limit)
    sql = (
        f"SELECT * FROM connections WHERE {' AND '.join(clauses)} "
        f"ORDER BY id DESC LIMIT ${len(args)}"
    )
    rows = await conn.fetch(sql, *args)
    return [_row_to_connection(r) for r in rows]


async def attach_connection(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    *,
    session_id: str,
) -> Connection:
    """Bind a detached connection to a session (single_session mode)."""
    try:
        row = await conn.fetchrow(
            """
            UPDATE connections
               SET session_id = $2,
                   attached_at = now(),
                   updated_at = now()
             WHERE id = $1
               AND archived_at IS NULL
               AND session_id IS NULL
               AND session_template_id IS NULL
            RETURNING *
            """,
            connection_id,
            session_id,
        )
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            f"session {session_id} not found",
            detail={"session_id": session_id},
        ) from exc
    if row is None:
        await _raise_for_failed_mode_transition(conn, connection_id)
    return _row_to_connection(row)


async def detach_connection(conn: asyncpg.Connection[Any], connection_id: str) -> Connection:
    """Drop the single_session binding, leaving the connection detached.

    Already-spawned per_chat sessions on this connection (if any) are
    not cascaded — they remain alive with ``spawned_from_connection_id``
    still pointing at this row.
    """
    row = await conn.fetchrow(
        """
        UPDATE connections
           SET session_id = NULL,
               attached_at = NULL,
               updated_at = now()
         WHERE id = $1
           AND archived_at IS NULL
           AND session_id IS NOT NULL
        RETURNING *
        """,
        connection_id,
    )
    if row is None:
        existing = await conn.fetchrow(
            "SELECT archived_at, session_id FROM connections WHERE id = $1",
            connection_id,
        )
        if existing is None:
            raise NotFoundError(
                f"connection {connection_id} not found",
                detail={"id": connection_id},
            )
        if existing["archived_at"] is not None:
            raise ConflictError(
                f"connection {connection_id} is archived",
                detail={"id": connection_id},
            )
        raise ConflictError(
            f"connection {connection_id} is not in single_session mode",
            detail={"id": connection_id},
        )
    return _row_to_connection(row)


async def configure_per_chat_connection(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    *,
    session_template_id: str,
) -> Connection:
    """Switch the connection into per_chat mode."""
    try:
        row = await conn.fetchrow(
            """
            UPDATE connections
               SET session_template_id = $2,
                   attached_at = now(),
                   updated_at = now()
             WHERE id = $1
               AND archived_at IS NULL
               AND session_id IS NULL
               AND session_template_id IS NULL
            RETURNING *
            """,
            connection_id,
            session_template_id,
        )
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            f"session template {session_template_id} not found",
            detail={"session_template_id": session_template_id},
        ) from exc
    if row is None:
        await _raise_for_failed_mode_transition(conn, connection_id)
    return _row_to_connection(row)


async def unconfigure_connection(conn: asyncpg.Connection[Any], connection_id: str) -> Connection:
    """Drop the per_chat configuration, leaving the connection detached.

    Already-spawned per_chat sessions are not cascaded — they remain
    alive with ``spawned_from_connection_id`` still pointing at this row.
    """
    row = await conn.fetchrow(
        """
        UPDATE connections
           SET session_template_id = NULL,
               attached_at = NULL,
               updated_at = now()
         WHERE id = $1
           AND archived_at IS NULL
           AND session_template_id IS NOT NULL
        RETURNING *
        """,
        connection_id,
    )
    if row is None:
        existing = await conn.fetchrow(
            "SELECT archived_at, session_template_id FROM connections WHERE id = $1",
            connection_id,
        )
        if existing is None:
            raise NotFoundError(
                f"connection {connection_id} not found",
                detail={"id": connection_id},
            )
        if existing["archived_at"] is not None:
            raise ConflictError(
                f"connection {connection_id} is archived",
                detail={"id": connection_id},
            )
        raise ConflictError(
            f"connection {connection_id} is not in per_chat mode",
            detail={"id": connection_id},
        )
    return _row_to_connection(row)


async def archive_connection(conn: asyncpg.Connection[Any], connection_id: str) -> Connection:
    row = await conn.fetchrow(
        "UPDATE connections SET archived_at = now(), updated_at = now() "
        "WHERE id = $1 AND archived_at IS NULL RETURNING *",
        connection_id,
    )
    if row is None:
        raise NotFoundError(
            f"connection {connection_id} not found or already archived",
            detail={"id": connection_id},
        )
    return _row_to_connection(row)


# ─── connection_chat_sessions (per_chat ledger) ─────────────────────────────


async def lookup_chat_session(
    conn: asyncpg.Connection[Any], connection_id: str, chat_id: str
) -> str | None:
    """Existing session_id for ``(connection_id, chat_id)``, else ``None``."""
    val: str | None = await conn.fetchval(
        "SELECT session_id FROM connection_chat_sessions WHERE connection_id = $1 AND chat_id = $2",
        connection_id,
        chat_id,
    )
    return val


async def insert_chat_session(
    conn: asyncpg.Connection[Any],
    *,
    connection_id: str,
    chat_id: str,
    session_id: str,
) -> str:
    """Race-safe insert: returns the session_id stored after the call.

    On conflict (a concurrent inbound for the same chat already wrote the
    row) returns the *existing* session_id; the caller is then on the
    hook to discard the just-created session as an orphan.
    """
    row = await conn.fetchrow(
        """
        INSERT INTO connection_chat_sessions (connection_id, chat_id, session_id)
        VALUES ($1, $2, $3)
        ON CONFLICT (connection_id, chat_id) DO NOTHING
        RETURNING session_id
        """,
        connection_id,
        chat_id,
        session_id,
    )
    if row is not None:
        return str(row["session_id"])
    existing = await lookup_chat_session(conn, connection_id, chat_id)
    if existing is None:
        # CONFLICT means the row existed at INSERT time; if it's gone now
        # the chat session was hard-deleted between the two queries.
        raise NotFoundError(
            f"chat session for ({connection_id}, {chat_id}) vanished after CONFLICT",
            detail={"connection_id": connection_id, "chat_id": chat_id},
        )
    return existing


# ─── connector_inbound_acks (dedup ledger) ──────────────────────────────────


async def flip_idle_to_pending(conn: asyncpg.Connection[Any], session_id: str) -> None:
    """Flip ``sessions.status`` from ``idle`` to ``pending`` if currently idle.

    Called after appending a user message so polling orchestrators can
    distinguish queued-but-not-started from turn-finished.  Other states
    (running / rescheduling / terminated) are left alone — the worker
    owns the running status, and changing rescheduling would lose the
    retry-in-progress signal.
    """
    await conn.execute(
        "UPDATE sessions SET status = 'pending', updated_at = now() "
        "WHERE id = $1 AND status = 'idle'",
        session_id,
    )


async def try_record_inbound_ack(
    conn: asyncpg.Connection[Any],
    *,
    connector: str,
    account: str,
    event_id: str,
    appended_seq: int,
) -> bool:
    """Insert a dedup-ledger row, returning ``True`` iff it actually inserted.

    Called from the worker's inbound handler in the same transaction as
    :func:`append_event`.  The PK ``(connector, account, event_id)``
    enforces at-most-once event append: a duplicate inbound (same ULID
    re-emitted on connector reconnect because the previous worker
    crashed before acking) hits ``ON CONFLICT DO NOTHING`` and the
    caller rolls back the txn so no second event lands.

    The ``appended_seq`` is the gapless seq the in-flight ``append_event``
    just allocated; it makes the ledger row queryable for the operator
    debugging "did this message land?".
    """
    row = await conn.fetchrow(
        """
        INSERT INTO connector_inbound_acks (connector, account, event_id, appended_seq)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT DO NOTHING
        RETURNING 1
        """,
        connector,
        account,
        event_id,
        appended_seq,
    )
    return row is not None


# ─── session_templates ──────────────────────────────────────────────────────


def _row_to_session_template(row: asyncpg.Record) -> SessionTemplate:
    raw_metadata = row["metadata"]
    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
    return SessionTemplate(
        id=row["id"],
        name=row["name"],
        agent_id=row["agent_id"],
        agent_version=row["agent_version"],
        environment_id=row["environment_id"],
        vault_ids=list(row["vault_ids"] or []),
        memory_store_ids=list(row["memory_store_ids"] or []),
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def insert_session_template(
    conn: asyncpg.Connection[Any],
    *,
    name: str,
    agent_id: str,
    environment_id: str,
    agent_version: int | None,
    vault_ids: list[str],
    memory_store_ids: list[str],
    metadata: dict[str, Any],
) -> SessionTemplate:
    new_id = make_id(SESSION_TEMPLATE)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO session_templates
                (id, name, agent_id, agent_version, environment_id,
                 vault_ids, memory_store_ids, metadata)
            VALUES ($1, $2, $3, $4, $5, $6::text[], $7::text[], $8::jsonb)
            RETURNING *
            """,
            new_id,
            name,
            agent_id,
            agent_version,
            environment_id,
            vault_ids,
            memory_store_ids,
            json.dumps(metadata),
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"a session template named {name!r} already exists",
            detail={"name": name},
        ) from exc
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            "agent or environment not found",
            detail={"agent_id": agent_id, "environment_id": environment_id},
        ) from exc
    assert row is not None
    return _row_to_session_template(row)


async def get_session_template(conn: asyncpg.Connection[Any], template_id: str) -> SessionTemplate:
    row = await conn.fetchrow("SELECT * FROM session_templates WHERE id = $1", template_id)
    if row is None:
        raise NotFoundError(
            f"session template {template_id} not found",
            detail={"id": template_id},
        )
    return _row_to_session_template(row)


async def list_session_templates(
    conn: asyncpg.Connection[Any], *, limit: int = 50, after: str | None = None
) -> list[SessionTemplate]:
    if after is None:
        rows = await conn.fetch(
            "SELECT * FROM session_templates WHERE archived_at IS NULL ORDER BY id DESC LIMIT $1",
            limit,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM session_templates WHERE archived_at IS NULL AND id < $1 "
            "ORDER BY id DESC LIMIT $2",
            after,
            limit,
        )
    return [_row_to_session_template(r) for r in rows]


async def update_session_template(
    conn: asyncpg.Connection[Any],
    template_id: str,
    *,
    name: str | None = None,
    agent_id: str | None = None,
    agent_version: int | None = _UNSET,
    environment_id: str | None = None,
    vault_ids: list[str] | None = None,
    memory_store_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> SessionTemplate:
    sets: list[str] = []
    args: list[Any] = [template_id]
    if name is not None:
        args.append(name)
        sets.append(f"name = ${len(args)}")
    if agent_id is not None:
        args.append(agent_id)
        sets.append(f"agent_id = ${len(args)}")
    if agent_version is not _UNSET:
        args.append(agent_version)
        sets.append(f"agent_version = ${len(args)}")
    if environment_id is not None:
        args.append(environment_id)
        sets.append(f"environment_id = ${len(args)}")
    if vault_ids is not None:
        args.append(vault_ids)
        sets.append(f"vault_ids = ${len(args)}::text[]")
    if memory_store_ids is not None:
        args.append(memory_store_ids)
        sets.append(f"memory_store_ids = ${len(args)}::text[]")
    if metadata is not None:
        args.append(json.dumps(metadata))
        sets.append(f"metadata = ${len(args)}::jsonb")
    if not sets:
        return await get_session_template(conn, template_id)
    sets.append("updated_at = now()")
    sql = f"UPDATE session_templates SET {', '.join(sets)} WHERE id = $1 RETURNING *"
    try:
        row = await conn.fetchrow(sql, *args)
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"a session template named {name!r} already exists",
            detail={"name": name},
        ) from exc
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            "agent or environment not found",
            detail={"agent_id": agent_id, "environment_id": environment_id},
        ) from exc
    if row is None:
        raise NotFoundError(
            f"session template {template_id} not found",
            detail={"id": template_id},
        )
    return _row_to_session_template(row)


async def archive_session_template(
    conn: asyncpg.Connection[Any], template_id: str
) -> SessionTemplate:
    """Soft-delete the template.  Already-spawned per_chat sessions keep
    working; new chat sessions on connections referencing this template
    will fail at the inbound handler.
    """
    row = await conn.fetchrow(
        "UPDATE session_templates SET archived_at = now(), updated_at = now() "
        "WHERE id = $1 AND archived_at IS NULL RETURNING *",
        template_id,
    )
    if row is None:
        raise NotFoundError(
            f"session template {template_id} not found or already archived",
            detail={"id": template_id},
        )
    return _row_to_session_template(row)


# ─── memory stores ──────────────────────────────────────────────────────────


def _row_to_memory_store(row: asyncpg.Record) -> MemoryStore:
    raw_metadata = row["metadata"]
    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
    return MemoryStore(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


def _row_to_memory(row: asyncpg.Record, *, include_content: bool) -> Memory:
    return Memory(
        id=row["id"],
        memory_store_id=row["memory_store_id"],
        memory_version_id=row["current_version_id"],
        path=row["path"],
        content=row["content"] if include_content else None,
        content_sha256=row["content_sha256"],
        content_size_bytes=row["content_size_bytes"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_memory_version(row: asyncpg.Record, *, include_content: bool) -> MemoryVersion:
    redacted = row["redacted_at"] is not None
    redacted_by: Actor | None = None
    if redacted and row["redacted_by_type"] is not None:
        redacted_by = _build_actor(row["redacted_by_type"], row["redacted_by_ref"])
    return MemoryVersion(
        id=row["id"],
        memory_store_id=row["memory_store_id"],
        memory_id=row["memory_id"],
        operation=row["operation"],
        path=row["path"],
        content=row["content"] if include_content and not redacted else None,
        content_sha256=row["content_sha256"],
        content_size_bytes=row["content_size_bytes"],
        created_by=_build_actor(row["created_by_type"], row["created_by_ref"]),
        created_at=row["created_at"],
        redacted_at=row["redacted_at"],
        redacted_by=redacted_by,
    )


def _build_actor(actor_type: str, actor_ref: str) -> Actor:
    if actor_type == "session_actor":
        return Actor(type="session_actor", session_id=actor_ref)
    return Actor(type="api_actor", api_key_id=actor_ref)


# Stores ───────────────────────────────────────────────────────────────────


async def insert_memory_store(
    conn: asyncpg.Connection[Any],
    *,
    name: str,
    description: str,
    metadata: dict[str, Any],
) -> MemoryStore:
    row = await conn.fetchrow(
        """
        INSERT INTO memory_stores (id, name, description, metadata)
        VALUES ($1, $2, $3, $4::jsonb)
        RETURNING *
        """,
        make_id(MEMORY_STORE),
        name,
        description,
        json.dumps(metadata),
    )
    assert row is not None
    return _row_to_memory_store(row)


async def get_memory_store(
    conn: asyncpg.Connection[Any], store_id: str, *, allow_archived: bool = True
) -> MemoryStore:
    row = await conn.fetchrow("SELECT * FROM memory_stores WHERE id = $1", store_id)
    if row is None:
        raise NotFoundError(f"memory store {store_id} not found", detail={"id": store_id})
    store = _row_to_memory_store(row)
    if not allow_archived and store.archived_at is not None:
        raise MemoryStoreArchivedError(
            f"memory store {store_id} is archived",
            detail={"id": store_id},
        )
    return store


async def list_memory_stores(
    conn: asyncpg.Connection[Any], *, include_archived: bool = False, limit: int = 100
) -> list[MemoryStore]:
    if include_archived:
        rows = await conn.fetch("SELECT * FROM memory_stores ORDER BY id DESC LIMIT $1", limit)
    else:
        rows = await conn.fetch(
            "SELECT * FROM memory_stores WHERE archived_at IS NULL ORDER BY id DESC LIMIT $1",
            limit,
        )
    return [_row_to_memory_store(r) for r in rows]


async def update_memory_store(
    conn: asyncpg.Connection[Any],
    store_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MemoryStore:
    sets: list[str] = []
    args: list[Any] = [store_id]
    if name is not None:
        args.append(name)
        sets.append(f"name = ${len(args)}")
    if description is not None:
        args.append(description)
        sets.append(f"description = ${len(args)}")
    if metadata is not None:
        args.append(json.dumps(metadata))
        sets.append(f"metadata = ${len(args)}::jsonb")
    if not sets:
        return await get_memory_store(conn, store_id)
    sets.append("updated_at = now()")
    sql = f"UPDATE memory_stores SET {', '.join(sets)} WHERE id = $1 RETURNING *"
    row = await conn.fetchrow(sql, *args)
    if row is None:
        raise NotFoundError(f"memory store {store_id} not found", detail={"id": store_id})
    return _row_to_memory_store(row)


async def archive_memory_store(conn: asyncpg.Connection[Any], store_id: str) -> MemoryStore:
    row = await conn.fetchrow(
        "UPDATE memory_stores SET archived_at = now(), updated_at = now() "
        "WHERE id = $1 AND archived_at IS NULL RETURNING *",
        store_id,
    )
    if row is None:
        raise NotFoundError(
            f"memory store {store_id} not found or already archived",
            detail={"id": store_id},
        )
    return _row_to_memory_store(row)


async def delete_memory_store(conn: asyncpg.Connection[Any], store_id: str) -> None:
    result = await conn.execute("DELETE FROM memory_stores WHERE id = $1", store_id)
    if result == "DELETE 0":
        raise NotFoundError(f"memory store {store_id} not found", detail={"id": store_id})


# Memory + version (single-txn helpers) ────────────────────────────────────


async def _allocate_version_seq(conn: asyncpg.Connection[Any], store_id: str) -> int:
    """Bump ``last_version_seq`` on the store row and return the allocated seq.

    Mirror of the events seq allocation at append_event: row-lock the parent,
    increment, return. Caller must be inside a transaction so the seq is
    bound to the version insert that follows.
    """
    row = await conn.fetchrow(
        "UPDATE memory_stores SET last_version_seq = last_version_seq + 1, "
        "updated_at = now() WHERE id = $1 AND archived_at IS NULL "
        "RETURNING last_version_seq",
        store_id,
    )
    if row is None:
        existing = await conn.fetchrow(
            "SELECT archived_at FROM memory_stores WHERE id = $1", store_id
        )
        if existing is None:
            raise NotFoundError(f"memory store {store_id} not found", detail={"id": store_id})
        raise MemoryStoreArchivedError(
            f"memory store {store_id} is archived",
            detail={"id": store_id},
        )
    seq: int = row["last_version_seq"]
    return seq


async def insert_memory_with_version(
    conn: asyncpg.Connection[Any],
    *,
    store_id: str,
    path: str,
    content: str,
    content_sha256: str,
    actor_type: str,
    actor_ref: str,
) -> Memory:
    """Insert a new memory + its initial ``created`` version in one txn.

    On path collision raises :class:`MemoryPathConflictError` carrying the
    existing memory id. The caller can decide between updating that memory
    and surfacing the error.
    """
    size_bytes = len(content.encode("utf-8"))
    memory_id = make_id(MEMORY)
    version_id = make_id(MEMORY_VERSION)

    try:
        async with conn.transaction():
            seq = await _allocate_version_seq(conn, store_id)

            # Version first — its `memory_id` column is non-FK, so the
            # not-yet-inserted memory row doesn't block this. Memory row
            # references back via current_version_id.
            await conn.execute(
                """
                INSERT INTO memory_versions
                    (id, memory_store_id, memory_id, seq, operation, path,
                     content, content_sha256, content_size_bytes,
                     created_by_type, created_by_ref)
                VALUES ($1, $2, $3, $4, 'created', $5, $6, $7, $8, $9, $10)
                """,
                version_id,
                store_id,
                memory_id,
                seq,
                path,
                content,
                content_sha256,
                size_bytes,
                actor_type,
                actor_ref,
            )

            row = await conn.fetchrow(
                """
                INSERT INTO memories
                    (id, memory_store_id, path, content, content_sha256,
                     content_size_bytes, current_version_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
                """,
                memory_id,
                store_id,
                path,
                content,
                content_sha256,
                size_bytes,
                version_id,
            )
    except asyncpg.UniqueViolationError as exc:
        # Re-issue the lookup outside the rolled-back transaction so the
        # error envelope can carry the existing memory id.
        existing = await conn.fetchrow(
            "SELECT id FROM memories WHERE memory_store_id = $1 AND path = $2 "
            "AND deleted_at IS NULL",
            store_id,
            path,
        )
        conflicting_id = existing["id"] if existing is not None else None
        raise MemoryPathConflictError(
            f"path {path!r} is already used by {conflicting_id!r}; use update to modify it",
            detail={
                "conflicting_memory_id": conflicting_id,
                "conflicting_path": path,
            },
        ) from exc
    assert row is not None
    return _row_to_memory(row, include_content=False)


async def get_memory(
    conn: asyncpg.Connection[Any],
    store_id: str,
    memory_id: str,
    *,
    include_content: bool = True,
) -> Memory:
    row = await conn.fetchrow(
        "SELECT * FROM memories WHERE memory_store_id = $1 AND id = $2 AND deleted_at IS NULL",
        store_id,
        memory_id,
    )
    if row is None:
        raise NotFoundError(
            f"memory {memory_id} not found in store {store_id}",
            detail={"id": memory_id, "memory_store_id": store_id},
        )
    return _row_to_memory(row, include_content=include_content)


async def get_memory_by_path(
    conn: asyncpg.Connection[Any],
    store_id: str,
    path: str,
    *,
    include_content: bool = True,
) -> Memory | None:
    row = await conn.fetchrow(
        "SELECT * FROM memories WHERE memory_store_id = $1 AND path = $2 AND deleted_at IS NULL",
        store_id,
        path,
    )
    if row is None:
        return None
    return _row_to_memory(row, include_content=include_content)


async def list_active_memory_paths_and_content(
    conn: asyncpg.Connection[Any], store_id: str
) -> list[tuple[str, str]]:
    """Bulk-fetch ``(path, content)`` for every non-deleted memory in the store.

    Used by sandbox materialization, which needs all live memories in one
    DB roundtrip rather than ``list_memories`` (metadata only) followed by
    a per-memory ``get_memory(include_content=True)`` fan-out.
    """
    rows = await conn.fetch(
        "SELECT path, content FROM memories WHERE memory_store_id = $1 AND deleted_at IS NULL",
        store_id,
    )
    return [(r["path"], r["content"]) for r in rows]


async def list_memories(
    conn: asyncpg.Connection[Any],
    store_id: str,
    *,
    path_prefix: str | None = None,
    order_by: str = "created_at",
    depth: int | None = None,
) -> list[Memory | MemoryPrefix]:
    """List memories, optionally filtered by ``path_prefix`` and depth-clipped.

    ``depth`` requires ``order_by='path'`` (matches Anthropic's wire validation).
    With depth set, paths whose component count under the prefix exceeds
    ``depth`` are collapsed into ``memory_prefix`` synthetic entries.
    """
    if depth is not None and order_by != "path":
        raise ConflictError(
            "depth requires order_by=path",
            detail={"order_by": order_by, "depth": depth},
        )

    where = "memory_store_id = $1 AND deleted_at IS NULL"
    args: list[Any] = [store_id]
    if path_prefix:
        args.append(path_prefix)
        where += f" AND (path = ${len(args)} OR path LIKE ${len(args)} || '%')"
    order_sql = "path ASC" if order_by == "path" else "created_at DESC"
    rows = await conn.fetch(f"SELECT * FROM memories WHERE {where} ORDER BY {order_sql}", *args)

    memories = [_row_to_memory(r, include_content=False) for r in rows]
    if depth is None:
        return list(memories)

    base = path_prefix.rstrip("/") if path_prefix else ""
    out: list[Memory | MemoryPrefix] = []
    seen_prefixes: set[str] = set()
    for memory in memories:
        rest = memory.path[len(base) :] if memory.path.startswith(base) else memory.path
        # rest looks like "/segment/segment/file"; strip the leading "/" and split
        parts = rest.lstrip("/").split("/")
        if len(parts) <= depth:
            out.append(memory)
            continue
        prefix_path = base + "/" + "/".join(parts[:depth]) + "/"
        if prefix_path in seen_prefixes:
            continue
        seen_prefixes.add(prefix_path)
        out.append(MemoryPrefix(path=prefix_path))
    return out


async def update_memory_with_version(
    conn: asyncpg.Connection[Any],
    *,
    store_id: str,
    memory_id: str,
    new_content: str | None,
    new_content_sha256: str | None,
    new_path: str | None,
    precondition_sha256: str | None,
    actor_type: str,
    actor_ref: str,
) -> Memory:
    """Update content and/or path; record a ``modified`` version.

    Precondition (when set) is content-only — renames are unconditional,
    matching Anthropic's wire semantics. If both content and path are None
    the call is a no-op and returns the current row.
    """
    if new_content is None and new_path is None:
        return await get_memory(conn, store_id, memory_id, include_content=False)

    next_path_for_conflict: str | None = None
    try:
        async with conn.transaction():
            cur = await conn.fetchrow(
                "SELECT * FROM memories WHERE memory_store_id = $1 AND id = $2 "
                "AND deleted_at IS NULL FOR UPDATE",
                store_id,
                memory_id,
            )
            if cur is None:
                raise NotFoundError(
                    f"memory {memory_id} not found in store {store_id}",
                    detail={"id": memory_id, "memory_store_id": store_id},
                )

            if (
                precondition_sha256 is not None
                and new_content is not None
                and cur["content_sha256"] != precondition_sha256
            ):
                raise MemoryPreconditionFailedError(
                    "precondition content_sha256 failed: content has changed",
                    detail={
                        "expected": precondition_sha256,
                        "actual": cur["content_sha256"],
                    },
                )

            next_content: str = new_content if new_content is not None else cur["content"]
            next_sha: str = (
                new_content_sha256 if new_content_sha256 is not None else cur["content_sha256"]
            )
            next_size: int = len(next_content.encode("utf-8"))
            next_path: str = new_path if new_path is not None else cur["path"]
            next_path_for_conflict = next_path

            seq = await _allocate_version_seq(conn, store_id)
            version_id = make_id(MEMORY_VERSION)
            await conn.execute(
                """
                INSERT INTO memory_versions
                    (id, memory_store_id, memory_id, seq, operation, path,
                     content, content_sha256, content_size_bytes,
                     created_by_type, created_by_ref)
                VALUES ($1, $2, $3, $4, 'modified', $5, $6, $7, $8, $9, $10)
                """,
                version_id,
                store_id,
                memory_id,
                seq,
                next_path,
                next_content,
                next_sha,
                next_size,
                actor_type,
                actor_ref,
            )

            row = await conn.fetchrow(
                "UPDATE memories SET content = $1, content_sha256 = $2, "
                "content_size_bytes = $3, path = $4, current_version_id = $5, "
                "updated_at = now() "
                "WHERE memory_store_id = $6 AND id = $7 RETURNING *",
                next_content,
                next_sha,
                next_size,
                next_path,
                version_id,
                store_id,
                memory_id,
            )
    except asyncpg.UniqueViolationError as exc:
        assert next_path_for_conflict is not None
        existing = await conn.fetchrow(
            "SELECT id FROM memories WHERE memory_store_id = $1 AND path = $2 "
            "AND id != $3 AND deleted_at IS NULL",
            store_id,
            next_path_for_conflict,
            memory_id,
        )
        conflicting_id = existing["id"] if existing is not None else None
        raise MemoryPathConflictError(
            f"path {next_path_for_conflict!r} is already used by {conflicting_id!r}",
            detail={
                "conflicting_memory_id": conflicting_id,
                "conflicting_path": next_path_for_conflict,
            },
        ) from exc
    assert row is not None
    return _row_to_memory(row, include_content=False)


async def delete_memory_with_version(
    conn: asyncpg.Connection[Any],
    *,
    store_id: str,
    memory_id: str,
    actor_type: str,
    actor_ref: str,
) -> None:
    """Soft-delete: tombstone version row + ``deleted_at`` on the memory."""
    async with conn.transaction():
        cur = await conn.fetchrow(
            "SELECT path FROM memories WHERE memory_store_id = $1 AND id = $2 "
            "AND deleted_at IS NULL FOR UPDATE",
            store_id,
            memory_id,
        )
        if cur is None:
            raise NotFoundError(
                f"memory {memory_id} not found in store {store_id}",
                detail={"id": memory_id, "memory_store_id": store_id},
            )

        seq = await _allocate_version_seq(conn, store_id)
        version_id = make_id(MEMORY_VERSION)
        await conn.execute(
            """
            INSERT INTO memory_versions
                (id, memory_store_id, memory_id, seq, operation, path,
                 created_by_type, created_by_ref)
            VALUES ($1, $2, $3, $4, 'deleted', $5, $6, $7)
            """,
            version_id,
            store_id,
            memory_id,
            seq,
            cur["path"],
            actor_type,
            actor_ref,
        )

        await conn.execute(
            "UPDATE memories SET deleted_at = now(), updated_at = now() "
            "WHERE memory_store_id = $1 AND id = $2",
            store_id,
            memory_id,
        )


# Versions ─────────────────────────────────────────────────────────────────


async def list_memory_versions(
    conn: asyncpg.Connection[Any],
    store_id: str,
    *,
    memory_id: str | None = None,
    limit: int = 100,
) -> list[MemoryVersion]:
    where = "memory_store_id = $1"
    args: list[Any] = [store_id]
    if memory_id is not None:
        args.append(memory_id)
        where += f" AND memory_id = ${len(args)}"
    args.append(limit)
    rows = await conn.fetch(
        f"SELECT * FROM memory_versions WHERE {where} ORDER BY created_at DESC LIMIT ${len(args)}",
        *args,
    )
    return [_row_to_memory_version(r, include_content=False) for r in rows]


async def get_memory_version(
    conn: asyncpg.Connection[Any], store_id: str, version_id: str
) -> MemoryVersion:
    row = await conn.fetchrow(
        "SELECT * FROM memory_versions WHERE memory_store_id = $1 AND id = $2",
        store_id,
        version_id,
    )
    if row is None:
        raise NotFoundError(
            f"memory version {version_id} not found",
            detail={"id": version_id, "memory_store_id": store_id},
        )
    return _row_to_memory_version(row, include_content=True)


async def redact_memory_version(
    conn: asyncpg.Connection[Any],
    *,
    store_id: str,
    version_id: str,
    actor_type: str,
    actor_ref: str,
) -> MemoryVersion:
    """Strip content fields from a version while keeping the audit trail.

    Rejects redacting the current head of a live (non-deleted) memory:
    write a new version first, or delete the parent memory.
    """
    async with conn.transaction():
        ver = await conn.fetchrow(
            "SELECT * FROM memory_versions WHERE memory_store_id = $1 AND id = $2 FOR UPDATE",
            store_id,
            version_id,
        )
        if ver is None:
            raise NotFoundError(
                f"memory version {version_id} not found",
                detail={"id": version_id, "memory_store_id": store_id},
            )

        head_check = await conn.fetchrow(
            "SELECT 1 FROM memories WHERE memory_store_id = $1 AND id = $2 "
            "AND current_version_id = $3 AND deleted_at IS NULL",
            store_id,
            ver["memory_id"],
            version_id,
        )
        if head_check is not None:
            raise ConflictError(
                "this version is the live head; write a new version first, "
                "or delete the memory to make all versions redactable",
                detail={"id": version_id, "memory_id": ver["memory_id"]},
            )

        if ver["redacted_at"] is not None:
            return _row_to_memory_version(ver, include_content=False)

        row = await conn.fetchrow(
            "UPDATE memory_versions SET path = NULL, content = NULL, "
            "content_sha256 = NULL, content_size_bytes = NULL, "
            "redacted_at = now(), redacted_by_type = $1, redacted_by_ref = $2 "
            "WHERE memory_store_id = $3 AND id = $4 RETURNING *",
            actor_type,
            actor_ref,
            store_id,
            version_id,
        )
    assert row is not None
    return _row_to_memory_version(row, include_content=False)


# Session attachment ───────────────────────────────────────────────────────


async def attach_memory_stores_to_session(
    conn: asyncpg.Connection[Any],
    session_id: str,
    resources: list[MemoryStoreResource],
) -> None:
    """Insert ``session_memory_stores`` rows for each resource, snapshotting
    name + description from the parent store at attach time. Validates that
    every referenced store exists and is non-archived; rejects duplicate
    snapshotted names (mount-path collision)."""
    if not resources:
        return
    seen_names: set[str] = set()
    for rank, res in enumerate(resources):
        store = await get_memory_store(conn, res.memory_store_id, allow_archived=False)
        if store.name in seen_names:
            raise ConflictError(
                f"two attached memory stores share the name {store.name!r}; "
                "rename one before attaching",
                detail={
                    "memory_store_id": res.memory_store_id,
                    "conflicting_name": store.name,
                },
            )
        seen_names.add(store.name)
        await conn.execute(
            """
            INSERT INTO session_memory_stores
                (session_id, memory_store_id, rank, access, instructions,
                 name_at_attach, description_at_attach)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            session_id,
            res.memory_store_id,
            rank,
            res.access,
            res.instructions,
            store.name,
            store.description,
        )


async def list_session_memory_store_echoes(
    conn: asyncpg.Connection[Any], session_id: str
) -> list[MemoryStoreResourceEcho]:
    rows = await conn.fetch(
        "SELECT * FROM session_memory_stores WHERE session_id = $1 ORDER BY rank",
        session_id,
    )
    return [
        MemoryStoreResourceEcho(
            memory_store_id=r["memory_store_id"],
            access=r["access"],
            instructions=r["instructions"],
            name=r["name_at_attach"],
            description=r["description_at_attach"],
            mount_path=f"/mnt/memory/{r['name_at_attach']}",
        )
        for r in rows
    ]
