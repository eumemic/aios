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
from types import EllipsisType
from typing import Any

import asyncpg

from aios.crypto.vault import EncryptedBlob
from aios.errors import ConflictError, NotFoundError
from aios.ids import (
    AGENT,
    CHANNEL_BINDING,
    CONNECTION,
    ENVIRONMENT,
    EVENT,
    ROUTING_RULE,
    SESSION,
    SKILL,
    VAULT,
    VAULT_CREDENTIAL,
    make_id,
)
from aios.models.agents import Agent, AgentVersion, McpServerSpec, ToolSpec
from aios.models.channel_bindings import ChannelBinding
from aios.models.connections import Connection
from aios.models.environments import Environment, EnvironmentConfig
from aios.models.events import Event, EventKind
from aios.models.routing_rules import RoutingRule, SessionParams
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
) -> Session:
    """Insert a fresh session row.

    ``workspace_path`` defaults to ``settings.workspace_root / session_id``.
    Caller sets up vault bindings via :func:`set_session_vaults` after.
    Raises :class:`NotFoundError` if either the agent or environment FK
    is unsatisfied.
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
                status, workspace_volume_path, env
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, 'idle', $7, $8::jsonb)
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


async def model_token_ratio(
    conn: asyncpg.Connection[Any],
    model: str,
    *,
    n: int = 100,
) -> float:
    """Per-model actual/local token correction.

    Returns ``SUM(actual) / SUM(local)`` over the most recent ``n``
    successful ``model_request_end`` spans for ``model``.  Below ``n``
    samples: returns ``1.0`` — R is too noisy to trust and applying it
    would churn the prefix cache.  At or above ``n``: aggregates exactly
    the last ``n`` samples.  The single parameter governs both the
    activation threshold and the sliding-window size, so per-step drift
    in R is bounded by the window itself.

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
    row = await conn.fetchrow(
        """
        WITH recent AS (
            SELECT
                (data->'model_usage'->>'input_tokens')::bigint AS it,
                (data->>'local_tokens')::bigint                 AS lt
            FROM events
            WHERE kind = 'span'
              AND data->>'event' = 'model_request_end'
              AND (data->>'is_error')::boolean = false
              AND data->>'model' = $1
              AND data ? 'local_tokens'
              AND (data->>'local_tokens')::bigint > 0
            ORDER BY seq DESC
            LIMIT $2
        )
        SELECT
            COUNT(*)                                AS k,
            COALESCE(SUM(it), 0)::bigint            AS total_actual,
            COALESCE(SUM(lt), 0)::bigint            AS total_local
        FROM recent
        """,
        model,
        n,
    )
    assert row is not None
    if row["k"] < n:
        return 1.0
    return float(row["total_actual"]) / float(row["total_local"])


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


async def read_windowed_events(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    window_min: int,
    window_max: int,
    model: str,
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

    ``model`` must be the session's currently-active mind string —
    ``agent.model`` on the session's pinned agent/version.  The same
    string is what :func:`~aios.harness.loop.run_session_step` stamps on
    ``model_request_end`` spans, so stamp-side and query-side stay
    partitioned on identical keys.

    Prefix-cache invariant: the plain chunked-snap algorithm gave a
    *strict* guarantee of byte-identical prompt prefix within a snap
    chunk.  With the ratio correction this weakens to a
    *quantitatively-bounded* guarantee: R can shift slightly between
    consecutive reads as new calibration samples land, which can nudge
    ``drop_local`` across an event boundary and invalidate the prefix
    cache for that turn.  With ``n=100`` samples, per-step drift in R is
    <1 % for the models we've measured, so the expected invalidation rate
    is well below Anthropic's ~5-minute cache TTL — accept-the-noise
    tradeoff documented here for the next reader.  Revisit the ``n``
    default in :func:`model_token_ratio` if a newly-onboarded model
    shows per-sample CV above ~5 %, or if prefix-cache invalidation ever
    shows up in telemetry for a steady-state workload.

    Falls back to :func:`read_message_events` (loading all events) when
    cumulative data is not available (pre-backfill sessions or rolling
    deploys) or when the entire session fits within ``window_max``.
    """
    # Index seek: total cumulative tokens from the latest message event.
    total = await _latest_cumulative_tokens(conn, session_id)

    # Fallback: no cumulative data yet — load everything.
    if total is None:
        return await read_message_events(conn, session_id)

    # Skip the ratio lookup when the session cannot possibly need a drop.
    # ``total <= window_min`` guarantees ``total * R <= window_max`` for
    # any ``R <= window_max / window_min`` — a ceiling of 3.0 for the
    # default 50k/150k config, well above any measured per-model ratio.
    # Saves one DB query on the common small-session path.
    if total <= window_min:
        return await read_message_events(conn, session_id)

    ratio = await model_token_ratio(conn, model)

    from aios.harness.tokens import tokens_to_drop

    # Forward-convert local → effective with plain rounding: best-estimate
    # of the provider-token total.  Back-convert effective → local with
    # ceil: deliberately asymmetric so the post-drop remaining fits under
    # ``window_max`` even when ratio error would otherwise leave one
    # message straddling the boundary.
    total_effective = round(total * ratio)
    drop_effective = tokens_to_drop(total_effective, window_min=window_min, window_max=window_max)
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


def _row_to_connection(row: asyncpg.Record) -> Connection:
    raw_metadata = row["metadata"]
    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
    return Connection(
        id=row["id"],
        connector=row["connector"],
        account=row["account"],
        mcp_url=row["mcp_url"],
        vault_id=row["vault_id"],
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def insert_connection(
    conn: asyncpg.Connection[Any],
    *,
    connector: str,
    account: str,
    mcp_url: str,
    vault_id: str,
    metadata: dict[str, Any],
) -> Connection:
    new_id = make_id(CONNECTION)
    metadata_json = json.dumps(metadata)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO connections (id, connector, account, mcp_url, vault_id, metadata)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            RETURNING *
            """,
            new_id,
            connector,
            account,
            mcp_url,
            vault_id,
            metadata_json,
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"a connection for ({connector!r}, {account!r}) already exists",
            detail={"connector": connector, "account": account},
        ) from exc
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            f"vault {vault_id} not found",
            detail={"vault_id": vault_id},
        ) from exc
    assert row is not None
    return _row_to_connection(row)


async def get_connection(conn: asyncpg.Connection[Any], connection_id: str) -> Connection:
    row = await conn.fetchrow("SELECT * FROM connections WHERE id = $1", connection_id)
    if row is None:
        raise NotFoundError(
            f"connection {connection_id} not found",
            detail={"id": connection_id},
        )
    return _row_to_connection(row)


async def list_connections(
    conn: asyncpg.Connection[Any], *, limit: int = 50, after: str | None = None
) -> list[Connection]:
    if after is None:
        rows = await conn.fetch(
            "SELECT * FROM connections WHERE archived_at IS NULL ORDER BY id DESC LIMIT $1",
            limit,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM connections WHERE archived_at IS NULL AND id < $1 "
            "ORDER BY id DESC LIMIT $2",
            after,
            limit,
        )
    return [_row_to_connection(r) for r in rows]


async def update_connection(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    *,
    mcp_url: str | None = None,
    vault_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Connection:
    sets: list[str] = []
    args: list[Any] = [connection_id]
    if mcp_url is not None:
        args.append(mcp_url)
        sets.append(f"mcp_url = ${len(args)}")
    if vault_id is not None:
        args.append(vault_id)
        sets.append(f"vault_id = ${len(args)}")
    if metadata is not None:
        args.append(json.dumps(metadata))
        sets.append(f"metadata = ${len(args)}::jsonb")
    if not sets:
        return await get_connection(conn, connection_id)
    sets.append("updated_at = now()")
    sql = f"UPDATE connections SET {', '.join(sets)} WHERE id = $1 RETURNING *"
    try:
        row = await conn.fetchrow(sql, *args)
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            "vault not found",
            detail={"vault_id": vault_id},
        ) from exc
    if row is None:
        raise NotFoundError(
            f"connection {connection_id} not found",
            detail={"id": connection_id},
        )
    return _row_to_connection(row)


async def list_connections_by_ids(
    conn: asyncpg.Connection[Any],
    ids: list[str],
) -> list[Connection]:
    """Active connections with the given ``ids``.  Empty input → no roundtrip.

    Results are ordered by ``c.id`` so callers can feed them into the
    system prompt in a stable order — prompt-cache stability depends
    on it.  Used by :func:`aios.harness.channels.list_bindings_and_connections`
    where the ``ids`` come from the distinct ``binding.connection_id``
    values of a session's bindings.
    """
    if not ids:
        return []
    rows = await conn.fetch(
        "SELECT * FROM connections WHERE id = ANY($1::text[]) AND archived_at IS NULL ORDER BY id",
        ids,
    )
    return [_row_to_connection(r) for r in rows]


async def get_connections_by_pairs(
    conn: asyncpg.Connection[Any],
    pairs: list[tuple[str, str]],
) -> list[Connection]:
    """Active connections where ``(connector, account)`` is in ``pairs``.

    Empty input → no roundtrip.  Results are ordered by ``c.id`` so the
    caller can feed them into the system prompt in a stable order — a
    prerequisite for prompt-cache stability across steps, since the
    caller-side ``pairs`` are typically built from a set.
    """
    if not pairs:
        return []
    connectors = [p[0] for p in pairs]
    accounts = [p[1] for p in pairs]
    rows = await conn.fetch(
        """
        SELECT c.*
          FROM connections c
          JOIN unnest($1::text[], $2::text[]) AS p(connector, account)
            ON c.connector = p.connector AND c.account = p.account
         WHERE c.archived_at IS NULL
         ORDER BY c.id
        """,
        connectors,
        accounts,
    )
    return [_row_to_connection(r) for r in rows]


async def get_connection_vault_for_url(
    conn: asyncpg.Connection[Any], mcp_server_url: str
) -> str | None:
    """Vault id of the active connection owning ``mcp_server_url``, else ``None``."""
    val: str | None = await conn.fetchval(
        "SELECT vault_id FROM connections WHERE mcp_url = $1 AND archived_at IS NULL LIMIT 1",
        mcp_server_url,
    )
    return val


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


async def count_active_bindings_for_connection(
    conn: asyncpg.Connection[Any], connection_id: str
) -> int:
    """Count active channel bindings owned by this connection.  Used to
    block connection archival while sessions are still reachable via
    those bindings — archiving the connection would silently drop the
    connection-provided MCP tools from any live session.
    """
    val: int = await conn.fetchval(
        "SELECT COUNT(*) FROM channel_bindings WHERE connection_id = $1 AND archived_at IS NULL",
        connection_id,
    )
    return val


# ─── channel bindings ───────────────────────────────────────────────────────
#
# The display ``address`` is computed on read by joining connections —
# storage is normalized as ``(connection_id, path)``.  All selects go
# through ``_BINDING_SELECT`` which tacks the connector/account columns
# onto the row, consumed by ``_row_to_channel_binding`` to rebuild
# ``address`` for the API response.

_BINDING_SELECT = """
    SELECT cb.*, c.connector, c.account
      FROM channel_bindings cb
      JOIN connections c ON c.id = cb.connection_id
"""


def _row_to_channel_binding(row: asyncpg.Record) -> ChannelBinding:
    address = f"{row['connector']}/{row['account']}/{row['path']}"
    return ChannelBinding(
        id=row["id"],
        connection_id=row["connection_id"],
        path=row["path"],
        address=address,
        session_id=row["session_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
        notification_mode=row["notification_mode"],
    )


async def insert_binding(
    conn: asyncpg.Connection[Any],
    *,
    connection_id: str,
    path: str,
    session_id: str,
) -> ChannelBinding:
    """Insert a ``(connection_id, path) → session_id`` binding.

    Returns the fully-populated read view (address reconstructed from the
    owning connection).  A CTE does the insert + connection join in one
    roundtrip.
    """
    new_id = make_id(CHANNEL_BINDING)
    try:
        row = await conn.fetchrow(
            """
            WITH inserted AS (
                INSERT INTO channel_bindings (id, connection_id, path, session_id)
                VALUES ($1, $2, $3, $4)
                RETURNING *
            )
            SELECT i.*, c.connector, c.account
              FROM inserted i
              JOIN connections c ON c.id = i.connection_id
            """,
            new_id,
            connection_id,
            path,
            session_id,
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"an active binding for path {path!r} on connection {connection_id} already exists",
            detail={"connection_id": connection_id, "path": path},
        ) from exc
    except asyncpg.ForeignKeyViolationError as exc:
        # Could be either the session FK or the connection FK — look up to
        # produce the correct message.  (The common case is a bad
        # ``session_id``; ``connection_id`` is resolved by the service layer
        # before the insert.)
        raise NotFoundError(
            "session or connection not found for binding",
            detail={"connection_id": connection_id, "session_id": session_id},
        ) from exc
    assert row is not None
    return _row_to_channel_binding(row)


async def get_binding(conn: asyncpg.Connection[Any], binding_id: str) -> ChannelBinding:
    row = await conn.fetchrow(_BINDING_SELECT + " WHERE cb.id = $1", binding_id)
    if row is None:
        raise NotFoundError(
            f"channel binding {binding_id} not found",
            detail={"id": binding_id},
        )
    return _row_to_channel_binding(row)


async def get_binding_by_connection_and_path(
    conn: asyncpg.Connection[Any], connection_id: str, path: str
) -> ChannelBinding | None:
    """Fast-path lookup used by the channel resolver (binding hit tier)."""
    row = await conn.fetchrow(
        _BINDING_SELECT
        + " WHERE cb.connection_id = $1 AND cb.path = $2 AND cb.archived_at IS NULL",
        connection_id,
        path,
    )
    if row is None:
        return None
    return _row_to_channel_binding(row)


async def list_bindings(
    conn: asyncpg.Connection[Any],
    *,
    session_id: str | None = None,
    limit: int = 50,
    after: str | None = None,
) -> list[ChannelBinding]:
    clauses: list[str] = ["cb.archived_at IS NULL"]
    args: list[Any] = []
    if session_id is not None:
        args.append(session_id)
        clauses.append(f"cb.session_id = ${len(args)}")
    if after is not None:
        args.append(after)
        clauses.append(f"cb.id < ${len(args)}")
    args.append(limit)
    sql = _BINDING_SELECT + f" WHERE {' AND '.join(clauses)} ORDER BY cb.id DESC LIMIT ${len(args)}"
    rows = await conn.fetch(sql, *args)
    return [_row_to_channel_binding(r) for r in rows]


async def list_session_bindings(
    conn: asyncpg.Connection[Any], session_id: str
) -> list[ChannelBinding]:
    """Every active binding for ``session_id``, unpaginated.

    The step function consumes this in one shot (see
    :func:`aios.harness.channels.list_bindings_and_connections`).
    """
    rows = await conn.fetch(
        _BINDING_SELECT + " WHERE cb.session_id = $1 AND cb.archived_at IS NULL ORDER BY cb.id",
        session_id,
    )
    return [_row_to_channel_binding(r) for r in rows]


async def archive_binding(conn: asyncpg.Connection[Any], binding_id: str) -> ChannelBinding:
    row = await conn.fetchrow(
        """
        WITH updated AS (
            UPDATE channel_bindings
               SET archived_at = now(), updated_at = now()
             WHERE id = $1 AND archived_at IS NULL
            RETURNING *
        )
        SELECT u.*, c.connector, c.account
          FROM updated u
          JOIN connections c ON c.id = u.connection_id
        """,
        binding_id,
    )
    if row is None:
        raise NotFoundError(
            f"channel binding {binding_id} not found or already archived",
            detail={"id": binding_id},
        )
    return _row_to_channel_binding(row)


# ─── routing rules ──────────────────────────────────────────────────────────


def _row_to_routing_rule(row: asyncpg.Record) -> RoutingRule:
    raw_params = row["session_params"]
    params_data = json.loads(raw_params) if isinstance(raw_params, str) else raw_params
    return RoutingRule(
        id=row["id"],
        connection_id=row["connection_id"],
        prefix=row["prefix"],
        target=row["target"],
        session_params=SessionParams.model_validate(params_data),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def insert_routing_rule(
    conn: asyncpg.Connection[Any],
    *,
    connection_id: str,
    prefix: str,
    target: str,
    session_params: SessionParams,
) -> RoutingRule:
    new_id = make_id(ROUTING_RULE)
    params_json = json.dumps(session_params.model_dump())
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO routing_rules (id, connection_id, prefix, target, session_params)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            RETURNING *
            """,
            new_id,
            connection_id,
            prefix,
            target,
            params_json,
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"an active routing rule for prefix {prefix!r} on connection "
            f"{connection_id} already exists",
            detail={"connection_id": connection_id, "prefix": prefix},
        ) from exc
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            f"connection {connection_id} not found",
            detail={"connection_id": connection_id},
        ) from exc
    assert row is not None
    return _row_to_routing_rule(row)


async def get_routing_rule(
    conn: asyncpg.Connection[Any], connection_id: str, rule_id: str
) -> RoutingRule:
    """Get a rule scoped to the given connection.

    The connection scope in the URL is load-bearing: a 404 on
    ``connections/A/routing-rules/<rid-belonging-to-B>`` is correct
    behavior (rather than leaking connection A's knowledge of rule B).
    """
    row = await conn.fetchrow(
        "SELECT * FROM routing_rules WHERE id = $1 AND connection_id = $2",
        rule_id,
        connection_id,
    )
    if row is None:
        raise NotFoundError(
            f"routing rule {rule_id} not found on connection {connection_id}",
            detail={"id": rule_id, "connection_id": connection_id},
        )
    return _row_to_routing_rule(row)


async def list_routing_rules(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    *,
    limit: int = 50,
    after: str | None = None,
) -> list[RoutingRule]:
    clauses: list[str] = ["connection_id = $1", "archived_at IS NULL"]
    args: list[Any] = [connection_id]
    if after is not None:
        args.append(after)
        clauses.append(f"id < ${len(args)}")
    args.append(limit)
    sql = (
        f"SELECT * FROM routing_rules WHERE {' AND '.join(clauses)} "
        f"ORDER BY id DESC LIMIT ${len(args)}"
    )
    rows = await conn.fetch(sql, *args)
    return [_row_to_routing_rule(r) for r in rows]


async def update_routing_rule(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    rule_id: str,
    *,
    target: str | None = None,
    session_params: SessionParams | None = None,
) -> RoutingRule:
    sets: list[str] = []
    args: list[Any] = [rule_id, connection_id]
    if target is not None:
        args.append(target)
        sets.append(f"target = ${len(args)}")
    if session_params is not None:
        args.append(json.dumps(session_params.model_dump()))
        sets.append(f"session_params = ${len(args)}::jsonb")
    if not sets:
        return await get_routing_rule(conn, connection_id, rule_id)
    sets.append("updated_at = now()")
    sql = (
        f"UPDATE routing_rules SET {', '.join(sets)} "
        "WHERE id = $1 AND connection_id = $2 RETURNING *"
    )
    row = await conn.fetchrow(sql, *args)
    if row is None:
        raise NotFoundError(
            f"routing rule {rule_id} not found on connection {connection_id}",
            detail={"id": rule_id, "connection_id": connection_id},
        )
    return _row_to_routing_rule(row)


async def archive_routing_rule(
    conn: asyncpg.Connection[Any], connection_id: str, rule_id: str
) -> RoutingRule:
    row = await conn.fetchrow(
        "UPDATE routing_rules SET archived_at = now(), updated_at = now() "
        "WHERE id = $1 AND connection_id = $2 AND archived_at IS NULL RETURNING *",
        rule_id,
        connection_id,
    )
    if row is None:
        raise NotFoundError(
            f"routing rule {rule_id} not found on connection {connection_id} or already archived",
            detail={"id": rule_id, "connection_id": connection_id},
        )
    return _row_to_routing_rule(row)


async def find_matching_rule(
    conn: asyncpg.Connection[Any], connection_id: str, path: str
) -> RoutingRule | None:
    """Longest-matching segment-aware prefix lookup within a connection.

    Three disjuncts:

    * ``prefix = ''`` is the per-connection catch-all and sorts last
      under ``ORDER BY length DESC``.
    * ``path = prefix`` for exact-match rules (e.g. rule prefix ``chat-a``
      and inbound path ``chat-a``).
    * Segment-aware extension: the path begins with ``prefix || '/'``, so
      ``group/thread-1`` matches a rule of ``group`` but not ``grou``.
      Substring equality (not ``LIKE``) keeps ``%``/``_`` in prefixes
      literal.
    """
    row = await conn.fetchrow(
        """
        SELECT * FROM routing_rules
        WHERE archived_at IS NULL
          AND connection_id = $1
          AND (
              prefix = ''
              OR $2 = prefix
              OR (
                  length($2) > length(prefix)
                  AND substring($2, 1, length(prefix) + 1) = prefix || '/'
              )
          )
        ORDER BY length(prefix) DESC
        LIMIT 1
        """,
        connection_id,
        path,
    )
    if row is None:
        return None
    return _row_to_routing_rule(row)
