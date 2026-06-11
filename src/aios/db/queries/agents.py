"""Agent queries.

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from aios.db.queries import (
    _archive_scoped,
    _get_scoped,
    _list_scoped,
    parse_jsonb,
)
from aios.errors import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from aios.ids import (
    AGENT,
    make_id,
)
from aios.models.agents import Agent, AgentVersion, HttpServerSpec, McpServerSpec, ToolSpec
from aios.models.skills import AgentSkillRef

# ─── agents ───────────────────────────────────────────────────────────────────


def _row_to_agent(row: asyncpg.Record) -> Agent:
    tools_data = parse_jsonb(row["tools"])
    skills_data = parse_jsonb(row["skills"])
    mcp_data = parse_jsonb(row.get("mcp_servers", []))
    http_data = parse_jsonb(row.get("http_servers", []))
    metadata = parse_jsonb(row["metadata"])
    litellm_extra = parse_jsonb(row["litellm_extra"])
    return Agent(
        id=row["id"],
        version=row["version"],
        name=row["name"],
        model=row["model"],
        system=row["system"],
        tools=[ToolSpec.model_validate(t) for t in tools_data],
        skills=[AgentSkillRef.model_validate(s) for s in skills_data],
        mcp_servers=[McpServerSpec.model_validate(s) for s in (mcp_data or [])],
        http_servers=[HttpServerSpec.model_validate(s) for s in (http_data or [])],
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
    tools_data = parse_jsonb(row["tools"])
    skills_data = parse_jsonb(row["skills"])
    mcp_data = parse_jsonb(row.get("mcp_servers", []))
    http_data = parse_jsonb(row.get("http_servers", []))
    litellm_extra = parse_jsonb(row["litellm_extra"])
    return AgentVersion(
        agent_id=row["agent_id"],
        version=row["version"],
        model=row["model"],
        system=row["system"],
        tools=[ToolSpec.model_validate(t) for t in tools_data],
        skills=[AgentSkillRef.model_validate(s) for s in skills_data],
        mcp_servers=[McpServerSpec.model_validate(s) for s in (mcp_data or [])],
        http_servers=[HttpServerSpec.model_validate(s) for s in (http_data or [])],
        litellm_extra=litellm_extra or {},
        window_min=row["window_min"],
        window_max=row["window_max"],
        created_at=row["created_at"],
    )


async def insert_agent(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    name: str,
    model: str,
    system: str,
    tools: list[ToolSpec],
    skills_json: str = "[]",
    mcp_servers: list[McpServerSpec],
    http_servers: list[HttpServerSpec],
    description: str | None,
    metadata: dict[str, Any],
    litellm_extra: dict[str, Any],
    window_min: int,
    window_max: int,
) -> Agent:
    if window_min >= window_max:
        raise ValidationError(
            "window_min must be strictly less than window_max",
            detail={"window_min": window_min, "window_max": window_max},
        )
    new_id = make_id(AGENT)
    tools_json = json.dumps([t.model_dump() for t in tools])
    mcp_json = json.dumps([s.model_dump() for s in mcp_servers])
    http_json = json.dumps([s.model_dump() for s in http_servers])
    metadata_json = json.dumps(metadata)
    extra_json = json.dumps(litellm_extra)
    try:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO agents (
                    id, name, model, system, tools, skills, mcp_servers, http_servers,
                    description, metadata, litellm_extra,
                    window_min, window_max, version, account_id
                )
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb,
                        $9, $10::jsonb, $11::jsonb, $12, $13, 1, $14)
                RETURNING *
                """,
                new_id,
                name,
                model,
                system,
                tools_json,
                skills_json,
                mcp_json,
                http_json,
                description,
                metadata_json,
                extra_json,
                window_min,
                window_max,
                account_id,
            )
            assert row is not None
            # Snapshot version 1 into agent_versions.
            await conn.execute(
                """
                INSERT INTO agent_versions (
                    agent_id, version, model, system, tools, skills, mcp_servers, http_servers,
                    litellm_extra, window_min, window_max, account_id
                )
                VALUES ($1, 1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7::jsonb,
                        $8::jsonb, $9, $10, $11)
                """,
                new_id,
                model,
                system,
                tools_json,
                skills_json,
                mcp_json,
                http_json,
                extra_json,
                window_min,
                window_max,
                account_id,
            )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"an agent named {name!r} already exists",
            detail={"name": name},
        ) from exc
    return _row_to_agent(row)


async def get_agent(conn: asyncpg.Connection[Any], agent_id: str, *, account_id: str) -> Agent:
    return await _get_scoped(
        conn,
        table="agents",
        id_=agent_id,
        account_id=account_id,
        row=_row_to_agent,
        noun="agent",
    )


async def list_agents(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    limit: int = 50,
    after: str | None = None,
    name: str | None = None,
) -> list[Agent]:
    return await _list_scoped(
        conn,
        table="agents",
        account_id=account_id,
        row=_row_to_agent,
        limit=limit,
        after=after,
        filters=[("name", name)],
    )


async def archive_agent(conn: asyncpg.Connection[Any], agent_id: str, *, account_id: str) -> None:
    await _archive_scoped(
        conn,
        table="agents",
        id_=agent_id,
        account_id=account_id,
        noun="agent",
    )


async def update_agent(
    conn: asyncpg.Connection[Any],
    agent_id: str,
    *,
    account_id: str,
    expected_version: int,
    name: str | None = None,
    model: str | None = None,
    system: str | None = None,
    tools: list[ToolSpec] | None = None,
    skills_json: str | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    http_servers: list[HttpServerSpec] | None = None,
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
    # ``get_agent`` reads ``current`` for the merge below and for the
    # archived check. The version match is *not* pre-checked here — the
    # in-transaction UPDATE's ``AND version = $expected_version`` is the
    # authoritative serialization point; pre-checking would just emit a
    # different error message for the obvious-stale case while letting
    # the racy-concurrent case slip past, which was the prior bug.
    current = await get_agent(conn, agent_id, account_id=account_id)
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
    new_http = http_servers if http_servers is not None else current.http_servers
    new_desc = description if description is not None else current.description
    new_meta = metadata if metadata is not None else current.metadata
    new_extra = litellm_extra if litellm_extra is not None else current.litellm_extra
    new_wmin = window_min if window_min is not None else current.window_min
    new_wmax = window_max if window_max is not None else current.window_max
    if new_wmin >= new_wmax:
        # Partial-merge semantics: a one-sided update (e.g. ``window_max``
        # alone, set at-or-below the current ``window_min``) only
        # produces an invalid pair AFTER the merge, so the check has to
        # live on the resolved values rather than the input kwargs.
        # Without this, the row commits and the next session step
        # ``ZeroDivisionError``s in :func:`aios.harness.tokens.tokens_to_drop`.
        raise ValidationError(
            "window_min must be strictly less than window_max",
            detail={"window_min": new_wmin, "window_max": new_wmax},
        )

    # No-op detection.
    if (
        new_name == current.name
        and new_model == current.model
        and new_system == current.system
        and new_tools == current.tools
        and new_skills_json == cur_skills_json
        and new_mcp == current.mcp_servers
        and new_http == current.http_servers
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
    http_json = json.dumps([s.model_dump() for s in new_http])
    meta_json = json.dumps(new_meta)
    extra_json = json.dumps(new_extra)

    async with conn.transaction():
        # ``AND version = $expected_version`` is the authoritative
        # optimistic-concurrency check: the pre-transaction guard above
        # is racy because two writers can both read the same
        # ``current.version`` and both pass. Putting the version in the
        # UPDATE ``WHERE`` lets the DB serialize the writers — exactly
        # one matches and bumps to ``new_version``; the other matches
        # zero rows and we raise ConflictError. Without this, the
        # subsequent ``INSERT INTO agent_versions`` collides on
        # ``agent_versions_pkey`` and leaks ``UniqueViolationError``
        # as HTTP 500 to the loser instead of a clean 409.
        row = await conn.fetchrow(
            """
            UPDATE agents
               SET version = $2, name = $3, model = $4, system = $5,
                   tools = $6::jsonb, skills = $7::jsonb, mcp_servers = $8::jsonb,
                   http_servers = $9::jsonb,
                   description = $10, metadata = $11::jsonb,
                   litellm_extra = $12::jsonb,
                   window_min = $13, window_max = $14,
                   updated_at = now()
             WHERE id = $1 AND account_id = $15 AND version = $16
               AND archived_at IS NULL
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
            http_json,
            new_desc,
            meta_json,
            extra_json,
            new_wmin,
            new_wmax,
            account_id,
            expected_version,
        )
        if row is None:
            # No row matched the (id, account_id, version, NOT archived)
            # tuple.  Re-read to distinguish the three possible causes:
            # racing archive, stale ``expected_version``, or concurrent
            # version bump.  READ COMMITTED means this statement-level
            # snapshot sees the concurrent writer's committed state.
            fresh = await get_agent(conn, agent_id, account_id=account_id)
            if fresh.archived_at is not None:
                raise ConflictError(f"agent {agent_id} is archived", detail={"id": agent_id})
            raise ConflictError(
                f"version mismatch: expected {expected_version}, current is {fresh.version}",
                detail={
                    "expected": expected_version,
                    "current": fresh.version,
                    "id": agent_id,
                },
            )
        await conn.execute(
            """
            INSERT INTO agent_versions (
                agent_id, version, model, system, tools, skills, mcp_servers, http_servers,
                litellm_extra, window_min, window_max, account_id
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb,
                    $9::jsonb, $10, $11, $12)
            """,
            agent_id,
            new_version,
            new_model,
            new_system,
            tools_json,
            new_skills_json,
            mcp_json,
            http_json,
            extra_json,
            new_wmin,
            new_wmax,
            account_id,
        )
    return _row_to_agent(row)


async def get_agent_version(
    conn: asyncpg.Connection[Any],
    agent_id: str,
    version: int,
    *,
    account_id: str,
) -> AgentVersion:
    row = await conn.fetchrow(
        "SELECT * FROM agent_versions WHERE agent_id = $1 AND version = $2 AND account_id = $3",
        agent_id,
        version,
        account_id,
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
    account_id: str,
    limit: int = 50,
    after: int | None = None,
) -> list[AgentVersion]:
    """List versions in descending order (newest first)."""
    if after is None:
        rows = await conn.fetch(
            "SELECT * FROM agent_versions WHERE agent_id = $1 AND account_id = $2 "
            "ORDER BY version DESC LIMIT $3",
            agent_id,
            account_id,
            limit,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM agent_versions WHERE agent_id = $1 AND version < $2 "
            "AND account_id = $3 ORDER BY version DESC LIMIT $4",
            agent_id,
            after,
            account_id,
            limit,
        )
    return [_row_to_agent_version(r) for r in rows]
