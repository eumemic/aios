"""Session-template queries.

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.
"""

from __future__ import annotations

import json
from types import EllipsisType
from typing import Any

import asyncpg

from aios.db import queries
from aios.db.queries import (
    _archive_scoped,
    _build_set_assignments,
    _get_scoped,
    _list_scoped,
)
from aios.errors import (
    ConflictError,
    NotFoundError,
)
from aios.ids import (
    SESSION_TEMPLATE,
    make_id,
)
from aios.models.session_templates import SessionTemplate

# ─── session_templates ──────────────────────────────────────────────────────


def _row_to_session_template(row: asyncpg.Record) -> SessionTemplate:
    raw_metadata = row["metadata"]
    metadata = raw_metadata
    return SessionTemplate(
        id=row["id"],
        name=row["name"],
        agent_id=row["agent_id"],
        agent_version=row["agent_version"],
        environment_id=row["environment_id"],
        vault_ids=list(row["vault_ids"] or []),
        memory_store_ids=list(row["memory_store_ids"] or []),
        metadata=metadata,
        archive_when_idle=row["archive_when_idle"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def insert_session_template(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    name: str,
    agent_id: str,
    environment_id: str,
    agent_version: int | None,
    vault_ids: list[str],
    memory_store_ids: list[str],
    metadata: dict[str, Any],
    archive_when_idle: bool = False,
) -> SessionTemplate:
    new_id = make_id(SESSION_TEMPLATE)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO session_templates
                (id, name, agent_id, agent_version, environment_id,
                 vault_ids, memory_store_ids, metadata, account_id, archive_when_idle)
            VALUES ($1, $2, $3, $4, $5, $6::text[], $7::text[], $8::jsonb, $9, $10)
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
            account_id,
            archive_when_idle,
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


async def get_session_template(
    conn: asyncpg.Connection[Any], template_id: str, *, account_id: str
) -> SessionTemplate:
    return await _get_scoped(
        conn,
        table="session_templates",
        id_=template_id,
        account_id=account_id,
        row=_row_to_session_template,
        noun="session template",
    )


async def list_session_templates(
    conn: asyncpg.Connection[Any], *, account_id: str, limit: int = 50, after: str | None = None
) -> list[SessionTemplate]:
    return await _list_scoped(
        conn,
        table="session_templates",
        account_id=account_id,
        row=_row_to_session_template,
        limit=limit,
        after=after,
    )


async def update_session_template(
    conn: asyncpg.Connection[Any],
    template_id: str,
    *,
    account_id: str,
    name: str | None = None,
    agent_id: str | None = None,
    agent_version: int | None | EllipsisType = ...,
    environment_id: str | None = None,
    vault_ids: list[str] | None = None,
    memory_store_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    archive_when_idle: bool | None = None,
) -> SessionTemplate:
    # Refuse updates to archived templates: the resolver's
    # ``_spawn_per_chat_session`` already drops inbounds that target an
    # archived template (``ResolveDrop.ARCHIVED_TEMPLATE``), so a
    # rewrite of an archived row has no downstream effect — but the
    # bare UPDATE below would still commit the new values and the
    # RETURNING-built response would lie back to the caller as if the
    # update took.  Mirrors the symmetric raise on archived rows in
    # ``update_agent`` / ``update_environment``.
    current = await queries.get_session_template(conn, template_id, account_id=account_id)
    if current.archived_at is not None:
        raise ConflictError(
            f"session template {template_id} is archived",
            detail={"id": template_id},
        )

    args: list[Any] = [template_id]
    fields: list[tuple[str, Any, str | None]] = []
    if name is not None:
        fields.append(("name", name, None))
    if agent_id is not None:
        fields.append(("agent_id", agent_id, None))
    if agent_version is not ...:
        fields.append(("agent_version", agent_version, None))
    if environment_id is not None:
        fields.append(("environment_id", environment_id, None))
    if vault_ids is not None:
        fields.append(("vault_ids", vault_ids, "text[]"))
    if memory_store_ids is not None:
        fields.append(("memory_store_ids", memory_store_ids, "text[]"))
    if metadata is not None:
        fields.append(("metadata", metadata, "jsonb"))
    if archive_when_idle is not None:
        fields.append(("archive_when_idle", archive_when_idle, None))
    sets = _build_set_assignments(fields, args)
    if not sets:
        return await queries.get_session_template(conn, template_id, account_id=account_id)
    sets.append("updated_at = now()")
    args.append(account_id)
    sql = (
        f"UPDATE session_templates SET {', '.join(sets)} "
        f"WHERE id = $1 AND account_id = ${len(args)} AND archived_at IS NULL RETURNING *"
    )
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
        raise ConflictError(
            f"session template {template_id} is archived",
            detail={"id": template_id},
        )
    return _row_to_session_template(row)


async def archive_session_template(
    conn: asyncpg.Connection[Any],
    template_id: str,
    *,
    account_id: str,
) -> SessionTemplate:
    """Soft-delete the template.  Already-spawned per_chat sessions keep
    working; new chat sessions on connections referencing this template
    will fail at the inbound handler.
    """
    row = await _archive_scoped(
        conn,
        table="session_templates",
        id_=template_id,
        account_id=account_id,
        noun="session template",
    )
    return _row_to_session_template(row)
