"""Environment queries.

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
    _build_set_assignments,
    _get_scoped,
    _list_scoped,
    parse_jsonb,
)
from aios.errors import (
    ConflictError,
)
from aios.ids import (
    ENVIRONMENT,
    make_id,
)
from aios.models.environments import Environment, EnvironmentConfig

# ─── environments ─────────────────────────────────────────────────────────────


def _row_to_environment(row: asyncpg.Record) -> Environment:
    raw_config = row["config"]
    config_data = parse_jsonb(raw_config)
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
    account_id: str,
    name: str,
    config: EnvironmentConfig | None = None,
) -> Environment:
    new_id = make_id(ENVIRONMENT)
    config_json = json.dumps((config or EnvironmentConfig()).model_dump(exclude_none=True))
    try:
        row = await conn.fetchrow(
            "INSERT INTO environments (id, name, config, account_id) VALUES ($1, $2, $3::jsonb, $4) RETURNING *",
            new_id,
            name,
            config_json,
            account_id,
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"an environment named {name!r} already exists",
            detail={"name": name},
        ) from exc
    assert row is not None
    return _row_to_environment(row)


async def get_environment(
    conn: asyncpg.Connection[Any], env_id: str, *, account_id: str
) -> Environment:
    return await _get_scoped(
        conn,
        table="environments",
        id_=env_id,
        account_id=account_id,
        row=_row_to_environment,
        noun="environment",
    )


async def list_environments(
    conn: asyncpg.Connection[Any], *, account_id: str, limit: int = 50, after: str | None = None
) -> list[Environment]:
    return await _list_scoped(
        conn,
        table="environments",
        account_id=account_id,
        row=_row_to_environment,
        limit=limit,
        after=after,
    )


async def archive_environment(
    conn: asyncpg.Connection[Any], env_id: str, *, account_id: str
) -> None:
    await _archive_scoped(
        conn,
        table="environments",
        id_=env_id,
        account_id=account_id,
        noun="environment",
        bump_updated_at=False,
    )


async def update_environment(
    conn: asyncpg.Connection[Any],
    env_id: str,
    *,
    account_id: str,
    name: str | None = None,
    config: EnvironmentConfig | None = None,
) -> Environment:
    """Update an environment. Omitted fields are preserved."""
    # Upfront read distinguishes 404 vs 409; the ``archived_at IS NULL``
    # clause on the UPDATE closes the read->UPDATE race.
    current = await get_environment(conn, env_id, account_id=account_id)
    if current.archived_at is not None:
        raise ConflictError(f"environment {env_id} is archived", detail={"id": env_id})

    args: list[Any] = [env_id]
    fields: list[tuple[str, Any, str | None]] = []
    if name is not None:
        fields.append(("name", name, None))
    if config is not None:
        fields.append(("config", config.model_dump(exclude_none=True), "jsonb"))
    sets = _build_set_assignments(fields, args)
    if not sets:
        return current

    args.append(account_id)
    sql = (
        f"UPDATE environments SET {', '.join(sets)} "
        f"WHERE id = $1 AND account_id = ${len(args)} AND archived_at IS NULL RETURNING *"
    )
    try:
        row = await conn.fetchrow(sql, *args)
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"an environment named {name!r} already exists",
            detail={"name": name},
        ) from exc
    if row is None:
        raise ConflictError(f"environment {env_id} is archived", detail={"id": env_id})
    return _row_to_environment(row)


async def get_environment_config_for_session(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> EnvironmentConfig | None:
    """Return the environment config for a session, or None if not found.

    Filters both ``s.account_id`` AND ``e.account_id`` against the same
    caller account: ``insert_session`` only relies on the
    ``environment_id REFERENCES environments(id)`` FK and does not
    validate cross-account ownership. As of issue #755 the service-layer
    create paths (``create_session`` and ``create_run``) verify
    environment ownership before insert, so the normal path no longer
    binds a session to a foreign-tenant environment. The ``e.account_id``
    predicate remains as defense-in-depth for rows created via paths that
    bypass the service layer (direct ``insert_session`` in tests,
    pre-existing rows): without it this read would surface the foreign
    tenant's ``EnvironmentConfig`` — env vars, networking, packages —
    inside the worker's step context [security].
    """
    row = await conn.fetchrow(
        """
        SELECT e.config FROM environments e
        JOIN sessions s ON s.environment_id = e.id
        WHERE s.id = $1
          AND s.account_id = $2
          AND e.account_id = $2
        """,
        session_id,
        account_id,
    )
    if row is None:
        return None
    raw_config = row["config"]
    config_data = parse_jsonb(raw_config)
    return EnvironmentConfig.model_validate(config_data)
