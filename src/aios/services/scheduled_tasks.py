"""Service layer for scheduled tasks (#636).

Thin pool-acquiring wrappers around the queries in
:mod:`aios.db.queries`. Owns the business logic around when to recompute
``next_fire`` — on add, on schedule change, on re-enable — and exposes
granular add/remove/update/list operations to the API + tool layers.

Deliberately no whole-list-replace primitive; per #270, the only
mutation surface is granular ops.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import EllipsisType
from typing import Any

import asyncpg

from aios.db import queries
from aios.models.scheduled_tasks import (
    ScheduledTaskCreate,
    ScheduledTaskEcho,
    ScheduledTaskUpdate,
    compute_next_fire,
)


async def add_task(
    pool: asyncpg.Pool[Any],
    session_id: str,
    spec: ScheduledTaskCreate,
    *,
    account_id: str,
) -> ScheduledTaskEcho:
    """Add a scheduled task to a session.

    Initial ``next_fire`` is computed from the schedule iff the task is
    enabled; disabled tasks store ``next_fire = NULL``.
    """
    next_fire = compute_next_fire(spec.schedule, datetime.now(UTC)) if spec.enabled else None
    async with pool.acquire() as conn:
        return await queries.add_scheduled_task(
            conn,
            session_id,
            name=spec.name,
            schedule=spec.schedule,
            command=spec.command,
            enabled=spec.enabled,
            timeout_seconds=spec.timeout_seconds,
            max_output_bytes=spec.max_output_bytes,
            metadata=spec.metadata,
            next_fire=next_fire,
            account_id=account_id,
        )


async def remove_task(
    pool: asyncpg.Pool[Any],
    session_id: str,
    name: str,
    *,
    account_id: str,
) -> None:
    async with pool.acquire() as conn:
        await queries.remove_scheduled_task(conn, session_id, name, account_id=account_id)


async def update_task(
    pool: asyncpg.Pool[Any],
    session_id: str,
    name: str,
    update: ScheduledTaskUpdate,
    *,
    account_id: str,
) -> ScheduledTaskEcho:
    """Update a scheduled task by name.

    Recomputes ``next_fire`` when the schedule changes, when the task is
    re-enabled, or clears it when the task is being disabled. Other
    updates (timeout, command, metadata) leave ``next_fire`` alone.
    """
    async with pool.acquire() as conn, conn.transaction():
        current = await queries.get_scheduled_task_by_name(
            conn, session_id, name, account_id=account_id
        )

        new_enabled = update.enabled if update.enabled is not None else current.enabled
        new_schedule = update.schedule if update.schedule is not None else current.schedule

        next_fire: datetime | None | EllipsisType = ...  # ... = leave alone
        if not new_enabled and current.enabled:
            # Disabling: clear next_fire.
            next_fire = None
        elif new_enabled and (update.schedule is not None or not current.enabled):
            # Schedule changed while enabled, or task re-enabled: recompute.
            next_fire = compute_next_fire(new_schedule, datetime.now(UTC))

        return await queries.update_scheduled_task(
            conn,
            session_id,
            name,
            schedule=update.schedule,
            command=update.command,
            enabled=update.enabled,
            timeout_seconds=update.timeout_seconds,
            max_output_bytes=update.max_output_bytes,
            metadata=update.metadata,
            next_fire=next_fire,
            account_id=account_id,
        )


async def list_tasks(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
) -> list[ScheduledTaskEcho]:
    async with pool.acquire() as conn:
        return await queries.list_scheduled_tasks(conn, session_id, account_id=account_id)
