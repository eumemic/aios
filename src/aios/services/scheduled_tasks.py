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
from aios.errors import ValidationError
from aios.models.scheduled_tasks import (
    ScheduledTaskCreate,
    ScheduledTaskEcho,
    ScheduledTaskUpdate,
    compute_initial_next_fire,
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

    Initial ``next_fire`` is computed from the schedule (cron) or copied
    from ``fire_at`` (one-shot), unless the task is disabled — in which
    case ``next_fire`` stays ``NULL``.
    """
    next_fire = (
        compute_initial_next_fire(spec.schedule, spec.fire_at, datetime.now(UTC))
        if spec.enabled
        else None
    )
    async with pool.acquire() as conn:
        return await queries.add_scheduled_task(
            conn,
            session_id,
            name=spec.name,
            schedule=spec.schedule,
            fire_at=spec.fire_at,
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

    Uses ``model_fields_set`` to distinguish 'explicit None to clear' from
    'field not provided' for the trigger fields (``schedule``, ``fire_at``)
    so a cron→one-shot or one-shot→cron conversion via PATCH is expressible.
    Validates the merged row state's XOR invariant before writing so the
    caller gets a typed :class:`ValidationError` instead of a raw DB
    constraint violation.

    Recomputes ``next_fire`` when either trigger changes, when the task is
    re-enabled, or clears it when the task is being disabled. Other
    updates (timeout, command, metadata) leave ``next_fire`` alone.
    """
    fields_set = update.model_fields_set
    schedule_explicit = "schedule" in fields_set
    fire_at_explicit = "fire_at" in fields_set

    async with pool.acquire() as conn, conn.transaction():
        current = await queries.get_scheduled_task_by_name(
            conn, session_id, name, account_id=account_id
        )

        new_enabled = update.enabled if update.enabled is not None else current.enabled
        new_schedule = update.schedule if schedule_explicit else current.schedule
        new_fire_at = update.fire_at if fire_at_explicit else current.fire_at

        # Cross-merge XOR invariant. The DB CHECK constraint enforces this
        # too, but doing the check here gives the caller a typed error with
        # a clearer message than asyncpg's CheckViolationError.
        if (new_schedule is None) == (new_fire_at is None):
            raise ValidationError(
                "after merging the PATCH, scheduled task must have exactly "
                "one of `schedule` (cron) or `fire_at` (one-shot) set — got "
                f"schedule={new_schedule!r}, fire_at={new_fire_at!r}"
            )

        next_fire: datetime | None | EllipsisType = ...  # ... = leave alone
        trigger_changed = schedule_explicit or fire_at_explicit
        if not new_enabled and current.enabled:
            # Disabling: clear next_fire.
            next_fire = None
        elif new_enabled and (trigger_changed or not current.enabled):
            # Trigger changed, or task re-enabled: recompute from merged state.
            if new_fire_at is not None:
                next_fire = new_fire_at
            else:
                assert new_schedule is not None  # XOR check above guarantees
                next_fire = compute_next_fire(new_schedule, datetime.now(UTC))

        schedule_arg: str | None | EllipsisType = update.schedule if schedule_explicit else ...
        fire_at_arg: datetime | None | EllipsisType = update.fire_at if fire_at_explicit else ...
        return await queries.update_scheduled_task(
            conn,
            session_id,
            name,
            schedule=schedule_arg,
            fire_at=fire_at_arg,
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
