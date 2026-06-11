"""Service layer for triggers (#818).

Thin pool-acquiring wrappers around the queries in
:mod:`aios.db.queries`. Owns the business logic around when to recompute
``next_fire`` — on add, on source change, on re-enable — and exposes
granular add/remove/update/list operations to the API + tool layers.

Deliberately no whole-list-replace primitive; per #270, the only mutation
surface is granular ops.

Triggers do NOT feed :func:`aios.sandbox.spec.build_spec_from_session`
(the runner reads them per-fire), so mutations here force no sandbox
eviction and get no Layer 2 ``spec_version`` trigger (#713). The NOTIFY
trigger on ``triggers`` (migration 0080, byte-identical to 0059's) is
untouched by service-layer writes other than the gated columns.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import EllipsisType
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.db import queries
from aios.errors import RateLimitedError, ValidationError
from aios.models.triggers import (
    MAX_TRIGGERS_PER_SESSION,
    OneShotSource,
    TriggerCreate,
    TriggerEcho,
    TriggerUpdate,
    compute_initial_next_fire,
)


async def add_trigger(
    pool: asyncpg.Pool[Any],
    session_id: str,
    spec: TriggerCreate,
    *,
    account_id: str,
) -> TriggerEcho:
    """Add a trigger to a session.

    Initial ``next_fire`` is computed from the source (cron: next slot;
    one-shot: ``fire_at``), unless the trigger is disabled — in which case
    ``next_fire`` stays ``NULL``.

    Enforces two caps, both serialized by a per-account transaction-scoped
    advisory lock so they're contractual against concurrent adds:

    - ``Settings.triggers_per_account_max`` against the account's count of
      enabled rows on non-archived sessions. Disabled rows and rows on
      archived sessions don't consume a slot.
    - ``MAX_TRIGGERS_PER_SESSION`` against the session's total row count
      (enabled + disabled), so a single session can't consume the whole
      per-account quota.
    """
    cap = get_settings().triggers_per_account_max
    next_fire = compute_initial_next_fire(spec.source, datetime.now(UTC)) if spec.enabled else None
    async with pool.acquire() as conn, conn.transaction():
        await queries.acquire_account_triggers_lock(conn, account_id)
        existing_session = await queries.count_session_triggers(
            conn, session_id=session_id, account_id=account_id
        )
        if existing_session >= MAX_TRIGGERS_PER_SESSION:
            raise RateLimitedError(
                f"session at triggers cap "
                f"({existing_session}/{MAX_TRIGGERS_PER_SESSION}); "
                "remove an existing trigger in this session to free a slot"
            )
        if spec.enabled:
            existing_account = await queries.count_account_triggers(
                conn, account_id=account_id, enabled_only=True
            )
            if existing_account >= cap:
                raise RateLimitedError(
                    f"account at active-timer cap ({existing_account}/{cap}); "
                    "remove or disable an existing trigger to free a slot"
                )
        return await queries.add_trigger(
            conn,
            session_id,
            name=spec.name,
            source=spec.source.kind,
            source_spec=spec.source.model_dump(mode="json", exclude={"kind"}),
            action=spec.action.model_dump(mode="json"),
            enabled=spec.enabled,
            metadata=spec.metadata,
            next_fire=next_fire,
            account_id=account_id,
        )


async def remove_trigger(
    pool: asyncpg.Pool[Any],
    session_id: str,
    name: str,
    *,
    account_id: str,
) -> None:
    async with pool.acquire() as conn:
        await queries.remove_trigger(conn, session_id, name, account_id=account_id)


async def update_trigger(
    pool: asyncpg.Pool[Any],
    session_id: str,
    name: str,
    update: TriggerUpdate,
    *,
    account_id: str,
) -> TriggerEcho:
    """Update a trigger by name (§2.4 of the design contract).

    ``source`` / ``action`` are replaced WHOLESALE when provided (a
    cron↔one-shot or sandbox↔wake conversion is just a different object —
    invalid shapes are unrepresentable, caught by the discriminated-union
    422). The business rules below are keyed on the MERGED final state:

    - Disabling (true→false): clears ``next_fire``.
    - Re-enabling (false→true): recomputes ``next_fire`` from the merged
      source, under the per-account active-timer cap (a disabled row didn't
      hold a slot), and rejects a one-shot whose merged ``fire_at`` is
      already in the past.
    - Source replaced on a row whose final state is enabled: recomputes
      ``next_fire`` (no cap re-check — an already-enabled row holds its
      slot), with the same past-``fire_at`` rejection.
    - action / metadata / no-op: ``next_fire`` untouched.

    ``updated_at`` always bumps (handled in the query layer) so a no-op
    PATCH is still visible to ``updated_at > since`` pollers.
    """
    source_provided = update.source is not None
    new_source = update.source.kind if update.source is not None else None
    new_source_spec = (
        update.source.model_dump(mode="json", exclude={"kind"})
        if update.source is not None
        else None
    )
    new_action = update.action.model_dump(mode="json") if update.action is not None else None

    async with pool.acquire() as conn, conn.transaction():
        current = await queries.get_trigger_by_name(conn, session_id, name, account_id=account_id)
        now = datetime.now(UTC)
        new_enabled = update.enabled if update.enabled is not None else current.enabled
        merged_source = update.source if update.source is not None else current.source

        next_fire: datetime | None | EllipsisType = ...  # ... = leave alone
        if not new_enabled and current.enabled:
            # Disabling: clear next_fire.
            next_fire = None
        elif new_enabled and (source_provided or not current.enabled):
            # Re-enabling, or replacing the source on a row whose final
            # state is enabled: recompute next_fire from the merged source.
            if not current.enabled:
                # Re-enable consumes a per-account active-timer slot (a
                # disabled row didn't); take the lock + cap check so this
                # can't race past the cap against concurrent adds.
                await queries.acquire_account_triggers_lock(conn, account_id)
                cap = get_settings().triggers_per_account_max
                existing = await queries.count_account_triggers(
                    conn, account_id=account_id, enabled_only=True
                )
                if existing >= cap:
                    raise RateLimitedError(
                        f"account at active-timer cap ({existing}/{cap}); remove "
                        "or disable another trigger before re-enabling this one"
                    )
            # Reject a one-shot whose merged fire_at is already in the past:
            # silently firing immediately with a stale wake reason is the
            # worse failure mode. Applies to re-enable AND source-replace on
            # an already-enabled row (today's behavior).
            if isinstance(merged_source, OneShotSource) and merged_source.fire_at <= now:
                raise ValidationError(
                    f"one-shot fire_at {merged_source.fire_at.isoformat()} is not in the "
                    "future; set a fresh fire_at before enabling (or send a new fire_at "
                    "in this same request)"
                )
            next_fire = compute_initial_next_fire(merged_source, now)

        return await queries.update_trigger(
            conn,
            session_id,
            name,
            source=new_source,
            source_spec=new_source_spec,
            action=new_action,
            enabled=update.enabled,
            metadata=update.metadata,
            next_fire=next_fire,
            account_id=account_id,
        )


async def list_triggers(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
) -> list[TriggerEcho]:
    async with pool.acquire() as conn:
        return await queries.list_triggers(conn, session_id, account_id=account_id)
