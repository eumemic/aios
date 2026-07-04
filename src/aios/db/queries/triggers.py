"""Trigger and trigger-run queries.

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import EllipsisType
from typing import Any, NamedTuple

import asyncpg

from aios.errors import (
    ConflictError,
    NotFoundError,
)
from aios.ids import (
    TRIGGER,
    TRIGGER_RUN,
    make_id,
)
from aios.models.triggers import (
    TRIGGER_ACTION_ADAPTER,
    TRIGGER_SOURCE_ADAPTER,
    AccountTriggerEcho,
    TriggerAction,
    TriggerEcho,
    TriggerFireStatus,
    TriggerRunEcho,
    compute_next_fire,
)

# ─── triggers ───────────────────────────────────────────────────────────────


class TriggerRow(NamedTuple):
    """Internal record for the trigger fire-handler + scheduler tick.

    Carries ``owner_session_id``, ``account_id``, and the owning session's
    ``session_archived_at`` alongside the definition fields so the unscoped
    fire-job handler (which only has the trigger id) can resolve the owner
    and verify it hasn't been archived between claim and fire — without an
    extra round-trip.

    ``source`` is the raw discriminator text and ``source_spec`` the raw
    parsed dict (the scheduler/runner branch lifecycle on the source
    string); ``action`` is the validated union (the runner dispatches on
    ``action.kind``). ``environment_id`` is the first-class FK column —
    non-NULL iff the action kind is ``workflow`` (the iff CHECK).
    ``session_parent_run_id`` is the owning session's own (immutable) parent
    run — the lineage a timer-fired workflow action threads, projected here
    off the JOIN the row already pays for.
    """

    id: str
    owner_session_id: str
    account_id: str
    name: str
    source: str
    source_spec: dict[str, Any]
    action: TriggerAction
    enabled: bool
    next_fire: datetime | None
    running_since: datetime | None
    last_fire_at: datetime | None
    last_fire_status: TriggerFireStatus | None
    consecutive_failures: int
    environment_id: str | None
    ingest_token_hash: str | None
    session_archived_at: datetime | None
    session_parent_run_id: str | None


def _row_to_trigger_echo(row: asyncpg.Record) -> TriggerEcho:
    """Reconstruct a :class:`TriggerEcho` from a ``triggers`` row.

    The source union is reconstructed as ``{"kind": source, **source_spec}``
    and the action union from the raw ``action`` jsonb. Both go through the
    module-level adapters, which are STRUCTURE-ONLY — they accept every row
    the write path ever accepted (no cron occurrence re-validation on read,
    or a legally-persisted rare cron would 500 every ``GET /v1/sessions``).
    """
    return TriggerEcho(
        id=row["id"],
        name=row["name"],
        source=TRIGGER_SOURCE_ADAPTER.validate_python(
            {"kind": row["source"], **row["source_spec"]}
        ),
        action=TRIGGER_ACTION_ADAPTER.validate_python(row["action"]),
        enabled=row["enabled"],
        next_fire=row["next_fire"],
        last_fire_at=row["last_fire_at"],
        last_fire_status=row["last_fire_status"],
        consecutive_failures=row["consecutive_failures"],
        metadata=row["metadata"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def add_trigger(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    name: str,
    source: str,
    source_spec: dict[str, Any],
    action: dict[str, Any],
    enabled: bool,
    metadata: dict[str, Any],
    next_fire: datetime | None,
    environment_id: str | None,
    ingest_token_hash: str | None,
    account_id: str,
) -> TriggerEcho:
    """Insert a trigger.

    ``source`` / ``source_spec`` / ``action`` are the serialized union forms
    (the service layer materializes action defaults and the union models
    enforce shape; the DB CHECK is a backstop). ``environment_id`` is a
    REQUIRED kwarg (no default) so mypy forces every call site through the
    resolution decision: the owner session's environment for ``workflow``
    actions, ``None`` otherwise (``services.triggers.validate_trigger_spec``).
    Maps unique-name violations to :class:`ConflictError` and FK violations
    to :class:`NotFoundError` — the session FK is the only one reachable
    today (the environment FK is pre-validated in the same transaction by the
    service layer, and environments have no delete path; if an env-delete
    feature ever ships, this blanket mapping must learn to tell the two
    apart or it will name the wrong missing resource).
    """
    trigger_id = make_id(TRIGGER)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO triggers
                (id, owner_session_id, account_id, name, source, source_spec,
                 action, enabled, next_fire, environment_id, ingest_token_hash,
                 metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING *
            """,
            trigger_id,
            session_id,
            account_id,
            name,
            source,
            json.dumps(source_spec),
            json.dumps(action),
            enabled,
            next_fire,
            environment_id,
            ingest_token_hash,
            json.dumps(metadata),
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"trigger name {name!r} already exists in this session",
            detail={"name": name, "session_id": session_id},
        ) from exc
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            f"session {session_id!r} not found",
            detail={"session_id": session_id},
        ) from exc
    assert row is not None
    return _row_to_trigger_echo(row)


async def remove_trigger(
    conn: asyncpg.Connection[Any],
    session_id: str,
    name: str,
    *,
    account_id: str,
) -> None:
    """Delete a trigger by name. Raises :class:`NotFoundError` if absent."""
    result = await conn.execute(
        "DELETE FROM triggers WHERE owner_session_id = $1 AND name = $2 AND account_id = $3",
        session_id,
        name,
        account_id,
    )
    if result == "DELETE 0":
        raise NotFoundError(
            f"trigger {name!r} not found",
            detail={"name": name, "session_id": session_id},
        )


async def get_trigger_by_name(
    conn: asyncpg.Connection[Any],
    session_id: str,
    name: str,
    *,
    account_id: str,
) -> TriggerEcho:
    row = await conn.fetchrow(
        "SELECT * FROM triggers WHERE owner_session_id = $1 AND name = $2 AND account_id = $3",
        session_id,
        name,
        account_id,
    )
    if row is None:
        raise NotFoundError(
            f"trigger {name!r} not found",
            detail={"name": name, "session_id": session_id},
        )
    return _row_to_trigger_echo(row)


async def list_triggers(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> list[TriggerEcho]:
    rows = await conn.fetch(
        "SELECT * FROM triggers "
        "WHERE owner_session_id = $1 AND account_id = $2 "
        "ORDER BY created_at",
        session_id,
        account_id,
    )
    return [_row_to_trigger_echo(r) for r in rows]


async def list_account_triggers(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    enabled_only: bool = True,
) -> list[AccountTriggerEcho]:
    """List every trigger in an account (across all its sessions) as the
    liveness-audit projection (#1673).

    The account-wide analog of :func:`list_triggers` (which is session-scoped):
    the ops-agent O7 trigger-liveness auditor sweeps EVERY enabled trigger in
    its account — dev-pipeline cron, reconciler cron, reaper, telemetry-observer,
    future sentinels, each on its own session — and reads each one's
    ``next_fire`` to catch the #925 zombie class (``enabled=true,
    next_fire=NULL`` cron rows the scheduler filters out and never fires).

    JOINs ``sessions`` and filters ``s.archived_at IS NULL`` for the same reason
    the scheduler's claim/MIN queries do: a trigger on an archived session can
    never fire regardless of its own ``enabled`` flag, so it isn't part of the
    live-liveness population. Defaults to ``enabled_only=True`` (the population
    the auditor's non-null-``next_fire`` invariant is about); pass
    ``enabled_only=False`` for an operator-visible total.

    Projects the discriminator text (``t.source AS source_kind``) rather than
    the full source union — the auditor branches on the kind (``cron`` →
    schedulable → ``next_fire`` must be non-null; ``run_completion`` /
    ``external_event`` → reactive → exempt), not the schedule payload. Ordered
    ``owner_session_id, name`` for a stable account-wide listing.
    """
    where_enabled = "AND t.enabled" if enabled_only else ""
    rows = await conn.fetch(
        f"""
        SELECT t.id, t.name, t.owner_session_id, t.source AS source_kind,
               t.enabled, t.next_fire, t.last_fire_status,
               t.consecutive_failures
        FROM triggers AS t
        JOIN sessions AS s ON s.id = t.owner_session_id
        WHERE t.account_id = $1
          AND s.archived_at IS NULL
          {where_enabled}
        ORDER BY t.owner_session_id, t.name
        """,
        account_id,
    )
    return [
        AccountTriggerEcho(
            id=r["id"],
            name=r["name"],
            owner_session_id=r["owner_session_id"],
            source_kind=r["source_kind"],
            enabled=r["enabled"],
            next_fire=r["next_fire"],
            last_fire_status=r["last_fire_status"],
            consecutive_failures=r["consecutive_failures"],
        )
        for r in rows
    ]


async def batch_list_session_triggers(
    conn: asyncpg.Connection[Any],
    session_ids: list[str],
    *,
    account_id: str,
) -> dict[str, list[TriggerEcho]]:
    """Batch-fetch trigger echoes for multiple sessions.

    Returns a dict keyed by session_id; sessions with no triggers map to an
    empty list. Mirrors the batch pattern used by
    :func:`batch_list_session_memory_store_echoes`. Ordered
    ``owner_session_id, created_at`` for stable per-session echo order.
    """
    if not session_ids:
        return {}
    rows = await conn.fetch(
        "SELECT * FROM triggers "
        "WHERE owner_session_id = ANY($1) AND account_id = $2 "
        "ORDER BY owner_session_id, created_at",
        session_ids,
        account_id,
    )
    result: dict[str, list[TriggerEcho]] = {sid: [] for sid in session_ids}
    for r in rows:
        result[str(r["owner_session_id"])].append(_row_to_trigger_echo(r))
    return result


async def update_trigger(
    conn: asyncpg.Connection[Any],
    session_id: str,
    name: str,
    *,
    source: str | None = None,
    source_spec: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
    enabled: bool | None = None,
    metadata: dict[str, Any] | None = None,
    next_fire: datetime | None | EllipsisType = ...,
    environment_id: str | None | EllipsisType = ...,
    ingest_token_hash: str | None | EllipsisType = ...,
    reset_consecutive_failures: bool = False,
    account_id: str,
) -> TriggerEcho:
    """Update fields by name. Raises :class:`NotFoundError` if absent.

    ``reset_consecutive_failures`` is set by the service on the enabled
    false→true flip: a re-enable is a fresh start for the failure counter, so
    the ``== MAX`` auto-disable gate can trip again on a still-broken trigger
    (without the reset, a counter parked past the threshold would never equal
    it again and the 5-strike breaker would be permanently disarmed).

    ``source`` / ``source_spec`` are set together (wholesale replacement of
    the source union) or both left alone (``None``). ``action`` is replaced
    wholesale or left alone. ``next_fire`` and ``environment_id`` use ``...``
    (Ellipsis) as the leave-alone sentinel because ``None`` is a meaningful
    clear-to-null. The service layer MUST pass ``environment_id`` whenever it
    passes ``action`` (the resolved env for ``workflow``, ``None`` otherwise)
    so the jsonb and the column flip in ONE UPDATE — the iff CHECK catches a
    forgotten kind conversion loudly, but a same-kind workflow→workflow
    replacement with a stale column would be silent. Source/action SHAPES are
    validated upstream by the discriminated-union models (a 422 fires before
    the DB CHECK), so no merged-XOR re-validation is needed here — invalid
    shapes are unrepresentable.
    """
    set_clauses: list[str] = []
    args: list[Any] = []

    def add(col: str, value: Any) -> None:
        args.append(value)
        set_clauses.append(f"{col} = ${len(args)}")

    if source is not None:
        add("source", source)
    if source_spec is not None:
        add("source_spec", json.dumps(source_spec))
    if action is not None:
        add("action", json.dumps(action))
    if enabled is not None:
        add("enabled", enabled)
    if metadata is not None:
        add("metadata", json.dumps(metadata))
    if not isinstance(next_fire, EllipsisType):
        add("next_fire", next_fire)
    if not isinstance(environment_id, EllipsisType):
        add("environment_id", environment_id)
    if not isinstance(ingest_token_hash, EllipsisType):
        add("ingest_token_hash", ingest_token_hash)
    if reset_consecutive_failures:
        set_clauses.append("consecutive_failures = 0")

    # Always bump ``updated_at`` so a no-op PATCH still records a write —
    # external pollers using ``updated_at > <since>`` mustn't miss it.
    set_clauses.append("updated_at = now()")
    args.extend([session_id, name, account_id])
    sql = f"""
        UPDATE triggers
        SET {", ".join(set_clauses)}
        WHERE owner_session_id = ${len(args) - 2}
          AND name = ${len(args) - 1}
          AND account_id = ${len(args)}
        RETURNING *
    """
    row = await conn.fetchrow(sql, *args)
    if row is None:
        raise NotFoundError(
            f"trigger {name!r} not found",
            detail={"name": name, "session_id": session_id},
        )
    return _row_to_trigger_echo(row)


async def unscoped_get_trigger_row(
    conn: asyncpg.Connection[Any],
    trigger_id: str,
) -> TriggerRow:
    """Fetch a trigger by id WITHOUT account_id scope.

    Used by the fire-job handler, which runs cross-tenant in the worker;
    each row carries its ``account_id`` denormalized. INTENTIONALLY unscoped
    — do not "fix" with account scoping. JOINs ``sessions`` so the handler
    can re-check the owning session's archive state and skip a fire whose
    session was archived between claim and execute.
    """
    row = await conn.fetchrow(
        "SELECT t.id, t.owner_session_id, t.account_id, t.name, t.source, "
        "t.source_spec, t.action, t.enabled, t.next_fire, t.running_since, "
        "t.last_fire_at, t.last_fire_status, t.consecutive_failures, "
        "t.environment_id, t.ingest_token_hash, "
        "s.archived_at AS session_archived_at, "
        "s.parent_run_id AS session_parent_run_id "
        "FROM triggers AS t "
        "JOIN sessions AS s ON s.id = t.owner_session_id "
        "WHERE t.id = $1",
        trigger_id,
    )
    if row is None:
        raise NotFoundError(
            f"trigger {trigger_id!r} not found",
            detail={"id": trigger_id},
        )
    return TriggerRow(
        id=row["id"],
        owner_session_id=row["owner_session_id"],
        account_id=row["account_id"],
        name=row["name"],
        source=row["source"],
        source_spec=row["source_spec"],
        action=TRIGGER_ACTION_ADAPTER.validate_python(row["action"]),
        enabled=row["enabled"],
        next_fire=row["next_fire"],
        running_since=row["running_since"],
        last_fire_at=row["last_fire_at"],
        last_fire_status=row["last_fire_status"],
        consecutive_failures=row["consecutive_failures"],
        environment_id=row["environment_id"],
        ingest_token_hash=row["ingest_token_hash"],
        session_archived_at=row["session_archived_at"],
        session_parent_run_id=row["session_parent_run_id"],
    )


async def fetch_and_claim_due_triggers(
    conn: asyncpg.Connection[Any],
    *,
    now_utc: datetime,
    limit: int = 100,
    stale_threshold_seconds: int = 7200,
) -> list[TriggerRow]:
    """Atomically claim due triggers for the scheduler tick.

    In a single transaction: SELECT enabled triggers whose owning session is
    not archived, whose ``next_fire <= now``, and which are either not
    running (``running_since IS NULL``) or stuck-running for more than
    ``stale_threshold_seconds`` (recovers from worker crashes mid-fire).
    For each claimed cron row, sets ``running_since`` to now and advances
    ``next_fire`` to the next scheduled instant so subsequent ticks skip the
    row; one-shot rows leave ``next_fire`` alone (the runner deletes them).
    Returns the claimed rows.

    Archive is the lifecycle boundary — once ``sessions.archived_at`` is
    set, none of the session's triggers fire again, regardless of their own
    ``enabled`` flag. Unarchiving (if supported) restores firing. The rows
    themselves are preserved.

    The claim SELECT projects ``source_spec ->> 'schedule' AS schedule``
    (text extraction in SQL) so the cron-advance loop stays byte-identical
    to today and sidesteps asyncpg's jsonb-as-raw-string decode trap on the
    hot path. The explicit ``next_fire IS NOT NULL`` predicate is
    load-bearing for future reactive sources: a row without ``next_fire`` is
    unschedulable by the tick BY PREDICATE, not merely by index shape.

    Must be called inside an outer transaction so the SELECT FOR UPDATE SKIP
    LOCKED + UPDATE chain executes atomically against concurrent workers.
    """
    from datetime import timedelta

    stale_cutoff = now_utc - timedelta(seconds=stale_threshold_seconds)
    rows = await conn.fetch(
        """
        SELECT t.id, t.owner_session_id, t.account_id, t.name, t.source,
               t.source_spec, t.source_spec ->> 'schedule' AS schedule,
               t.action, t.enabled, t.next_fire, t.running_since,
               t.last_fire_at, t.last_fire_status, t.consecutive_failures,
               t.environment_id, t.ingest_token_hash,
               s.archived_at AS session_archived_at,
               s.parent_run_id AS session_parent_run_id
        FROM triggers AS t
        JOIN sessions AS s ON s.id = t.owner_session_id
        WHERE t.enabled
          AND s.archived_at IS NULL
          AND t.next_fire IS NOT NULL
          AND t.next_fire <= $1
          AND (t.running_since IS NULL OR t.running_since <= $2)
        ORDER BY t.next_fire
        FOR UPDATE OF t SKIP LOCKED
        LIMIT $3
        """,
        now_utc,
        stale_cutoff,
        limit,
    )
    claimed: list[TriggerRow] = []
    for r in rows:
        if r["source"] == "cron":
            # Cron row — advance next_fire so subsequent ticks skip until the
            # next scheduled slot. ``schedule`` is the text projection of
            # ``source_spec ->> 'schedule'`` (NULL for one-shot rows).
            new_next_fire = compute_next_fire(r["schedule"], now_utc)
            await conn.execute(
                """
                UPDATE triggers
                SET running_since = $1, next_fire = $2, updated_at = $1
                WHERE id = $3
                """,
                now_utc,
                new_next_fire,
                r["id"],
            )
        else:
            # One-shot row — leave next_fire alone; the runner deletes the
            # row after the fire completes, so we never need to skip-by-
            # next_fire.
            new_next_fire = r["next_fire"]
            await conn.execute(
                """
                UPDATE triggers
                SET running_since = $1, updated_at = $1
                WHERE id = $2
                """,
                now_utc,
                r["id"],
            )
        claimed.append(
            TriggerRow(
                id=r["id"],
                owner_session_id=r["owner_session_id"],
                account_id=r["account_id"],
                name=r["name"],
                source=r["source"],
                source_spec=r["source_spec"],
                action=TRIGGER_ACTION_ADAPTER.validate_python(r["action"]),
                enabled=r["enabled"],
                # next_fire on the returned row reflects the *advanced* value
                # for cron, or the unchanged value for one-shot.
                next_fire=new_next_fire,
                running_since=now_utc,
                last_fire_at=r["last_fire_at"],
                last_fire_status=r["last_fire_status"],
                consecutive_failures=r["consecutive_failures"],
                environment_id=r["environment_id"],
                ingest_token_hash=r["ingest_token_hash"],
                session_archived_at=r["session_archived_at"],
                session_parent_run_id=r["session_parent_run_id"],
            )
        )
    return claimed


async def record_trigger_fire(
    conn: asyncpg.Connection[Any],
    trigger_id: str,
    *,
    status: TriggerFireStatus,
    fired_at: datetime,
    clear_running_since: bool = True,
) -> int | None:
    """Record the outcome of a fire (clearing ``running_since`` for TICK
    fires); return the post-update ``consecutive_failures``.

    ``clear_running_since=False`` is the EVENT-fire form: event fires never
    held the tick's ``running_since`` claim, and unconditionally clearing it
    here could release a CONCURRENT tick fire's overlap lease (reachable when
    a mid-flight source conversion lets a straggler event fire finish while a
    freshly-converted cron row's tick fire is executing) — re-opening the row
    for an overlapping fire, the exact invariant ``running_since`` exists for.

    The counter is computed SQL-side (ok → 0, skipped → unchanged, else +1):
    run_completion fires are unserialized (no ``running_since`` claim, no
    per-trigger queueing lock), so a Python read-modify-write would lose
    updates under concurrent fires — bursts could fail forever without ever
    reaching the auto-disable threshold, or a stale failure could clobber a
    success's reset. The UPDATE's row lock serializes concurrent records, so
    the RETURNING value is the true serialized count; behavior-identical for
    single-flight cron. Returns ``None`` when the row vanished mid-fire (an
    API DELETE racing the fire — benign; callers must tolerate it).

    Does NOT touch ``next_fire`` — that was advanced by the tick when the
    row was claimed (and is permanently NULL for run_completion rows).
    """
    result: int | None = await conn.fetchval(
        """
        UPDATE triggers
        SET running_since = CASE WHEN $4 THEN NULL ELSE running_since END,
            last_fire_at = $1,
            last_fire_status = $2,
            consecutive_failures = CASE
                WHEN $2 = 'ok' THEN 0
                WHEN $2 = 'skipped' THEN consecutive_failures
                ELSE consecutive_failures + 1
            END,
            updated_at = $1
        WHERE id = $3
        RETURNING consecutive_failures
        """,
        fired_at,
        status,
        trigger_id,
        clear_running_since,
    )
    return result


async def disable_trigger(
    conn: asyncpg.Connection[Any],
    trigger_id: str,
) -> None:
    """Disable a trigger by id.

    Sets ``enabled = false``, clears ``next_fire``, and leaves
    ``running_since`` untouched (in case a fire is still in flight; the
    handler clears it on completion).
    """
    await conn.execute(
        """
        UPDATE triggers
        SET enabled = false, next_fire = NULL, updated_at = now()
        WHERE id = $1
        """,
        trigger_id,
    )


async def delete_trigger_by_id(
    conn: asyncpg.Connection[Any],
    trigger_id: str,
) -> None:
    """Delete a trigger by id, unscoped.

    Used by the runner for one-shot rows after the fire completes — the
    marker event the action delivered is the receipt; the row's job is done
    and keeping it would let it fire again on the next tick (next_fire is
    still in the past for one-shot rows). No-op if the row doesn't exist
    (e.g. raced with an API DELETE).
    """
    await conn.execute(
        "DELETE FROM triggers WHERE id = $1",
        trigger_id,
    )


async def release_trigger_claim(
    conn: asyncpg.Connection[Any],
    trigger_id: str,
) -> None:
    """Compensating reset for a claim whose downstream defer/enqueue failed.

    Clears ``running_since`` so the next scheduler cycle can re-claim the
    row. For cron rows, ``next_fire`` was already advanced by
    :func:`fetch_and_claim_due_triggers` — the released row fires at the next
    scheduled slot, effectively skipping the current slot whose defer failed
    (acceptable churn for a transient broker error). For one-shot rows,
    ``next_fire = fire_at`` is still in the past, so the row is re-claimed
    immediately.
    """
    await conn.execute(
        "UPDATE triggers SET running_since = NULL, updated_at = now() WHERE id = $1",
        trigger_id,
    )


async def count_account_triggers(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    enabled_only: bool = True,
) -> int:
    """Count trigger rows owned by ``account_id``.

    Backs the per-account cap enforced in
    ``services.triggers.add_trigger``. Defaults to counting only enabled
    rows on non-archived sessions — paused/disabled entries don't consume a
    "slot" against the cap, and rows attached to archived sessions are
    permanently unable to fire (the scheduler's claim and MIN queries both
    filter ``s.archived_at IS NULL``), so they shouldn't count either. Pass
    ``enabled_only=False`` for an operator-visible total that ignores both
    filters.
    """
    if not enabled_only:
        result: int | None = await conn.fetchval(
            "SELECT COUNT(*) FROM triggers WHERE account_id = $1",
            account_id,
        )
        return result or 0
    result = await conn.fetchval(
        """
        SELECT COUNT(*)
        FROM triggers AS t
        JOIN sessions AS s ON s.id = t.owner_session_id
        WHERE t.account_id = $1
          AND t.enabled
          AND s.archived_at IS NULL
        """,
        account_id,
    )
    return result or 0


async def count_session_triggers(
    conn: asyncpg.Connection[Any],
    *,
    session_id: str,
    account_id: str,
) -> int:
    """Count trigger rows attached to one session.

    Backs the per-session cap (``MAX_TRIGGERS_PER_SESSION``) enforced in
    ``services.triggers.add_trigger``. Counts all rows (enabled or disabled)
    because the per-session cap is about the session's resource footprint,
    not just its active-trigger load.
    """
    result: int | None = await conn.fetchval(
        """
        SELECT COUNT(*)
        FROM triggers
        WHERE owner_session_id = $1 AND account_id = $2
        """,
        session_id,
        account_id,
    )
    return result or 0


# ─── trigger_runs: per-fire audit + run_completion dispatch carrier (#819) ───


class TriggerFireRef(NamedTuple):
    """A dispatchable run_completion fire: the carrier row + its trigger."""

    trigger_run_id: str
    trigger_id: str


def _row_to_trigger_run_echo(row: asyncpg.Record) -> TriggerRunEcho:
    return TriggerRunEcho(
        id=row["id"],
        trigger_id=row["trigger_id"],
        trigger_context=row["trigger_context"],
        event=row["event"] if row["event"] is not None else None,
        status=row["status"],
        result_id=row["result_id"],
        error_summary=row["error_summary"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


async def insert_run_completion_fires(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    workflow_id: str,
    run_id: str,
    status: str,
) -> list[TriggerFireRef]:
    """Match watching ``run_completion`` triggers and insert one ``pending``
    fire-carrier row per match.

    MUST run inside the run-completion transaction (the single consistency
    point: commit makes "the run completed" and "these fires are owed"
    atomic). The ``t.account_id = $1`` conjunct is the tenant boundary —
    write-path validation is UX, this is enforcement: a trigger is only ever
    handed run data its owner could already read via the account-scoped run
    reads. Matching is covered by the ``triggers_run_completion_watch``
    partial index; the insert loop is bounded by the per-account
    enabled-trigger cap.
    """
    rows = await conn.fetch(
        """
        SELECT t.id, t.account_id, t.owner_session_id, t.name
        FROM triggers AS t
        JOIN sessions AS s ON s.id = t.owner_session_id
        WHERE t.source = 'run_completion'
          AND t.account_id = $1
          AND t.source_spec ->> 'workflow_id' = $2
          AND t.source_spec -> 'statuses' ? $3
          AND t.enabled
          AND s.archived_at IS NULL
        ORDER BY t.created_at
        """,
        account_id,
        workflow_id,
        status,
    )
    if not rows:
        # The overwhelmingly common case (no watchers) — never pay an
        # executemany round-trip inside the run-completion terminal txn.
        return []
    event = json.dumps({"run_id": run_id, "workflow_id": workflow_id, "status": status})
    refs = [TriggerFireRef(make_id(TRIGGER_RUN), r["id"]) for r in rows]
    # One pipelined statement for the whole batch — this runs inside the
    # run-completion transaction, so per-row round-trips would extend the
    # terminal commit window for nothing.
    await conn.executemany(
        """
        INSERT INTO trigger_runs
            (id, trigger_id, account_id, owner_session_id, trigger_name,
             trigger_context, event, status)
        VALUES ($1, $2, $3, $4, $5, 'run_completion', $6::jsonb, 'pending')
        """,
        [
            (ref.trigger_run_id, r["id"], r["account_id"], r["owner_session_id"], r["name"], event)
            for ref, r in zip(refs, rows, strict=True)
        ],
    )
    return refs


class ResolvedExternalEventTrigger(NamedTuple):
    """The single ``external_event`` trigger an ingest token resolves to.

    The token IS the tenant proof — the matched row's own ``account_id``
    becomes the authenticated scope for the fire it dispatches, exactly as
    ``resolve_runtime_token`` makes the matched row's account the auth scope.
    """

    trigger_id: str
    account_id: str
    owner_session_id: str
    trigger_name: str


async def resolve_external_event_trigger(
    conn: asyncpg.Connection[Any],
    *,
    ingest_token_hash: str,
) -> ResolvedExternalEventTrigger | None:
    """Resolve an ingest-token hash to the single enabled ``external_event``
    trigger on a non-archived session, or ``None``.

    No ``account_id`` parameter: this is the account-key-free ingress edge.
    The matched row's own ``account_id`` is the authenticated scope (the token
    IS the tenant proof). A unique partial index on ``ingest_token_hash``
    guarantees at most one match. ``None`` (miss/disabled/archived/revoked)
    becomes a uniform 404 at the ingress — the ``lookup_account_by_key_hash``
    no-oracle stance: an attacker cannot distinguish wrong-token from
    disabled-trigger.
    """
    row = await conn.fetchrow(
        """
        SELECT t.id, t.account_id, t.owner_session_id, t.name
        FROM triggers AS t
        JOIN sessions AS s ON s.id = t.owner_session_id
        WHERE t.source = 'external_event'
          AND t.ingest_token_hash = $1
          AND t.enabled
          AND s.archived_at IS NULL
        """,
        ingest_token_hash,
    )
    if row is None:
        return None
    return ResolvedExternalEventTrigger(
        trigger_id=row["id"],
        account_id=row["account_id"],
        owner_session_id=row["owner_session_id"],
        trigger_name=row["name"],
    )


async def insert_external_event_fire(
    conn: asyncpg.Connection[Any],
    *,
    trigger_id: str,
    account_id: str,
    owner_session_id: str,
    trigger_name: str,
    event: dict[str, Any],
) -> str:
    """Insert one ``pending`` ``trigger_runs`` carrier for an external event.

    Mirrors :func:`insert_run_completion_fires`' VALUES tuple but is a
    single-row insert (one ingress call = one fire), so no ``executemany``.
    ``trigger_context = 'external_event'`` and ``event`` is the inbound body
    verbatim. Returns the new ``trigger_run_id``.
    """
    trigger_run_id = make_id(TRIGGER_RUN)
    await conn.execute(
        """
        INSERT INTO trigger_runs
            (id, trigger_id, account_id, owner_session_id, trigger_name,
             trigger_context, event, status)
        VALUES ($1, $2, $3, $4, $5, 'external_event', $6::jsonb, 'pending')
        """,
        trigger_run_id,
        trigger_id,
        account_id,
        owner_session_id,
        trigger_name,
        json.dumps(event),
    )
    return trigger_run_id


class ClaimedTriggerRun(NamedTuple):
    """A claimed fire-carrier row: its ``event`` body + originating context.

    ``trigger_context`` distinguishes the two reactive carrier kinds at fire
    time (``run_completion`` vs ``external_event``) — the runner branches on it
    to derive lineage and the delivered envelope's ``source``, NEVER on
    ``event is not None`` (both carrier kinds carry a dict ``event``, but an
    ``external_event`` body has no ``run_id`` to read a completing run from).
    """

    event: dict[str, Any]
    trigger_context: str


async def claim_trigger_run(
    conn: asyncpg.Connection[Any],
    trigger_run_id: str,
    *,
    started_at: datetime,
) -> ClaimedTriggerRun | None:
    """Claim a fire-carrier row (``pending`` → ``running``); return its event
    and originating ``trigger_context``.

    ``None`` means the row was already claimed (a duplicate job — the sweep
    re-deferred a fire whose live job won the race) and the caller must exit
    without firing. ``started_at`` is the runner's single fire timestamp.

    The ``assert isinstance(event, dict)`` stays valid for BOTH carrier kinds:
    run_completion rows carry ``{run_id, workflow_id, status}`` and the
    external-event ingress rejects any non-object JSON body (422) before the
    carrier row is ever inserted.
    """
    row = await conn.fetchrow(
        """
        UPDATE trigger_runs
        SET status = 'running', started_at = $2
        WHERE id = $1 AND status = 'pending'
        RETURNING event, trigger_context
        """,
        trigger_run_id,
        started_at,
    )
    if row is None:
        return None
    event = row["event"]
    # Both reactive carrier kinds always carry a dict event (run_completion's
    # synthesized {run_id,…}; external_event's ingress-validated object body).
    assert isinstance(event, dict)
    return ClaimedTriggerRun(event=event, trigger_context=row["trigger_context"])


async def finalize_trigger_run(
    conn: asyncpg.Connection[Any],
    trigger_run_id: str,
    *,
    status: str,
    error_summary: str | None = None,
    result_id: str | None = None,
) -> None:
    """Record an event fire's outcome on its carrier row.

    First-writer-wins via the ``status = 'running'`` guard (the
    ``set_run_status`` terminal-guard idiom): a re-dispatched job racing a
    finished one cannot overwrite the outcome.
    """
    await conn.execute(
        """
        UPDATE trigger_runs
        SET status = $2, error_summary = $3, result_id = $4, finished_at = now()
        WHERE id = $1 AND status = 'running'
        """,
        trigger_run_id,
        status,
        error_summary,
        result_id,
    )


async def record_trigger_run(
    conn: asyncpg.Connection[Any],
    *,
    trigger_id: str,
    account_id: str,
    owner_session_id: str,
    trigger_name: str,
    trigger_context: str,
    status: str,
    error_summary: str | None,
    result_id: str | None,
    started_at: datetime,
) -> str:
    """Timer-fire audit writer: one complete row at fire completion.

    Used for cron fires (inside ``record_trigger_fire``'s transaction, so the
    echo cache and the audit can never disagree), one-shot fires (standalone
    post-action — by then the trigger row is already deleted, making this row
    the only persistent record the fire ever happened), and the one-shot
    skip tombstone. Timer rows are NEVER written at tick-claim time: the tick
    tail is contractually frozen (task-id-only payload, per-trigger
    queueing_lock whose coalesce would orphan a claim-time row as 'pending').
    """
    trigger_run_id = make_id(TRIGGER_RUN)
    await conn.execute(
        """
        INSERT INTO trigger_runs
            (id, trigger_id, account_id, owner_session_id, trigger_name,
             trigger_context, status, error_summary, result_id,
             started_at, finished_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, now())
        """,
        trigger_run_id,
        trigger_id,
        account_id,
        owner_session_id,
        trigger_name,
        trigger_context,
        status,
        error_summary,
        result_id,
        started_at,
    )
    return trigger_run_id


async def list_trigger_runs(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    session_id: str,
    trigger_name: str,
    limit: int = 50,
) -> list[TriggerRunEcho]:
    """List a trigger's fires, newest first — keyed by the DENORMALIZED
    ``(account_id, owner_session_id, trigger_name)``, never by resolving the
    live trigger row: one-shot tombstones and deleted-trigger audits must stay
    reachable (the audit outlives its trigger by design). Name reuse merges
    incarnations in the listing; rows carry ``trigger_id`` to disambiguate.

    Served by the ``trigger_runs_by_owner_name`` index.
    """
    rows = await conn.fetch(
        """
        SELECT * FROM trigger_runs
        WHERE account_id = $1 AND owner_session_id = $2 AND trigger_name = $3
        ORDER BY created_at DESC
        LIMIT $4
        """,
        account_id,
        session_id,
        trigger_name,
        limit,
    )
    return [_row_to_trigger_run_echo(r) for r in rows]


async def list_pending_trigger_run_refs(
    conn: asyncpg.Connection[Any],
    *,
    older_than_seconds: float,
) -> list[TriggerFireRef]:
    """Fires whose post-commit defer was lost (worker crash between the
    run-completion commit and ``defer_async``). The periodic sweep re-defers
    these; the ``pending → running`` claim makes a re-defer racing a live job
    safe. Covered by the ``trigger_runs_unfinished`` partial index.
    """
    rows = await conn.fetch(
        """
        SELECT id, trigger_id FROM trigger_runs
        WHERE status = 'pending' AND created_at < now() - make_interval(secs => $1)
        """,
        older_than_seconds,
    )
    return [TriggerFireRef(r["id"], r["trigger_id"]) for r in rows]


async def count_stuck_running_trigger_runs(
    conn: asyncpg.Connection[Any],
    *,
    older_than_seconds: float,
) -> int:
    """Count fires claimed ``running`` longer than the threshold — a worker
    crashed mid-fire. Deliberately surfaced (sweep warning log) but NEVER
    retried: re-firing could double-launch a run; the at-most-once-after-claim
    trade, observable instead of silent.
    """
    result: int | None = await conn.fetchval(
        """
        SELECT COUNT(*) FROM trigger_runs
        WHERE status = 'running' AND created_at < now() - make_interval(secs => $1)
        """,
        older_than_seconds,
    )
    return result or 0


async def prune_trigger_runs(
    conn: asyncpg.Connection[Any],
    *,
    retention_days: int,
) -> int:
    """Delete audit rows older than the retention window; returns the count.

    Age-keyed on ``created_at`` (the intent timestamp operators sort by).
    Time-based only — a count-cap could evict a young ``run_completion`` claim
    row inside the dispatch-recovery horizon and re-arm a duplicate fire; the
    multi-day retention floor makes that structurally unreachable.
    """
    result = await conn.execute(
        "DELETE FROM trigger_runs WHERE created_at < now() - make_interval(days => $1)",
        retention_days,
    )
    # asyncpg returns e.g. "DELETE 3"
    return int(result.split()[-1])


async def acquire_account_triggers_lock(
    conn: asyncpg.Connection[Any],
    account_id: str,
) -> None:
    """Per-account transaction-scoped advisory lock for cap enforcement.

    Held for the duration of the surrounding transaction; serializes
    concurrent ``count_account_triggers`` + INSERT pairs across workers so
    the cap is contractual instead of approximate. The lock key text
    ``'aios_st_cap:'`` is kept BYTE-IDENTICAL across the rename — it is a
    cross-process hashed string; renaming it buys nothing and would split
    the lock across a deploy window (a 64-bit hash that won't collide with
    other modules' advisory locks).
    """
    await conn.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
        f"aios_st_cap:{account_id}",
    )


async def fetch_next_trigger_event(
    conn: asyncpg.Connection[Any],
    *,
    stale_threshold_seconds: int = 7200,
) -> datetime | None:
    """Return when the event-driven scheduler should next wake.

    Computed as ``MIN(GREATEST(next_fire, running_since + stale_threshold))``
    across enabled rows on non-archived sessions:

    - Idle rows (``running_since IS NULL``) contribute ``next_fire`` — the
      earliest of these is the next genuine fire.
    - Running rows contribute ``running_since + stale_threshold`` — they're
      either in-flight (and the handler clears ``running_since`` long before
      the threshold elapses) or stuck (in which case the threshold is when
      we'll re-claim them).

    Returns ``None`` when no enabled rows exist — the scheduler then sleeps
    until either a NOTIFY or the cold-path heartbeat.

    Cheap: the ``triggers_due`` partial index covers the WHERE clause.
    """
    from datetime import timedelta

    stale_threshold = timedelta(seconds=stale_threshold_seconds)
    result: datetime | None = await conn.fetchval(
        """
        SELECT MIN(
            CASE
                WHEN t.running_since IS NULL THEN t.next_fire
                ELSE GREATEST(t.next_fire, t.running_since + $1)
            END
        )
        FROM triggers AS t
        JOIN sessions AS s ON s.id = t.owner_session_id
        WHERE t.enabled
          AND s.archived_at IS NULL
          AND t.next_fire IS NOT NULL
        """,
        stale_threshold,
    )
    return result
