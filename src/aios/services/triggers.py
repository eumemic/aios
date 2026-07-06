"""Service layer for triggers (#818).

Thin pool-acquiring wrappers around the queries in
:mod:`aios.db.queries`. Owns the business logic around when to recompute
``next_fire`` — on add, on source change, and on re-enable — and exposes
granular add/remove/update/list operations to the API + tool layers.

Arming invariant: an enabled schedulable (cron / one_shot) trigger always has
a non-NULL ``next_fire``. This is enforced by the schema, not the service: the
``triggers_schedulable_enabled_armed`` CHECK (migration 0130) — the
schedulable complement of 0108's reactive ``triggers_reactive_no_next_fire``
— makes the #925 incident state (``enabled=true, next_fire=NULL``, produced by
a manual ``UPDATE … SET enabled=true``) unrepresentable for every writer alike,
including manual psql. So the #957 runtime heal that used to re-arm such rows is
gone: the offending out-of-band write now fails loudly at write time (fail-hard),
and the honest recovery recipe is ``SET enabled=true, next_fire=now()`` (the
claim path recomputes the proper cron slot on the next tick). run_completion /
external_event rows stay ``next_fire`` NULL by design (the 0108 reactive arm).

Deliberately no whole-list-replace primitive; per #270, the only mutation
surface is granular ops.

Triggers do NOT feed :func:`aios.sandbox.spec.build_spec_from_session`
(the runner reads them per-fire), so mutations here force no sandbox
eviction and get no Layer 2 ``spec_version`` trigger (#713). The NOTIFY
trigger on ``triggers`` (migration 0080, byte-identical to 0059's) is
untouched by service-layer writes other than the gated columns.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime
from types import EllipsisType
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.db import queries
from aios.db.queries import workflows as wf_queries
from aios.errors import ConflictError, RateLimitedError, ValidationError
from aios.models.sessions import Session
from aios.models.triggers import (
    MAX_TRIGGERS_PER_SESSION,
    AccountTriggerEcho,
    CronSource,
    ExternalEventSource,
    OneShotSource,
    RunCompletionSource,
    SandboxCommandAction,
    TriggerCreate,
    TriggerCreated,
    TriggerEcho,
    TriggerRunEcho,
    TriggerUpdate,
    WakeOwnerAction,
    WakeSessionAction,
    WorkflowAction,
    compute_initial_next_fire,
)

# Per-trigger ingest secret (external_event). Mirrors the runtime_tokens
# precedent: `aios_evt_<32-byte url-safe>` (256 bits of CSPRNG entropy),
# sha256-at-rest, surfaced plaintext exactly once on create / source-replace.
_INGEST_TOKEN_PREFIX = "aios_evt_"
_INGEST_TOKEN_BYTES = 32


def _mint_ingest_token() -> tuple[str, str]:
    """Return ``(plaintext, sha256_hex)`` for a fresh ingest secret."""
    plaintext = _INGEST_TOKEN_PREFIX + secrets.token_urlsafe(_INGEST_TOKEN_BYTES)
    return plaintext, hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def mint_ingest_token_hash() -> str:
    """Mint a fresh ingest secret and return ONLY its hash (plaintext dropped).

    Used by the session-create attach path, whose response carries no
    per-trigger token surface — the row needs a stored hash to satisfy the
    iff CHECK, and the owner rotates via ``update_trigger`` to obtain a
    usable plaintext.
    """
    return _mint_ingest_token()[1]


async def validate_trigger_spec(
    conn: asyncpg.Connection[Any],
    source: CronSource | OneShotSource | RunCompletionSource | ExternalEventSource | None,
    action: SandboxCommandAction | WakeOwnerAction | WakeSessionAction | WorkflowAction | None,
    *,
    session_id: str,
    account_id: str,
    session: Session | None = None,
) -> str | None:
    """Shared write-path validation for a trigger's source/action references;
    returns the resolved ``environment_id`` (``workflow`` actions) or ``None``.

    Called from EVERY write path into ``triggers`` — ``add_trigger``,
    ``update_trigger`` (provided pieces only), and ``create_session``'s
    ``SessionCreate.triggers`` attachment — so the three stay consistent. The
    typed ``action`` is part of the contract: only PARSED union members reach
    this chokepoint (the Replace variants are subclasses), so a caller cannot
    silently skip workflow validation with an unparsed dict.

    The account-scoped session read is FIRST and UNCONDITIONAL: the HTTP path
    takes ``session_id`` from the URL, and without this check account A could
    attach a trigger to account B's session (for ``sandbox_command`` that is
    cross-tenant code execution at fire time). 404s like every other scoped
    read. ``create_session``'s attach loop passes its just-inserted ``session``
    instead — same-transaction, same-account by construction — so attaching N
    specs doesn't re-read the row N times.

    Reference checks are correctness, not authority (authority is enforced at
    every fire by ``create_run``, where it cannot go stale): the watched
    workflow must exist account-scoped (a foreign/absent watch never matches,
    so it would sit silently dead forever — the silent-dead-cron analog of the
    occurrence-horizon check); the action's workflow must exist account-scoped,
    and an integer ``workflow_version`` pin must equal its CURRENT version
    ("resolve-latest-at-write": the write resolves what the pin freezes;
    mismatch is a stale optimistic token → 409, mirroring ``update_workflow``).
    Deliberately NO vault checks: ``action.vault_ids`` are re-checked against
    the owner's live vaults at every fire — a write-time subset check would go
    stale in both directions.
    """
    if session is None:
        session = await queries.get_session_bare(conn, session_id, account_id=account_id)
    else:
        # The bypass exists ONLY for create_session's attach loop (the row it
        # just inserted under this account); a misaimed session would silently
        # skip the cross-tenant check this helper was built to enforce.
        assert session.id == session_id
    if isinstance(source, RunCompletionSource):
        await wf_queries.get_workflow(conn, source.workflow_id, account_id=account_id)
    if isinstance(source, ExternalEventSource):
        # Valid standalone: an external_event source references no watched
        # workflow (the inbound webhook body IS the event). Explicit no-op
        # branch so a future reviewer sees the case was considered, not missed.
        pass
    if not isinstance(action, WorkflowAction):
        return None
    workflow = await wf_queries.get_workflow(conn, action.workflow_id, account_id=account_id)
    if action.workflow_version is not None and action.workflow_version != workflow.version:
        raise ConflictError(
            f"workflow_version pin {action.workflow_version} does not match current "
            f"version {workflow.version}; re-read the workflow and pin the version "
            "you reviewed",
            detail={"pinned": action.workflow_version, "current": workflow.version},
        )
    # Sessions' environment is immutable, so resolving it at write time equals
    # fire-time resolution; environment_id is deliberately NOT a wire field (a
    # caller-chosen env would bypass the call_workflow builtin's same refusal).
    return session.environment_id


async def add_trigger(
    pool: asyncpg.Pool[Any],
    session_id: str,
    spec: TriggerCreate,
    *,
    account_id: str,
) -> TriggerCreated:
    """Add a trigger to a session.

    For an ``external_event`` source, mints a per-trigger ingest secret
    (``aios_evt_…``), stores only its SHA-256 hash, and surfaces the plaintext
    EXACTLY ONCE on the returned :class:`TriggerCreated` (``ingest_token``);
    every other source returns ``ingest_token=None``. The plaintext is never
    persisted — losing it means rotating via ``update_trigger``.

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
    ingest_plaintext: str | None = None
    ingest_token_hash: str | None = None
    if isinstance(spec.source, ExternalEventSource):
        ingest_plaintext, ingest_token_hash = _mint_ingest_token()
    async with pool.acquire() as conn, conn.transaction():
        # Validate the spec's references (and account-scope the session) BEFORE
        # the advisory lock — a doomed spec never serializes other writers.
        environment_id = await validate_trigger_spec(
            conn, spec.source, spec.action, session_id=session_id, account_id=account_id
        )
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
                    f"account at active-trigger cap ({existing_account}/{cap}); "
                    "remove or disable an existing trigger to free a slot"
                )
        echo = await queries.add_trigger(
            conn,
            session_id,
            name=spec.name,
            source=spec.source.kind,
            source_spec=spec.source.model_dump(mode="json", exclude={"kind"}),
            action=spec.action.model_dump(mode="json"),
            enabled=spec.enabled,
            metadata=spec.metadata,
            next_fire=next_fire,
            environment_id=environment_id,
            ingest_token_hash=ingest_token_hash,
            account_id=account_id,
        )
        return TriggerCreated(**echo.model_dump(), ingest_token=ingest_plaintext)


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
) -> TriggerCreated:
    """Update a trigger by name (§2.4 of the design contract).

    Source-replace semantics for the ingest secret (mirrors ``add_trigger``):
    replacing the source TO ``external_event`` mints a fresh secret + hash and
    surfaces the plaintext once on the returned :class:`TriggerCreated`
    (re-minting an already-external_event source = rotation); replacing AWAY
    from ``external_event`` NULLs the stored hash. A non-source-touching update
    leaves the hash alone and returns ``ingest_token=None``. external_event and
    run_completion stay ``next_fire`` NULL throughout (the 0108 reactive arm).

    ``source`` / ``action`` are replaced WHOLESALE when provided (a
    cron↔one-shot or sandbox↔wake conversion is just a different object —
    invalid shapes are unrepresentable, caught by the discriminated-union
    422). The business rules below are keyed on the MERGED final state:

    - Disabling (true→false): clears ``next_fire``.
    - Re-enabling (false→true): recomputes ``next_fire`` from the merged
      source, under the per-account active-trigger cap (a disabled row didn't
      hold a slot), and rejects a one-shot whose merged ``fire_at`` is
      already in the past.
    - Source replaced on a row whose final state is enabled: recomputes
      ``next_fire`` (no cap re-check — an already-enabled row holds its
      slot), with the same past-``fire_at`` rejection.
    - action / metadata / no-op: ``next_fire`` untouched.

    The invariant "an enabled schedulable (cron / one_shot) trigger always has
    non-NULL ``next_fire``" is enforced by the ``triggers_schedulable_enabled_armed``
    CHECK (migration 0130), not by a runtime heal — the #925 incident state
    (manual ``UPDATE … SET enabled=true`` leaving ``next_fire`` NULL) is
    unrepresentable, so it fails at the offending write rather than being
    silently corrected later.

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
        # Validate only the PROVIDED pieces (re-validating an untouched stored
        # pin must not 409 an unrelated enable-flip). environment_id is
        # recomputed WHENEVER action is provided — the resolved env for a
        # workflow action, NULL otherwise — so the jsonb and the column flip in
        # one UPDATE (the iff CHECK catches forgotten kind conversions; a
        # same-kind workflow→workflow replacement with a stale column would be
        # silent without this).
        environment_id: str | None | EllipsisType = ...
        if update.source is not None or update.action is not None:
            resolved = await validate_trigger_spec(
                conn, update.source, update.action, session_id=session_id, account_id=account_id
            )
            if update.action is not None:
                environment_id = resolved
        now = datetime.now(UTC)
        new_enabled = update.enabled if update.enabled is not None else current.enabled
        merged_source = update.source if update.source is not None else current.source

        next_fire: datetime | None | EllipsisType = ...  # ... = leave alone
        reenabled = new_enabled and not current.enabled
        # The arming invariant ("an enabled schedulable trigger always has
        # non-NULL next_fire") is enforced by the triggers_schedulable_enabled_armed
        # CHECK (migration 0130), so there is no heal disjunct here: the only
        # producers of a NULL-next_fire-on-enabled row (the #925 manual
        # `UPDATE … SET enabled=true`) are now rejected at write time. We recompute
        # next_fire only on a genuine re-enable (false->true) or a source replace
        # on an already-enabled row; the per-account cap re-check and the
        # consecutive-failures reset are gated on `not current.enabled` / `reenabled`
        # below, so they apply only to a genuine re-enable.
        if not new_enabled and current.enabled:
            # Disabling: clear next_fire.
            next_fire = None
        elif new_enabled and (source_provided or not current.enabled):
            # Re-enabling, or replacing the source on a row whose final
            # state is enabled: recompute next_fire from the merged source.
            if not current.enabled:
                # Re-enable consumes a per-account active-trigger slot (a
                # disabled row didn't); take the lock + cap check so this
                # can't race past the cap against concurrent adds.
                await queries.acquire_account_triggers_lock(conn, account_id)
                cap = get_settings().triggers_per_account_max
                existing = await queries.count_account_triggers(
                    conn, account_id=account_id, enabled_only=True
                )
                if existing >= cap:
                    raise RateLimitedError(
                        f"account at active-trigger cap ({existing}/{cap}); remove "
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

        # Ingest-secret lifecycle keyed on a PROVIDED source replacement only
        # (a no-op or action/metadata/enabled-only update leaves the column
        # alone — Ellipsis sentinel). Replace TO external_event mints+surfaces
        # (re-mint of an already-external_event source is rotation); replace
        # AWAY NULLs the hash (the iff CHECK would otherwise reject the row).
        ingest_token_hash: str | None | EllipsisType = ...
        ingest_plaintext: str | None = None
        if source_provided:
            if isinstance(update.source, ExternalEventSource):
                ingest_plaintext, ingest_token_hash = _mint_ingest_token()
            else:
                ingest_token_hash = None

        echo = await queries.update_trigger(
            conn,
            session_id,
            name,
            source=new_source,
            source_spec=new_source_spec,
            action=new_action,
            enabled=update.enabled,
            metadata=update.metadata,
            next_fire=next_fire,
            environment_id=environment_id,
            ingest_token_hash=ingest_token_hash,
            # A re-enable is a fresh start for the failure counter — without
            # the reset, a counter parked past MAX_CONSECUTIVE_FAILURES would
            # never EQUAL it again and the auto-disable breaker would be
            # permanently disarmed for a still-broken trigger.
            reset_consecutive_failures=reenabled,
            account_id=account_id,
        )
        return TriggerCreated(**echo.model_dump(), ingest_token=ingest_plaintext)


async def list_triggers(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
) -> list[TriggerEcho]:
    async with pool.acquire() as conn:
        return await queries.list_triggers(conn, session_id, account_id=account_id)


async def list_account_triggers(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    enabled_only: bool = True,
) -> list[AccountTriggerEcho]:
    """List every trigger in ``account_id`` (across all its sessions) — the
    account-wide read backing the ops-agent O7 trigger-liveness sweep (#1673).

    The account-wide analog of :func:`list_triggers` (session-scoped): a caller
    that legitimately holds the whole account (a workflow run, dispatched
    account-scoped to ``run.account_id``) enumerates every enabled trigger and
    reads each ``next_fire`` to catch the #925 zombie class. A thin
    pool-acquiring wrapper — a READ only; the #925 zombie state is now
    unrepresentable at write time (the ``triggers_schedulable_enabled_armed``
    CHECK, migration 0130), so this sweep is a defense-in-depth liveness read,
    not a corrective.
    """
    async with pool.acquire() as conn:
        return await queries.list_account_triggers(
            conn, account_id=account_id, enabled_only=enabled_only
        )


async def list_trigger_runs(
    pool: asyncpg.Pool[Any],
    session_id: str,
    name: str,
    *,
    account_id: str,
    limit: int = 50,
) -> list[TriggerRunEcho]:
    async with pool.acquire() as conn:
        return await queries.list_trigger_runs(
            conn, account_id=account_id, session_id=session_id, trigger_name=name, limit=limit
        )
