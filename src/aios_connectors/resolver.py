"""Three-tier per-chat resolver for inbound (#328).

Given ``(connection, chat_id)`` find or spawn the target session:

1. **chat_sessions ledger** — operator-curated overrides + previously
   spawned per_chat sessions. If a row exists, dispatch to its
   ``session_id`` directly.
2. **routing_rules prefix demux** (#328 PR 2/4) — iterate the
   connection's active binding's rules. The first rule whose
   ``prefix`` matches ``chat_id`` decides the target:
   - ``target_type='session'``: drop into the named session and
     stamp the chat_sessions ledger so future inbounds short-circuit.
   - ``target_type='session_template'``: spawn from the named
     template and stamp the ledger.
3. **bindings.mode fallback** — read the connection's active binding:
   ``single_session`` dispatches to ``binding.session_id``;
   ``per_chat`` spawns from ``binding.session_template_id`` and stamps
   the ledger. No active binding → :data:`ResolveDrop.DETACHED`.

All spawn-and-insert paths are race-safe via the existing
``chat_sessions`` ``ON CONFLICT DO NOTHING RETURNING *`` pattern: the
loser of a race gets the winner's ``session_id`` and the just-spawned
session is left as an orphan for operator cleanup.

This is the only module in ``aios_connectors`` that synthesizes new
sessions; the rest of the subsystem reads against existing tables.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, NamedTuple

import asyncpg

from aios.db import queries
from aios.models.connections import Connection
from aios.services import sessions as sessions_service


class ResolveDrop(StrEnum):
    """Why the resolver refused to produce a target session."""

    DETACHED = "detached"
    ARCHIVED_TEMPLATE = "archived_template"


class ResolveResult(NamedTuple):
    session_id: str | None
    drop: ResolveDrop | None


async def _session_is_archived(
    pool: asyncpg.Pool[Any], session_id: str, *, account_id: str
) -> bool:
    """Return ``True`` iff ``session_id`` exists for ``account_id`` and is archived.

    Shared by the tier-1 ledger and tier-3 single_session checks so both
    surfaces refuse to route inbounds to a session ``append_event``
    (post-#523) would reject — see the call sites for the full
    ``DETACHED`` → 422 cascade rationale.
    """
    async with pool.acquire() as conn:
        archived_at = await conn.fetchval(
            "SELECT archived_at FROM sessions WHERE id = $1 AND account_id = $2",
            session_id,
            account_id,
        )
    return archived_at is not None


async def resolve_target_session(
    pool: asyncpg.Pool[Any], *, account_id: str, connection: Connection, chat_id: str
) -> ResolveResult:
    """Resolve ``(connection, chat_id)`` → ``ResolveResult``.

    Returns ``(session_id, None)`` on success or ``(None, drop_reason)``
    on a terminal refusal (no binding, archived template).
    """
    # Tier 1: chat_sessions ledger — fast path, no spawn.
    async with pool.acquire() as conn:
        existing = await queries.lookup_chat_session(
            conn, connection.id, chat_id, account_id=account_id
        )
    if existing is not None:
        # The ledger entry was stamped at spawn (tier-3) or operator-bind
        # time (when the session was live), but the operator may have
        # archived the bound session since.  Without this check the
        # resolver returns the archived id with ``drop=None``,
        # ``handle_inbound`` proceeds to ``append_event`` which (post-
        # #523) raises NotFoundError, the inbound surfaces as
        # ``SESSION_MISSING`` → HTTP 500, and well-behaved connectors
        # retry-forever on 5xx (same retry-loop pathology PR #526 closed
        # for tier-3 single_session).  ``DETACHED`` (→ 422) is the
        # terminal signal that tells the connector to stop retrying.
        if await _session_is_archived(pool, existing, account_id=account_id):
            return ResolveResult(session_id=None, drop=ResolveDrop.DETACHED)
        return ResolveResult(session_id=existing, drop=None)

    # Tier 2: routing_rules prefix demux on the active binding.
    async with pool.acquire() as conn:
        rules = await queries.list_routing_rules_for_connection(
            conn, connection.id, account_id=account_id
        )
    for prefix, target_type, target_id in rules:
        if not chat_id.startswith(prefix):
            continue
        return await _dispatch_routing_target(
            pool,
            connection=connection,
            chat_id=chat_id,
            target_type=target_type,
            target_id=target_id,
            account_id=account_id,
        )

    # Tier 3: active binding's mode + target.
    async with pool.acquire() as conn:
        binding = await queries.get_active_binding(conn, connection.id, account_id=account_id)
    if binding is None:
        return ResolveResult(session_id=None, drop=ResolveDrop.DETACHED)
    if binding.mode == "single_session":
        assert binding.session_id is not None
        # Check whether the bound session was archived after this
        # binding was created. Without this, the resolver would return
        # the archived id, ``handle_inbound`` would fail at
        # ``append_event`` (post-#523) with ``SESSION_MISSING`` (mapped
        # to 500), and well-behaved connectors would retry forever
        # because 5xx looks transient. ``DETACHED`` (→ 422) is the
        # correct terminal signal — same shape the resolver already
        # uses for ``ARCHIVED_TEMPLATE`` upstream of the append. No
        # ledger stamp here so the next inbound goes through the same
        # check (no poisoning), and if the operator detaches+rebinds
        # to a live session the resolver picks it up immediately.
        if await _session_is_archived(pool, binding.session_id, account_id=account_id):
            return ResolveResult(session_id=None, drop=ResolveDrop.DETACHED)
        # No ledger insert — operators opt into per-chat overrides explicitly.
        return ResolveResult(session_id=binding.session_id, drop=None)
    assert binding.mode == "per_chat"
    assert binding.session_template_id is not None
    return await _spawn_per_chat_session(
        pool,
        connection=connection,
        chat_id=chat_id,
        template_id=binding.session_template_id,
        account_id=account_id,
    )


async def _dispatch_routing_target(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    connection: Connection,
    chat_id: str,
    target_type: str,
    target_id: str,
) -> ResolveResult:
    """Apply a matched routing_rule. Stamps the chat_sessions ledger so
    future inbounds for this ``chat_id`` short-circuit at tier 1.
    """
    if target_type == "session":
        target_session_id = target_id
    elif target_type == "session_template":
        spawn = await _spawn_per_chat_session(
            pool,
            connection=connection,
            chat_id=chat_id,
            template_id=target_id,
            account_id=account_id,
        )
        # Spawn already stamped the ledger; surface its result verbatim.
        return spawn
    else:
        # The schema CHECK constraint forbids other values, so this is
        # a real bug (post-migration data corruption or a future
        # target_type we forgot to handle).
        raise AssertionError(f"unknown routing_rules.target_type: {target_type!r}")

    # Direct-session dispatch: stamp the ledger so future inbounds
    # short-circuit. Race-safe via ON CONFLICT DO NOTHING. Also takes
    # SELECT FOR UPDATE on the connections row + re-checks archived
    # — without it, a concurrent detach+archive can commit between
    # the resolver's binding load and the insert, leaving the
    # chat_session row on an archived connection (chat_sessions.
    # connection_id FK accepts archived rows). Symmetric to
    # bind_chat_to_session's #663 fix; archive_connection's
    # chat_sessions presence check (also #663) closes the opposite
    # race direction.
    async with pool.acquire() as conn, conn.transaction():
        locked = await conn.fetchrow(
            "SELECT archived_at FROM connections WHERE id = $1 AND account_id = $2 FOR UPDATE",
            connection.id,
            account_id,
        )
        if locked is None or locked["archived_at"] is not None:
            return ResolveResult(session_id=None, drop=ResolveDrop.DETACHED)
        registered = await queries.insert_chat_session(
            conn,
            connection_id=connection.id,
            chat_id=chat_id,
            session_id=target_session_id,
            account_id=account_id,
        )
    return ResolveResult(session_id=registered, drop=None)


async def _spawn_per_chat_session(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    connection: Connection,
    chat_id: str,
    template_id: str,
) -> ResolveResult:
    """Spawn a fresh per_chat session from ``template_id`` and stamp the ledger."""
    async with pool.acquire() as conn:
        template = await queries.get_session_template(conn, template_id, account_id=account_id)
    if template.archived_at is not None:
        return ResolveResult(session_id=None, drop=ResolveDrop.ARCHIVED_TEMPLATE)

    focal_channel = f"{connection.connector}/{connection.external_account_id}/{chat_id}"
    session = await sessions_service.create_session(
        pool,
        agent_id=template.agent_id,
        environment_id=template.environment_id,
        agent_version=template.agent_version,
        title=None,
        metadata={},
        vault_ids=template.vault_ids or None,
        focal_channel=focal_channel,
        focal_locked=True,
        account_id=account_id,
    )

    # Race-safe register: loser gets the winner's session_id; just-spawned
    # orphan is left for operator cleanup (same posture as pre-#328
    # ``_resolve_per_chat``). Also takes SELECT FOR UPDATE on the
    # connections row + re-checks archived — see _dispatch_routing_target
    # above for the rationale. The just-created session may end up an
    # orphan when DETACHED is returned; that matches the existing
    # "loser orphan" posture and operator cleanup expectation.
    async with pool.acquire() as conn, conn.transaction():
        locked = await conn.fetchrow(
            "SELECT archived_at FROM connections WHERE id = $1 AND account_id = $2 FOR UPDATE",
            connection.id,
            account_id,
        )
        if locked is None or locked["archived_at"] is not None:
            return ResolveResult(session_id=None, drop=ResolveDrop.DETACHED)
        registered = await queries.insert_chat_session(
            conn,
            connection_id=connection.id,
            chat_id=chat_id,
            session_id=session.id,
            account_id=account_id,
        )
    return ResolveResult(session_id=registered, drop=None)


__all__: list[str] = ["ResolveDrop", "ResolveResult", "resolve_target_session"]
