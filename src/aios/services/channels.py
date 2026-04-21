"""Channel resolution and routing-resource business logic.

The headline function is :func:`resolve_channel`: given a channel
*address* (``{connector}/{account}/{path}``), return the session it
maps to.  Two-tier resolution:

1. If an explicit :class:`~aios.models.channel_bindings.ChannelBinding`
   exists for this address, return its session.  Fast path.
2. Otherwise, find the longest-matching segment-aware prefix
   :class:`~aios.models.routing_rules.RoutingRule`, parse its target,
   and either look up the existing session (``session:`` target) or
   create a fresh one (``agent:`` target).  Either way, persist a
   binding so the next message short-circuits.
3. If neither tier matches, raise :class:`~aios.errors.NoRouteError`.

This module also owns CRUD for both bindings and routing rules so the
target-validation logic (``agent:`` requires ``environment_id``,
``session:`` rejects ``session_params``) lives next to the resolver
that depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import asyncpg

from aios.db import queries
from aios.errors import NoRouteError, NotFoundError, ValidationError
from aios.models._paths import validate_path_segments
from aios.models.channel_bindings import ChannelBinding
from aios.models.connections import Connection
from aios.models.routing_rules import RoutingRule, SessionParams

# ─── target parsing ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class AgentTarget:
    kind: Literal["agent"]
    agent_id: str
    agent_version: int | None


@dataclass(slots=True)
class SessionTarget:
    kind: Literal["session"]
    session_id: str


Target = AgentTarget | SessionTarget


def parse_target(s: str) -> Target:
    """Parse a routing-rule target string.

    Format: ``agent:<id>[@<version>]`` or ``session:<id>``.
    Raises :class:`ValueError` on anything else.
    """
    if s.startswith("agent:"):
        rest = s[len("agent:") :]
        if not rest:
            raise ValueError(f"invalid target: {s!r}")
        if "@" in rest:
            agent_id, _, version_str = rest.rpartition("@")
            if not agent_id or not version_str:
                raise ValueError(f"invalid target: {s!r}")
            try:
                version = int(version_str)
            except ValueError as exc:
                raise ValueError(f"invalid target: {s!r}") from exc
            return AgentTarget(kind="agent", agent_id=agent_id, agent_version=version)
        return AgentTarget(kind="agent", agent_id=rest, agent_version=None)
    if s.startswith("session:"):
        rest = s[len("session:") :]
        if not rest:
            raise ValueError(f"invalid target: {s!r}")
        return SessionTarget(kind="session", session_id=rest)
    raise ValueError(f"invalid target: {s!r}")


def _validate_rule_target(target: str, session_params: SessionParams) -> None:
    """Enforce the cross-field rules between ``target`` and ``session_params``.

    Raises :class:`ValidationError` (422) on failure.
    """
    try:
        parsed = parse_target(target)
    except ValueError as exc:
        raise ValidationError(str(exc), detail={"target": target}) from exc

    if parsed.kind == "agent":
        if session_params.environment_id is None:
            raise ValidationError(
                "agent: targets require session_params.environment_id",
                detail={"target": target},
            )
    else:
        # session: targets must not carry session_params
        if (
            session_params.environment_id is not None
            or session_params.vault_ids
            or session_params.title is not None
            or session_params.metadata
        ):
            raise ValidationError(
                "session: targets must have empty session_params",
                detail={"target": target},
            )


# ─── routing-rule CRUD ──────────────────────────────────────────────────────
#
# ``target`` and ``session_params`` are NOT NULL columns, so ``None`` in the
# update path cleanly means "don't change this field" — no _UNSET sentinel
# needed (matches the vaults.update_vault style).


async def create_routing_rule(
    pool: asyncpg.Pool[Any],
    connection_id: str,
    *,
    prefix: str,
    target: str,
    session_params: SessionParams,
) -> RoutingRule:
    _validate_rule_target(target, session_params)
    async with pool.acquire() as conn:
        return await queries.insert_routing_rule(
            conn,
            connection_id=connection_id,
            prefix=prefix,
            target=target,
            session_params=session_params,
        )


async def get_routing_rule(
    pool: asyncpg.Pool[Any], connection_id: str, rule_id: str
) -> RoutingRule:
    async with pool.acquire() as conn:
        return await queries.get_routing_rule(conn, connection_id, rule_id)


async def list_routing_rules(
    pool: asyncpg.Pool[Any],
    connection_id: str,
    *,
    limit: int = 50,
    after: str | None = None,
) -> list[RoutingRule]:
    async with pool.acquire() as conn:
        return await queries.list_routing_rules(conn, connection_id, limit=limit, after=after)


async def update_routing_rule(
    pool: asyncpg.Pool[Any],
    connection_id: str,
    rule_id: str,
    *,
    target: str | None = None,
    session_params: SessionParams | None = None,
) -> RoutingRule:
    if target is not None or session_params is not None:
        async with pool.acquire() as conn:
            current = await queries.get_routing_rule(conn, connection_id, rule_id)
        new_target = target if target is not None else current.target
        new_params = session_params if session_params is not None else current.session_params
        _validate_rule_target(new_target, new_params)

    async with pool.acquire() as conn:
        return await queries.update_routing_rule(
            conn,
            connection_id,
            rule_id,
            target=target,
            session_params=session_params,
        )


async def archive_routing_rule(
    pool: asyncpg.Pool[Any], connection_id: str, rule_id: str
) -> RoutingRule:
    async with pool.acquire() as conn:
        return await queries.archive_routing_rule(conn, connection_id, rule_id)


# ─── binding CRUD ───────────────────────────────────────────────────────────


def _parse_address(address: str) -> tuple[str, str, str]:
    """Split ``{connector}/{account}/{path}`` into its three parts.

    The path portion is validated (non-empty, no empty/.. segments) so
    the resolver can trust it downstream.  Raises :class:`ValidationError`
    on anything malformed.
    """
    parts = address.split("/", 2)
    if len(parts) < 3:
        raise ValidationError(
            "address must be {connector}/{account}/{path}",
            detail={"address": address},
        )
    connector, account, path = parts
    if not connector or not account:
        raise ValidationError(
            "address must be {connector}/{account}/{path}",
            detail={"address": address},
        )
    try:
        validate_path_segments(path, allow_empty=False)
    except ValueError as exc:
        raise ValidationError(f"address path {exc}", detail={"address": address}) from exc
    return connector, account, path


async def create_binding(
    pool: asyncpg.Pool[Any], *, address: str, session_id: str
) -> ChannelBinding:
    connector, account, path = _parse_address(address)
    async with pool.acquire() as conn:
        # Resolve the (connector, account) pair to a registered connection.
        pairs = await queries.get_connections_by_pairs(conn, [(connector, account)])
        if not pairs:
            raise NotFoundError(
                f"no registered connection for {connector}/{account}",
                detail={"connector": connector, "account": account},
            )
        return await queries.insert_binding(
            conn,
            connection_id=pairs[0].id,
            path=path,
            session_id=session_id,
        )


async def get_binding(pool: asyncpg.Pool[Any], binding_id: str) -> ChannelBinding:
    async with pool.acquire() as conn:
        return await queries.get_binding(conn, binding_id)


async def list_bindings(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str | None = None,
    limit: int = 50,
    after: str | None = None,
) -> list[ChannelBinding]:
    async with pool.acquire() as conn:
        return await queries.list_bindings(conn, session_id=session_id, limit=limit, after=after)


async def archive_binding(pool: asyncpg.Pool[Any], binding_id: str) -> ChannelBinding:
    async with pool.acquire() as conn:
        return await queries.archive_binding(conn, binding_id)


# ─── resolve_channel ────────────────────────────────────────────────────────


@dataclass(slots=True)
class ResolveResult:
    session_id: str
    binding_id: str
    created_session: bool


def _render_title(template: str | None, address: str) -> str | None:
    if template is None:
        return None
    return template.replace("{address}", address)


async def resolve_channel(
    pool: asyncpg.Pool[Any], connection: Connection, path: str
) -> ResolveResult:
    """Resolve a ``(connection, path)`` to a session, creating one if a rule matches.

    Binding/rule match runs under a single transaction so the binding's
    FK to ``sessions.id`` is satisfied atomically.
    """
    address = f"{connection.connector}/{connection.account}/{path}"
    async with pool.acquire() as conn, conn.transaction():
        # Optimistic: most resolves are binding hits and need no lock.
        existing = await queries.get_binding_by_connection_and_path(conn, connection.id, path)
        if existing is not None:
            return ResolveResult(
                session_id=existing.session_id,
                binding_id=existing.id,
                created_session=False,
            )

        # Per-address advisory lock so concurrent first-time resolves of
        # the same address don't both reach insert_binding and spuriously
        # 409.  ``hashtextextended`` (64-bit) avoids collisions across
        # unrelated addresses in the lock space.
        await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", address)
        existing = await queries.get_binding_by_connection_and_path(conn, connection.id, path)
        if existing is not None:
            return ResolveResult(
                session_id=existing.session_id,
                binding_id=existing.id,
                created_session=False,
            )

        rule = await queries.find_matching_rule(conn, connection.id, path)
        if rule is None:
            raise NoRouteError(
                f"no binding or rule matches address {address}",
                detail={"address": address},
            )

        target = parse_target(rule.target)

        if isinstance(target, SessionTarget):
            # Binding to an archived session would silently resurrect it.
            session = await queries.get_session(conn, target.session_id)
            if session.archived_at is not None:
                raise NotFoundError(
                    f"session {target.session_id} is archived",
                    detail={"id": target.session_id},
                )
            session_id = session.id
            created = False
        else:
            # Validation at create/update guarantees environment_id is set.
            assert rule.session_params.environment_id is not None
            session = await queries.insert_session(
                conn,
                agent_id=target.agent_id,
                agent_version=target.agent_version,
                environment_id=rule.session_params.environment_id,
                title=_render_title(rule.session_params.title, address),
                metadata=rule.session_params.metadata,
            )
            if rule.session_params.vault_ids:
                await queries.set_session_vaults(conn, session.id, rule.session_params.vault_ids)
            session_id = session.id
            created = True

        binding = await queries.insert_binding(
            conn, connection_id=connection.id, path=path, session_id=session_id
        )
        return ResolveResult(
            session_id=session_id,
            binding_id=binding.id,
            created_session=created,
        )
