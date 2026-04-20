"""Routing rule endpoints nested under a connection.

Rules are per-connection resources — the connector/account part of a
channel address is implicit from the owning connection, so rule prefixes
are just the path portion (or ``""`` as the per-connection catch-all).

The ``connection`` dependency 404s before any handler logic runs, so
endpoints can trust the path-param connection id is valid.  ``DELETE``
soft-archives; archived rules are retained for audit.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AuthDep, ConnectionDep, PoolDep
from aios.models.common import ListResponse
from aios.models.routing_rules import (
    RoutingRule,
    RoutingRuleCreate,
    RoutingRuleUpdate,
)
from aios.services import channels as service

router = APIRouter(
    prefix="/v1/connections/{connection_id}/routing-rules",
    tags=["routing-rules"],
)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(
    body: RoutingRuleCreate,
    connection: ConnectionDep,
    pool: PoolDep,
    _auth: AuthDep,
) -> RoutingRule:
    return await service.create_routing_rule(
        pool,
        connection.id,
        prefix=body.prefix,
        target=body.target,
        session_params=body.session_params,
    )


@router.get("")
async def list_(
    connection: ConnectionDep,
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
    after: str | None = None,
) -> ListResponse[RoutingRule]:
    items = await service.list_routing_rules(pool, connection.id, limit=limit, after=after)
    return ListResponse[RoutingRule](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{rule_id}")
async def get(
    rule_id: str,
    connection: ConnectionDep,
    pool: PoolDep,
    _auth: AuthDep,
) -> RoutingRule:
    return await service.get_routing_rule(pool, connection.id, rule_id)


@router.put("/{rule_id}")
async def update(
    rule_id: str,
    body: RoutingRuleUpdate,
    connection: ConnectionDep,
    pool: PoolDep,
    _auth: AuthDep,
) -> RoutingRule:
    return await service.update_routing_rule(
        pool,
        connection.id,
        rule_id,
        target=body.target,
        session_params=body.session_params,
    )


@router.delete("/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(
    rule_id: str,
    connection: ConnectionDep,
    pool: PoolDep,
    _auth: AuthDep,
) -> None:
    await service.archive_routing_rule(pool, connection.id, rule_id)
