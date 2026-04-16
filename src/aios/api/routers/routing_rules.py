"""Routing rule endpoints — fallback prefix-match for unbound addresses."""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AuthDep, PoolDep
from aios.models.common import ListResponse
from aios.models.routing_rules import (
    RoutingRule,
    RoutingRuleCreate,
    RoutingRuleUpdate,
)
from aios.services import channels as service

router = APIRouter(prefix="/v1/routing-rules", tags=["routing-rules"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(body: RoutingRuleCreate, pool: PoolDep, _auth: AuthDep) -> RoutingRule:
    return await service.create_routing_rule(
        pool,
        prefix=body.prefix,
        target=body.target,
        session_params=body.session_params,
    )


@router.get("")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
    after: str | None = None,
) -> ListResponse[RoutingRule]:
    items = await service.list_routing_rules(pool, limit=limit, after=after)
    return ListResponse[RoutingRule](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{rule_id}")
async def get(rule_id: str, pool: PoolDep, _auth: AuthDep) -> RoutingRule:
    return await service.get_routing_rule(pool, rule_id)


@router.put("/{rule_id}")
async def update(
    rule_id: str, body: RoutingRuleUpdate, pool: PoolDep, _auth: AuthDep
) -> RoutingRule:
    return await service.update_routing_rule(
        pool,
        rule_id,
        target=body.target,
        session_params=body.session_params,
    )


@router.delete("/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(rule_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    await service.archive_routing_rule(pool, rule_id)
