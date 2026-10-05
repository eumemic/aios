"""Operator-owned triggers (#2473): ``/v1/triggers``.

A trigger an operator owns has no session. It fires on a timer (``cron`` or
``one_shot``) and launches a workflow run with a required ``budget_usd``; the
run is an operator run, like one from ``POST /v1/runs``. Account-scoped via
``AccountIdDep``, keyed by name (unique among the account's operator triggers).

Agents can't reach these: their trigger tools and the session routes
(``/v1/sessions/{id}/triggers``) are keyed by an owner session, which an operator
trigger doesn't have. ``POST /v1/triggers/ingest/{token}`` shares the prefix, so
the name ``ingest`` is reserved.

An operator trigger's failures reach no session. Its last fire status, failure
count and auto-disable are on the trigger, and each fire is in
``GET /v1/triggers/{name}/runs``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from aios.api.deps import AccountIdDep, PoolDep
from aios.models.common import ListResponse
from aios.models.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from aios.models.triggers import (
    OperatorTriggerCreate,
    OperatorTriggerEcho,
    OperatorTriggerUpdate,
    TriggerRunEcho,
)
from aios.services import triggers as triggers_service

router = APIRouter(prefix="/v1/triggers", tags=["triggers"])


@router.post(
    "",
    operation_id="create_operator_trigger",
    status_code=status.HTTP_201_CREATED,
)
async def create_operator_trigger(
    body: OperatorTriggerCreate, pool: PoolDep, account_id: AccountIdDep
) -> OperatorTriggerEcho:
    """Create an operator trigger. Its source is ``cron`` or ``one_shot`` and its
    action a ``workflow`` with ``budget_usd``; each fire launches an operator run
    in ``environment_id``. ``warnings`` carries lint findings."""
    return await triggers_service.add_operator_trigger(pool, body, account_id=account_id)


@router.get("", operation_id="list_operator_triggers")
async def list_operator_triggers(
    pool: PoolDep, account_id: AccountIdDep
) -> ListResponse[OperatorTriggerEcho]:
    """List the account's operator triggers."""
    triggers = await triggers_service.list_operator_triggers(pool, account_id=account_id)
    return ListResponse[OperatorTriggerEcho](data=triggers)


@router.get("/{name}", operation_id="get_operator_trigger")
async def get_operator_trigger(
    name: str, pool: PoolDep, account_id: AccountIdDep
) -> OperatorTriggerEcho:
    """Get an operator trigger by name, with its last fire status and failure count."""
    return await triggers_service.get_operator_trigger(pool, name, account_id=account_id)


@router.put("/{name}", operation_id="update_operator_trigger")
async def update_operator_trigger(
    name: str, body: OperatorTriggerUpdate, pool: PoolDep, account_id: AccountIdDep
) -> OperatorTriggerEcho:
    """Replace an operator trigger's source/action/enabled/metadata by name. Omitted
    fields unchanged; ``source`` / ``action`` replace wholesale."""
    return await triggers_service.update_operator_trigger(pool, name, body, account_id=account_id)


@router.delete(
    "/{name}",
    operation_id="delete_operator_trigger",
    status_code=status.HTTP_204_NO_CONTENT,
    openapi_extra={"x-codegen": {"mcp": {"destructiveHint": True}}},
)
async def delete_operator_trigger(name: str, pool: PoolDep, account_id: AccountIdDep) -> None:
    """Delete an operator trigger by name. Runs it already launched keep running, and
    its fire history stays readable at ``/{name}/runs``."""
    await triggers_service.remove_operator_trigger(pool, name, account_id=account_id)


@router.get("/{name}/runs", operation_id="list_operator_trigger_runs")
async def list_operator_trigger_runs(
    name: str,
    pool: PoolDep,
    account_id: AccountIdDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = DEFAULT_PAGE_LIMIT,
) -> ListResponse[TriggerRunEcho]:
    """List an operator trigger's fires (the per-fire audit), newest first. Keyed by
    name, so a deleted trigger's history stays readable until retention prunes it."""
    runs = await triggers_service.list_operator_trigger_runs(
        pool, name, account_id=account_id, limit=limit
    )
    return ListResponse[TriggerRunEcho](data=runs)
