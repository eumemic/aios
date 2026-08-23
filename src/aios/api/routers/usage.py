"""Account-wide inference attribution endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from aios.api.deps import AccountIdDep, PoolDep
from aios.models.accounting import (
    DEFAULT_USAGE_WINDOW_SECONDS,
    MAX_USAGE_WINDOW_SECONDS,
    MIN_USAGE_WINDOW_SECONDS,
    UsageConsumersResponse,
    UsageMetric,
)
from aios.services import accounting as service

router = APIRouter(prefix="/v1/usage", tags=["usage"])


@router.get("/consumers", operation_id="list_usage_consumers")
async def list_usage_consumers(
    pool: PoolDep,
    account_id: AccountIdDep,
    window_seconds: Annotated[
        int, Query(ge=MIN_USAGE_WINDOW_SECONDS, le=MAX_USAGE_WINDOW_SECONDS)
    ] = DEFAULT_USAGE_WINDOW_SECONDS,
    metric: UsageMetric = "cost_microusd",
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> UsageConsumersResponse:
    """Rank root consumers by rolling creation-subtree inference rate.

    Root consumers are additive: every session/run belongs to exactly one root
    through immutable creation edges, so ``share`` values never double-count
    shared invocation work. Archived descendants remain in the rollup. Rates
    update on every inference charge and are normalized per hour over the
    requested rolling window.
    """
    return await service.ranked_consumers(
        pool,
        account_id=account_id,
        window_seconds=window_seconds,
        metric=metric,
        limit=limit,
    )
