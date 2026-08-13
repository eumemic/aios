"""Root-only administrative introspection."""

from __future__ import annotations

from fastapi import APIRouter

from aios.api.deps import AccountIdDep, PoolDep
from aios.db import queries
from aios.errors import NotFoundError
from aios.models.db_stats import DatabaseStats
from aios.services import db_stats

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.get("/db-stats", operation_id="get_database_stats")
async def get_database_stats(pool: PoolDep, account_id: AccountIdDep) -> DatabaseStats:
    """Return cached catalog and planner estimates of database storage use.

    Root-account keys only. Non-root callers receive 404 so the global
    introspection surface is not disclosed to tenants.
    """
    async with pool.acquire() as conn:
        account = await queries.get_account(conn, account_id)
    if account is None or account.parent_account_id is not None:
        raise NotFoundError("not found")
    return await db_stats.collect_database_stats(pool)
