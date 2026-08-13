"""Root-only administrative introspection."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from aios.api.deps import PoolDep, require_root_account
from aios.models.db_stats import DatabaseStats
from aios.services import db_stats

router = APIRouter(
    prefix="/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_root_account)],
)


@router.get("/db-stats", operation_id="get_database_stats", include_in_schema=False)
async def get_database_stats(pool: PoolDep) -> DatabaseStats:
    """Return cached catalog and planner estimates of database storage use.

    Root-account keys only. Non-root callers receive 404 so the global
    introspection surface is not disclosed to tenants.
    """
    return await db_stats.collect_database_stats(pool)
