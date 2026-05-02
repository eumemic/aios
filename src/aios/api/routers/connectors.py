"""Connector admin endpoints — stubbed in PR1, wired in PR2/PR3.

PR2 adds the supervisor that spawns connector subprocesses and the
procrastinate ``connector_call`` task.  PR3 wires inbound handling.
Until then every endpoint returns 503 so callers (the operator CLI in
particular) see an explicit "not yet" rather than a confusing 404.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from aios.api.deps import AuthDep, PoolDep

router = APIRouter(prefix="/v1/connectors", tags=["connectors"])

_NOT_READY = "connector subprocess machinery ships in PR2/PR3"


@router.get("")
async def list_(_pool: PoolDep, _auth: AuthDep) -> dict[str, str]:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_NOT_READY)


@router.get("/{name}/accounts")
async def list_accounts(name: str, _pool: PoolDep, _auth: AuthDep) -> dict[str, str]:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_NOT_READY)


@router.get("/{name}/tools")
async def list_tools(name: str, _pool: PoolDep, _auth: AuthDep) -> dict[str, str]:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_NOT_READY)


@router.post("/{name}/call")
async def call(name: str, _pool: PoolDep, _auth: AuthDep) -> dict[str, str]:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_NOT_READY)
