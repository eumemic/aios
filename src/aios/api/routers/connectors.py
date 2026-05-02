"""Connector admin endpoints — stubbed.

The connector subprocess supervisor and inbound handling aren't wired
yet; every endpoint returns 503 so operator tooling gets an explicit
"not implemented" rather than a confusing 404.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from aios.api.deps import AuthDep, PoolDep

router = APIRouter(prefix="/v1/connectors", tags=["connectors"])

_NOT_READY = "connectors not yet implemented"


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
