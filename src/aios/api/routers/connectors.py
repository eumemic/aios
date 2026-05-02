"""Connector admin endpoints.

The connector subprocesses live on the worker process; the API process
talks to them indirectly via procrastinate jobs that NOTIFY back when
done.  Each handler:

1. Mints a ``call_id`` ULID and ``LISTEN``s on the result channel
   first, before enqueuing — the LISTEN-before-action invariant from
   :mod:`aios.db.listen` makes sure NOTIFY can't fire into a dead
   subscriber and get dropped.
2. Defers the matching ``harness.connector_*`` task.
3. Awaits one NOTIFY payload with a 60-second ceiling.
4. Translates ``error`` envelopes into HTTP status codes.

408 (Request Timeout) means the worker didn't NOTIFY within 60s — the
job may still be queued / running; the operator should retry or look
at the worker log.  503 (Service Unavailable) means the worker isn't
running connector machinery (no supervisor, name not enabled) or the
connector itself is down.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict
from ulid import ULID

from aios.api.deps import AuthDep, DbUrlDep
from aios.db.listen import listen_for_connector_result
from aios.harness.connector_tasks import (
    defer_connector_call,
    defer_connector_status,
    defer_connector_tools,
)

router = APIRouter(prefix="/v1/connectors", tags=["connectors"])

_RESULT_TIMEOUT_S = 60.0


class ConnectorCallBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, Any] = {}
    meta: dict[str, Any] | None = None


async def _rpc(db_url: str, defer: Callable[[str], Awaitable[None]]) -> dict[str, Any]:
    """LISTEN-then-defer-then-await for one connector RPC round-trip.

    ``defer`` is the call-site closure that enqueues the matching
    procrastinate task with the minted ``call_id``.  Order matters:
    LISTEN must be live before defer, otherwise a fast worker could
    NOTIFY before the listener is attached and we'd hang waiting for
    a payload that's already been dropped.
    """
    call_id = str(ULID())
    async with listen_for_connector_result(db_url, call_id) as queue:
        await defer(call_id)
        try:
            payload = await asyncio.wait_for(queue.get(), timeout=_RESULT_TIMEOUT_S)
        except TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="worker did not respond within 60s",
            ) from exc
    envelope: dict[str, Any] = json.loads(payload)
    _raise_for_error(envelope)
    return envelope


_CODE_TO_STATUS: dict[str, int] = {
    "not_enabled": status.HTTP_404_NOT_FOUND,
    "not_ready": status.HTTP_503_SERVICE_UNAVAILABLE,
    "circuit_open": status.HTTP_503_SERVICE_UNAVAILABLE,
    "transport_error": status.HTTP_503_SERVICE_UNAVAILABLE,
    "tool_error": status.HTTP_502_BAD_GATEWAY,
}


def _raise_for_error(envelope: dict[str, Any]) -> None:
    """Map the worker's error envelope onto the right HTTP status.

    Producer-side codes are the contract; the human-readable
    ``error`` string is for the operator's eyes.  An unknown / missing
    code falls through to 502 so a future code addition that we don't
    yet recognize doesn't masquerade as success.
    """
    err = envelope.get("error")
    if not err:
        return
    raw_code = envelope.get("code")
    code = raw_code if isinstance(raw_code, str) else ""
    raise HTTPException(
        status_code=_CODE_TO_STATUS.get(code, status.HTTP_502_BAD_GATEWAY),
        detail=err,
    )


@router.get("")
async def list_(db_url: DbUrlDep, _auth: AuthDep) -> dict[str, Any]:
    return await _rpc(db_url, lambda cid: defer_connector_status(call_id=cid, name=None))


@router.get("/{name}/accounts")
async def list_accounts(name: str, db_url: DbUrlDep, _auth: AuthDep) -> dict[str, Any]:
    envelope = await _rpc(db_url, lambda cid: defer_connector_status(call_id=cid, name=name))
    connector = envelope["connector"]
    return {"name": connector["name"], "accounts": connector["accounts"]}


@router.get("/{name}/tools")
async def list_tools(name: str, db_url: DbUrlDep, _auth: AuthDep) -> dict[str, Any]:
    return await _rpc(db_url, lambda cid: defer_connector_tools(call_id=cid, name=name))


@router.post("/{name}/call")
async def call(
    name: str,
    body: ConnectorCallBody,
    db_url: DbUrlDep,
    _auth: AuthDep,
) -> dict[str, Any]:
    return await _rpc(
        db_url,
        lambda cid: defer_connector_call(
            call_id=cid,
            name=name,
            tool=body.tool,
            arguments=body.arguments,
            meta=body.meta,
        ),
    )
