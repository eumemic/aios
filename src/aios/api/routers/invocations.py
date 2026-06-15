"""The API caller's request-*writer*: ``POST /v1/invocations`` (#1128).

One kind-agnostic endpoint that, for an external/operator caller, **writes the
trusted request edge** (#1123) and **resolves-or-creates a servicer** (a session
or a run), returning a structured handle. The API caller is *ephemeral* —
nothing to wake; the handle is its continuation, awaited via the already-shipped
completion endpoints (``GET /v1/sessions/{id}/await`` with ``request_id``,
``GET /v1/runs/{id}/wait``). The caller picks the matching awaiter off the
handle's ``servicer_kind``.

The handle is **not** an auth boundary (``await`` re-authorizes by
``account_id``), so it ships as plain JSON fields — no opaque encoding.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AccountIdDep, CryptoBoxDep, PoolDep
from aios.logging import get_logger
from aios.models.invocations import InvocationHandle, InvocationRequest
from aios.services import sessions as service

log = get_logger("aios.api.routers.invocations")

router = APIRouter(prefix="/v1/invocations", tags=["invocations"])


@router.post("", operation_id="invoke", status_code=status.HTTP_201_CREATED)
async def invoke(
    body: InvocationRequest,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    account_id: AccountIdDep,
) -> InvocationHandle:
    """Write a trusted request edge + resolve-or-create a servicer, returning a handle.

    ``target_kind=agent`` creates a **session** servicer (env-bound) and injects a
    channel-less request; ``target_kind=workflow`` creates a **run** servicer;
    ``target_kind=session`` invokes an existing same-account session by id. The
    returned ``request_id`` correlates the matching awaiter
    (``GET /v1/sessions/{id}/await?request_id=`` for sessions, ``GET /v1/runs/{id}/wait``
    for runs). A cross-tenant ``target`` 404s before any edge is written; a supplied
    ``environment_id`` is ownership-checked against the caller's account.
    """
    return await service.invoke(
        pool,
        account_id=account_id,
        target_kind=body.target_kind,
        target=body.target,
        input=body.input,
        output_schema=body.output_schema,
        environment_id=body.environment_id,
        crypto_box=crypto_box,
    )
