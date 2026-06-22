"""The API caller's request-*writer* + completion *awaiter* (``/v1/tasks``).

``POST /v1/tasks`` is one kind-agnostic endpoint that, for an
external/operator caller, **writes the trusted request edge** (#1123) and
**resolves-or-creates a servicer** (a session or a run), returning a structured
handle. The API caller is *ephemeral* — nothing to wake; the handle is its
continuation, awaited via ``GET /v1/tasks/{task_id}/await`` — the **one
awaiter** over both servicer kinds, returning one
:class:`~aios.models.tasks.AwaitResponse`.

The handle is **not** an auth boundary (``await`` re-authorizes by
``account_id``), so it ships as plain JSON fields — no opaque encoding.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Query, status

from aios.api.deps import AccountIdDep, CryptoBoxDep, DbUrlDep, PoolDep
from aios.errors import ValidationError
from aios.ids import servicer_kind
from aios.logging import get_logger
from aios.models.tasks import AwaitResponse, TaskHandle, TaskRequest
from aios.services import sessions as service
from aios.services import tasks as tasks_service

log = get_logger("aios.api.routers.tasks")

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])


def _servicer_kind(task_id: str) -> Literal["session", "run"]:
    """The servicer kind of an await ``task_id`` (= the ``servicer_id``), off its id prefix.

    Adapts :func:`aios.ids.servicer_kind`'s ``ValueError`` (malformed or non-servicer id)
    into a 422 for the HTTP surface.
    """
    try:
        return servicer_kind(task_id)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


@router.post("", operation_id="create_task", status_code=status.HTTP_201_CREATED)
async def create_task(
    body: TaskRequest,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    account_id: AccountIdDep,
) -> TaskHandle:
    """Write a trusted request edge + resolve-or-create a servicer, returning a handle.

    ``target_kind=agent`` creates a **session** servicer (env-bound) and injects a
    channel-less request; ``target_kind=workflow`` creates a **run** servicer;
    ``target_kind=session`` invokes an existing same-account session by id. Await the
    handle at ``GET /v1/tasks/{servicer_id}/await`` — for a session servicer pass
    ``?request_id=`` to correlate the response; a run resolves off its terminal row. A
    cross-tenant ``target`` 404s before any edge is written; a supplied ``environment_id``
    is ownership-checked against the caller's account.
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


@router.get("/{task_id}/await", operation_id="await_task")
async def await_task(
    task_id: str,
    db_url: DbUrlDep,
    pool: PoolDep,
    account_id: AccountIdDep,
    request_id: str | None = None,
    timeout_seconds: Annotated[int, Query(alias="timeout", ge=0, le=60)] = 30,
) -> AwaitResponse:
    """Block until the task reaches a terminal state, or ``timeout`` seconds.

    The **one awaiter** over both servicer kinds: ``task_id`` is the ``servicer_id``
    from the POST handle and its kind is read off the id prefix. A ``session``
    servicer needs ``?request_id=`` to correlate its response; a ``run`` resolves off
    its terminal row (``request_id`` ignored). On timeout returns ``outcome=null`` so
    the caller re-polls — a plain request/response (MCP-usable) so an agent can await
    a sub-task and join. A cross-tenant/missing servicer 404s.
    """
    return await tasks_service.await_task(
        pool,
        db_url,
        servicer_kind=_servicer_kind(task_id),
        servicer_id=task_id,
        request_id=request_id,
        account_id=account_id,
        timeout_seconds=timeout_seconds,
    )


@router.post(
    "/{task_id}/cancel",
    operation_id="cancel_task",
    status_code=status.HTTP_202_ACCEPTED,
    openapi_extra={"x-codegen": {"mcp": {"destructiveHint": True}}},
)
async def cancel_task(
    task_id: str,
    pool: PoolDep,
    account_id: AccountIdDep,
    request_id: str,
) -> None:
    """Cancel the task identified by ``task_id`` (the servicer id) + ``request_id``.

    Seeds the cancel on the servicer (a run via its cancel signal, a session via its
    cancel-marker); the servicer harvests it under its own single-writer step lock and answers
    the request ``cancelled``. **202 Accepted**: the cancel is *requested*, finalized on the
    servicer's next step. Account-scoped; idempotent (re-cancel is a no-op); a cross-tenant or
    missing servicer 404s. ``task_id``'s kind is read off the id prefix (``wfr``/``sess``).
    """
    await tasks_service.cancel_task(
        pool,
        servicer_kind=_servicer_kind(task_id),
        servicer_id=task_id,
        request_id=request_id,
        account_id=account_id,
    )
