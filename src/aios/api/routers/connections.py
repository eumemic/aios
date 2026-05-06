"""Connection endpoints — CRUD plus mode-binding transitions.

Created in detached mode; switch to single_session via ``attach`` or
per_chat via ``configure-per-chat``.  ``DELETE`` soft-archives — but
only on detached connections (the service layer enforces this so
operators can't silently break inbound delivery for live single_session
connections or orphan ``spawned_from_connection_id`` pointers on
per_chat-spawned sessions).

``attach`` validates the connection's ``account`` against the connector
subprocess's current snapshot (audit fix #1, design §5.6) — without
this guard an operator can permanently attach a connection for an
account the connector no longer serves, and inbound silently drops with
the ``account_drift`` counter ticking up.  The validation flows through
procrastinate RPC like the rest of ``/v1/connectors/*``: API mints a
ULID, LISTENs on ``connector_result_<call_id>``, defers
``harness.connector_status``, awaits NOTIFY.  ~50-200ms latency on an
admin-only endpoint is invisible.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from aios.api.connector_rpc import connector_rpc
from aios.api.deps import AuthDep, DbUrlDep, PoolDep
from aios.errors import AccountDriftError
from aios.harness.connector_tasks import defer_connector_status
from aios.models.common import ListResponse
from aios.models.connections import (
    BindChatRequest,
    BoundChat,
    Connection,
    ConnectionAttach,
    ConnectionConfigurePerChat,
    ConnectionCreate,
    ConnectionMode,
    RecentChat,
)
from aios.services import connections as service

router = APIRouter(prefix="/v1/connections", tags=["connections"])

# Snapshot lookup is cheap (worker reads in-memory state and returns) —
# tighter than the 60s call timeout but generous enough that a busy
# worker queue doesn't false-positive a drift error on the operator.
_DRIFT_CHECK_TIMEOUT_S = 10.0


async def _assert_account_in_snapshot(db_url: str, *, connector: str, account: str) -> None:
    """Raise :class:`AccountDriftError` if ``account`` isn't in the connector's snapshot.

    Treats supervisor-down (``not_ready`` / ``circuit_open``) as a
    distinct condition: those surface as 503, not 409 — the operator
    should retry once the connector boots, not assume drift.  The
    ``not_enabled`` envelope from the worker means the connector isn't
    in ``connectors_enabled`` at all and is also a 503.

    The connector's account snapshot dicts each carry an ``id`` field
    (per SDK convention via :func:`aios_connector.make_account`); the
    snapshot match is on that field.
    """
    envelope: dict[str, Any] = await connector_rpc(
        db_url,
        lambda cid: defer_connector_status(call_id=cid, connector=connector),
        timeout_s=_DRIFT_CHECK_TIMEOUT_S,
    )
    err = envelope.get("error")
    if err:
        # Supervisor not running / connector not booted → can't tell
        # whether the account is valid.  Surface as 503; the caller
        # should retry rather than assume drift.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"connector {connector!r} snapshot unavailable: {err}",
        )
    # ``connector_status(connector=<c>)`` returns ``{"connectors": [<one
    # snapshot per instance>]}`` — aggregate across instances so the drift
    # check works on multi-instance deployments (e.g. ``telegram:support``,
    # ``telegram:alerts``) without caring which instance owns the account.
    instances = envelope.get("connectors") or []
    statuses = {entry.get("status") for entry in instances if isinstance(entry, dict)}
    if "running" not in statuses:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"connector {connector!r} has no running instances "
                f"(states: {sorted(s for s in statuses if s)}); "
                "snapshot unavailable — retry once it's running"
            ),
        )
    known_ids: set[Any] = {
        account_entry.get("id")
        for entry in instances
        if isinstance(entry, dict)
        for account_entry in (entry.get("accounts") or [])
        if isinstance(account_entry, dict)
    }
    if not known_ids:
        # Boot-window race: supervisor flips ``status = "running"`` after MCP
        # ``initialize`` returns, but the SDK only emits
        # ``notifications/aios/accounts`` once the supervisor's
        # ``notifications/initialized`` has round-tripped — at least one stdio
        # leg of asymmetry.  An attach during that window must be 503 (retry
        # shortly), not 409 drift; the inbound handler exempts the same case
        # at ``connector_supervisor._handle_inbound``.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(f"connector {connector!r} snapshot not yet populated; retry shortly"),
        )
    if account not in known_ids:
        raise AccountDriftError(
            f"account {account!r} is not in {connector!r}'s current snapshot",
            detail={"connector": connector, "account": account},
        )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(body: ConnectionCreate, pool: PoolDep, _auth: AuthDep) -> Connection:
    """Create a detached connection, **idempotent on ``(connector, account)``**.

    Per plan decision #5, this endpoint and the supervisor's
    auto-create-on-first-inbound path race-safely converge on a single row:
    posting twice with the same ``(connector, account)`` returns 201 with the
    existing row rather than 409.  The ``id`` may differ from a freshly-allocated
    one if a concurrent writer landed first; the response always reflects the
    canonical active row.
    """
    return await service.create_connection(
        pool,
        connector=body.connector,
        account=body.account,
        metadata=body.metadata,
    )


@router.get("")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    connector: str | None = None,
    session_id: str | None = None,
    mode: ConnectionMode | None = None,
    limit: int = 50,
    after: str | None = None,
) -> ListResponse[Connection]:
    items = await service.list_connections(
        pool,
        connector=connector,
        session_id=session_id,
        mode=mode,
        limit=limit,
        after=after,
    )
    return ListResponse[Connection](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{connection_id}")
async def get(connection_id: str, pool: PoolDep, _auth: AuthDep) -> Connection:
    return await service.get_connection(pool, connection_id)


@router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(connection_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    await service.archive_connection(pool, connection_id)


@router.post("/{connection_id}/attach")
async def attach(
    connection_id: str,
    body: ConnectionAttach,
    pool: PoolDep,
    db_url: DbUrlDep,
    _auth: AuthDep,
) -> Connection:
    """Attach a connection to a session, after validating the account against the snapshot.

    The drift check runs BEFORE the DB write so a stale account can't
    move into ``single_session`` mode and silently swallow inbound.
    """
    connection = await service.get_connection(pool, connection_id)
    await _assert_account_in_snapshot(
        db_url, connector=connection.connector, account=connection.account
    )
    return await service.attach_connection(pool, connection_id, session_id=body.session_id)


@router.post("/{connection_id}/detach")
async def detach(connection_id: str, pool: PoolDep, _auth: AuthDep) -> Connection:
    return await service.detach_connection(pool, connection_id)


@router.post("/{connection_id}/configure-per-chat")
async def configure_per_chat(
    connection_id: str, body: ConnectionConfigurePerChat, pool: PoolDep, _auth: AuthDep
) -> Connection:
    return await service.configure_per_chat(
        pool, connection_id, session_template_id=body.session_template_id
    )


@router.post("/{connection_id}/unconfigure")
async def unconfigure(connection_id: str, pool: PoolDep, _auth: AuthDep) -> Connection:
    return await service.unconfigure_connection(pool, connection_id)


@router.post("/{connection_id}/bind-chat", status_code=status.HTTP_201_CREATED)
async def bind_chat(
    connection_id: str, body: BindChatRequest, pool: PoolDep, _auth: AuthDep
) -> BoundChat:
    """Operator-curate a chat → session mapping (#215).

    A row in ``connection_chat_sessions`` overrides the connection's
    mode-default fallback for that ``chat_id``.  Operators use this to
    point different chats on a single account at different existing
    sessions — the middle case the unified ``connections`` shape didn't
    cover after #205.

    Idempotent on ``(connection_id, chat_id)``: a second call with the
    same chat returns the existing row (its ``session_id`` may differ
    from the requested one if a concurrent writer landed first or the
    supervisor pre-populated it via per_chat spawn).
    """
    return await service.bind_chat_to_session(
        pool, connection_id, chat_id=body.chat_id, session_id=body.session_id
    )


@router.delete("/{connection_id}/bind-chat/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unbind_chat(connection_id: str, chat_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    """Drop the operator-curated row.  Idempotent — repeat calls 204."""
    await service.unbind_chat(pool, connection_id, chat_id)


@router.get("/{connection_id}/bound-chats")
async def bound_chats(connection_id: str, pool: PoolDep, _auth: AuthDep) -> ListResponse[BoundChat]:
    items = await service.list_bound_chats(pool, connection_id)
    return ListResponse[BoundChat](data=items)


@router.get("/{connection_id}/recent-chats")
async def recent_chats(
    connection_id: str,
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
) -> ListResponse[RecentChat]:
    items = await service.list_recent_chats(pool, connection_id, limit=limit)
    return ListResponse[RecentChat](data=items)
