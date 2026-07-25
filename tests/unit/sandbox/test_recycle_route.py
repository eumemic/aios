"""Sandbox recycle admission and self-service route."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aios.db import queries
from aios.sandbox.tool_broker import ToolBroker


@pytest.fixture
async def broker() -> AsyncIterator[ToolBroker]:
    instance = ToolBroker()
    await instance.start()
    try:
        yield instance
    finally:
        await instance.stop()


async def test_self_recycle_is_bound_to_secret_session(broker: ToolBroker) -> None:
    broker.register_session("sess_own", "secret")
    event = MagicMock()
    event.model_dump.return_value = {
        "kind": "lifecycle",
        "data": {"event": "sandbox_recycle_requested", "requested_by": "self"},
    }
    with (
        patch("aios.harness.runtime.require_pool", return_value=MagicMock()),
        patch(
            "aios.services.sessions.load_session_account_id",
            AsyncMock(return_value="acct_1"),
        ),
        patch(
            "aios.services.sessions.request_sandbox_recycle", AsyncMock(return_value=event)
        ) as call,
        patch("aios.jobs.app.defer_sandbox_recycle", AsyncMock()),
    ):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{broker.port}/v1/secret/sandbox/recycle",
                json={"discard_unsalvaged": True},
            )
    assert response.status_code == 202
    assert response.json() == event.model_dump.return_value
    call.assert_awaited_once()
    awaited = call.await_args
    assert awaited is not None
    assert awaited.args[1] == "sess_own"
    assert awaited.kwargs["requested_by"] == "self"


async def test_self_recycle_requires_discard_ack(broker: ToolBroker) -> None:
    broker.register_session("sess_own", "secret")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://127.0.0.1:{broker.port}/v1/secret/sandbox/recycle", json={}
        )
    assert response.status_code == 422
    assert "discard_unsalvaged" in response.json()["error"]


async def test_recycle_rate_limit_is_atomic() -> None:
    from aios.services import sessions

    conn = AsyncMock()
    transaction = MagicMock()
    transaction.__aenter__ = AsyncMock(return_value=None)
    transaction.__aexit__ = AsyncMock(return_value=None)
    conn.transaction = MagicMock(return_value=transaction)
    conn.fetchval.return_value = 3
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    with (
        patch.object(sessions, "get_settings") as settings,
        patch.object(queries, "get_session_bare", AsyncMock()),
    ):
        settings.return_value.sandbox_recycle_hourly_limit = 3
        try:
            await sessions.request_sandbox_recycle(
                pool,
                "sess_1",
                account_id="acct_1",
                requested_by="operator",
                discard_unsalvaged=True,
            )
        except Exception as exc:
            assert getattr(exc, "status_code", None) == 429
        else:
            raise AssertionError("expected rate limit")
    conn.execute.assert_awaited_once_with(
        "SELECT 1 FROM sessions WHERE id = $1 AND account_id = $2 FOR UPDATE", "sess_1", "acct_1"
    )


async def test_two_admitted_requests_each_get_their_own_job(in_memory_app: Any) -> None:
    """Finding 2: admission must be one-to-one with execution and outcome.

    ``request_sandbox_recycle`` commits a distinct ``sandbox_recycle_requested``
    event (counted against the 3/hour budget, answered 202) BEFORE the enqueue.
    The enqueue therefore may not silently coalesce two distinct admissions:
    with the old session-wide ``queueing_lock=recycle:{session_id}``, a second
    admitted request arriving while the first job was still ``todo`` raised
    ``AlreadyEnqueued``, which ``defer_sandbox_recycle`` swallowed — two request
    events, one job, one terminal event.

    Keying the queueing lock on the admitting event's id makes the identity
    per-request, so both admissions get a job (and both terminal events).
    """
    from aios.jobs.app import defer_sandbox_recycle

    await defer_sandbox_recycle("sess_1", requested_by="operator", request_id="evt_a")
    await defer_sandbox_recycle("sess_1", requested_by="self", request_id="evt_b")

    jobs = list(in_memory_app.connector.jobs.values())
    assert len(jobs) == 2, f"each admitted request needs its own job, got {jobs}"
    assert {j["queueing_lock"] for j in jobs} == {"recycle:evt_a", "recycle:evt_b"}
    # ...but they still serialize on the session: ``lock`` is the RUNNING-job
    # mutex, so two recycles never race on the same sandbox.
    assert {j["lock"] for j in jobs} == {"sess_1"}
    assert {j["args"]["request_id"] for j in jobs} == {"evt_a", "evt_b"}
    assert {j["args"]["session_id"] for j in jobs} == {"sess_1"}


async def test_redefer_of_same_request_is_idempotent(in_memory_app: Any) -> None:
    """A re-defer of ONE admission (at-least-once retry) still coalesces."""
    from aios.jobs.app import defer_sandbox_recycle

    await defer_sandbox_recycle("sess_1", requested_by="operator", request_id="evt_a")
    await defer_sandbox_recycle("sess_1", requested_by="operator", request_id="evt_a")

    assert len(in_memory_app.connector.jobs) == 1


async def test_route_defers_with_the_admitting_event_id() -> None:
    """The route must pass the committed request event's id, not a fresh one."""
    from aios.api.routers import sessions as sessions_router
    from aios.models.sessions import SandboxRecycleRequest

    event = MagicMock()
    event.id = "evt_admitted"
    with (
        patch(
            "aios.api.routers.sessions.service.request_sandbox_recycle",
            AsyncMock(return_value=event),
        ),
        patch("aios.api.routers.sessions.defer_sandbox_recycle", AsyncMock()) as defer,
    ):
        returned = await sessions_router.recycle_sandbox(
            "sess_1",
            SandboxRecycleRequest(discard_unsalvaged=True),
            MagicMock(),
            "acct_1",
        )

    assert returned is event
    defer.assert_awaited_once_with("sess_1", requested_by="operator", request_id="evt_admitted")
