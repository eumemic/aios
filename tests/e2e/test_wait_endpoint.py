"""E2E tests for the session/run watch endpoints: the ``GET /v1/sessions/{id}/wait`` SSE
long-poll (issue #40), the unified ``GET /v1/tasks/{task_id}/await`` awaiter, and the
``GET /v1/sessions/{id}/await`` watermark quiescence alias."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import httpx
import pytest

from tests.helpers.connections import authed_client, wired_app


def _uniq() -> str:
    return secrets.token_hex(4)


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=wired_app(pool))
    async with authed_client(
        "http://testserver",
        aios_env["AIOS_API_KEY"],
        transport=transport,
    ) as client:
        yield client


@pytest.fixture
async def session_id(pool: Any) -> str:
    account_id = "acc_test_stub"  # PR 3 scaffolding
    from aios.db import queries
    from aios.services import agents as agents_svc
    from aios.services import sessions as sessions_svc

    async with pool.acquire() as conn:
        env = await queries.insert_environment(
            conn, name=f"wait-env-{_uniq()}", account_id=account_id
        )
    agent = await agents_svc.create_agent(
        pool,
        name=f"wait-agent-{_uniq()}",
        model="openai/gpt-4o-mini",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
        account_id=account_id,
    )
    session = await sessions_svc.create_session(
        pool,
        agent_id=agent.id,
        environment_id=env.id,
        title=None,
        metadata={},
        account_id=account_id,
    )
    return session.id


class TestWaitEndpoint:
    async def test_returns_existing_events_immediately(
        self, http_client: httpx.AsyncClient, pool: Any, session_id: str
    ) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import sessions as sessions_svc

        await sessions_svc.append_user_message(pool, session_id, "hello", account_id=account_id)

        r = await http_client.get(
            f"/v1/sessions/{session_id}/wait",
            params={"after": 0, "timeout": 30},
        )

        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["events"]) >= 1
        assert any(
            e["kind"] == "message" and e["data"].get("content") == "hello" for e in body["events"]
        )
        assert body["next_after"] == body["events"][-1]["seq"]
        assert body["session_status"] in {"active", "idle"}

    async def test_empty_after_timeout(
        self, http_client: httpx.AsyncClient, session_id: str
    ) -> None:
        r = await http_client.get(
            f"/v1/sessions/{session_id}/wait",
            params={"after": 9999, "timeout": 1},
        )

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["events"] == []
        assert body["next_after"] == 9999

    async def test_unblocks_on_new_event(
        self, http_client: httpx.AsyncClient, pool: Any, session_id: str
    ) -> None:
        from aios.services import sessions as sessions_svc

        async def wait_call() -> httpx.Response:
            return await http_client.get(
                f"/v1/sessions/{session_id}/wait",
                params={"after": 0, "timeout": 10},
            )

        async def delayed_append() -> None:
            account_id = "acc_test_stub"  # PR 3 scaffolding
            await asyncio.sleep(0.3)
            await sessions_svc.append_user_message(
                pool, session_id, "late hello", account_id=account_id
            )

        wait_task = asyncio.create_task(wait_call())
        post_task = asyncio.create_task(delayed_append())

        r, _ = await asyncio.gather(wait_task, post_task)

        assert r.status_code == 200, r.text
        body = r.json()
        assert any(
            e["kind"] == "message" and e["data"].get("content") == "late hello"
            for e in body["events"]
        )

    async def test_rejects_timeout_above_cap(
        self, http_client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Request-level validation rejects out-of-range ``timeout``."""
        r = await http_client.get(
            f"/v1/sessions/{session_id}/wait",
            params={"after": 0, "timeout": 3600},
        )
        assert r.status_code == 422, r.text

    async def test_delta_notifications_do_not_short_circuit_wait(
        self, http_client: httpx.AsyncClient, pool: Any, session_id: str
    ) -> None:
        """Streaming delta pokes on the session channel must not return an empty response.

        ``_notify_delta`` in ``aios.harness.completion`` fires transient
        ``pg_notify`` payloads on ``events_<session_id>`` for every token; these
        payloads start with ``{`` and carry no DB row. A long-poll handler that
        re-reads events on every notification would return empty immediately
        and cause clients to hot-loop on any streaming session.
        """

        async def fire_delta_then_append() -> None:
            account_id = "acc_test_stub"  # PR 3 scaffolding
            from aios.services import sessions as sessions_svc

            async with pool.acquire() as conn:
                for _ in range(3):
                    await conn.execute(
                        "SELECT pg_notify($1, $2)",
                        f"events_{session_id}",
                        '{"delta": "token"}',
                    )
                    await asyncio.sleep(0.05)
            # Real event arrives later; the wait should latch onto this,
            # not any of the deltas.
            await asyncio.sleep(0.3)
            await sessions_svc.append_user_message(pool, session_id, "real", account_id=account_id)

        async def wait_call() -> httpx.Response:
            return await http_client.get(
                f"/v1/sessions/{session_id}/wait",
                params={"after": 0, "timeout": 5},
            )

        wait_task = asyncio.create_task(wait_call())
        poke_task = asyncio.create_task(fire_delta_then_append())

        r, _ = await asyncio.gather(wait_task, poke_task)

        assert r.status_code == 200, r.text
        body = r.json()
        assert any(
            e["kind"] == "message" and e["data"].get("content") == "real" for e in body["events"]
        ), body


@pytest.fixture
async def run_id(pool: Any) -> str:
    """A freshly created (``pending``, never-stepped) run for the run-wait endpoint.

    ``defer_run_wake`` is patched out so creation doesn't need a procrastinate worker;
    the run stays ``pending`` (nothing steps it), which is exactly the non-terminal state
    the timeout test exercises."""
    account_id = "acc_test_stub"  # PR 3 scaffolding
    from aios.db import queries
    from aios.services import workflows as wf_svc

    async with pool.acquire() as conn:
        env = await queries.insert_environment(
            conn, name=f"runwait-env-{_uniq()}", account_id=account_id
        )
    wf = await wf_svc.create_workflow(
        pool,
        account_id=account_id,
        name=f"runwait-wf-{_uniq()}",
        script="async def main(input):\n    return 1",
    )
    with mock.patch("aios.workflows.service.defer_run_wake", new=mock.AsyncMock()):
        run = await wf_svc.create_run(
            pool, account_id=account_id, workflow_id=wf.id, environment_id=env.id, input=None
        )
    return run.id


class TestTaskAwaitEndpoint:
    """``GET /v1/tasks/{task_id}/await`` — the one awaiter over both servicer kinds.

    Exercises the HTTP wiring (route registration, the ``timeout`` query alias, the
    ``AwaitResponse`` serialization, kind-dispatch off the id prefix, account scoping) over a
    real ASGI client. The blocking/completion behavior of the service is covered in
    ``tests/integration/test_wf_step.py`` (run arm) and ``test_await_session.py`` (session arm);
    here ``timeout=0`` keeps every case non-blocking."""

    async def test_run_pending_returns_outcome_null(
        self, http_client: httpx.AsyncClient, run_id: str
    ) -> None:
        """A never-stepped run is non-terminal; ``timeout=0`` returns at once with
        ``outcome=null`` (the re-poll contract) — proving the run-arm dispatch + model wiring."""
        r = await http_client.get(f"/v1/tasks/{run_id}/await", params={"timeout": 0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["outcome"] is None
        assert body["result"] is None and body["error"] is None

    async def test_session_request_pending_returns_outcome_null(
        self, http_client: httpx.AsyncClient, session_id: str
    ) -> None:
        """An unanswered ``request_id`` on a session servicer is non-terminal → ``outcome=null``."""
        r = await http_client.get(
            f"/v1/tasks/{session_id}/await", params={"request_id": "nope", "timeout": 0}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["outcome"] is None and body["result"] is None

    async def test_session_without_request_id_422(
        self, http_client: httpx.AsyncClient, session_id: str
    ) -> None:
        """A session servicer needs ``request_id`` to correlate; omitting it is a 422."""
        r = await http_client.get(f"/v1/tasks/{session_id}/await", params={"timeout": 0})
        assert r.status_code == 422, r.text

    async def test_rejects_timeout_above_cap(
        self, http_client: httpx.AsyncClient, run_id: str
    ) -> None:
        """The ``le=60`` bound on the ``timeout`` query param is enforced (422)."""
        r = await http_client.get(f"/v1/tasks/{run_id}/await", params={"timeout": 3600})
        assert r.status_code == 422, r.text

    async def test_non_servicer_task_id_422(self, http_client: httpx.AsyncClient) -> None:
        """A well-formed id of a non-awaitable kind (e.g. an agent) is a 422, not a 404."""
        r = await http_client.get(
            "/v1/tasks/agent_0000000000000000000000abcd/await", params={"timeout": 0}
        )
        assert r.status_code == 422, r.text

    async def test_unknown_run_404s_before_subscribing(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """A run id the caller can't see 404s — and the scope check runs BEFORE any LISTEN is
        opened, so an unauthorized caller never opens a connection on the channel."""
        r = await http_client.get(
            "/v1/tasks/wfr_00000000000000000000000000/await", params={"timeout": 0}
        )
        assert r.status_code == 404, r.text


class TestSessionAwaitEndpoint:
    """``GET /v1/sessions/{id}/await`` — the session quiescence drive-and-join (watermark only).

    Exercises the HTTP wiring (route registration, the ``timeout`` query alias, the ``watermark``
    param, the ``SessionAwaitResponse`` serialization, account scoping) over a real ASGI client.
    Request correlation lives on the unified awaiter (``/v1/tasks/{id}/await``); here
    ``timeout=0`` keeps each case non-blocking."""

    async def test_default_watermark_unmet_done_false(
        self, http_client: httpx.AsyncClient, pool: Any, session_id: str
    ) -> None:
        """A pending user stimulus (last_stimulus_seq > last_reacted_seq) with the default
        watermark is unmet at ``timeout=0`` → ``done=false`` with ``last_reacted_seq == 0``."""
        from aios.services import sessions as sessions_svc

        await sessions_svc.append_user_message(pool, session_id, "hi", account_id="acc_test_stub")

        r = await http_client.get(f"/v1/sessions/{session_id}/await", params={"timeout": 0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["done"] is False
        assert body["last_reacted_seq"] == 0

    async def test_met_watermark_done_true(
        self, http_client: httpx.AsyncClient, session_id: str
    ) -> None:
        """A fresh session has ``last_reacted_seq == 0``; ``watermark=0`` is met → ``done``."""
        r = await http_client.get(
            f"/v1/sessions/{session_id}/await", params={"watermark": 0, "timeout": 0}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["done"] is True

    async def test_rejects_timeout_above_cap(
        self, http_client: httpx.AsyncClient, session_id: str
    ) -> None:
        """The ``le=60`` bound on the ``timeout`` query param is enforced (422)."""
        r = await http_client.get(f"/v1/sessions/{session_id}/await", params={"timeout": 3600})
        assert r.status_code == 422, r.text

    async def test_unknown_session_404(self, http_client: httpx.AsyncClient) -> None:
        """A session id the caller can't see 404s before any LISTEN opens."""
        r = await http_client.get("/v1/sessions/sess_nope/await", params={"timeout": 0})
        assert r.status_code == 404, r.text
