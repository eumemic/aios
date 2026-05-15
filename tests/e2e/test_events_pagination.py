"""E2E tests for ``GET /v1/sessions/{id}/events`` pagination round-trip.

Regression coverage for issue #389: the query param was named ``after_seq``
while the response field was ``next_after``, so a naive client following the
documented pattern (use ``next_after`` as the next call's ``?after=``) saw
the value silently ignored and got the first page repeated forever.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import httpx
import pytest

from tests.helpers.connections import authed_client


def _uniq() -> str:
    return secrets.token_hex(4)


@pytest.fixture
async def pool(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    from aios.api.app import create_app
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox

    settings = get_settings()
    app = create_app()
    app.state.pool = pool
    app.state.crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    app.state.db_url = settings.db_url
    app.state.procrastinate = mock.MagicMock()

    transport = httpx.ASGITransport(app=app)
    async with authed_client(
        "http://testserver",
        aios_env["AIOS_API_KEY"],
        transport=transport,
    ) as client:
        yield client


@pytest.fixture
async def session_with_events(pool: Any) -> str:
    from aios.db import queries
    from aios.services import agents as agents_svc
    from aios.services import sessions as sessions_svc

    account_id = "acc_test_stub"
    async with pool.acquire() as conn:
        env = await queries.insert_environment(
            conn, name=f"events-env-{_uniq()}", account_id=account_id
        )
    agent = await agents_svc.create_agent(
        pool,
        name=f"events-agent-{_uniq()}",
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
    for i in range(5):
        await sessions_svc.append_user_message(
            pool, session.id, f"message {i}", account_id=account_id
        )
    return session.id


class TestEventsPaginationRoundTrip:
    """The ``next_after`` cursor returned by one call must be acceptable as
    ``?after=`` on the next call. Forward-paginating walks all events without
    repeats. This was issue #389.
    """

    async def test_after_advances_pagination(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        # First page, limit=2 → seqs 1..2 with has_more=True
        r1 = await http_client.get(
            f"/v1/sessions/{session_with_events}/events",
            params={"limit": 2},
        )
        assert r1.status_code == 200, r1.text
        page1 = r1.json()
        assert len(page1["data"]) == 2
        assert page1["has_more"] is True
        assert page1["next_after"] is not None
        page1_last_seq = page1["data"][-1]["seq"]
        assert str(page1_last_seq) == page1["next_after"]

        # Second page, ?after=<page1.next_after> → seqs MUST be > page1_last_seq.
        # Pre-fix this returned seqs 1..2 again because `after_seq` defaulted to 0.
        r2 = await http_client.get(
            f"/v1/sessions/{session_with_events}/events",
            params={"limit": 2, "after": page1["next_after"]},
        )
        assert r2.status_code == 200, r2.text
        page2 = r2.json()
        assert len(page2["data"]) == 2
        for event in page2["data"]:
            assert event["seq"] > page1_last_seq, (
                f"page 2 returned seq {event['seq']}; expected > {page1_last_seq}. "
                f"Pre-fix symptom: `?after=` was ignored and the same page came back."
            )

    async def test_full_walk_terminates_without_repeats(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        # Walk the full event stream paginated; assert no seq appears twice and
        # the loop terminates (has_more eventually False or empty data).
        seen: set[int] = set()
        cursor: str | None = None
        for iteration in range(20):  # guardrail; should terminate in ~3 iterations
            params: dict[str, Any] = {"limit": 2}
            if cursor is not None:
                params["after"] = cursor
            r = await http_client.get(
                f"/v1/sessions/{session_with_events}/events",
                params=params,
            )
            assert r.status_code == 200, r.text
            body = r.json()
            for event in body["data"]:
                assert event["seq"] not in seen, (
                    f"seq {event['seq']} returned twice during forward walk "
                    f"(iteration {iteration}). Symptom of issue #389."
                )
                seen.add(event["seq"])
            if not body["has_more"]:
                break
            cursor = body["next_after"]
            assert cursor is not None
        else:
            pytest.fail("Pagination did not terminate within 20 iterations")
