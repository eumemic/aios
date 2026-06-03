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


_ACCOUNT_ID = "acc_test_stub"


async def _make_session_with_events(pool: Any, n_events: int = 5) -> str:
    """Create a session with ``n_events`` user messages. Returns the session id."""
    from aios.db import queries
    from aios.services import agents as agents_svc
    from aios.services import sessions as sessions_svc

    async with pool.acquire() as conn:
        env = await queries.insert_environment(
            conn, name=f"events-env-{_uniq()}", account_id=_ACCOUNT_ID
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
        account_id=_ACCOUNT_ID,
    )
    session = await sessions_svc.create_session(
        pool,
        agent_id=agent.id,
        environment_id=env.id,
        title=None,
        metadata={},
        account_id=_ACCOUNT_ID,
    )
    for i in range(n_events):
        await sessions_svc.append_user_message(
            pool, session.id, f"message {i}", account_id=_ACCOUNT_ID
        )
    return session.id


async def _append_event(pool: Any, session_id: str, kind: str, data: dict[str, Any]) -> int:
    """Append one raw event of an arbitrary ``kind``; returns its seq."""
    from aios.db import queries

    async with pool.acquire() as conn:
        event = await queries.append_event(
            conn,
            account_id=_ACCOUNT_ID,
            session_id=session_id,
            kind=kind,  # type: ignore[arg-type]
            data=data,
        )
    return event.seq


@pytest.fixture
async def session_with_events(pool: Any) -> str:
    return await _make_session_with_events(pool, n_events=5)


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


class TestLimitValidation:
    """Gap 1: limit > 500 must return 422; limit = 500 must be accepted."""

    async def test_limit_over_500_returns_422(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r = await http_client.get(
            f"/v1/sessions/{session_with_events}/events",
            params={"limit": 501},
        )
        assert r.status_code == 422, r.text

    async def test_limit_500_accepted(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r = await http_client.get(
            f"/v1/sessions/{session_with_events}/events",
            params={"limit": 500},
        )
        assert r.status_code == 200, r.text


class TestHasMoreAccuracy:
    """Gap 2: has_more must be a true off-by-one-free signal."""

    async def test_has_more_false_when_at_exact_boundary(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        # session_with_events has exactly 5 events; requesting limit=5
        # must return has_more=False (not True as the paginate() helper
        # previously signalled when len==limit).
        r = await http_client.get(
            f"/v1/sessions/{session_with_events}/events",
            params={"limit": 5},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["data"]) == 5
        assert body["has_more"] is False, (
            "has_more should be False when all events fit in exactly one page"
        )

    async def test_has_more_true_with_valid_next_after(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        # 5 events, request limit=3 → has_more=True and next_after is set
        r = await http_client.get(
            f"/v1/sessions/{session_with_events}/events",
            params={"limit": 3},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["data"]) == 3
        assert body["has_more"] is True
        assert body["next_after"] is not None


class TestAfterSeqAlias:
    """Gap 3: ?after_seq=N must work as an alias for ?after=N."""

    async def test_after_seq_filters_events(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        # Fetch first page to get a real seq value
        r1 = await http_client.get(
            f"/v1/sessions/{session_with_events}/events",
            params={"limit": 2},
        )
        assert r1.status_code == 200, r1.text
        page1 = r1.json()
        boundary_seq = page1["data"][-1]["seq"]

        # Use after_seq alias; all returned events must have seq > boundary_seq
        r2 = await http_client.get(
            f"/v1/sessions/{session_with_events}/events",
            params={"after_seq": boundary_seq},
        )
        assert r2.status_code == 200, r2.text
        page2 = r2.json()
        assert len(page2["data"]) > 0, "Expected events after the boundary seq"
        for event in page2["data"]:
            assert event["seq"] > boundary_seq, (
                f"after_seq={boundary_seq} was ignored; got seq {event['seq']}"
            )


class TestSessionEventStats:
    """Gaps 4/5: last_event_at and total_events on the Session resource."""

    async def test_last_event_at_populated(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r = await http_client.get(f"/v1/sessions/{session_with_events}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["last_event_at"] is not None, (
            "last_event_at should be non-null after events are appended"
        )

    async def test_total_events_count(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r = await http_client.get(f"/v1/sessions/{session_with_events}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total_events"] == 5, f"Expected total_events=5, got {body['total_events']}"

    async def test_fresh_session_has_zero_total_events(
        self, http_client: httpx.AsyncClient, pool: Any
    ) -> None:
        # Create a session with no messages
        session_id = await _make_session_with_events(pool, n_events=0)
        r = await http_client.get(f"/v1/sessions/{session_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total_events"] == 0, (
            f"Expected total_events=0 for fresh session, got {body['total_events']}"
        )
        assert body["last_event_at"] is None, (
            "last_event_at should be None for session with no events"
        )


class TestGetSingleEvent:
    """Gap 6: GET /v1/sessions/{id}/events/{event_id} must be a real endpoint."""

    async def test_get_event_by_id(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        # List events to get a real event id
        r_list = await http_client.get(
            f"/v1/sessions/{session_with_events}/events",
            params={"limit": 1},
        )
        assert r_list.status_code == 200, r_list.text
        first_event = r_list.json()["data"][0]
        event_id = first_event["id"]

        r = await http_client.get(f"/v1/sessions/{session_with_events}/events/{event_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["id"] == event_id
        assert body["seq"] == first_event["seq"]

    async def test_get_event_missing_returns_404(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r = await http_client.get(f"/v1/sessions/{session_with_events}/events/evt_notexist")
        assert r.status_code == 404, r.text

    async def test_get_event_wrong_session_returns_404(
        self, http_client: httpx.AsyncClient, pool: Any, session_with_events: str
    ) -> None:
        # Get a real event id from session_with_events
        r_list = await http_client.get(
            f"/v1/sessions/{session_with_events}/events",
            params={"limit": 1},
        )
        assert r_list.status_code == 200, r_list.text
        event_id = r_list.json()["data"][0]["id"]

        # Create a different session; use its id with the real event id
        other_session_id = await _make_session_with_events(pool, n_events=0)
        r = await http_client.get(f"/v1/sessions/{other_session_id}/events/{event_id}")
        assert r.status_code == 404, r.text


class TestBackwardPagination:
    """Tail-anchored ``?before=N`` paging: newest-first pages that walk into the
    past, with ``next_after`` chaining as the next ``?before=`` cursor.
    """

    async def test_before_returns_newest_first_page(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        # 5 events (seq 1..5). before=6 (last_event_seq + 1), limit=2 → [5, 4].
        r = await http_client.get(
            f"/v1/sessions/{session_with_events}/events",
            params={"before": 6, "limit": 2},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        seqs = [e["seq"] for e in body["data"]]
        assert seqs == [5, 4], f"expected newest-first [5, 4], got {seqs}"
        assert body["has_more"] is True
        # next_after is the SMALLEST seq in the page → the next ?before= cursor.
        assert body["next_after"] == "4"

    async def test_before_walks_into_the_past_without_repeats(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        seen: list[int] = []
        cursor = 6  # last_event_seq + 1 for a 5-event session
        for _ in range(20):  # guardrail; should terminate in ~3 iterations
            r = await http_client.get(
                f"/v1/sessions/{session_with_events}/events",
                params={"before": cursor, "limit": 2},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            page_seqs = [e["seq"] for e in body["data"]]
            assert page_seqs == sorted(page_seqs, reverse=True), (
                f"backward page not newest-first: {page_seqs}"
            )
            seen.extend(page_seqs)
            if not body["has_more"]:
                break
            cursor = int(body["next_after"])
        else:
            pytest.fail("backward pagination did not terminate within 20 iterations")
        assert seen == [5, 4, 3, 2, 1], f"expected full descending walk, got {seen}"
        assert len(seen) == len(set(seen)), "backward walk repeated a seq"

    async def test_before_includes_newest_event(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        # Off-by-one guard: before=last_event_seq+1 must include the newest event.
        r_session = await http_client.get(f"/v1/sessions/{session_with_events}")
        last_seq = r_session.json()["last_event_seq"]
        r = await http_client.get(
            f"/v1/sessions/{session_with_events}/events",
            params={"before": last_seq + 1, "limit": 1},
        )
        assert r.status_code == 200, r.text
        assert r.json()["data"][0]["seq"] == last_seq

    async def test_before_and_after_together_is_422(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r = await http_client.get(
            f"/v1/sessions/{session_with_events}/events",
            params={"before": 6, "after": 1},
        )
        assert r.status_code == 422, r.text

    async def test_before_and_after_seq_together_is_422(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r = await http_client.get(
            f"/v1/sessions/{session_with_events}/events",
            params={"before": 6, "after_seq": 1},
        )
        assert r.status_code == 422, r.text

    async def test_before_chains_across_non_contiguous_filtered_seq(
        self, http_client: httpx.AsyncClient, pool: Any
    ) -> None:
        # Interleave message and lifecycle events so message seqs are
        # non-contiguous (1, 3, 5). Backward paging with kind=message must chain
        # via next_after across the gaps — proving the cursor is the real
        # returned seq, not lastSeq-limit arithmetic.
        from aios.services import sessions as sessions_svc

        session_id = await _make_session_with_events(pool, n_events=0)
        await sessions_svc.append_user_message(
            pool, session_id, "m1", account_id=_ACCOUNT_ID
        )  # seq 1 (message)
        await _append_event(pool, session_id, "lifecycle", {"phase": "x"})  # seq 2
        await sessions_svc.append_user_message(
            pool, session_id, "m3", account_id=_ACCOUNT_ID
        )  # seq 3 (message)
        await _append_event(pool, session_id, "lifecycle", {"phase": "y"})  # seq 4
        await sessions_svc.append_user_message(
            pool, session_id, "m5", account_id=_ACCOUNT_ID
        )  # seq 5 (message)

        seen: list[int] = []
        cursor = 6
        for _ in range(20):
            r = await http_client.get(
                f"/v1/sessions/{session_id}/events",
                params={"before": cursor, "kind": "message", "limit": 1},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            for e in body["data"]:
                assert e["kind"] == "message"
                seen.append(e["seq"])
            if not body["has_more"]:
                break
            cursor = int(body["next_after"])
        else:
            pytest.fail("filtered backward pagination did not terminate")
        assert seen == [5, 3, 1], f"expected message seqs newest-first, got {seen}"
