"""E2E tests for ``GET /v1/sessions/{id}/events`` opaque-cursor pagination.

The first page selects a direction (``?dir=forward|backward``, default forward)
plus optional ``?kind=``/``?error_only=`` and ``?limit=``; every subsequent page
is just ``?cursor=<next_cursor>``, the opaque token from the previous response.
Forward walks oldest→newest; backward loads the newest-first tail and pages into
the past.
"""

from __future__ import annotations

import base64
import json
import secrets
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from aios.models.events import EventKind
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


_ACCOUNT_ID = "acc_test_stub"
_EVENTS = "/v1/sessions/{sid}/events"


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


async def _append_event(pool: Any, session_id: str, kind: EventKind, data: dict[str, Any]) -> int:
    """Append one raw event of an arbitrary ``kind``; returns its seq."""
    from aios.db import queries

    async with pool.acquire() as conn:
        event = await queries.append_event(
            conn,
            account_id=_ACCOUNT_ID,
            session_id=session_id,
            kind=kind,
            data=data,
        )
    return event.seq


@pytest.fixture
async def session_with_events(pool: Any) -> str:
    return await _make_session_with_events(pool, n_events=5)


async def _walk(
    http_client: httpx.AsyncClient, session_id: str, first_params: dict[str, Any]
) -> list[int]:
    """Walk every page of the events endpoint, returning seqs in page order.

    The first request carries ``first_params``; each subsequent request sends
    only ``?cursor=<next_cursor>`` — proving the token is self-contained.
    """
    seqs: list[int] = []
    params = dict(first_params)
    for _ in range(20):  # guardrail; the test sessions terminate in ~3 pages
        r = await http_client.get(_EVENTS.format(sid=session_id), params=params)
        assert r.status_code == 200, r.text
        body = r.json()
        seqs.extend(e["seq"] for e in body["data"])
        if not body["has_more"]:
            assert body["next_cursor"] is None
            break
        assert body["next_cursor"] is not None
        params = {"cursor": body["next_cursor"]}
    else:
        pytest.fail("pagination did not terminate within 20 iterations")
    return seqs


class TestForwardPagination:
    """Default ``?dir=forward`` walks oldest→newest; ``next_cursor`` chains as the
    next ``?cursor=`` with no repeats."""

    async def test_first_page_is_oldest_first(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r = await http_client.get(_EVENTS.format(sid=session_with_events), params={"limit": 2})
        assert r.status_code == 200, r.text
        body = r.json()
        assert [e["seq"] for e in body["data"]] == [1, 2]
        assert body["has_more"] is True
        assert body["next_cursor"]  # opaque, non-empty

    async def test_full_forward_walk_no_repeats(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        seqs = await _walk(http_client, session_with_events, {"dir": "forward", "limit": 2})
        assert seqs == [1, 2, 3, 4, 5]
        assert len(seqs) == len(set(seqs))


class TestSequenceAnchoredPagination:
    async def test_after_seq_walks_forward_from_exclusive_anchor(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        assert await _walk(http_client, session_with_events, {"after_seq": 2, "limit": 2}) == [
            3,
            4,
            5,
        ]

    async def test_bounded_window_is_preserved_across_pages(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        assert await _walk(
            http_client,
            session_with_events,
            {"after_seq": 1, "before_seq": 5, "limit": 2},
        ) == [2, 3, 4]


class TestLimitValidation:
    """limit > 500 must return 422; limit = 500 must be accepted."""

    async def test_limit_over_500_returns_422(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r = await http_client.get(_EVENTS.format(sid=session_with_events), params={"limit": 501})
        assert r.status_code == 422, r.text

    async def test_limit_500_accepted(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r = await http_client.get(_EVENTS.format(sid=session_with_events), params={"limit": 500})
        assert r.status_code == 200, r.text


class TestHasMoreAccuracy:
    """has_more must be an off-by-one-free signal; next_cursor is null on the last page."""

    async def test_has_more_false_at_exact_boundary(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        # 5 events, limit=5 → one full page, no next page.
        r = await http_client.get(_EVENTS.format(sid=session_with_events), params={"limit": 5})
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["data"]) == 5
        assert body["has_more"] is False
        assert body["next_cursor"] is None

    async def test_has_more_true_mints_cursor(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r = await http_client.get(_EVENTS.format(sid=session_with_events), params={"limit": 3})
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["data"]) == 3
        assert body["has_more"] is True
        assert body["next_cursor"]


class TestBackwardPagination:
    """``?dir=backward`` loads the newest-first tail; ``next_cursor`` walks into the past."""

    async def test_first_page_is_newest_first(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r = await http_client.get(
            _EVENTS.format(sid=session_with_events),
            params={"dir": "backward", "limit": 2},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert [e["seq"] for e in body["data"]] == [5, 4]
        assert body["has_more"] is True
        assert body["next_cursor"]

    async def test_full_backward_walk_no_repeats(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        seqs = await _walk(http_client, session_with_events, {"dir": "backward", "limit": 2})
        assert seqs == [5, 4, 3, 2, 1]
        assert len(seqs) == len(set(seqs))

    async def test_backward_chains_across_non_contiguous_filtered_seq(
        self, http_client: httpx.AsyncClient, pool: Any
    ) -> None:
        # Interleave message and lifecycle events so message seqs are
        # non-contiguous (1, 3, 5). A kind=message backward walk must chain via
        # the opaque cursor across the gaps — proving the token, not lastSeq-limit
        # arithmetic, drives the walk and that filters survive in the token.
        from aios.services import sessions as sessions_svc

        session_id = await _make_session_with_events(pool, n_events=0)
        await sessions_svc.append_user_message(pool, session_id, "m1", account_id=_ACCOUNT_ID)
        await _append_event(pool, session_id, "lifecycle", {"phase": "x"})
        await sessions_svc.append_user_message(pool, session_id, "m3", account_id=_ACCOUNT_ID)
        await _append_event(pool, session_id, "lifecycle", {"phase": "y"})
        await sessions_svc.append_user_message(pool, session_id, "m5", account_id=_ACCOUNT_ID)

        seqs = await _walk(
            http_client, session_id, {"dir": "backward", "kind": "message", "limit": 1}
        )
        assert seqs == [5, 3, 1]


class TestCursorValidation:
    """A ``?cursor=`` request is self-contained: any other param is a 422, and a
    garbage token is a 422."""

    async def _first_cursor(self, http_client: httpx.AsyncClient, session_id: str) -> str:
        r = await http_client.get(_EVENTS.format(sid=session_id), params={"limit": 2})
        assert r.status_code == 200, r.text
        cursor = r.json()["next_cursor"]
        assert cursor
        return str(cursor)

    async def test_cursor_with_limit_is_422(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        cursor = await self._first_cursor(http_client, session_with_events)
        r = await http_client.get(
            _EVENTS.format(sid=session_with_events), params={"cursor": cursor, "limit": 2}
        )
        assert r.status_code == 422, r.text

    async def test_cursor_with_kind_is_422(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        cursor = await self._first_cursor(http_client, session_with_events)
        r = await http_client.get(
            _EVENTS.format(sid=session_with_events),
            params={"cursor": cursor, "kind": "message"},
        )
        assert r.status_code == 422, r.text

    async def test_cursor_alone_continues(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        cursor = await self._first_cursor(http_client, session_with_events)
        r = await http_client.get(
            _EVENTS.format(sid=session_with_events), params={"cursor": cursor}
        )
        assert r.status_code == 200, r.text
        assert [e["seq"] for e in r.json()["data"]] == [3, 4]

    async def test_malformed_cursor_is_422(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r = await http_client.get(
            _EVENTS.format(sid=session_with_events), params={"cursor": "not-a-real-token!!!"}
        )
        assert r.status_code == 422, r.text

    async def test_forged_non_int_keyset_is_422_not_500(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        # A well-formed-but-tampered token: it decodes cleanly (the cursor is
        # unsigned base64url(JSON)), but carries a non-int keyset ``c``. This
        # endpoint coerces the keyset to int (``seq = int(st.cursor)``); a forged
        # non-int must surface as a malformed-cursor 422, not crash the coercion
        # into an unhandled 500.
        forged = (
            base64.urlsafe_b64encode(
                json.dumps({"v": 1, "c": "not-an-int", "d": "f", "f": {}, "l": 50}).encode()
            )
            .decode()
            .rstrip("=")
        )
        r = await http_client.get(
            _EVENTS.format(sid=session_with_events), params={"cursor": forged}
        )
        assert r.status_code == 422, r.text


class TestSessionEventStats:
    """last_event_at and total_events on the Session resource."""

    async def test_last_event_at_populated(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r = await http_client.get(f"/v1/sessions/{session_with_events}")
        assert r.status_code == 200, r.text
        assert r.json()["last_event_at"] is not None

    async def test_total_events_count(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r = await http_client.get(f"/v1/sessions/{session_with_events}")
        assert r.status_code == 200, r.text
        assert r.json()["total_events"] == 5

    async def test_fresh_session_has_zero_total_events(
        self, http_client: httpx.AsyncClient, pool: Any
    ) -> None:
        session_id = await _make_session_with_events(pool, n_events=0)
        r = await http_client.get(f"/v1/sessions/{session_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total_events"] == 0
        assert body["last_event_at"] is None


class TestGetSingleEvent:
    """GET /v1/sessions/{id}/events/{event_id} fetches one event."""

    async def test_get_event_by_id(
        self, http_client: httpx.AsyncClient, session_with_events: str
    ) -> None:
        r_list = await http_client.get(_EVENTS.format(sid=session_with_events), params={"limit": 1})
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
        r_list = await http_client.get(_EVENTS.format(sid=session_with_events), params={"limit": 1})
        assert r_list.status_code == 200, r_list.text
        event_id = r_list.json()["data"][0]["id"]

        other_session_id = await _make_session_with_events(pool, n_events=0)
        r = await http_client.get(f"/v1/sessions/{other_session_id}/events/{event_id}")
        assert r.status_code == 404, r.text
