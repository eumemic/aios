"""Integration repro for #744 — the CORRECTED root cause: the connector SSE
subscribe-time backfill surfaces weeks-stale, unresolved connector sends for
transmission with **no age guard**.

Background — what actually fired the metals-factchecker incident
================================================================
On the 2026-06-08 worker restart, 17 ``signal_send`` messages authored
2026-05-22 (~2.5 weeks dormant) were transmitted.  The mechanism is NOT the
harness confirmed-tool dispatch resolver (``list_confirmed_unresolved_tool_calls``):
``signal_send`` is ``always_allow``, so the incident calls carry **no**
``tool_confirmed`` lifecycle event for that resolver to key on.

The real path is the **connector SSE pending-call backfill**:

* ``queries.list_pending_calls_for_connector``  (db/queries/__init__.py:2256)
* invoked at SSE subscribe time in ``runtime_connector_calls_stream``
  (api/sse.py:218): when the ``aios-signal`` connector reconnects after the
  worker restart, re-subscribes, and backfills every pending call.

That function delegates to ``_unresolved_tool_calls`` (db/queries/__init__.py:2427),
which surfaces a tool_call whose ``function.name`` is in the connector type's
``tools_schema`` and has **no paired tool_result**, on **ANY assistant turn**.
Commit ``775554e`` section #2 (#741, the "awaiting" fix) changed the backing
predicate from latest-assistant-only (``_latest_unresolved_tool_calls``, a
``DISTINCT ON (session_id) ... ORDER BY seq DESC``) to all-assistants
(``_unresolved_tool_calls``).  Read the query body: there is **no**
``created_at`` reference, **no** ``interval`` / ``now()`` age bound, and **no**
windowing.  So a dormant ``signal_send`` from weeks ago — sitting on a
non-latest assistant turn, never resolved — is recovered for transmission the
moment the connector reconnects.

The invariant this test pins
============================
A connector send that is

  * **unresolved** (no paired tool_result), and
  * authored on a **non-latest** assistant turn (a later assistant exists), and
  * **~21 days old** (the incident calls were ~2.5 weeks old)

MUST NOT be surfaced by ``list_pending_calls_for_connector`` for transmission.

Assertions (after the fix):

  * ``test_stale_connector_send_excluded`` — the fixed contract: ``tc_stale``
    is NOT in the backfill output (the ~21-day-old send is dropped by the age
    guard).  This was the RED repro before the fix; it is GREEN now.
  * ``test_fresh_pending_send_still_surfaced`` — the over-reach guard: a fresh,
    unresolved send on a NON-latest assistant (``tc_fresh``) is STILL surfaced
    while ``tc_stale`` is excluded.  ``tc_fresh`` proves the #741 all-assistants
    backfill is preserved for non-stale calls — the fix bounds by age only, it
    does not collapse back to latest-assistant-only.

The remediation (commit on top of this repro) is an age guard scoped to the
transmit/backfill path ONLY (``list_pending_calls_for_connector`` passes an
``max_age_seconds`` to ``_unresolved_tool_calls``; ~1h via
``settings.connector_backfill_max_age_seconds``) — skip-not-expire (the event
log is untouched) and NOT a change to ``Session.awaiting`` (the read-model
sibling, which legitimately surfaces all unresolved calls regardless of age,
#741).  These tests assert only the user-facing connector-backfill contract.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from typing import Any, Literal

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

# The metals-factchecker incident's stale ``signal_send`` calls were ~2.5 weeks
# old; 21 days is comfortably past any sane transmit window while staying a
# concrete, single value the test can backdate to.
STALE_AGE = dt.timedelta(days=21)

CONNECTOR = "signal"
SEND_TOOL = "signal_send"


def _assistant(tool_call_id: str, name: str = SEND_TOOL) -> dict[str, Any]:
    """An assistant message carrying a single custom tool_call (no result)."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": name, "arguments": '{"text": "hi"}'},
            }
        ],
    }


@pytest.fixture
async def bound_signal_session_with_stale_send(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, session_id, connection_id)`` for a ``signal``
    connection bound to a session whose event log is::

        A1[tc_stale signal_send]  → user → A2[tc_fresh signal_send]

    Neither send has a tool_result (both unresolved).  ``tc_stale`` is on the
    NON-latest assistant A1, and its assistant event is backdated ~21 days.
    ``tc_fresh`` is on the latest assistant A2 and is fresh.  The connector
    type's ``tools_schema`` registers ``signal_send`` so the backfill's
    name-gate admits both.

    ``session_id`` and ``connection_id`` are exposed so the NOTIFY-tail test
    can call ``list_pending_calls_for_session_and_connection`` directly (the
    tail's per-session fetch), not just the subscribe-time backfill.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        account_id = "acc_stale_connector_backfill"
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, $2)",
                account_id,
                "stale-connector-backfill-test",
            )

        # A session-shaped scaffold; the agent's tools don't gate the backfill
        # (the connector type's tools_schema does), only the event log matters.
        _agent, _env, session = await seed_agent_env_session(
            pool,
            account_id=account_id,
            prefix="stale-connector-backfill",
            tools=[],
        )
        sid = session.id

        async with pool.acquire() as conn:
            # A signal connection bound to the session: this is what makes the
            # session "bound" for list_pending_calls_for_connector's roster
            # join (connections + bindings).
            connection = await queries.insert_connection(
                conn,
                account_id=account_id,
                connector=CONNECTOR,
                external_account_id="+15550042",
                metadata={},
            )
            await queries.insert_binding(
                conn,
                account_id=account_id,
                connection_id=connection.id,
                mode="single_session",
                session_id=sid,
            )
            # Register signal_send in the per-type tools_schema so the
            # backfill's name-gate (name_set, built from connectors.tools_schema)
            # admits it. insert_connection upserts the connectors row with the
            # default empty '[]' schema (ON CONFLICT DO NOTHING), so we publish
            # the real schema afterward, mirroring the connector container's
            # startup PUT /v1/connectors/{connector}/tools_schema.
            await queries.update_connector_tools_schema(
                conn,
                CONNECTOR,
                account_id=account_id,
                tools_schema=[
                    {
                        "type": "custom",
                        "name": SEND_TOOL,
                        "description": "Send a Signal message.",
                        "input_schema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                ],
            )

        async def append(
            kind: Literal["message", "lifecycle", "span", "interrupt"], data: dict[str, Any]
        ) -> None:
            async with pool.acquire() as conn:
                await queries.append_event(
                    conn, account_id=account_id, session_id=sid, kind=kind, data=data
                )

        # A1 carries the stale send (later confirmed by time-passing, never
        # resolved); a user message lifts the session; A2 (the latest assistant)
        # carries a fresh send.  Neither send gets a tool_result → both remain
        # unresolved.  tc_stale's parent A1 is deliberately NOT the latest
        # assistant, so it only surfaces via the all-assistants predicate (#741).
        await append("message", _assistant("tc_stale"))
        await append("message", {"role": "user", "content": "are you still there?"})
        await append("message", _assistant("tc_fresh"))

        # Backdate the stale send's age. append_event stamps created_at = now()
        # (the table default); rewrite it on the A1 assistant event that issued
        # tc_stale so the call is ~21 days old — well past any sane transmit
        # window.  created_at is the ONLY signal an age guard could key on; the
        # backfill path currently ignores it entirely.
        old = dt.datetime.now(dt.UTC) - STALE_AGE
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE events SET created_at = $1 "
                "WHERE session_id = $2 AND account_id = $3 "
                "AND kind = 'message' AND role = 'assistant' "
                "AND data->'tool_calls' @> jsonb_build_array("
                "    jsonb_build_object('id', 'tc_stale'::text))",
                old,
                sid,
                account_id,
            )

        yield pool, account_id, sid, connection.id
    finally:
        await pool.close()


class TestStaleConnectorBackfill:
    async def test_stale_connector_send_excluded(
        self,
        bound_signal_session_with_stale_send: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """#744 fixed contract: a ~21-day-old, unresolved ``signal_send`` on a
        non-latest assistant MUST NOT be surfaced by the connector backfill for
        transmission.

        Was RED before the fix: ``list_pending_calls_for_connector`` →
        ``_unresolved_tool_calls`` had no ``created_at`` / age guard, so the
        dormant send was recovered and (at SSE subscribe time, sse.py:218)
        re-transmitted on the next connector reconnect — exactly the
        metals-factchecker incident (17 weeks-old signal_send messages re-sent
        on a worker restart).  GREEN now: the backfill passes an age bound
        (``settings.connector_backfill_max_age_seconds``, ~1h) to
        ``_unresolved_tool_calls``, which excludes the stale parent turn.
        """
        pool, account_id, _sid, _conn_id = bound_signal_session_with_stale_send
        async with pool.acquire() as conn:
            backfill = await queries.list_pending_calls_for_connector(
                conn, CONNECTOR, account_id=account_id
            )
        surfaced = {c["tool_call_id"] for c in backfill}
        assert "tc_stale" not in surfaced, (
            f"a ~{STALE_AGE.days}-day-old unresolved {SEND_TOOL} (tc_stale, on "
            "the non-latest assistant A1) was surfaced by "
            "list_pending_calls_for_connector for transmission — the connector "
            "SSE backfill (sse.py:218) has no age guard, so it re-sends "
            "weeks-stale dormant connector calls on reconnect (#744). "
            f"backfill surfaced: {sorted(surfaced)}"
        )

    async def test_fresh_pending_send_still_surfaced(
        self,
        bound_signal_session_with_stale_send: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """Over-reach guard: the age fix excludes ``tc_stale`` but must NOT
        collapse the backfill back to latest-assistant-only.

        ``list_pending_calls_for_connector`` surfaces EXACTLY ``tc_fresh`` and
        not ``tc_stale``:

        * ``tc_fresh`` — a fresh, unresolved send; it is STILL surfaced, which
          proves the #741 all-assistants backfill is preserved for non-stale
          calls (the fix bounds by age, it does not re-introduce the
          latest-assistant-only window).
        * ``tc_stale`` — the same shape but ~21 days old; excluded by the age
          guard, which is the #744 fix.

        Pairs with ``test_stale_connector_send_excluded``: together they pin
        the exact delta — age-bounded, not behavior-collapsed.
        """
        pool, account_id, _sid, _conn_id = bound_signal_session_with_stale_send
        async with pool.acquire() as conn:
            backfill = await queries.list_pending_calls_for_connector(
                conn, CONNECTOR, account_id=account_id
            )
        surfaced = {c["tool_call_id"] for c in backfill}
        assert surfaced == {"tc_fresh"}, (
            "expected the fixed connector backfill to surface ONLY the fresh "
            "send (tc_fresh) and exclude the ~21-day-old dormant send "
            "(tc_stale): the age guard (#744) drops tc_stale while the #741 "
            "all-assistants behavior is preserved for non-stale calls. "
            f"got: {sorted(surfaced)}"
        )
        # The surfaced record is transmit-ready: it carries the connection_id
        # the runtime fans out on and the original tool arguments.
        by_id = {c["tool_call_id"]: c for c in backfill}
        assert by_id["tc_fresh"]["name"] == SEND_TOOL
        assert by_id["tc_fresh"]["connection_id"]
        assert by_id["tc_fresh"]["arguments"] == '{"text": "hi"}'
        # #816: the awaiting read-model now stamps each unresolved tool_call
        # with a per-row ``pending_since`` (and an internal ``_pending_since``
        # carrier from ``_unresolved_tool_calls``). The connector backfill
        # builds its own explicit output dict, so neither key must leak into
        # the transmit payload — its shape is exactly the documented set.
        assert set(by_id["tc_fresh"]) == {
            "session_id",
            "tool_call_id",
            "name",
            "arguments",
            "connection_id",
            "focal_channel",
            "workspace_path",
        }


class TestStaleConnectorTail:
    """The SSE NOTIFY *tail* — the second transmit path — must be age-bounded
    identically to the subscribe-time backfill (#744).

    ``runtime_connector_calls_stream`` (api/sse.py) has TWO emit paths sharing
    one ``emitted`` dedup set:

    1. subscribe-time backfill → ``list_pending_calls_for_connector`` (bounded).
    2. the NOTIFY tail (the ``while True`` loop) →
       ``list_pending_calls_for_session_and_connection``.

    The ``emitted`` set only suppresses calls the *backfill* already yielded.
    Because the backfill now SKIPS stale calls, a stale send is NOT in
    ``emitted`` — so if the tail's per-session fetch were unbounded, a session
    that carries a weeks-stale unresolved connector send and then emits a new
    event (firing the per-session NOTIFY) would hit the tail, fetch the stale
    send, find it absent from ``emitted``, and transmit it: the exact #744 harm
    (a weeks-stale send delivered out of nowhere), triggered by session
    re-activation instead of connector reconnect.  This test exercises that
    tail fetch directly and asserts the stale send is excluded while the fresh
    one is surfaced.
    """

    async def test_tail_excludes_stale_surfaces_fresh(
        self,
        bound_signal_session_with_stale_send: tuple[asyncpg.Pool[Any], str, str, str],
    ) -> None:
        """The tail fetch (``list_pending_calls_for_session_and_connection``)
        for the re-activated session surfaces EXACTLY ``tc_fresh`` and excludes
        ``tc_stale``.

        Was RED before the tail fix: that function called
        ``_unresolved_tool_calls`` with NO ``max_age_seconds``, returning ALL
        unresolved calls for the session unbounded — so ``tc_stale`` (~21 days
        old, absent from the stream's ``emitted`` set because the backfill
        skipped it) slipped through and was transmitted on the next per-session
        NOTIFY.  GREEN now: the tail passes the same
        ``settings.connector_backfill_max_age_seconds`` bound the backfill does,
        so the stale parent turn is excluded on BOTH transmit paths.
        """
        pool, account_id, sid, conn_id = bound_signal_session_with_stale_send
        async with pool.acquire() as conn:
            pending = await queries.list_pending_calls_for_session_and_connection(
                conn,
                session_id=sid,
                connection_id=conn_id,
                account_id=account_id,
            )
        surfaced = {c["tool_call_id"] for c in pending}
        assert surfaced == {"tc_fresh"}, (
            "expected the SSE NOTIFY-tail fetch "
            "(list_pending_calls_for_session_and_connection) to surface ONLY "
            "the fresh send (tc_fresh) and exclude the ~21-day-old dormant send "
            "(tc_stale): the tail is the second transmit path into "
            "runtime_connector_calls_stream and must be age-bounded identically "
            "to the backfill (#744). The stale send is absent from the stream's "
            "`emitted` dedup (the backfill skipped it), so an unbounded tail "
            "would re-transmit it the instant the session emits a new event. "
            f"got: {sorted(surfaced)}"
        )
        # The surfaced record is transmit-ready, exactly as the backfill path's.
        by_id = {c["tool_call_id"]: c for c in pending}
        assert by_id["tc_fresh"]["name"] == SEND_TOOL
        assert by_id["tc_fresh"]["connection_id"] == conn_id
        assert by_id["tc_fresh"]["arguments"] == '{"text": "hi"}'
