"""``_filter_incomplete_batches`` must fetch ONLY what it inspects (#1729).

Before #1729 the batch filter ran two unbounded lifetime scans per full sweep:
``ALL_RESULT_ROWS_SQL`` (every ``role='tool'`` message ever) and
``ALL_ASST_ROWS_SQL`` (every assistant-with-tool_calls message ever, with the
FULL ``data`` payload — 126 MB of JSONB observed on the largest production
session). Both were pulled over the wire and JSON-decoded row-by-row on the
worker event loop on every step, a 12-17s pre-model stall that grows linearly
and forever with session size.

These tests encode the fix as a **structural** contract on the SQL the filter
issues, driven through a fake connection (no docker):

  * it NEVER issues the two unbounded lifetime scans;
  * its assistant fetch is bounded by tool_call_id containment (``@>``) and
    projects only ``tool_calls[].id`` (no ``data`` payload);
  * its result fetch is bounded to the referenced batch ids (``= ANY``);
  * a session with no unreacted events never touches ``events`` for
    assistant/result rows at all.

Behaviour parity (a session is/ isn't admitted for the same reasons as before)
is covered by the docker-backed sweep e2e suite; these are the perf-shape
guards that a well-meaning refactor back to a lifetime scan would trip.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from aios.harness import sweep
from aios.harness.inflight_tool_registry import InflightToolRegistry
from tests.unit.conftest import fake_pool_yielding_conn


def _dispatching_conn(handlers: list[tuple[str, list[dict[str, Any]]]]) -> Any:
    """Build a fake ``conn`` whose ``fetch`` returns rows based on the SQL text.

    ``handlers`` is a list of ``(needle, rows)``: the first needle found as a
    substring of the SQL passed to ``conn.fetch`` wins. Records the SQL of every
    call on ``conn.fetched_sql`` for assertions.
    """
    conn = MagicMock()
    conn.fetched_sql = []

    async def _fetch(sql: str, *args: Any) -> list[dict[str, Any]]:
        conn.fetched_sql.append(sql)
        for needle, rows in handlers:
            if needle in sql:
                return rows
        raise AssertionError(f"unexpected SQL fetched by batch filter:\n{sql}")

    conn.fetch = AsyncMock(side_effect=_fetch)
    return conn


async def test_filter_never_issues_unbounded_lifetime_scans() -> None:
    """The batch filter must not run the removed whole-history queries — the
    projected assistant/result select and the ``@>`` containment bound are the
    load-bearing fix (#1729)."""
    unreacted = [{"session_id": "sess_a", "data": {"role": "tool", "tool_call_id": "tc_1"}}]
    asst = [{"session_id": "sess_a", "tool_call_ids": ["tc_1"]}]
    results = [{"session_id": "sess_a", "tool_call_id": "tc_1"}]

    conn = _dispatching_conn(
        [
            ("s.last_reacted_seq", unreacted),  # UNREACTED_ROWS_SQL
            ("jsonb_path_query_array", asst),  # REFERENCED_ASST_BATCH_SQL
            ("data->>'tool_call_id' = ANY", results),  # BATCH_RESULT_ROWS_SQL
        ]
    )
    pool = fake_pool_yielding_conn(conn)

    out = await sweep._filter_incomplete_batches(pool, InflightToolRegistry(), {"sess_a"})

    # Fully-resolved single batch → session is ready.
    assert out == {"sess_a"}

    joined = "\n".join(conn.fetched_sql)
    # The projected assistant fetch selects ONLY the id array. The removed
    # ``ALL_ASST_ROWS_SQL`` selected the full ``data`` payload gated only by
    # ``jsonb_array_length(... 'tool_calls') > 0`` — assert that whole-payload
    # unbounded shape is gone (the seq-bounded ``UNREACTED_ROWS_SQL`` also selects
    # ``e.data`` but carries a ``last_reacted_seq`` bound, so key on the removed
    # query's distinguishing predicate).
    assert "jsonb_path_query_array(e.data->'tool_calls'" in joined
    assert "jsonb_array_length(COALESCE(NULLIF(e.data->'tool_calls'" not in joined
    # The assistant fetch is bounded by tool_call_id containment, not a scan of
    # every assistant-with-tool_calls row.
    assert "e.data->'tool_calls' @> ANY" in joined
    # The result fetch is bounded to the referenced batch ids.
    assert "e.data->>'tool_call_id' = ANY" in joined
    # None of the fetches is the removed unbounded lifetime scan (session set
    # only, no tcid/containment bound).
    assert (
        "e.session_id = ANY($1::text[])\n       AND e.kind = 'message'\n       AND e.role = 'tool'\n\""
        not in joined
    )


async def test_no_unreacted_events_touches_no_history() -> None:
    """A candidate with an empty unreacted set is decided from the unreacted
    query alone — no assistant/result fetch fires (it can't need bounding what
    it never reads)."""
    conn = _dispatching_conn([("s.last_reacted_seq", [])])
    pool = fake_pool_yielding_conn(conn)

    out = await sweep._filter_incomplete_batches(pool, InflightToolRegistry(), {"sess_idle"})

    # No unreacted events and nothing in-flight → ready.
    assert out == {"sess_idle"}
    # Exactly one fetch: the seq-bounded unreacted query.
    assert conn.fetch.await_count == 1
    assert "s.last_reacted_seq" in conn.fetched_sql[0]


async def test_incomplete_batch_not_admitted() -> None:
    """A referenced batch whose sibling ids lack results is NOT admitted — the
    session is still waiting on in-flight tools (parity with pre-#1729)."""
    unreacted = [{"session_id": "sess_a", "data": {"role": "tool", "tool_call_id": "tc_1"}}]
    # Batch owns tc_1 and tc_2, but only tc_1 has a result.
    asst = [{"session_id": "sess_a", "tool_call_ids": ["tc_1", "tc_2"]}]
    results = [{"session_id": "sess_a", "tool_call_id": "tc_1"}]

    conn = _dispatching_conn(
        [
            ("s.last_reacted_seq", unreacted),
            ("jsonb_path_query_array", asst),
            ("data->>'tool_call_id' = ANY", results),
        ]
    )
    pool = fake_pool_yielding_conn(conn)

    out = await sweep._filter_incomplete_batches(pool, InflightToolRegistry(), {"sess_a"})
    assert out == set()


async def test_user_message_bypasses_batch_fetch() -> None:
    """An unreacted user message admits immediately, without any batch fetch."""
    unreacted = [{"session_id": "sess_u", "data": {"role": "user", "content": "hi"}}]
    conn = _dispatching_conn([("s.last_reacted_seq", unreacted)])
    pool = fake_pool_yielding_conn(conn)

    out = await sweep._filter_incomplete_batches(pool, InflightToolRegistry(), {"sess_u"})
    assert out == {"sess_u"}
    # Only the unreacted query — the user branch short-circuits before any
    # assistant/result fetch.
    assert conn.fetch.await_count == 1


async def test_removed_unbounded_constants_are_gone() -> None:
    """The two unbounded lifetime scans the batch filter used are removed from
    its query surface. ``ALL_RESULT_ROWS_SQL`` survives ONLY for ghost repair
    (bounded upstream by ``open_tool_call_count > 0``); ``ALL_ASST_ROWS_SQL`` —
    the 126 MB payload fetch — is deleted outright."""
    assert not hasattr(sweep, "ALL_ASST_ROWS_SQL")
    # New bounded queries exist and carry their bounds.
    assert "@> ANY" in sweep.REFERENCED_ASST_BATCH_SQL
    assert "jsonb_path_query_array" in sweep.REFERENCED_ASST_BATCH_SQL
    assert "e.data" not in sweep.REFERENCED_ASST_BATCH_SQL.split("FROM")[0].replace(
        "e.data->'tool_calls'", ""
    )
    assert "= ANY($2::text[])" in sweep.BATCH_RESULT_ROWS_SQL
