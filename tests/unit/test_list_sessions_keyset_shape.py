"""Unit test: ``list_sessions`` emits a STRUCTURALLY correct keyset predicate.

The prior ordering test (``test_list_sessions_ordering.py``) asserts only SQL
substrings against a stub connection, so — as the #1939 review pointed out — it
"would still pass if the entire keyset WHERE clause were deleted or inverted".
This test closes that hole *without a database* by parsing the emitted SQL with
sqlglot and asserting the keyset comparison is present, points the right way,
and binds the anchor parameters. It is the cheap structural guard; the real
round-trip/no-gaps guarantee lives in the DB-backed integration test
(``tests/integration/test_list_sessions_pagination.py``).

Over-correction guard: a fix that made the ordering test pass by weakening the
query (dropping the tiebreaker, flipping ``<`` to ``>``) would slip past a
substring check. Here each assertion is anchored to a parsed comparison node, so
deleting or inverting the keyset fails loudly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
import sqlglot
from sqlglot import expressions as exp

from aios.db.queries import sessions as session_queries


class _CapturingConn:
    def __init__(self) -> None:
        self.sql: str | None = None
        self.args: tuple[Any, ...] = ()

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.sql = sql
        self.args = args
        return []


def _comparisons(sql: str, order_col: str) -> dict[str, list[exp.Expression]]:
    """Parse ``sql`` and collect the keyset comparisons over ``order_col`` and id.

    Returns the ``<`` comparisons whose left side is the order column and the
    ``<`` comparisons whose left side is ``sessions.id``. A deleted keyset yields
    empty lists; an inverted one (``>``) yields empty ``<`` lists.
    """
    tree = sqlglot.parse_one(sql, read="postgres")
    order_lt: list[exp.Expression] = []
    id_lt: list[exp.Expression] = []
    for node in tree.find_all(exp.LT):
        left = node.this.sql()
        if left.endswith(order_col):
            order_lt.append(node)
        elif left.endswith("sessions.id"):
            id_lt.append(node)
    return {"order_lt": order_lt, "id_lt": id_lt}


class TestListSessionsKeysetShape:
    @pytest.mark.parametrize(
        ("order_by", "order_col"),
        [
            ("created_at", "created_at"),
            ("updated_at", "updated_at"),
            ("last_event_at", "last_event_at"),
        ],
    )
    async def test_keyset_predicate_is_strictly_descending(
        self, order_by: str, order_col: str
    ) -> None:
        conn = _CapturingConn()
        anchor = datetime(2026, 7, 12, 23, 25, tzinfo=UTC)
        await session_queries.list_sessions(
            conn,
            account_id="acc_x",
            order_by=order_by,  # type: ignore[arg-type]
            after=(anchor, "sess_anchor"),
        )
        assert conn.sql is not None
        cmps = _comparisons(conn.sql, order_col)

        # The keyset MUST contain a strict-less-than on the order column and a
        # strict-less-than tiebreaker on id. Deleting the WHERE clause empties
        # both; inverting to ``>`` empties the ``<`` collections. Either way this
        # fails — unlike the substring assertion in test_list_sessions_ordering.
        assert cmps["order_lt"], f"missing `{order_col} < $anchor` keyset bound"
        assert cmps["id_lt"], "missing `sessions.id < $anchor_id` tiebreaker"

        # The anchor and its id must be bound as parameters (not inlined).
        assert anchor in conn.args
        assert "sess_anchor" in conn.args

    async def test_no_keyset_without_cursor(self) -> None:
        """Positive control: the FIRST page (no ``after``) has no keyset bound at
        all, only the account/archive filters — so the structural check above is
        actually reacting to the cursor, not always-true."""
        conn = _CapturingConn()
        await session_queries.list_sessions(conn, account_id="acc_x", order_by="created_at")
        assert conn.sql is not None
        cmps = _comparisons(conn.sql, "created_at")
        assert not cmps["order_lt"]
        assert not cmps["id_lt"]
