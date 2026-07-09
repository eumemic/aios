"""Integration pin: the ``events.role`` column stays byte-equivalent to
``data->>'role'`` for every message event (modulo the non-string-NULL rule).

Follow-up from #1755 (tier-4 gate review of #1751, UNREACTED_ROWS_SQL
projection fix). ``UNREACTED_ROWS_SQL`` filters/projects the maintained
``role`` COLUMN instead of the JSONB payload — a read-time optimization
(#1738) that is only correct as long as ``append_event``'s write-time
promotion (``src/aios/db/queries/events.py``: role is set iff
``kind == 'message'`` and ``data['role']`` is a ``str``) and migration
0022's identical backfill both keep that column in lockstep with the
payload. If a future writer bypasses the promotion (hand-rolled INSERT,
a new append path, a refactor that drops the ``role=`` argument), the
sweep's WHERE silently diverges from what the payload actually says —
a session could go unreacted-forever or get falsely woken with no
loud failure anywhere.

This test appends every representative message shape through the real
``append_event`` write path and asserts, per row,
``role IS NOT DISTINCT FROM data->>'role'`` — with the one documented
exception: a non-string ``data['role']`` (e.g. ``None``, a number, an
object) promotes to a NULL column while ``data->>'role'`` may itself
render a non-NULL text (e.g. ``5`` -> ``'5'``, or an object -> its JSON
text). We assert the promotion rule directly instead of the raw
equality for those rows, and plain byte-equivalence for the ordinary
string-role rows.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_and_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_role_inv', NULL, TRUE, 'role-invariant-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_role_inv", prefix="role-inv-test"
        )
        yield pool, "acc_role_inv", session.id
    finally:
        await pool.close()


async def test_role_column_matches_payload_for_message_events(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Append a representative mix of message events through
    ``append_event`` and assert the promoted ``role`` column tracks
    ``data->>'role'`` exactly for ordinary string roles, and follows the
    documented non-string-NULL exception otherwise.

    Red if the promotion in ``append_event`` (or migration 0022's backfill
    formula) breaks — e.g. someone forgets the ``isinstance(raw_role, str)``
    guard, or stops passing ``role`` on the INSERT. Green on master.
    """
    pool, account_id, session_id = pool_and_session

    string_role_cases: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
    ]
    # Non-string ``role`` values: append_event's write-time promotion only
    # sets the column when ``isinstance(data.get("role"), str)`` — a
    # non-string role (missing/None/number/object) must promote to NULL,
    # even though ``data->>'role'`` may itself be non-NULL text for some of
    # these (e.g. ``->>'role'`` on a number renders its digits).
    non_string_role_cases: list[dict[str, Any]] = [
        {"content": "no role key at all"},
        {"role": None, "content": "explicit null role"},
        {"role": 5, "content": "numeric role"},
        {"role": {"nested": "object"}, "content": "object role"},
    ]

    async with pool.acquire() as conn:
        for data in string_role_cases + non_string_role_cases:
            await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data=data,
            )

        rows = await conn.fetch(
            "SELECT role, data, data->>'role' AS payload_role "
            "FROM events WHERE session_id = $1 AND kind = 'message' ORDER BY seq",
            session_id,
        )

    assert len(rows) == len(string_role_cases) + len(non_string_role_cases)

    for row in rows:
        raw_role = row["data"].get("role") if isinstance(row["data"], dict) else None
        if isinstance(raw_role, str):
            # Ordinary case: the column must be byte-equivalent to the
            # JSONB projection (modulo SQL NULL-vs-Python-None spelling,
            # handled by IS NOT DISTINCT FROM).
            assert row["role"] == row["payload_role"] == raw_role, (
                "events.role diverged from data->>'role' for a string-role "
                f"message event: role={row['role']!r} "
                f"payload_role={row['payload_role']!r} raw={raw_role!r}. "
                "The append_event write-time promotion (or migration 0022's "
                "backfill) is broken — the sweep's WHERE on the role column "
                "no longer matches the payload (#1755)."
            )
        else:
            # Documented exception: a non-string data['role'] must promote
            # to a NULL column, regardless of what data->>'role' renders.
            assert row["role"] is None, (
                "events.role must be NULL when data['role'] is not a string "
                f"(got role={row['role']!r} for data={row['data']!r}); "
                "append_event's isinstance(raw_role, str) guard appears to "
                "have regressed (#1755)."
            )
