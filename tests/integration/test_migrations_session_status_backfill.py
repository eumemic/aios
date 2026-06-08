"""Integration tests for migration 0066's session-status scalar backfill.

Migration 0066 adds five maintained scalar columns to ``sessions``
(``last_reacted_seq``, ``open_tool_call_count``, ``last_error_seq``,
``last_user_seq``, ``last_stimulus_seq``) and backfills them from the event
log for existing sessions, so ``/v1/sessions`` status derivation is O(1)
column arithmetic instead of an O(session-size) correlated scan (#748/#749,
landed in #750).

The defect these tests guard against: 0066's *first* cut wrote the backfill
as five **correlated** per-session subqueries (each ``SELECT ... WHERE
ate.session_id = s.id``, one a ``COUNT(*)`` over ``jsonb_array_elements`` with
a ``NOT EXISTS`` anti-join). That correlated form re-runs the very
O(session-size) event-log scan the feature exists to eliminate — once per
session, under ``ACCESS EXCLUSIVE`` — so a fresh ``alembic upgrade head`` on a
large existing DB (DR restore, new replica) hangs for minutes on a big
session. CI never caught it because testcontainer DBs are tiny. The fix is a
single **set-based** statement (a handful of grouped aggregates total,
independent of any one session's size).

These tests:

* seed events directly at revision 0065 (pre-0066), exercising every scalar —
  user messages, an assistant reply with ``reacting_to``, an assistant reply
  WITHOUT ``reacting_to`` (own-seq fallback), assistant ``tool_calls`` with and
  without paired results, an error lifecycle event, and the
  ``"tool_calls": null`` shape that the bare correlated form ERRORS on
  ("cannot extract elements from a scalar");
* run ``upgrade head`` so 0066's backfill fires;
* assert each scalar equals an independently-computed expected value;
* assert the migration source contains no per-session correlated subquery; and
* a bounded-scan smoke: many events for ONE session, asserting the backfill
  completes well under the wall-clock budget a correlated form would blow.

Mirrors ``test_migrations_workspace_path_backfill.py`` (seed-at-N then
upgrade-to-head against a fresh per-test Postgres).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from typing import Any

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import (
    PROJECT_ROOT,
    _alembic_url,
    _run_alembic,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def postgres() -> Iterator[object]:
    """Fresh function-scoped Postgres — each test mutates ``alembic_version``."""
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


# ─── direct-SQL seeding (bypasses the service layer) ─────────────────────────

# The backfill must run against rows that pre-date the scalar columns, so we
# seed raw events at revision 0065 rather than going through ``append_event``
# (which would maintain the scalars itself and hide a broken backfill).

_ACCOUNT_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root')
ON CONFLICT DO NOTHING
"""

_AGENT_SQL = """
INSERT INTO agents (
    id, account_id, name, model, system, tools, description, metadata,
    window_min, window_max, version, created_at, updated_at
)
VALUES (
    'agent_test', 'acc_root', 'test', 'openrouter/test', '', '[]'::jsonb,
    NULL, '{}'::jsonb, 50000, 150000, 1, now(), now()
)
ON CONFLICT DO NOTHING
"""

_ENV_SQL = """
INSERT INTO environments (id, account_id, name, config, created_at)
VALUES ('env_test', 'acc_root', 'test', '{}'::jsonb, now())
ON CONFLICT DO NOTHING
"""


async def _seed_session_row(conn: asyncpg.Connection[Any], session_id: str) -> None:
    # Only the NOT-NULL-without-default columns at revision 0065 are supplied;
    # everything else takes its column default. Notably there is no ``status``
    # column at 0065 — migration 0063 dropped it (status is derived from the
    # event log now), which is the whole reason 0066's scalars exist.
    await conn.execute(
        """
        INSERT INTO sessions (
            id, account_id, agent_id, environment_id, workspace_volume_path
        )
        VALUES ($1, 'acc_root', 'agent_test', 'env_test', '/ws/' || $1)
        """,
        session_id,
    )


async def _seed_event(
    conn: asyncpg.Connection[Any],
    *,
    session_id: str,
    seq: int,
    kind: str,
    data: dict[str, Any],
) -> None:
    """Insert one raw event, mirroring how ``append_event`` stamps columns.

    ``role`` is promoted to a physical column (migration 0022) kept
    byte-equivalent to ``data->>'role'``; the 0066 backfill reads it, so we
    populate it the same way ``append_event`` does.
    """
    role = data.get("role") if kind == "message" else None
    await conn.execute(
        """
        INSERT INTO events (id, session_id, seq, kind, data, role, account_id)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, 'acc_root')
        """,
        f"{session_id}-{seq}",
        session_id,
        seq,
        kind,
        json.dumps(data),
        role,
    )


async def _scalars(conn: asyncpg.Connection[Any], session_id: str) -> dict[str, int]:
    row = await conn.fetchrow(
        "SELECT last_reacted_seq, open_tool_call_count, last_error_seq, "
        "last_user_seq, last_stimulus_seq FROM sessions WHERE id = $1",
        session_id,
    )
    assert row is not None
    return {k: int(v) for k, v in dict(row).items()}


# ─── the rich-fixture backfill test ──────────────────────────────────────────


async def _seed_rich(db_url: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(_ACCOUNT_SQL)
        await conn.execute(_AGENT_SQL)
        await conn.execute(_ENV_SQL)

        # ── s_full: exercises every scalar ──────────────────────────────
        # seq1 user                              -> last_user_seq=1, stimulus=1
        # seq2 assistant reacting_to 1, 2 tool_calls (tc_a, tc_b)
        #                                        -> reacted=1, open +2
        # seq3 tool result tc_a                  -> stimulus=3, open -> 1 (tc_b open)
        # seq4 assistant reply, NO reacting_to   -> reacted=4 (own-seq fallback)
        # seq5 lifecycle error                   -> last_error_seq=5
        # seq6 user (would clear the error)      -> last_user_seq=6, stimulus=6
        await _seed_session_row(conn, "s_full")
        await _seed_event(
            conn,
            session_id="s_full",
            seq=1,
            kind="message",
            data={"role": "user", "content": "hi"},
        )
        await _seed_event(
            conn,
            session_id="s_full",
            seq=2,
            kind="message",
            data={
                "role": "assistant",
                "reacting_to": 1,
                "tool_calls": [{"id": "tc_a"}, {"id": "tc_b"}],
            },
        )
        await _seed_event(
            conn,
            session_id="s_full",
            seq=3,
            kind="message",
            data={"role": "tool", "tool_call_id": "tc_a", "content": "ok"},
        )
        await _seed_event(
            conn,
            session_id="s_full",
            seq=4,
            kind="message",
            data={"role": "assistant", "content": "more"},
        )
        await _seed_event(
            conn,
            session_id="s_full",
            seq=5,
            kind="lifecycle",
            data={"stop_reason": "error"},
        )
        await _seed_event(
            conn,
            session_id="s_full",
            seq=6,
            kind="message",
            data={"role": "user", "content": "retry"},
        )

        # ── s_empty: no events -> every scalar must backfill to 0 ────────
        await _seed_session_row(conn, "s_empty")

        # ── s_nullcalls: an assistant row with "tool_calls": null ────────
        # JSON null (key present, value null) satisfies ``data ? 'tool_calls'``
        # but ``jsonb_array_elements('null')`` raises "cannot extract elements
        # from a scalar" — the bare correlated form aborts the whole migration
        # on this shape. The set-based form's COALESCE(NULLIF(...)) guard makes
        # it contribute zero open calls.
        await _seed_session_row(conn, "s_nullcalls")
        await _seed_event(
            conn,
            session_id="s_nullcalls",
            seq=1,
            kind="message",
            data={"role": "user", "content": "x"},
        )
        await _seed_event(
            conn,
            session_id="s_nullcalls",
            seq=2,
            kind="message",
            data={"role": "assistant", "content": "reply", "reacting_to": 1, "tool_calls": None},
        )
    finally:
        await conn.close()


@needs_docker
def test_backfill_computes_every_scalar(postgres: object) -> None:
    """After ``upgrade head`` runs 0066's backfill, each scalar on each seeded
    session equals its independently-computed expected value."""
    db_url = _alembic_url(postgres)

    # Stop at 0065 so the scalar columns don't exist yet and we can seed the
    # legacy (pre-0066) event log directly.
    result = _run_alembic(["upgrade", "0065"], db_url)
    assert result.returncode == 0, f"upgrade 0065 failed:\n{result.stderr}\n{result.stdout}"

    asyncio.run(_seed_rich(db_url))

    # Run 0066's backfill.
    result = _run_alembic(["upgrade", "head"], db_url)
    assert result.returncode == 0, (
        "upgrade head (0066 backfill) failed — note the bare correlated form "
        f"ERRORS on the 'tool_calls': null shape:\n{result.stderr}\n{result.stdout}"
    )

    async def check() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            full = await _scalars(conn, "s_full")
            assert full == {
                "last_user_seq": 6,  # MAX user seq (seq 6)
                "last_stimulus_seq": 6,  # MAX non-assistant message seq (user seq 6)
                "last_error_seq": 5,  # the error lifecycle at seq 5
                # MAX(COALESCE(reacting_to, seq)) over assistants:
                # seq2 reacting_to=1, seq4 no reacting_to -> own seq 4 => 4
                "last_reacted_seq": 4,
                # tc_a resolved, tc_b open -> 1
                "open_tool_call_count": 1,
            }, full

            empty = await _scalars(conn, "s_empty")
            assert empty == {
                "last_user_seq": 0,
                "last_stimulus_seq": 0,
                "last_error_seq": 0,
                "last_reacted_seq": 0,
                "open_tool_call_count": 0,
            }, empty

            nullcalls = await _scalars(conn, "s_nullcalls")
            assert nullcalls == {
                "last_user_seq": 1,
                "last_stimulus_seq": 1,
                "last_error_seq": 0,
                "last_reacted_seq": 1,  # assistant seq2 reacting_to=1
                "open_tool_call_count": 0,  # "tool_calls": null -> zero open
            }, nullcalls
        finally:
            await conn.close()

    asyncio.run(check())


# ─── structural guard: no per-session correlated subquery ────────────────────


def test_migration_backfill_is_not_correlated() -> None:
    """The 0066 backfill must be set-based, not a per-session correlated scan.

    A correlated subquery (``WHERE ... = s.id`` joining an aggregate back to the
    outer ``sessions`` row) re-runs the O(session-size) event-log scan once per
    session under ACCESS EXCLUSIVE — the exact hang #750 exists to remove. Guard
    the source text so a future edit can't silently reintroduce it.
    """
    raw = (PROJECT_ROOT / "migrations" / "versions" / "0066_session_status_scalars.py").read_text()

    # Strip Python comment lines (and the module docstring's prose) so we scan
    # only executable code — the rewrite intentionally *names* the old
    # correlated form in a comment to warn against it, which must not trip the
    # guard. Comment lines are whole-line ``#`` comments at any indent.
    src = "\n".join(line for line in raw.splitlines() if not line.lstrip().startswith("#"))

    # The original defect's signature: an inner ``events`` alias correlated back
    # to the outer ``sessions s`` row inside a per-column subquery.
    for needle in (
        "e.session_id = s.id",
        "ate.session_id = s.id",
        "tr.session_id = s.id",
    ):
        assert needle not in src, (
            f"0066 backfill reintroduced a correlated subquery ({needle!r}); "
            "the backfill must be a single set-based statement (see #750)."
        )

    # And it must still be a single UPDATE (one set-based pass), not five.
    assert src.count("UPDATE sessions") == 1, (
        "0066 backfill should be exactly one set-based UPDATE; found "
        f"{src.count('UPDATE sessions')}"
    )


# ─── bounded-scan smoke: many events, ONE session ────────────────────────────


@needs_docker
def test_backfill_bounded_on_a_large_session(postgres: object) -> None:
    """Seed many events for a SINGLE session and assert the backfill stays
    fast. A correlated per-session backfill would re-scan all N events for that
    one session (and again per other session); the set-based form scans the
    table a constant number of times. We don't need a 320k-event repro to make
    the point — a few thousand events backfill in well under a second
    set-based, whereas the correlated form's cost is visibly super-linear.

    The budget (60s) is deliberately generous: it brackets the WHOLE
    ``uv run alembic upgrade`` invocation — process spawn, env bootstrap, and
    the full migration ladder from 0065 — not just the 0066 statement, so it
    stays robust on cold CI while still failing a true minutes-long hang.
    """
    db_url = _alembic_url(postgres)

    result = _run_alembic(["upgrade", "0065"], db_url)
    assert result.returncode == 0, f"upgrade 0065 failed:\n{result.stderr}\n{result.stdout}"

    n_events = 5000

    async def seed() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute(_ACCOUNT_SQL)
            await conn.execute(_AGENT_SQL)
            await conn.execute(_ENV_SQL)
            await _seed_session_row(conn, "s_big")
            # Alternate user / assistant-with-tool / tool-result so every
            # CTE has real work and ``open_tool_call_count`` nets to a known
            # value. Each triple: user, assistant(1 tool_call), tool result.
            rows: list[tuple[str, str, int, str, str, str | None, str]] = []
            seq = 0
            for i in range(n_events // 3):
                seq += 1
                rows.append(
                    (
                        f"s_big-{seq}",
                        "s_big",
                        seq,
                        "message",
                        json.dumps({"role": "user", "content": f"u{i}"}),
                        "user",
                        "acc_root",
                    )
                )
                seq += 1
                rows.append(
                    (
                        f"s_big-{seq}",
                        "s_big",
                        seq,
                        "message",
                        json.dumps(
                            {
                                "role": "assistant",
                                "reacting_to": seq - 1,
                                "tool_calls": [{"id": f"tc_{i}"}],
                            }
                        ),
                        "assistant",
                        "acc_root",
                    )
                )
                seq += 1
                rows.append(
                    (
                        f"s_big-{seq}",
                        "s_big",
                        seq,
                        "message",
                        json.dumps({"role": "tool", "tool_call_id": f"tc_{i}", "content": "ok"}),
                        "tool",
                        "acc_root",
                    )
                )
            await conn.copy_records_to_table(
                "events",
                records=rows,
                columns=["id", "session_id", "seq", "kind", "data", "role", "account_id"],
            )
        finally:
            await conn.close()

    asyncio.run(seed())

    start = time.monotonic()
    result = _run_alembic(["upgrade", "head"], db_url)
    elapsed = time.monotonic() - start
    assert result.returncode == 0, f"upgrade head failed:\n{result.stderr}\n{result.stdout}"

    # The whole alembic invocation (process spawn + 0066) finishes well under
    # this; a correlated re-scan of a large session is what blows it.
    assert elapsed < 60.0, f"0066 backfill took {elapsed:.1f}s — suspect a re-scan"

    async def check() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            s = await _scalars(conn, "s_big")
            # Every tool_call was resolved by its paired result -> 0 open.
            assert s["open_tool_call_count"] == 0, s
            # Sanity: the watermarks advanced to near the tail.
            assert s["last_stimulus_seq"] > 0
            assert s["last_reacted_seq"] > 0
            assert s["last_user_seq"] > 0
        finally:
            await conn.close()

    asyncio.run(check())
