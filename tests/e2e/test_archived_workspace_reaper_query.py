"""E2E: the archived-session workspace reaper's candidate SQL against real PG (#40).

The unit tests inject the query result; this proves the SQL itself is valid and
selects the right rows against a real Postgres — the ``make_interval`` age floor,
the ``archived_at IS NOT NULL`` filter, and the composed
``session_active_predicate`` (whose columns live on ``sessions``). Mirrors the
existing ``unscoped_*`` reaper-query conventions.

Asserts:
* an archived, aged, not-active session IS returned (reap-eligible);
* a NON-archived (live) session is NOT returned (archived-only);
* a just-archived session is NOT returned while the age floor exceeds its age
  (min-age floor on DB archive time);
* an archived session with an unreacted stimulus (active) is NOT returned
  (the not-active defense-in-depth keep-set).
"""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

from aios.db import queries

pytestmark = pytest.mark.docker


async def _seed_fk_targets(pool: asyncpg.Pool[Any]) -> tuple[str, str]:
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO agents (id, name, model, account_id) "
            "VALUES ('agt_wsr', 'wsr', 'openrouter/x', 'acc_test_stub') "
            "ON CONFLICT (id) DO NOTHING"
        )
        await conn.execute(
            "INSERT INTO environments (id, name, account_id) "
            "VALUES ('env_wsr', 'env_wsr', 'acc_test_stub') "
            "ON CONFLICT (id) DO NOTHING"
        )
    return "agt_wsr", "env_wsr"


async def _insert_session(
    conn: asyncpg.Connection[Any],
    *,
    sid: str,
    archived_age_seconds: float | None,
    active: bool = False,
) -> None:
    """Insert a session row directly, controlling archived_at and active-ness.

    ``archived_age_seconds=None`` ⇒ live (archived_at NULL). A positive value ⇒
    ``archived_at = now() - that``. ``active=True`` sets an unreacted stimulus so
    the active predicate is true.
    """
    await conn.execute(
        """
        INSERT INTO sessions (
            id, agent_id, environment_id, title, metadata,
            workspace_volume_path, account_id, archived_at,
            last_stimulus_seq, last_reacted_seq, open_tool_call_count,
            last_error_seq, last_user_seq
        )
        VALUES (
            $1, 'agt_wsr', 'env_wsr', NULL, '{}'::jsonb,
            $2, 'acc_test_stub',
            CASE WHEN $3::double precision IS NULL THEN NULL
                 ELSE now() - make_interval(secs => $3::double precision) END,
            $4, 0, 0, 0, 0
        )
        ON CONFLICT (id) DO NOTHING
        """,
        sid,
        f"/var/lib/aios/workspaces/acc_test_stub/{sid}",
        archived_age_seconds,
        1 if active else 0,  # last_stimulus_seq > last_reacted_seq(0) ⇒ active
    )


async def test_candidate_query_selects_only_archived_aged_inactive(
    pool: asyncpg.Pool[Any],
) -> None:
    await _seed_fk_targets(pool)
    async with pool.acquire() as conn:
        # archived 2h ago, not active ⇒ eligible under a 1h floor.
        await _insert_session(conn, sid="ses_wsr_reap", archived_age_seconds=7200)
        # live (never archived) ⇒ excluded.
        await _insert_session(conn, sid="ses_wsr_live", archived_age_seconds=None)
        # archived only 60s ago ⇒ excluded by the 1h floor.
        await _insert_session(conn, sid="ses_wsr_fresh", archived_age_seconds=60)
        # archived 2h ago BUT active (unreacted stimulus) ⇒ excluded by keep-set.
        await _insert_session(conn, sid="ses_wsr_active", archived_age_seconds=7200, active=True)

        rows = await queries.unscoped_reapable_archived_workspaces(
            conn, min_archived_age_seconds=3600
        )

    ids = {r["id"] for r in rows}
    assert "ses_wsr_reap" in ids, "archived+aged+inactive must be reap-eligible"
    assert "ses_wsr_live" not in ids, "a live session must never be a candidate"
    assert "ses_wsr_fresh" not in ids, "a just-archived session is held by the age floor"
    assert "ses_wsr_active" not in ids, "an active (unreacted) session is held by the keep-set"

    # The returned row carries exactly the fields the reaper confinement needs.
    reap_row = next(r for r in rows if r["id"] == "ses_wsr_reap")
    assert reap_row["account_id"] == "acc_test_stub"
    assert reap_row["workspace_volume_path"].endswith("/acc_test_stub/ses_wsr_reap")
