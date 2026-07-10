from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness.invariant_sweep import INVARIANT_CANDIDATES_SQL, sweep_session_invariants


def test_invariant_sql_preserves_four_clause_contract() -> None:
    sql = " ".join(INVARIANT_CANDIDATES_SQL.split())
    assert "e.data->'caller'->>'id' = s.parent_run_id" in sql
    assert "e.data->'caller'->>'kind' = 'run'" in sql
    assert "error'->>'kind' = ANY($4::text[])" in sql
    assert "caller'->>'kind'='api'" in sql
    assert "deadline_seconds" in sql
    assert "NOT EXISTS" in sql
    assert "archive_when_idle" in sql


@pytest.mark.asyncio
async def test_dry_run_logs_all_candidates_without_actuation() -> None:
    conn = AsyncMock()
    conn.fetch.return_value = [
        {"session_id": "s1", "account_id": "a", "clause": "revoked_lease"},
        {"session_id": "s2", "account_id": "a", "clause": "abandoned_api_lease"},
    ]
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=conn)
    acquire.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquire

    with patch(
        "aios.harness.invariant_sweep.sessions_service.archive_session", new=AsyncMock()
    ) as archive:
        result = await sweep_session_invariants(pool, dry_run=True)

    assert result.selected == 2
    assert result.archived == 0
    archive.assert_not_awaited()


@pytest.mark.asyncio
async def test_armed_pass_archives_reclaiming_clauses_but_api_stays_log_only() -> None:
    conn = AsyncMock()
    conn.fetch.return_value = [
        {"session_id": "s1", "account_id": "a", "clause": "lease_ceiling"},
        {"session_id": "s2", "account_id": "a", "clause": "abandoned_api_lease"},
    ]
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=conn)
    acquire.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquire

    with patch(
        "aios.harness.invariant_sweep.sessions_service.archive_session", new=AsyncMock()
    ) as archive:
        result = await sweep_session_invariants(pool, dry_run=False)

    archive.assert_awaited_once_with(pool, "s1", account_id="a", idempotent=True)
    assert result.archived == 1
