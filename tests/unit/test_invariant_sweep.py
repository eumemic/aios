from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness.invariant_sweep import (
    _DEADLINE_SECONDS_REGEX,
    INVARIANT_CANDIDATES_SQL,
    sweep_session_invariants,
)


def test_invariant_sql_preserves_four_clause_contract() -> None:
    sql = " ".join(INVARIANT_CANDIDATES_SQL.split())
    assert "e.data->'caller'->>'id' = s.parent_run_id" in sql
    assert "e.data->'caller'->>'kind' = 'run'" in sql
    assert "error'->>'kind' = ANY($4::text[])" in sql
    assert "caller'->>'kind'='api'" in sql
    assert "deadline_seconds" in sql
    assert "NOT EXISTS" in sql
    assert "archive_when_idle" in sql
    # The single-session re-validation pin ($7) the read/actuation TOCTOU fix runs
    # under the session row lock.
    assert "$7::text IS NULL OR s.id = $7::text" in sql


def test_deadline_regex_is_bound_as_param_not_embedded_literal() -> None:
    """Finding #3: the fractional-deadline regex is a BOUND PARAM ($6), so its bytes reach
    PostgreSQL's regex engine with no SQL-string-literal escaping layer — no over-escaped
    ``\\\\.`` that would silently match a literal backslash and drop ``1.5``. The param
    itself carries a SINGLE escape (one backslash) for the literal decimal point."""
    sql = " ".join(INVARIANT_CANDIDATES_SQL.split())
    assert "(o.edge->>'deadline_seconds') ~ $6" in sql
    # The old embedded, over-escapable literal must be gone.
    assert "deadline_seconds') ~ '" not in sql
    # Exactly one backslash before the dot — a literal decimal point, not backslash+any.
    assert _DEADLINE_SECONDS_REGEX == r"^[0-9]+(\.[0-9]+)?$"
    assert _DEADLINE_SECONDS_REGEX.count("\\") == 1


def _mock_pool_returning(rows: list[dict[str, str]]) -> MagicMock:
    conn = AsyncMock()
    conn.fetch.return_value = rows
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=conn)
    acquire.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquire
    return pool


@pytest.mark.asyncio
async def test_dry_run_logs_all_candidates_without_actuation() -> None:
    pool = _mock_pool_returning(
        [
            {"session_id": "s1", "account_id": "a", "clause": "revoked_lease"},
            {"session_id": "s2", "account_id": "a", "clause": "abandoned_api_lease"},
        ]
    )
    with patch(
        "aios.harness.invariant_sweep._revalidate_and_archive", new=AsyncMock()
    ) as revalidate:
        result = await sweep_session_invariants(pool, dry_run=True)

    assert result.selected == 2
    assert result.archived == 0
    revalidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_armed_pass_actuates_reclaiming_clauses_but_api_stays_log_only() -> None:
    pool = _mock_pool_returning(
        [
            {"session_id": "s1", "account_id": "a", "clause": "lease_ceiling"},
            {"session_id": "s2", "account_id": "a", "clause": "abandoned_api_lease"},
        ]
    )
    # The armed reclaim goes through the under-lock re-validation seam, never the raw
    # unconditional archive; the measurement-only api clause is skipped before it.
    with patch(
        "aios.harness.invariant_sweep._revalidate_and_archive",
        new=AsyncMock(return_value=True),
    ) as revalidate:
        result = await sweep_session_invariants(pool, dry_run=False)

    revalidate.assert_awaited_once()
    assert revalidate.await_args is not None
    args = revalidate.await_args.args
    assert args[1] == "s1" and args[2] == "a"  # (pool, session_id, account_id)
    assert result.archived == 1


@pytest.mark.asyncio
async def test_armed_pass_skips_row_whose_clause_no_longer_holds() -> None:
    """A candidate that lost its clause between selection and actuation (revalidation
    returns False) is NOT counted as archived — the TOCTOU skip path."""
    pool = _mock_pool_returning([{"session_id": "s1", "account_id": "a", "clause": "expired_idle"}])
    with patch(
        "aios.harness.invariant_sweep._revalidate_and_archive",
        new=AsyncMock(return_value=False),
    ) as revalidate:
        result = await sweep_session_invariants(pool, dry_run=False)

    revalidate.assert_awaited_once()
    assert result.selected == 1
    assert result.archived == 0
