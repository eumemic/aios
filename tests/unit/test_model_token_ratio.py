"""Unit tests for :func:`aios.db.queries.model_token_ratio`.

The SQL itself — partial-index coverage, JSON extraction, is_error
filter, model-string partitioning, and LIMIT-based sliding window — is
exercised against a real Postgres in
``tests/e2e/test_model_token_ratio_sql.py``.  These tests pin the
Python-side contract only: below-N fallback and the SUM/SUM arithmetic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.db.queries import model_token_ratio


def _mock_conn(*, k: int, total_actual: int, total_local: int) -> MagicMock:
    conn = MagicMock()
    conn.fetchrow = AsyncMock(
        return_value={
            "k": k,
            "total_actual": total_actual,
            "total_local": total_local,
        }
    )
    return conn


class TestModelTokenRatio:
    @pytest.mark.asyncio
    async def test_below_n_returns_1(self) -> None:
        conn = _mock_conn(k=29, total_actual=15_000, total_local=10_000)
        ratio = await model_token_ratio(conn, "model-x", n=30)
        assert ratio == 1.0

    @pytest.mark.asyncio
    async def test_at_n_returns_sum_ratio(self) -> None:
        # 30 spans, actual sums to 1.5x the local sum -> R = 1.5 exactly.
        conn = _mock_conn(k=30, total_actual=1_500, total_local=1_000)
        ratio = await model_token_ratio(conn, "model-x", n=30)
        assert ratio == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_above_n_returns_sum_ratio(self) -> None:
        # LIMIT in the SQL caps the sample set at n; the helper trusts the
        # row it got back.  Just verify the division when k >= n.
        conn = _mock_conn(k=100, total_actual=11_824, total_local=10_000)
        ratio = await model_token_ratio(conn, "model-x", n=100)
        assert ratio == pytest.approx(1.1824)

    @pytest.mark.asyncio
    async def test_default_n_is_100(self) -> None:
        # With k=99 and no explicit n, the 100-default should return 1.0.
        conn = _mock_conn(k=99, total_actual=1_500, total_local=1_000)
        assert await model_token_ratio(conn, "model-x") == 1.0
        # k=100 activates the default threshold.
        conn2 = _mock_conn(k=100, total_actual=1_500, total_local=1_000)
        assert await model_token_ratio(conn2, "model-x") == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_passes_model_and_n_to_query(self) -> None:
        conn = _mock_conn(k=0, total_actual=0, total_local=0)
        await model_token_ratio(conn, "anthropic/claude-sonnet-4-6", n=200)
        args = conn.fetchrow.await_args
        assert args is not None
        # Positional args after the SQL string: (model, n)
        assert args.args[1] == "anthropic/claude-sonnet-4-6"
        assert args.args[2] == 200
