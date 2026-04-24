"""Unit tests for :func:`aios.db.queries.model_token_ratio`.

The SQL itself — partial-index coverage, JSON extraction, is_error
filter, model-string partitioning, and lifetime aggregation — is exercised
against a real Postgres in
``tests/e2e/test_model_token_ratio_sql.py``.  These tests pin the
Python-side contract only: insufficient-sample fallback, AVG/STDDEV
arithmetic, and standard-error bucketing.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.db.queries import _clear_model_token_ratio_cache, model_token_ratio


@pytest.fixture(autouse=True)
def _clear_ratio_cache() -> None:
    _clear_model_token_ratio_cache()


def _mock_conn(*, n: int, mean_ratio: float) -> MagicMock:
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value={"n": n, "mean_ratio": mean_ratio})
    return conn


class TestModelTokenRatio:
    @pytest.mark.asyncio
    async def test_below_min_samples_returns_1(self) -> None:
        conn = _mock_conn(n=4, mean_ratio=1.5)
        assert await model_token_ratio(conn, "model-x") == 1.0

    @pytest.mark.asyncio
    async def test_min_samples_quantizes_by_sigma_prior_bucket(self) -> None:
        # bucket = k_bucket * sigma_prior / sqrt(n) = 2 * 0.02 / sqrt(5)
        # ≈ 0.01789; 1.5 rounds to 84 * 0.01789 ≈ 1.5028.
        conn = _mock_conn(n=5, mean_ratio=1.5)
        assert await model_token_ratio(conn, "model-x") == pytest.approx(1.5028, abs=0.001)

    @pytest.mark.asyncio
    async def test_bucket_shrinks_with_sqrt_n(self) -> None:
        # n=100 → bucket = 0.004; 1.44 rounds to itself.
        conn = _mock_conn(n=100, mean_ratio=1.44)
        assert await model_token_ratio(conn, "model-x") == pytest.approx(1.44, abs=0.002)

    @pytest.mark.asyncio
    async def test_k_bucket_scales_bucket_width(self) -> None:
        # k_bucket=0.5 halves the bucket to 0.001 at n=100.
        conn = _mock_conn(n=100, mean_ratio=1.4445)
        ratio = await model_token_ratio(conn, "model-x", k_bucket=0.5)
        assert ratio == pytest.approx(1.444, abs=0.0005)

    @pytest.mark.asyncio
    async def test_passes_model_to_query(self) -> None:
        conn = _mock_conn(n=0, mean_ratio=0.0)
        await model_token_ratio(conn, "anthropic/claude-sonnet-4-6")
        args = conn.fetchrow.await_args
        assert args is not None
        assert args.args[1] == "anthropic/claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_sql_shape(self) -> None:
        """SQL invariants:

        * Does not sum cache_* breakdown fields on top of ``input_tokens``
          (regression for the pre-#163 double-count).
        * Does not select ``STDDEV`` — the bucket is derived from a fixed
          ``sigma_prior``, not observed spread (#170 / #171).
        * Lifetime aggregate: no ``LIMIT`` window.
        * Uses the unweighted ``AVG`` estimator.
        """
        conn = _mock_conn(n=0, mean_ratio=0.0)
        await model_token_ratio(conn, "model-x")
        sql = conn.fetchrow.await_args.args[0]
        assert "cache_read_input_tokens" not in sql
        assert "cache_creation_input_tokens" not in sql
        assert "STDDEV" not in sql
        assert "LIMIT" not in sql
        assert "AVG(" in sql
        assert "SUM(it)" not in sql

    @pytest.mark.asyncio
    async def test_invalid_bucket_rejected(self) -> None:
        conn = _mock_conn(n=5, mean_ratio=1.5)
        with pytest.raises(ValueError, match="k_bucket must be positive"):
            await model_token_ratio(conn, "model-x", k_bucket=0.0)

    @pytest.mark.asyncio
    async def test_min_ratio_clamp(self) -> None:
        conn = _mock_conn(n=5, mean_ratio=0.0001)
        assert await model_token_ratio(conn, "model-x") == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_calibrated_ratio_is_cached(self) -> None:
        conn = _mock_conn(n=5, mean_ratio=1.5)
        first = await model_token_ratio(conn, "model-cache")
        conn.fetchrow = AsyncMock(return_value={"n": 5, "mean_ratio": 2.0})
        second = await model_token_ratio(conn, "model-cache")
        assert second == first, f"expected cached reuse, got {second} vs first={first}"
        conn.fetchrow.assert_not_awaited()
