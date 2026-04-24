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


def _mock_conn(
    *,
    n: int,
    mean_ratio: float,
    stddev_ratio: float,
) -> MagicMock:
    conn = MagicMock()
    conn.fetchrow = AsyncMock(
        return_value={
            "n": n,
            "mean_ratio": mean_ratio,
            "stddev_ratio": stddev_ratio,
        }
    )
    return conn


class TestModelTokenRatio:
    @pytest.mark.asyncio
    async def test_below_min_samples_returns_1(self) -> None:
        conn = _mock_conn(n=4, mean_ratio=1.5, stddev_ratio=0.0)
        ratio = await model_token_ratio(conn, "model-x")
        assert ratio == 1.0

    @pytest.mark.asyncio
    async def test_min_samples_returns_mean_ratio_quantized_to_floor_bucket(self) -> None:
        # Zero observed spread uses the 0.001 floor bucket.  The raw ratio
        # 1.5004 rounds to the nearest floor bucket: 1.500.
        conn = _mock_conn(n=5, mean_ratio=1.5004, stddev_ratio=0.0)
        ratio = await model_token_ratio(conn, "model-x")
        assert ratio == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_standard_error_bucket_controls_quantization(self) -> None:
        # raw=1.44, stddev=0.15, n=9 -> SE=0.05, bucket=0.1.
        conn = _mock_conn(n=9, mean_ratio=1.44, stddev_ratio=0.15)
        ratio = await model_token_ratio(conn, "model-x")
        assert ratio == pytest.approx(1.4)

    @pytest.mark.asyncio
    async def test_k_bucket_tunes_standard_error_width(self) -> None:
        # raw=1.44, stddev=0.15, n=9.  k=1 halves the bucket to 0.05,
        # so the same aggregate rounds closer to the raw value.
        conn = _mock_conn(n=9, mean_ratio=1.44, stddev_ratio=0.15)
        ratio = await model_token_ratio(conn, "model-x", k_bucket=1.0)
        assert ratio == pytest.approx(1.45)

    @pytest.mark.asyncio
    async def test_passes_model_to_query(self) -> None:
        conn = _mock_conn(n=0, mean_ratio=0.0, stddev_ratio=0.0)
        await model_token_ratio(conn, "anthropic/claude-sonnet-4-6")
        args = conn.fetchrow.await_args
        assert args is not None
        # Positional args after the SQL string: (model,)
        assert args.args[1] == "anthropic/claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_sql_does_not_sum_cache_fields(self) -> None:
        """Regression: summing ``cache_*`` breakdown fields with
        ``input_tokens`` double-counts and roughly doubles R on
        cache-hot workloads."""
        conn = _mock_conn(n=0, mean_ratio=0.0, stddev_ratio=0.0)
        await model_token_ratio(conn, "model-x")
        sql = conn.fetchrow.await_args.args[0]
        assert "cache_read_input_tokens" not in sql
        assert "cache_creation_input_tokens" not in sql
        assert "LIMIT" not in sql
        assert "AVG(" in sql
        assert "SUM(it)" not in sql

    @pytest.mark.asyncio
    async def test_invalid_bucket_rejected(self) -> None:
        conn = _mock_conn(n=5, mean_ratio=1.5, stddev_ratio=0.0)
        with pytest.raises(ValueError, match="k_bucket must be positive"):
            await model_token_ratio(conn, "model-x", k_bucket=0.0)

    @pytest.mark.asyncio
    async def test_min_ratio_clamp(self) -> None:
        conn = _mock_conn(n=5, mean_ratio=0.0001, stddev_ratio=0.0)
        assert await model_token_ratio(conn, "model-x") == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_two_high_spread_samples_stay_neutral(self) -> None:
        conn = _mock_conn(n=2, mean_ratio=1.5, stddev_ratio=10.0)
        assert await model_token_ratio(conn, "model-x") == 1.0

    @pytest.mark.asyncio
    async def test_calibrated_ratio_is_cached(self) -> None:
        conn = _mock_conn(n=5, mean_ratio=1.5, stddev_ratio=0.0)
        assert await model_token_ratio(conn, "model-cache") == pytest.approx(1.5)

        conn.fetchrow = AsyncMock(return_value={"n": 5, "mean_ratio": 2.0, "stddev_ratio": 0.0})
        assert await model_token_ratio(conn, "model-cache") == pytest.approx(1.5)
        conn.fetchrow.assert_not_awaited()
