"""Unit tests for the per-model ratio application in ``read_windowed_events``.

These tests pin the arithmetic: how ``total * ratio`` drives
``tokens_to_drop``, and how the resulting provider-token boundary is
translated back to local units for the ``cumulative_tokens`` SQL scan.
Full SQL behavior (index usage, real event rows) is covered by the e2e
layer.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.db import queries


class _FakeConn:
    """Minimal asyncpg.Connection stand-in.

    ``fetchval`` serves ``_latest_cumulative_tokens`` (total local tokens).
    ``fetchrow`` serves ``model_token_ratio`` (the lifetime calibration
    aggregate row).  ``fetch`` captures the bounded range scan's args so
    tests can assert the computed ``drop_local``.
    """

    def __init__(
        self,
        *,
        total_local: int | None,
        ratio_n: int,
        ratio_mean: float,
    ) -> None:
        self.total_local = total_local
        self.ratio_row = {"n": ratio_n, "mean_ratio": ratio_mean}
        self.fetch_calls: list[tuple[Any, ...]] = []

    async def fetchval(self, _sql: str, *_args: Any) -> int | None:
        return self.total_local

    async def fetchrow(self, _sql: str, *_args: Any) -> dict[str, Any]:
        return self.ratio_row

    async def fetch(self, _sql: str, *args: Any) -> list[Any]:
        self.fetch_calls.append(args)
        return []


@pytest.fixture(autouse=True)
def _stub_read_message_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short-circuit ``read_message_events`` so no real DB is hit when the
    code path falls back to 'load everything'.  We sentinel its return so
    tests can detect the fallback."""
    queries._clear_model_token_ratio_cache()
    monkeypatch.setattr(
        queries,
        "read_message_events",
        AsyncMock(return_value=["_fallback_sentinel"]),
    )


@pytest.mark.asyncio
async def test_no_cumulative_falls_back_to_full_read() -> None:
    conn = _FakeConn(total_local=None, ratio_n=0, ratio_mean=0.0)
    result = await queries.read_windowed_events(
        conn, "sess_x", window_min=1_000, window_max=2_000, model="m", overhead_local=0
    )
    # Fallback short-circuit — ratio never consulted.
    assert result == ["_fallback_sentinel"]


@pytest.mark.asyncio
async def test_insufficient_ratio_1_matches_today() -> None:
    """Load-bearing backward-compatibility fence.  Do not delete.

    While model_token_ratio has too few samples (or on a model the
    DB has never seen), it returns 1.0 and ``read_windowed_events`` must behave
    byte-identically to the pre-ratio chunked-snap algorithm — otherwise
    the "gradual rollout" rollout property breaks.  This test pins that.
    """
    conn = _FakeConn(total_local=3_000, ratio_n=4, ratio_mean=0.0)
    # window_min=1000, window_max=2000 → chunk size 1000.
    # total=3000 → overshoot 1000 → drop 1000 (one chunk).
    await queries.read_windowed_events(
        conn, "sess_x", window_min=1_000, window_max=2_000, model="m", overhead_local=0
    )
    assert conn.fetch_calls, "expected bounded range scan to be called"
    # Second positional arg to conn.fetch is the drop value.
    _session_id, drop_local = conn.fetch_calls[-1]
    assert drop_local == 1_000


@pytest.mark.asyncio
async def test_ratio_above_1_drops_more() -> None:
    """ratio=1.5 inflates total_effective so the drop boundary crosses a
    snap, and the returned drop_local ceil-divides back.

    total_local=1500, ratio≈1.5 → total_effective≈2250.
    window_min=1000, window_max=2000, chunk=1000.
    overshoot=250 → drop_effective=1000.
    drop_local = ceil(1000 / ratio) = 667.

    Uses ratio_n=100 so the sigma_prior bucket (0.004 at n=100) is tight
    enough to quantize raw=1.5 to exactly 1.5.
    """
    conn = _FakeConn(total_local=1_500, ratio_n=100, ratio_mean=1.5)
    await queries.read_windowed_events(
        conn, "sess_x", window_min=1_000, window_max=2_000, model="m", overhead_local=0
    )
    _session_id, drop_local = conn.fetch_calls[-1]
    assert drop_local == 667


@pytest.mark.asyncio
async def test_ratio_below_1_drops_fewer() -> None:
    """ratio=0.5 deflates total_effective below window_max → no drop."""
    conn = _FakeConn(total_local=3_000, ratio_n=5, ratio_mean=0.5)
    result = await queries.read_windowed_events(
        conn, "sess_x", window_min=1_000, window_max=2_000, model="m", overhead_local=0
    )
    # total_effective = 1500 < 2000 → drop_effective = 0 → fallback.
    assert result == ["_fallback_sentinel"]
    assert not conn.fetch_calls


@pytest.mark.asyncio
async def test_ceil_div_never_overshoots_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-drop effective size must be <= window_max for any ratio > 1."""
    ratio = 1.37
    total_local = 10_000
    window_min, window_max = 3_000, 5_000

    monkeypatch.setattr(
        queries,
        "model_token_ratio",
        AsyncMock(return_value=ratio),
    )
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=total_local)

    captured: dict[str, int] = {}

    async def _fetch(_sql: str, *args: Any) -> list[Any]:
        captured["drop_local"] = args[1]
        return []

    conn.fetch = _fetch
    await queries.read_windowed_events(
        conn, "sess_x", window_min=window_min, window_max=window_max, model="m", overhead_local=0
    )
    drop_local = captured["drop_local"]
    remaining_local = total_local - drop_local
    remaining_effective = remaining_local * ratio
    assert remaining_effective <= window_max, (
        f"post-drop {remaining_effective} exceeds window_max={window_max}"
    )
