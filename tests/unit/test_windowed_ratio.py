"""Unit tests for the per-model ratio application in ``read_windowed_events``.

These tests pin the arithmetic: how ``total * ratio`` drives
``tokens_to_drop``, and how the resulting provider-token boundary is
translated back to local units for the ``cumulative_tokens`` SQL scan.
Full SQL behavior (index usage, real event rows) is covered by the e2e
layer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.db import queries
from aios.harness.window import WindowOmission

_BEGAN_AT = datetime(2026, 2, 19, 9, 0, 0, tzinfo=UTC)

# ``list[Any]`` so equality checks against ``WindowedEvents.events``
# (statically ``list[Event]``) don't trip mypy's comparison-overlap.
_FALLBACK_SENTINEL: list[Any] = ["_fallback_sentinel"]


class _FakeConn:
    """Minimal asyncpg.Connection stand-in.

    ``fetchval`` serves ``_latest_cumulative_tokens`` (total local tokens).
    ``fetchrow`` serves the omission-complement aggregate.  ``fetch`` now
    backs three queries, dispatched on SQL text (issue #1609):

    * the per-class calibration scan (``model_token_class_ratios``) — we
      return ``ratio_rows`` (empty => neutral all-1.0 fallback, the
      backward-compat default these tests exercise);
    * the retained-slate class-mass re-derivation (``_retained_class_mass``)
      — returns ``mass_rows`` (empty => no composition signal, blend folds to
      the coefficient mean = 1.0 under the neutral default);
    * the bounded retained range scan — captured into ``fetch_calls`` so
      tests can assert the computed ``drop_local``.
    """

    def __init__(
        self,
        *,
        total_local: int | None,
        ratio_n: int,
        ratio_mean: float,
        omission_row: dict[str, Any] | None = None,
        ratio_rows: list[Any] | None = None,
        mass_rows: list[Any] | None = None,
    ) -> None:
        self.total_local = total_local
        # Retained for signature back-compat; the per-class fit no longer
        # consumes a single aggregate row.
        self.ratio_row = {"n": ratio_n, "mean_ratio": ratio_mean}
        self.omission_row = omission_row or {
            "began_at": _BEGAN_AT,
            "omitted_messages": 7,
        }
        self.ratio_rows = ratio_rows or []
        self.mass_rows = mass_rows or []
        self.fetch_calls: list[tuple[Any, ...]] = []
        self.omission_calls: list[tuple[Any, ...]] = []

    async def fetchval(self, _sql: str, *_args: Any) -> int | None:
        return self.total_local

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any]:
        if "began_at" in sql:
            self.omission_calls.append(args)
            return self.omission_row
        return self.ratio_row

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        if "model_request_end" in sql:
            return self.ratio_rows
        if "WITH deltas" in sql:
            return self.mass_rows
        self.fetch_calls.append(args)
        return []


@pytest.fixture(autouse=True)
def _stub_read_context_events(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> None:
    """Short-circuit the fallback ``read_windowed_context_events`` so no real
    DB is hit when the code path falls back to 'load everything'.  We sentinel
    its return so tests can detect the fallback.

    Only the fallback paths call it via the package attribute; the retained
    range scan calls it bare (module-global), so this stub leaves that path
    to hit ``_FakeConn.fetch`` — which is what the drop-boundary assertions
    rely on."""
    queries._clear_model_token_ratio_cache()
    monkeypatch.setattr(
        queries,
        "read_windowed_context_events",
        AsyncMock(return_value=_FALLBACK_SENTINEL),
    )


@pytest.mark.asyncio
async def test_no_cumulative_falls_back_to_full_read() -> None:
    account_id = "acc_test_stub"  # PR 3 scaffolding
    conn = _FakeConn(total_local=None, ratio_n=0, ratio_mean=0.0)
    result = await queries.read_windowed_events(
        conn,
        "sess_x",
        window_min=1_000,
        window_max=2_000,
        model="m",
        overhead_local=0,
        account_id=account_id,
    )
    # Fallback short-circuit — ratio never consulted, no omission.
    assert result.events == _FALLBACK_SENTINEL
    assert result.omission is None


@pytest.mark.asyncio
async def test_insufficient_ratio_1_matches_today() -> None:
    """Load-bearing backward-compatibility fence.  Do not delete.

    While model_token_ratio has too few samples (or on a model the
    DB has never seen), it returns 1.0 and ``read_windowed_events`` must behave
    byte-identically to the pre-ratio chunked-snap algorithm — otherwise
    the "gradual rollout" rollout property breaks.  This test pins that.
    """
    account_id = "acc_test_stub"  # PR 3 scaffolding
    conn = _FakeConn(total_local=3_000, ratio_n=4, ratio_mean=0.0)
    # window_min=1000, window_max=2000 -> chunk size 1000.
    # total=3000 -> overshoot 1000 -> drop 1000 (one chunk).
    await queries.read_windowed_events(
        conn,
        "sess_x",
        window_min=1_000,
        window_max=2_000,
        model="m",
        overhead_local=0,
        account_id=account_id,
    )
    assert conn.fetch_calls, "expected bounded range scan to be called"
    # Second positional arg to conn.fetch is the drop value.
    _session_id, drop_local, *_ = conn.fetch_calls[-1]
    assert drop_local == 1_000


@pytest.mark.asyncio
async def test_ratio_above_1_drops_more(monkeypatch: pytest.MonkeyPatch) -> None:
    """A calibrated R_eff > 1 inflates total_effective so the drop boundary
    crosses a snap, and the returned drop_local ceil-divides back.

    With a uniform calibrated coefficient of 1.5 the blend is R_eff=1.5 for
    any composition, and the calibrated safety margin (x1.3) applies. So:

    total_local=1500, eff factor = 1.5*1.3 = 1.95 -> total_effective≈2925.
    window_min=1000, window_max=2000, chunk=1000.
    overshoot=925 -> drop_effective=1000 (one chunk).
    drop_local = ceil(1000 / 1.95) = 513.
    """
    account_id = "acc_test_stub"
    monkeypatch.setattr(
        queries,
        "model_token_class_ratios",
        AsyncMock(return_value={c: 1.5 for c in ("text", "tool_result", "thinking", "tool_use", "system", "tools")}),
    )
    conn = _FakeConn(total_local=1_500, ratio_n=100, ratio_mean=1.5)
    await queries.read_windowed_events(
        conn,
        "sess_x",
        window_min=1_000,
        window_max=2_000,
        model="m",
        overhead_local=0,
        account_id=account_id,
    )
    _session_id, drop_local, *_ = conn.fetch_calls[-1]
    import math

    assert drop_local == math.ceil(1_000 / (1.5 * 1.3))


@pytest.mark.asyncio
async def test_ratio_below_1_drops_fewer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A calibrated R_eff=0.5 deflates total_effective below window_max -> no
    drop, even after the x1.3 calibrated safety margin (0.5*1.3=0.65)."""
    account_id = "acc_test_stub"
    monkeypatch.setattr(
        queries,
        "model_token_class_ratios",
        AsyncMock(return_value={c: 0.5 for c in ("text", "tool_result", "thinking", "tool_use", "system", "tools")}),
    )
    conn = _FakeConn(total_local=3_000, ratio_n=5, ratio_mean=0.5)
    result = await queries.read_windowed_events(
        conn,
        "sess_x",
        window_min=1_000,
        window_max=2_000,
        model="m",
        overhead_local=0,
        account_id=account_id,
    )
    # total_effective = round(3000*0.5*1.3) = 1950 < 2000 -> no drop -> fallback.
    assert result.events == _FALLBACK_SENTINEL
    assert result.omission is None
    assert not conn.fetch_calls


# ─── omission metadata (#738) ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_windowed_read_reports_omission() -> None:
    """A real drop returns the omitted-span facts, queried against the
    SAME boundary value as the retained range scan (exact complements)."""
    account_id = "acc_test_stub"
    conn = _FakeConn(total_local=3_000, ratio_n=4, ratio_mean=0.0)
    result = await queries.read_windowed_events(
        conn,
        "sess_x",
        window_min=1_000,
        window_max=2_000,
        model="m",
        overhead_local=0,
        account_id=account_id,
    )
    assert result.omission == WindowOmission(began_at=_BEGAN_AT, omitted_messages=7)
    # Complement check: both queries saw the same drop boundary.
    assert conn.omission_calls, "expected the omission aggregate to be queried"
    _sid, retained_drop, *_ = conn.fetch_calls[-1]
    _sid2, omitted_drop, *_ = conn.omission_calls[-1]
    assert retained_drop == omitted_drop


@pytest.mark.asyncio
async def test_empty_complement_reports_no_omission() -> None:
    """drop > 0 but the boundary excludes nothing (oversized first event
    straddling it) -> omission is None, not a zero-count marker."""
    account_id = "acc_test_stub"
    conn = _FakeConn(
        total_local=3_000,
        ratio_n=4,
        ratio_mean=0.0,
        omission_row={"began_at": None, "omitted_messages": 0},
    )
    result = await queries.read_windowed_events(
        conn,
        "sess_x",
        window_min=1_000,
        window_max=2_000,
        model="m",
        overhead_local=0,
        account_id=account_id,
    )
    assert result.omission is None


@pytest.mark.asyncio
async def test_ceil_div_never_overshoots_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-drop effective size must be <= window_max for any ratio > 1."""
    account_id = "acc_test_stub"
    ratio = 1.37
    total_local = 10_000
    window_min, window_max = 3_000, 5_000

    # A uniform calibrated coefficient makes R_eff == ratio for any
    # composition; the windower then applies the x1.3 calibrated margin.
    monkeypatch.setattr(
        queries,
        "model_token_class_ratios",
        AsyncMock(return_value={c: ratio for c in ("text", "tool_result", "thinking", "tool_use", "system", "tools")}),
    )
    eff = ratio * 1.3  # calibrated safety margin
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=total_local)
    conn.fetchrow = AsyncMock(  # serves the omission-complement aggregate
        return_value={"began_at": None, "omitted_messages": 0}
    )

    captured: dict[str, int] = {}

    async def _fetch(sql: str, *args: Any) -> list[Any]:
        # Per-class calibration scan and retained-mass re-derivation also
        # route through fetch — only the bounded retained scan carries the
        # drop boundary as its second positional arg.
        if "model_request_end" in sql or "WITH deltas" in sql:
            return []
        captured["drop_local"] = args[1]
        return []

    conn.fetch = _fetch
    await queries.read_windowed_events(
        conn,
        "sess_x",
        window_min=window_min,
        window_max=window_max,
        model="m",
        overhead_local=0,
        account_id=account_id,
    )
    drop_local = captured["drop_local"]
    remaining_local = total_local - drop_local
    # Post-drop must fit under window_max in the *budgeted* effective space
    # (incl. the safety margin) — the strong form of the cap guarantee.
    remaining_effective = remaining_local * eff
    assert remaining_effective <= window_max, (
        f"post-drop {remaining_effective} exceeds window_max={window_max}"
    )


@pytest.mark.asyncio
async def test_overhead_clamp_never_drops_entire_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The windower must never drop the *entire* window.

    When per-step overhead exceeds ``window_min``, ``events_window_min``
    clamps to its 0 floor (events.py) — losing the floor that normally
    guarantees a non-empty tail (the chunked policy requires
    ``min_tokens >= 1``). The chunked snap then drops in full-window chunks,
    and the asymmetric ``ceil(drop_effective / ratio)`` back-conversion can
    push ``drop`` up to ``total``. The retained scan (``cumulative_tokens >
    drop``) then matches ZERO rows while the omission complement
    (``<= drop``) matches every row — so ``read_windowed_events`` returns an
    empty event list paired with a non-None omission. ``build_messages``
    relies on the inverse invariant and reads ``events[0].created_at`` to
    anchor the omission marker, crashing with IndexError — and since
    ``build_messages`` is pure replay, the session wedges permanently.

    The boundary must keep ``drop < total`` so the most recent event always
    survives (the last event's ``cumulative_tokens == total``; the scan is
    ``> drop``).
    """
    account_id = "acc_test_stub"
    ratio = 1.2
    total_local = 483
    window_min, window_max, overhead_local = 1_000, 2_000, 800
    # A uniform calibrated coef => R_eff=1.2; calibrated safety margin x1.3 =>
    # eff = 1.56. overhead_effective = round(800*1.56) = 1248 ->
    # events_window_max = 752, events_window_min = max(0, 1000-1248) = 0.
    # total_effective = round(483*1.56) = 753 > 752 -> drop_effective =
    # tokens_to_drop(753, min=0, max=752) = 752 -> ceil(752/1.56) = 483 ==
    # total: without the clamp the snap would drop the *entire* window.
    monkeypatch.setattr(
        queries,
        "model_token_class_ratios",
        AsyncMock(return_value={c: ratio for c in ("text", "tool_result", "thinking", "tool_use", "system", "tools")}),
    )
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=total_local)
    conn.fetchrow = AsyncMock(  # omission complement matches every row here
        return_value={"began_at": _BEGAN_AT, "omitted_messages": 7}
    )
    captured: dict[str, int] = {}

    async def _fetch(sql: str, *args: Any) -> list[Any]:
        if "model_request_end" in sql or "WITH deltas" in sql:
            return []
        captured["drop_local"] = args[1]
        return []

    conn.fetch = _fetch
    await queries.read_windowed_events(
        conn,
        "sess_x",
        window_min=window_min,
        window_max=window_max,
        model="m",
        overhead_local=overhead_local,
        account_id=account_id,
    )
    drop_local = captured["drop_local"]
    assert drop_local < total_local, (
        f"drop_local={drop_local} >= total={total_local} drops the entire "
        "window, leaving an empty retained scan paired with a non-None "
        "omission — which crashes build_messages on events[0]"
    )
