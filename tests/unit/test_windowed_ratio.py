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


class _Unset:
    """Sentinel distinguishing "arg omitted" from an explicit ``None``.

    ``omission_row`` uses ``None`` to mean "no boundary row" (the drop
    excluded nothing), so a plain ``None`` default couldn't tell "caller
    wants the no-boundary case" apart from "caller didn't specify one and
    wants the present-row default". This sentinel makes that distinction
    explicit.
    """


_UNSET = _Unset()


class _FakeConn:
    """Minimal asyncpg.Connection stand-in.

    Since the per-turn read path became O(1) (issue #1657) it reads from
    stored running-counter columns instead of re-aggregating the slate, so
    the stubs dispatch on SQL text:

    * ``fetchval`` serves ``_latest_cumulative_tokens`` (``cumulative_tokens``
      total local), and the O(1) ``began_at`` seek (``created_at`` of the
      first message). Both return scalars; dispatched on SQL text.
    * ``fetchrow`` serves two O(1) index seeks: ``_retained_class_mass``'s
      latest-row per-class cumulative mass (``mass_row``), and the omission
      boundary row (``omission_row``: its ``cumulative_messages`` running count
      + ``created_at``). Dispatched on SQL text.
    * ``fetch`` backs the per-class calibration scan
      (``model_token_class_ratios`` -> ``ratio_rows``) and the bounded retained
      range scan (captured into ``fetch_calls`` for the drop-boundary asserts).
    """

    def __init__(
        self,
        *,
        total_local: int | None,
        ratio_n: int,
        ratio_mean: float,
        omission_row: dict[str, Any] | _Unset | None = _UNSET,
        ratio_rows: list[Any] | None = None,
        mass_row: dict[str, Any] | None = None,
    ) -> None:
        self.total_local = total_local
        # Retained for signature back-compat; the per-class fit no longer
        # consumes a single aggregate row.
        self.ratio_row = {"n": ratio_n, "mean_ratio": ratio_mean}
        # The omission boundary row: a non-None ``cumulative_messages`` means
        # the drop excluded that many user/assistant messages; a None row means
        # the boundary excluded nothing. Omitting the arg entirely defaults to
        # a present row (7 omitted); passing ``omission_row=None`` explicitly
        # selects the no-boundary case.
        self.omission_row: dict[str, Any] | None = (
            {"cumulative_messages": 7, "created_at": _BEGAN_AT}
            if isinstance(omission_row, _Unset)
            else omission_row
        )
        self.ratio_rows = ratio_rows or []
        # The latest-message per-class cumulative mass row (all-None => no
        # composition signal, blend folds to the coefficient mean = 1.0 under
        # the neutral default).
        self.mass_row = mass_row
        self.fetch_calls: list[tuple[Any, ...]] = []
        self.omission_calls: list[tuple[Any, ...]] = []

    async def fetchval(self, sql: str, *args: Any) -> Any:
        if "SELECT created_at FROM events" in sql:
            # ``began_at`` O(1) seek — present iff there is an omitted message.
            row = self.omission_row
            if row is None:
                return None
            return row.get("created_at")
        if "count(*) FILTER" in sql:
            # Pre-backfill fallback count (only hit when the boundary row's
            # ``cumulative_messages`` is None); these tests seed it non-None.
            row = self.omission_row or {}
            return row.get("omitted_messages")
        # ``_latest_cumulative_tokens`` — the session's total local tokens.
        return self.total_local

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        if "cumulative_messages" in sql and "ORDER BY cumulative_tokens DESC" in sql:
            self.omission_calls.append(args)
            return self.omission_row
        if "cumulative_text_mass" in sql:
            # ``_retained_class_mass`` latest-row per-class mass seek.
            return self.mass_row
        return self.ratio_row

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        if "model_request_end" in sql:
            return self.ratio_rows
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

    While calibration has too few samples (or on a model the
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
        AsyncMock(
            return_value={
                c: 1.5 for c in ("text", "tool_result", "thinking", "tool_use", "system", "tools")
            }
        ),
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
async def test_ratio_below_1_never_inflates_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sub-neutral calibration cannot enlarge the retained window."""
    account_id = "acc_test_stub"
    monkeypatch.setattr(
        queries,
        "model_token_class_ratios",
        AsyncMock(
            return_value={
                c: 0.5 for c in ("text", "tool_result", "thinking", "tool_use", "system", "tools")
            }
        ),
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
    # The two-way margin clamps the conversion to neutral, so a low fit can
    # never inflate retention; this 3000-token slate must still be windowed.
    assert result.events != _FALLBACK_SENTINEL
    assert conn.fetch_calls


# ─── omission metadata (#738) ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_windowed_read_reports_omission() -> None:
    """A real drop returns the omitted-span facts, queried against the
    SAME boundary value as the retained range scan (exact complements).

    Post-#1657 the omitted count is the boundary row's ``cumulative_messages``
    running counter (O(1) index seek), not a ``count(*)`` scan — but the
    boundary value it is read at must still equal the retained scan's drop.
    """
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
    # Complement check: both the retained scan and the omission boundary seek
    # saw the same drop boundary.
    assert conn.omission_calls, "expected the omission boundary row to be queried"
    _sid, retained_drop, *_ = conn.fetch_calls[-1]
    _sid2, omitted_drop, *_ = conn.omission_calls[-1]
    assert retained_drop == omitted_drop


@pytest.mark.asyncio
async def test_empty_complement_reports_no_omission() -> None:
    """drop > 0 but the boundary excludes nothing (oversized first event
    straddling it) -> omission is None, not a zero-count marker.

    A None boundary row means no message satisfies ``cumulative_tokens <=
    drop`` — the O(1) seek returns nothing, so there is no omission.
    """
    account_id = "acc_test_stub"
    # ``omission_row=None`` explicitly selects the no-boundary case (omitting
    # the arg would default to a present row).
    conn = _FakeConn(
        total_local=3_000,
        ratio_n=4,
        ratio_mean=0.0,
        omission_row=None,
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
async def test_omission_prebackfill_falls_back_to_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A boundary row that predates migration 0127 carries a NULL
    ``cumulative_messages``; the omission count then falls back to the
    role-filtered ``count(*)`` over the omitted prefix — same value, bounded
    by the ``cumulative_tokens <= drop`` index cond (issue #1657)."""
    account_id = "acc_test_stub"
    # Boundary row present (created_at set) but cumulative_messages is None:
    # the un-backfilled tail. The fallback count(*) returns 5.
    conn = _FakeConn(
        total_local=3_000,
        ratio_n=4,
        ratio_mean=0.0,
        omission_row={
            "cumulative_messages": None,
            "created_at": _BEGAN_AT,
            "omitted_messages": 5,
        },
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
    assert result.omission == WindowOmission(began_at=_BEGAN_AT, omitted_messages=5)


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
        AsyncMock(
            return_value={
                c: ratio for c in ("text", "tool_result", "thinking", "tool_use", "system", "tools")
            }
        ),
    )
    eff = ratio * 1.3  # calibrated safety margin
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=total_local)

    # ``read_windowed_events`` issues two distinct fetchrow seeks: the
    # ``_retained_class_mass`` per-class mass row and the omission boundary
    # row. Both return None here — no per-class composition signal (blend
    # folds to the neutral mean) and no omission (oversized first event).
    conn.fetchrow = AsyncMock(return_value=None)

    captured: dict[str, int] = {}

    async def _fetch(sql: str, *args: Any) -> list[Any]:
        # Per-class calibration scan routes through fetch too — only the
        # bounded retained scan carries the drop boundary as its second
        # positional arg.
        if "model_request_end" in sql:
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
    """The windower must never drop the *entire* window — the load-bearing
    invariant behind the whole snap path.

    An empty retained scan (``cumulative_tokens > drop`` matching ZERO rows)
    paired with a non-None omission complement (``<= drop`` matching every
    row) crashes ``build_messages``, which reads ``events[0].created_at`` to
    anchor the omission marker and relies on the inverse invariant — and
    since ``build_messages`` is pure replay, the session wedges permanently.
    So the boundary must keep ``drop < total``: the most recent event always
    survives (its ``cumulative_tokens == total``; the scan is ``> drop``).

    This geometry originally reached ``drop == total`` through the overhead
    zeroing the floor (``events_window_min = max(0, 1000 - 1248) == 0``) and
    the asymmetric ``ceil(drop_effective / ratio)`` back-conversion rounding
    up. Issue #2289 removed that zeroing — the floor now clamps to 75% of the
    events budget (564 here) rather than to 0 — so the geometry no longer
    drives ``drop`` anywhere near ``total``. The invariant it defends is
    unchanged and still needs a fence: ``window_min=0`` is a *live* caller
    (the adaptive context-overflow retry) and restores exactly the
    full-budget chunk that produced the crash.
    """
    account_id = "acc_test_stub"
    ratio = 1.2
    total_local = 483
    window_min, window_max, overhead_local = 1_000, 2_000, 800
    # A uniform calibrated coef => R_eff=1.2; calibrated safety margin x1.3 =>
    # eff = 1.56. overhead_effective = round(800*1.56) = 1248 ->
    # events_window_max = 752, events_window_min = min(1000, int(752*0.75))
    # = 564 (clamped). total_effective = round(483*1.56) = 753 > 752 ->
    # chunk = 188, overshoot = 1 -> drop_effective = 188 ->
    # ceil(188/1.56) = 121, comfortably below total.
    monkeypatch.setattr(
        queries,
        "model_token_class_ratios",
        AsyncMock(
            return_value={
                c: ratio for c in ("text", "tool_result", "thinking", "tool_use", "system", "tools")
            }
        ),
    )
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=total_local)

    # Two distinct fetchrow seeks: the ``_retained_class_mass`` per-class mass
    # row (no composition signal here -> None, blend folds to the neutral
    # mean) and the omission boundary row present here (matches every row);
    # its ``cumulative_messages`` seek returns a count.
    async def _fetchrow(sql: str, *args: Any) -> dict[str, Any] | None:
        if "cumulative_text_mass" in sql:
            return None
        return {"cumulative_messages": 7, "created_at": _BEGAN_AT}

    conn.fetchrow = _fetchrow
    captured: dict[str, int] = {}

    async def _fetch(sql: str, *args: Any) -> list[Any]:
        if "model_request_end" in sql:
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


# --- issue #2289: ``window_min`` floors RETAINED HISTORY, not the whole prompt


def _uniform_ratio(monkeypatch: pytest.MonkeyPatch, coef: float = 1.5) -> float:
    """Force a uniform calibrated coefficient and return the effective factor.

    A uniform coef blends to ``R_eff == coef`` for ANY composition, and being
    != 1.0 it counts as calibrated, so the x1.3 safety margin applies. The
    returned value is what the windower multiplies both the overhead and the
    session total by.
    """
    monkeypatch.setattr(
        queries,
        "model_token_class_ratios",
        AsyncMock(
            return_value={
                c: coef for c in ("text", "tool_result", "thinking", "tool_use", "system", "tools")
            }
        ),
    )
    return coef * 1.3


@pytest.mark.asyncio
async def test_incident_geometry_retains_history_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the 2026-08-28 self-query incident (#2289).

    A fat tool prelude (121 MCP schemas, ~99k effective) against the default
    50k/150k band used to drive ``events_window_min`` to 0, so every snap took
    a full-budget chunk and left the agent with the single event the
    ``drop < total`` guard preserves — its own "history has scrolled out of
    view, search first" notice, which then commanded a search loop.

    With the floor no longer reduced by overhead, the retained slate must stay
    at or above the (clamped) floor, and the snap chunk must stay large enough
    that snaps remain rare.
    """
    account_id = "acc_test_stub"
    eff = _uniform_ratio(monkeypatch)  # 1.95
    total_local, overhead_local = 200_000, 50_000
    window_min, window_max = 50_000, 150_000
    # overhead_effective = round(50_000*1.95) = 97_500 -> events_window_max =
    # 52_500; floor = min(50_000, int(52_500*0.75)) = 39_375 (clamped).
    conn = _FakeConn(total_local=total_local, ratio_n=50, ratio_mean=0.0)
    result = await queries.read_windowed_events(
        conn,
        "sess_x",
        window_min=window_min,
        window_max=window_max,
        model="m",
        overhead_local=overhead_local,
        account_id=account_id,
    )
    floor = result.floor
    assert floor is not None
    assert floor.outcome == "clamped"
    assert floor.events_window_max == 52_500
    assert floor.effective == 39_375

    _session_id, drop_local, *_ = conn.fetch_calls[-1]
    retained_effective = (total_local - drop_local) * eff
    # The heart of the bug: retained history used to collapse below the floor
    # (to a single event). It must now land inside the band.
    assert floor.effective <= retained_effective <= floor.events_window_max, (
        f"retained {retained_effective} outside [{floor.effective}, {floor.events_window_max}]"
    )
    # And the snap chunk stays a meaningful fraction of the budget, so snaps
    # (= full prefix-cache misses) stay rare.
    chunk = floor.events_window_max - floor.effective
    assert chunk >= 0.25 * floor.events_window_max


@pytest.mark.asyncio
async def test_comfortable_band_honors_configured_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overhead small relative to the band: the floor passes through EXACTLY.

    This is the direct fence against re-introducing ``window_min - overhead``.
    """
    account_id = "acc_test_stub"
    _uniform_ratio(monkeypatch)
    conn = _FakeConn(total_local=500_000, ratio_n=50, ratio_mean=0.0)
    result = await queries.read_windowed_events(
        conn,
        "sess_x",
        window_min=10_000,
        window_max=200_000,
        model="m",
        overhead_local=1_000,  # -> 1_950 effective; budget 198_050
        account_id=account_id,
    )
    floor = result.floor
    assert floor is not None
    assert floor.outcome == "honored"
    assert floor.effective == 10_000  # NOT 10_000 - 1_950
    assert floor.overhead_effective == 1_950


@pytest.mark.asyncio
async def test_infeasible_floor_clamps_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A floor the band cannot afford degrades, it does not raise.

    Deliberate exception to the repo's fail-hard stance: the prelude grows
    whenever an upstream MCP server adds tools, so raising here would wedge
    every session in the fleet at wake time. The clamp is reported instead.
    """
    account_id = "acc_test_stub"
    _uniform_ratio(monkeypatch)
    conn = _FakeConn(total_local=500_000, ratio_n=50, ratio_mean=0.0)
    result = await queries.read_windowed_events(
        conn,
        "sess_x",
        window_min=140_000,
        window_max=150_000,
        model="m",
        overhead_local=51_282,  # -> 100_000 effective; budget 50_000
        account_id=account_id,
    )
    floor = result.floor
    assert floor is not None
    assert floor.outcome == "clamped"
    assert floor.events_window_max == 50_000
    assert floor.effective == 37_500  # 75% of the events budget
    assert floor.configured == 140_000


@pytest.mark.asyncio
async def test_overhead_exceeding_window_max_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one hard failure on this path survives the floor rework: with no
    events budget at all there is nothing to degrade to."""
    account_id = "acc_test_stub"
    _uniform_ratio(monkeypatch)
    conn = _FakeConn(total_local=5_000, ratio_n=50, ratio_mean=0.0)
    with pytest.raises(ValueError, match="no budget remains for events"):
        await queries.read_windowed_events(
            conn,
            "sess_x",
            window_min=500,
            window_max=1_000,
            model="m",
            overhead_local=1_000,  # -> 1_950 effective > window_max
            account_id=account_id,
        )


@pytest.mark.asyncio
async def test_adaptive_retry_zero_floor_is_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``window_min=0`` — the adaptive context-overflow retry (``loop.py``) —
    must stay a no-op under the clamp, and must NOT report a clamp.

    Otherwise every overflow recovery would trip the operator alarm.
    """
    account_id = "acc_test_stub"
    _uniform_ratio(monkeypatch)
    conn = _FakeConn(total_local=200_000, ratio_n=50, ratio_mean=0.0)
    result = await queries.read_windowed_events(
        conn,
        "sess_x",
        window_min=0,
        window_max=150_000,
        model="m",
        overhead_local=50_000,
        account_id=account_id,
    )
    floor = result.floor
    assert floor is not None
    assert floor.effective == 0
    assert floor.outcome == "honored"


@pytest.mark.asyncio
async def test_omission_always_pairs_with_non_empty_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_messages`` reads ``events[0]`` unguarded whenever an omission is
    present, so the two can never be (empty, non-None).

    Asserted here on the boundary rather than on the row list, which the fake
    conn does not materialize: ``drop < total`` is exactly the condition that
    keeps the retained scan (``cumulative_tokens > drop``) non-empty, since the
    newest event's ``cumulative_tokens == total``.
    """
    account_id = "acc_test_stub"
    _uniform_ratio(monkeypatch)
    total_local = 200_000
    conn = _FakeConn(total_local=total_local, ratio_n=50, ratio_mean=0.0)
    result = await queries.read_windowed_events(
        conn,
        "sess_x",
        window_min=50_000,
        window_max=150_000,
        model="m",
        overhead_local=50_000,
        account_id=account_id,
    )
    assert result.omission is not None
    _session_id, drop_local, *_ = conn.fetch_calls[-1]
    assert drop_local < total_local
