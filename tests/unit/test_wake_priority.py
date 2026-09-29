"""Foreground protection: background workflow work is demoted below foreground.

``defer_wake`` derives a procrastinate job ``priority`` from the **triggering
edge's up-link** (#1123's ``request_opened`` ``caller``), re-keyed off the run-only
``parent_run_id`` column (#1125). Every caller kind (api/session/run) demotes
uniformly when its ancestor is background, so a fan-out of background descendants
can't starve a user's message. ``defer_run_wake`` is always background.
Procrastinate (and the in-memory connector) fetch todo jobs in
``(priority DESC, id ASC)`` order, so a higher priority is served first.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from procrastinate import App
from procrastinate.testing import InMemoryConnector

from aios.jobs.app import (
    _BACKGROUND_PRIORITY,
    _FOREGROUND_PRIORITY,
    defer_run_wake,
    defer_wake,
)


@pytest.fixture(autouse=True)
def _idle_fair_share_meter() -> Any:
    """Default the per-account fair-share meter (#418) to an idle account (0
    outstanding wakes) so the tier tests see the bare tier priority; the #418
    tests below re-patch it with their own values."""
    with patch(
        "aios.jobs.app.queries.count_account_outstanding_session_wakes",
        AsyncMock(return_value=0),
    ) as mock:
        yield mock


def _jobs_by_session(app: App) -> dict[str, int]:
    connector = app.connector
    assert isinstance(connector, InMemoryConnector)
    return {j["args"]["session_id"]: j["priority"] for j in connector.jobs.values()}


def _ctx(is_background: bool) -> tuple[str, bool]:
    """A ``get_wake_priority_context`` result: ``(account_id, is_background)``."""
    return ("acc", is_background)


@pytest.mark.parametrize(
    "ctx_return, expected",
    [
        # A run-launched request-serving child wakes background (regression guard on
        # the run path: behavior-preserved from the old parent_run_id derivation).
        pytest.param(_ctx(True), _BACKGROUND_PRIORITY, id="run_launched_child"),
        # A background-rooted session-invoke child also demotes — the new behavior;
        # today (keyed on parent_run_id) it would have woken foreground.
        pytest.param(_ctx(True), _BACKGROUND_PRIORITY, id="session_launched_child"),
        # A root / fg-user (edgeless, or fg up-link) session stays foreground.
        pytest.param(_ctx(False), _FOREGROUND_PRIORITY, id="foreground"),
        # Deleted-session race: a missing row → foreground default → wake no-ops.
        pytest.param(None, _FOREGROUND_PRIORITY, id="missing_session_race"),
    ],
)
async def test_defer_wake_priority_from_edge_uplink(
    in_memory_app: App, ctx_return: tuple[str, bool] | None, expected: int
) -> None:
    """The triggering edge's up-link sets the priority: any background-rooted
    request-serving descendant (run- or session-launched) is demoted; a foreground
    session — or a vanished one — stays default."""
    pool = MagicMock()
    with (
        patch("aios.jobs.app.queries.append_event", AsyncMock()),
        patch(
            "aios.jobs.app.queries.get_wake_priority_context",
            AsyncMock(return_value=ctx_return),
        ),
    ):
        await defer_wake(pool, "sess_x", cause="message", account_id="acc")
    assert _jobs_by_session(in_memory_app)["sess_x"] == expected


async def test_delayed_background_wake_is_also_demoted(in_memory_app: App) -> None:
    """The reschedule-backoff path (``delay_seconds``) carries the priority too."""
    pool = MagicMock()
    with (
        patch("aios.jobs.app.queries.append_event", AsyncMock()),
        patch(
            "aios.jobs.app.queries.get_wake_priority_context",
            AsyncMock(return_value=_ctx(True)),
        ),
    ):
        await defer_wake(pool, "sess_bg2", cause="reschedule", delay_seconds=2, account_id="acc")
    assert _jobs_by_session(in_memory_app)["sess_bg2"] == _BACKGROUND_PRIORITY


async def test_reused_servicer_priority_is_per_stimulus(in_memory_app: App) -> None:
    """``defer_wake`` derives the job priority freshly per wake from
    ``get_wake_priority_context`` — not from a materialized servicer-row scalar — so a
    reused servicer's priority tracks *this* stimulus's edge. This tier **mocks** the
    query (two verdicts → two priorities), proving only the wiring; the SQL that
    actually picks the latest *open* edge over the oldest-ever one (the multi-edge
    distinction itself) is covered by
    ``tests/integration/test_wake_priority_context.py`` (``test_latest_open_edge_*``)."""
    pool = MagicMock()
    with patch("aios.jobs.app.queries.append_event", AsyncMock()):
        with patch(
            "aios.jobs.app.queries.get_wake_priority_context",
            AsyncMock(return_value=_ctx(True)),  # bg-edge stimulus
        ):
            await defer_wake(pool, "router_bg", cause="message", account_id="acc")
        with patch(
            "aios.jobs.app.queries.get_wake_priority_context",
            AsyncMock(return_value=_ctx(False)),  # fg-user stimulus
        ):
            await defer_wake(pool, "router_fg", cause="message", account_id="acc")
    priorities = _jobs_by_session(in_memory_app)
    assert priorities["router_bg"] == _BACKGROUND_PRIORITY
    assert priorities["router_fg"] == _FOREGROUND_PRIORITY


async def test_run_wake_is_background(in_memory_app: App) -> None:
    """``defer_run_wake`` (run-step priority) is unconditionally background and
    edge-independent — untouched by the #1125 edge re-key."""
    await defer_run_wake("wfr_x")
    connector = in_memory_app.connector
    assert isinstance(connector, InMemoryConnector)
    rows = list(connector.jobs.values())
    assert len(rows) == 1 and rows[0]["priority"] == _BACKGROUND_PRIORITY


async def test_foreground_outranks_background(in_memory_app: App) -> None:
    """The end goal: a foreground wake is fetched before a queued background wake."""
    pool = MagicMock()
    with patch("aios.jobs.app.queries.append_event", AsyncMock()):
        with patch(
            "aios.jobs.app.queries.get_wake_priority_context",
            AsyncMock(return_value=_ctx(True)),
        ):
            await defer_wake(pool, "sess_bg", cause="message", account_id="acc")  # enqueued first
        with patch(
            "aios.jobs.app.queries.get_wake_priority_context",
            AsyncMock(return_value=_ctx(False)),
        ):
            await defer_wake(pool, "sess_fg", cause="message", account_id="acc")  # enqueued later
    priorities = _jobs_by_session(in_memory_app)
    # Foreground enqueued *after* background but outranks it on (priority DESC, id ASC).
    assert priorities["sess_fg"] > priorities["sess_bg"]


# ─── #418: per-account fair-share term, background tier only ─────────────────


def _patch_meter(outstanding: int | Any) -> Any:
    """Patch the Postgres-backed per-account meter (outstanding session wakes)."""
    mock = (
        outstanding if isinstance(outstanding, AsyncMock) else AsyncMock(return_value=outstanding)
    )
    return patch("aios.jobs.app.queries.count_account_outstanding_session_wakes", mock)


async def test_background_wake_demoted_by_account_outstanding_wakes(in_memory_app: App) -> None:
    """A background wake is demoted one step per outstanding session wake the account
    already holds on the shared queue (the fair-share term)."""
    pool = MagicMock()
    with (
        patch("aios.jobs.app.queries.append_event", AsyncMock()),
        patch(
            "aios.jobs.app.queries.get_wake_priority_context",
            AsyncMock(return_value=_ctx(True)),
        ),
        _patch_meter(7) as meter,
    ):
        await defer_wake(pool, "sess_heavy", cause="message", account_id="acc")
    assert _jobs_by_session(in_memory_app)["sess_heavy"] == _BACKGROUND_PRIORITY - 7
    meter.assert_awaited_once()
    assert meter.await_args.args[1:] == ("acc", "sess_heavy")


async def test_fair_share_demotion_is_capped(
    in_memory_app: App, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The demotion saturates at ``wake_fair_share_max_demotion``."""
    from aios.config import get_settings

    monkeypatch.setenv("AIOS_WAKE_FAIR_SHARE_MAX_DEMOTION", "5")
    get_settings.cache_clear()
    try:
        pool = MagicMock()
        with (
            patch("aios.jobs.app.queries.append_event", AsyncMock()),
            patch(
                "aios.jobs.app.queries.get_wake_priority_context",
                AsyncMock(return_value=_ctx(True)),
            ),
            _patch_meter(500),
        ):
            await defer_wake(pool, "sess_capped", cause="message", account_id="acc")
    finally:
        get_settings.cache_clear()
    assert _jobs_by_session(in_memory_app)["sess_capped"] == _BACKGROUND_PRIORITY - 5


async def test_fair_share_disabled_at_zero_skips_meter(
    in_memory_app: App, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``wake_fair_share_max_demotion=0`` is the kill switch: no meter query, plain
    background priority (pre-#418 behavior)."""
    from aios.config import get_settings

    monkeypatch.setenv("AIOS_WAKE_FAIR_SHARE_MAX_DEMOTION", "0")
    get_settings.cache_clear()
    try:
        pool = MagicMock()
        with (
            patch("aios.jobs.app.queries.append_event", AsyncMock()),
            patch(
                "aios.jobs.app.queries.get_wake_priority_context",
                AsyncMock(return_value=_ctx(True)),
            ),
            _patch_meter(9) as meter,
        ):
            await defer_wake(pool, "sess_off", cause="message", account_id="acc")
    finally:
        get_settings.cache_clear()
    assert _jobs_by_session(in_memory_app)["sess_off"] == _BACKGROUND_PRIORITY
    meter.assert_not_awaited()


@pytest.mark.parametrize("ctx_return", [_ctx(False), None], ids=["foreground", "missing"])
async def test_foreground_never_fair_share_demoted(
    in_memory_app: App, ctx_return: tuple[str, bool] | None
) -> None:
    """Tiering stays strictly dominant: a heavy account's foreground wake is never
    demoted (and the meter is not even consulted on the foreground hot path)."""
    pool = MagicMock()
    with (
        patch("aios.jobs.app.queries.append_event", AsyncMock()),
        patch(
            "aios.jobs.app.queries.get_wake_priority_context",
            AsyncMock(return_value=ctx_return),
        ),
        _patch_meter(10_000) as meter,
    ):
        await defer_wake(pool, "sess_fg_heavy", cause="message", account_id="acc")
    assert _jobs_by_session(in_memory_app)["sess_fg_heavy"] == _FOREGROUND_PRIORITY
    meter.assert_not_awaited()


async def test_light_account_not_starved_by_heavy_account_burst(in_memory_app: App) -> None:
    """The issue's unit scenario: account A bursts 50 background wakes, then account
    B enqueues one. Fetching in procrastinate order (priority DESC, id ASC), B's wake
    is served within the first two fetches — not after all 50 of A's."""
    connector = in_memory_app.connector
    assert isinstance(connector, InMemoryConnector)
    account_of: dict[str, str] = {}

    async def meter(_conn: Any, account_id: str, session_id: str) -> int:
        # The real meter counts the account's todo/doing wake_session rows; mirror
        # it over the in-memory connector's rows.
        return sum(
            1
            for j in connector.jobs.values()
            if j["status"] in ("todo", "doing")
            and account_of.get(j["args"]["session_id"]) == account_id
            and j["args"]["session_id"] != session_id
        )

    async def ctx_for(_conn: Any, session_id: str) -> tuple[str, bool]:
        return (account_of[session_id], True)

    pool = MagicMock()
    with (
        patch("aios.jobs.app.queries.append_event", AsyncMock()),
        patch("aios.jobs.app.queries.get_wake_priority_context", ctx_for),
        _patch_meter(AsyncMock(side_effect=meter)),
    ):
        for i in range(50):
            sid = f"a_{i}"
            account_of[sid] = "acc_a"
            await defer_wake(pool, sid, cause="message", account_id="acc_a")
        account_of["b_0"] = "acc_b"
        await defer_wake(pool, "b_0", cause="message", account_id="acc_b")

    order = [
        j["args"]["session_id"]
        for j in sorted(connector.jobs.values(), key=lambda j: (-j["priority"], j["id"]))
    ]
    assert order.index("b_0") <= 1
    # And every one of A's jobs stays inside the background tier.
    assert all(
        _BACKGROUND_PRIORITY >= j["priority"] > _FOREGROUND_PRIORITY - 10_000
        for j in connector.jobs.values()
    )
