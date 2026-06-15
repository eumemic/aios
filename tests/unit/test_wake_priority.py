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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from procrastinate import App
from procrastinate.testing import InMemoryConnector

from aios.services.wake import (
    _BACKGROUND_PRIORITY,
    _FOREGROUND_PRIORITY,
    defer_run_wake,
    defer_wake,
)


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
        patch("aios.services.wake.sessions_service.append_event", AsyncMock()),
        patch(
            "aios.services.wake.queries.get_wake_priority_context",
            AsyncMock(return_value=ctx_return),
        ),
    ):
        await defer_wake(pool, "sess_x", cause="message", account_id="acc")
    assert _jobs_by_session(in_memory_app)["sess_x"] == expected


async def test_delayed_background_wake_is_also_demoted(in_memory_app: App) -> None:
    """The reschedule-backoff path (``delay_seconds``) carries the priority too."""
    pool = MagicMock()
    with (
        patch("aios.services.wake.sessions_service.append_event", AsyncMock()),
        patch(
            "aios.services.wake.queries.get_wake_priority_context",
            AsyncMock(return_value=_ctx(True)),
        ),
    ):
        await defer_wake(pool, "sess_bg2", cause="reschedule", delay_seconds=2, account_id="acc")
    assert _jobs_by_session(in_memory_app)["sess_bg2"] == _BACKGROUND_PRIORITY


async def test_reused_servicer_priority_is_per_stimulus(in_memory_app: App) -> None:
    """The multi-edge per-stimulus distinction the locked decision turns on: the
    same servicer woken by a bg-edge stimulus wakes background, and woken by a
    fg-user stimulus wakes foreground. ``defer_wake`` derives priority freshly per
    wake, so the carrier reflects *this* stimulus's edge — not a materialized
    servicer-row scalar."""
    pool = MagicMock()
    with patch("aios.services.wake.sessions_service.append_event", AsyncMock()):
        with patch(
            "aios.services.wake.queries.get_wake_priority_context",
            AsyncMock(return_value=_ctx(True)),  # bg-edge stimulus
        ):
            await defer_wake(pool, "router_bg", cause="message", account_id="acc")
        with patch(
            "aios.services.wake.queries.get_wake_priority_context",
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
    with patch("aios.services.wake.sessions_service.append_event", AsyncMock()):
        with patch(
            "aios.services.wake.queries.get_wake_priority_context",
            AsyncMock(return_value=_ctx(True)),
        ):
            await defer_wake(pool, "sess_bg", cause="message", account_id="acc")  # enqueued first
        with patch(
            "aios.services.wake.queries.get_wake_priority_context",
            AsyncMock(return_value=_ctx(False)),
        ):
            await defer_wake(pool, "sess_fg", cause="message", account_id="acc")  # enqueued later
    priorities = _jobs_by_session(in_memory_app)
    # Foreground enqueued *after* background but outranks it on (priority DESC, id ASC).
    assert priorities["sess_fg"] > priorities["sess_bg"]
