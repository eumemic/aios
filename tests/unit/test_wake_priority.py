"""Foreground protection: background workflow work is demoted below foreground.

``defer_wake`` derives a procrastinate job ``priority`` from the woken session's
origin (a workflow ``agent()`` child carries a ``parent_run_id``), so a fan-out of
background children can't starve a user's message. ``defer_run_wake`` is always
background. Procrastinate (and the in-memory connector) fetch todo jobs in
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


def _ctx(parent_run_id: str | None) -> tuple[str, str | None]:
    """A ``get_session_workflow_context`` result: ``(account_id, parent_run_id)``."""
    return ("acc", parent_run_id)


@pytest.mark.parametrize(
    "ctx_return, expected",
    [
        pytest.param(_ctx("wfr_1"), _BACKGROUND_PRIORITY, id="background_child"),
        pytest.param(_ctx(None), _FOREGROUND_PRIORITY, id="foreground"),
        pytest.param(None, _FOREGROUND_PRIORITY, id="missing_session_race"),
    ],
)
async def test_defer_wake_priority_by_origin(
    in_memory_app: App, ctx_return: tuple[str, str | None] | None, expected: int
) -> None:
    """The woken session's origin sets the priority: a workflow child (non-null
    parent_run_id) is demoted; a foreground session — or a vanished one — stays default."""
    pool = MagicMock()
    with (
        patch("aios.services.wake.sessions_service.append_event", AsyncMock()),
        patch(
            "aios.services.wake.queries.get_session_workflow_context",
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
            "aios.services.wake.queries.get_session_workflow_context",
            AsyncMock(return_value=_ctx("wfr_2")),
        ),
    ):
        await defer_wake(pool, "sess_bg2", cause="reschedule", delay_seconds=2, account_id="acc")
    assert _jobs_by_session(in_memory_app)["sess_bg2"] == _BACKGROUND_PRIORITY


async def test_run_wake_is_background(in_memory_app: App) -> None:
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
            "aios.services.wake.queries.get_session_workflow_context",
            AsyncMock(return_value=_ctx("wfr_3")),
        ):
            await defer_wake(pool, "sess_bg", cause="message", account_id="acc")  # enqueued first
        with patch(
            "aios.services.wake.queries.get_session_workflow_context",
            AsyncMock(return_value=_ctx(None)),
        ):
            await defer_wake(pool, "sess_fg", cause="message", account_id="acc")  # enqueued later
    priorities = _jobs_by_session(in_memory_app)
    # Foreground enqueued *after* background but outranks it on (priority DESC, id ASC).
    assert priorities["sess_fg"] > priorities["sess_bg"]
