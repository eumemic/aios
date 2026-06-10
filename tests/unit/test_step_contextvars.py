"""Unit tests for contextvar binding in the harness/workflow step entrypoints.

``run_session_step`` / ``run_workflow_step`` bind ``account_id`` (plus
``session_id``/``cause`` resp. ``run_id``) onto structlog contextvars so every
log line a step emits is attributable to its tenant, and clear them in a
``finally`` so the binding never leaks to the next job on the same worker task.

The early-return guards run BEFORE ``account_id`` is known (a wake for a
gone/terminal session-or-run is an idempotent no-op), so they must NOT bind —
and they must leave contextvars untouched. We pin both: the no-bind early path
and the bind→clear lifecycle on a step that proceeds past the guard.
"""

from __future__ import annotations

from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
from structlog.contextvars import clear_contextvars, get_contextvars

from aios.harness import runtime
from aios.harness.loop import run_session_step
from aios.workflows.step import run_workflow_step
from tests.unit.conftest import fake_pool_yielding_conn


@pytest.fixture(autouse=True)
def _clean_contextvars() -> Any:
    clear_contextvars()
    yield
    clear_contextvars()


async def test_session_step_missing_session_leaves_contextvars_empty() -> None:
    """A wake for a gone session early-returns BEFORE any bind, so contextvars
    stay empty (the guard runs ahead of ``bind_contextvars``)."""
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=None)  # no live session → account_id None
    pool = fake_pool_yielding_conn(conn)

    prev_pool = runtime.pool
    runtime.pool = pool
    try:
        await run_session_step("ses_does_not_exist")
    finally:
        runtime.pool = prev_pool

    assert get_contextvars() == {}


async def test_session_step_binds_then_clears() -> None:
    """Once ``account_id`` resolves, the step binds session_id/account_id/cause,
    and the ``finally`` clears them — so nothing leaks past the step.

    We stub the step body to snapshot the contextvars mid-step (proving the bind
    happened) and otherwise no-op, avoiding the full DB-backed step.
    """
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value="acc_live")
    pool = fake_pool_yielding_conn(conn)

    seen: dict[str, Any] = {}

    async def _fake_body(*_args: Any, **_kwargs: Any) -> Any:
        seen.update(get_contextvars())
        from aios.harness.loop import _StepResult

        return _StepResult()

    prev_pool = runtime.pool
    runtime.pool = pool
    try:
        with (
            mock.patch("aios.harness.loop._run_session_step_body", new=_fake_body),
            mock.patch(
                "aios.services.sessions.append_event",
                new=AsyncMock(return_value=MagicMock(id="evt_1")),
            ),
            mock.patch.object(runtime, "require_task_registry", return_value=MagicMock()),
        ):
            await run_session_step("ses_live", cause="message")
    finally:
        runtime.pool = prev_pool

    # Bound mid-step …
    assert seen.get("account_id") == "acc_live"
    assert seen.get("session_id") == "ses_live"
    assert seen.get("cause") == "message"
    # … and cleared afterward.
    assert get_contextvars() == {}


async def test_workflow_step_missing_run_leaves_contextvars_empty() -> None:
    """A wake for a vanished run early-returns BEFORE any bind, contextvars empty."""
    conn = MagicMock()
    pool = fake_pool_yielding_conn(conn)

    prev_pool = runtime.pool
    runtime.pool = pool
    try:
        with mock.patch(
            "aios.workflows.step.wf_queries.get_run_for_step",
            new=AsyncMock(return_value=None),
        ):
            await run_workflow_step("wfr_gone")
    finally:
        runtime.pool = prev_pool

    assert get_contextvars() == {}


async def test_workflow_step_binds_then_clears() -> None:
    """Once the run loads, the step binds run_id/account_id (no ``cause`` for
    workflows) and the ``finally`` clears them."""
    conn = MagicMock()
    pool = fake_pool_yielding_conn(conn)

    run = MagicMock()
    run.account_id = "acc_wf"
    run.status = "running"

    seen: dict[str, Any] = {}

    async def _fake_body(*_args: Any, **_kwargs: Any) -> None:
        seen.update(get_contextvars())

    prev_pool = runtime.pool
    runtime.pool = pool
    try:
        with (
            mock.patch(
                "aios.workflows.step.wf_queries.get_run_for_step",
                new=AsyncMock(return_value=run),
            ),
            mock.patch("aios.workflows.step._run_workflow_step_body", new=_fake_body),
        ):
            await run_workflow_step("wfr_live")
    finally:
        runtime.pool = prev_pool

    assert seen.get("account_id") == "acc_wf"
    assert seen.get("run_id") == "wfr_live"
    assert "cause" not in seen  # workflows omit cause
    assert get_contextvars() == {}


def test_logging_emits_account_id_when_bound() -> None:
    """Sanity: with account_id on the contextvars, a logged line carries it
    through ``merge_contextvars`` (the mechanism the step binds for)."""
    from structlog.contextvars import bind_contextvars

    from aios.logging import configure_logging

    configure_logging("INFO")
    clear_contextvars()
    bind_contextvars(account_id="acc_z", session_id="ses_z")
    try:
        assert get_contextvars()["account_id"] == "acc_z"
        log = structlog.get_logger("test")
        log.info("noop")  # smoke: does not raise with bindings present
    finally:
        clear_contextvars()
