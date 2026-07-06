"""Pin the queue a ``harness.*`` job lands on when deferred from a process
with no registered tasks (the aios-api process).

Regression for issue #1699: after ``ae0c0064`` the api process registers no
procrastinate tasks (worker-only), so ``configure_task`` can no longer read the
``@app.task(queue="sessions")`` registration and procrastinate falls back to the
``"default"`` queue. The worker only polls ``sessions``/``workflows``, so an
API-origin wake (``inbound`` / ``connector_tool_result``) sat ``todo`` forever
and its ``queueing_lock`` then swallowed every subsequent wake — a permanent
wedge of a live connector agent.

The fix is an EXPLICIT ``queue=`` on every API-side ``configure_task`` call, so
routing never depends on task-registration being present in the calling process.
The ``in_memory_app`` fixture deliberately does NOT import ``aios.harness.tasks``,
so it faithfully reproduces the api process's no-registered-tasks condition.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from procrastinate import App
from procrastinate.testing import InMemoryConnector


def _job_rows(app: App) -> list[dict[str, Any]]:
    connector = app.connector
    assert isinstance(connector, InMemoryConnector)
    return list(connector.jobs.values())


def _assert_no_registered_tasks(app: App) -> None:
    """Guard the premise: the fixture reproduces the api process (no tasks).

    If a future refactor made the fixture register ``aios.harness.tasks``,
    ``configure_task`` would read the registered queue and these tests would
    pass *without* the explicit ``queue=`` — silently defeating the regression
    guard. Assert the fixture stays task-free so the tests keep their teeth.
    """
    assert "harness.wake_session" not in app.tasks
    assert "harness.wake_workflow" not in app.tasks


class TestApiOriginWakeRouting:
    """Acceptance #1 + #2: API-origin wakes land on the worker-polled queue."""

    @pytest.mark.parametrize("cause", ["inbound", "connector_tool_result", "api_invoke"])
    async def test_wake_session_routes_to_sessions_not_default(
        self, in_memory_app: App, cause: str
    ) -> None:
        _assert_no_registered_tasks(in_memory_app)
        from aios.jobs.app import defer_wake

        pool = MagicMock()
        with patch("aios.jobs.app.queries.append_event", AsyncMock()):
            await defer_wake(pool, "sess_api", cause=cause, account_id="acc")

        rows = _job_rows(in_memory_app)
        assert len(rows) == 1
        # The crux of #1699: NOT "default".
        assert rows[0]["queue_name"] == "sessions"
        assert rows[0]["task_name"] == "harness.wake_session"

    async def test_delayed_wake_still_routes_to_sessions(self, in_memory_app: App) -> None:
        """The reschedule / debounce path (``delay_seconds``) routes too."""
        from aios.jobs.app import defer_wake

        pool = MagicMock()
        with patch("aios.jobs.app.queries.append_event", AsyncMock()):
            await defer_wake(
                pool, "sess_delayed", cause="reschedule", delay_seconds=2, account_id="acc"
            )

        rows = _job_rows(in_memory_app)
        assert len(rows) == 1
        assert rows[0]["queue_name"] == "sessions"

    async def test_run_wake_routes_to_workflows_not_default(self, in_memory_app: App) -> None:
        _assert_no_registered_tasks(in_memory_app)
        from aios.jobs.app import defer_run_wake

        await defer_run_wake("run_api")

        rows = _job_rows(in_memory_app)
        assert len(rows) == 1
        assert rows[0]["queue_name"] == "workflows"
        assert rows[0]["task_name"] == "harness.wake_workflow"

    async def test_trigger_fire_routes_to_sessions_not_default(self, in_memory_app: App) -> None:
        from aios.jobs.app import defer_trigger_fire

        await defer_trigger_fire("trig_1", "trun_1")

        rows = _job_rows(in_memory_app)
        assert len(rows) == 1
        assert rows[0]["queue_name"] == "sessions"
        assert rows[0]["task_name"] == "harness.run_trigger"


class TestRoutingMatchesRegistration:
    """The explicit ``queue=`` must equal what the worker actually registers.

    An explicit-but-wrong queue reintroduces the wedge. This test pins the
    hardcoded routing constants against the ``@app.task(queue=…)`` decorators in
    ``aios.harness.tasks`` (imported here — the worker's registration side).
    """

    async def test_explicit_queue_equals_registered_queue(self, in_memory_app: App) -> None:
        import aios.harness.tasks  # noqa: F401 — registers the tasks on the app
        from aios.jobs.app import defer_run_wake, defer_trigger_fire, defer_wake

        # Now the tasks ARE registered; read their decorator-level queues.
        registered = {name: task.queue for name, task in in_memory_app.tasks.items()}

        pool = MagicMock()
        with patch("aios.jobs.app.queries.append_event", AsyncMock()):
            await defer_wake(pool, "sess_reg", cause="message", account_id="acc")
        await defer_run_wake("run_reg")
        await defer_trigger_fire("trig_reg", "trun_reg")

        rows = {row["task_name"]: row["queue_name"] for row in _job_rows(in_memory_app)}
        for task_name, deferred_queue in rows.items():
            assert deferred_queue == registered[task_name], (
                f"{task_name} deferred to {deferred_queue!r} but registered on "
                f"{registered[task_name]!r} — routing drift wedges the worker (#1699)"
            )


class TestNoUnpolledQueue:
    """Acceptance #3: no ``harness.*`` job may be created on an unpolled queue."""

    def test_polled_queue_guard_accepts_worker_queues(self) -> None:
        from aios.jobs.app import (
            QUEUE_SESSIONS,
            QUEUE_WORKFLOWS,
            WORKER_POLLED_QUEUES,
            _polled_queue,
        )

        assert _polled_queue(QUEUE_SESSIONS) == "sessions"
        assert _polled_queue(QUEUE_WORKFLOWS) == "workflows"
        assert {QUEUE_SESSIONS, QUEUE_WORKFLOWS} == set(WORKER_POLLED_QUEUES)

    def test_polled_queue_guard_rejects_default(self) -> None:
        from aios.jobs.app import _polled_queue

        with pytest.raises(ValueError, match="default"):
            _polled_queue("default")

    def test_worker_polls_exactly_the_routing_constants(self) -> None:
        """The worker's polled set is the shared constant, not a separate list.

        Guards against the two drifting apart (a queue polled but unroutable,
        or routable but unpolled) — the class of bug #1699 is.
        """
        from aios.harness import worker
        from aios.jobs.app import WORKER_POLLED_QUEUES

        assert worker.WORKER_POLLED_QUEUES is WORKER_POLLED_QUEUES
