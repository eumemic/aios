"""Pin per-session ``lock`` and ``queueing_lock`` on wake_session jobs.

Procrastinate stores the ``@app.task(lock=...)`` decorator argument
verbatim — there is no kwarg-template substitution. So a decorator-level
``lock="{session_id}"`` would assign every wake job the same literal
lock value and procrastinate's per-lock serialization (the unique
partial indexes on ``lock``/``queueing_lock``) would gate all wakes to
one at a time across sessions. ``defer_wake`` therefore must pass the
real ``session_id`` per call to ``configure_task``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from procrastinate import App
from procrastinate.testing import InMemoryConnector


def _job_rows(app: App) -> list[dict]:
    connector = app.connector
    assert isinstance(connector, InMemoryConnector)
    return list(connector.jobs.values())


class TestDeferWakeLockValues:
    async def test_lock_is_session_id(self, in_memory_app: App) -> None:
        from aios.harness.wake import defer_wake

        pool = MagicMock()
        with patch("aios.harness.wake.sessions_service.append_event", AsyncMock()):
            await defer_wake(pool, "sess_alpha", cause="message")

        rows = _job_rows(in_memory_app)
        assert len(rows) == 1
        assert rows[0]["lock"] == "sess_alpha"

    async def test_queueing_lock_is_session_id(self, in_memory_app: App) -> None:
        from aios.harness.wake import defer_wake

        pool = MagicMock()
        with patch("aios.harness.wake.sessions_service.append_event", AsyncMock()):
            await defer_wake(pool, "sess_beta", cause="message")

        rows = _job_rows(in_memory_app)
        assert len(rows) == 1
        assert rows[0]["queueing_lock"] == "sess_beta"

    async def test_defer_retry_wake_uses_session_id_lock(self, in_memory_app: App) -> None:
        from aios.harness.wake import defer_retry_wake

        pool = MagicMock()
        with patch("aios.harness.wake.sessions_service.append_event", AsyncMock()):
            await defer_retry_wake(pool, "sess_gamma", delay_seconds=2)

        rows = _job_rows(in_memory_app)
        assert len(rows) == 1
        assert rows[0]["lock"] == "sess_gamma"
        assert rows[0]["queueing_lock"] == "sess_gamma"


class TestDeferWakeCrossSessionIsolation:
    async def test_distinct_sessions_do_not_coalesce(self, in_memory_app: App) -> None:
        """N defers for N distinct sessions ⇒ N queued jobs.

        ``defer_wake`` swallows ``AlreadyEnqueued``, so a regression of
        the original bug wouldn't raise — it would surface as fewer
        queued jobs than sessions.
        """
        from aios.harness.wake import defer_wake

        pool = MagicMock()
        n = 5

        with patch("aios.harness.wake.sessions_service.append_event", AsyncMock()):
            for i in range(n):
                await defer_wake(pool, f"sess_{i}", cause="message")

        rows = _job_rows(in_memory_app)
        assert len(rows) == n
        assert sorted(row["lock"] for row in rows) == [f"sess_{i}" for i in range(n)]

    async def test_same_session_still_coalesces(self, in_memory_app: App) -> None:
        """Same-session defers must still dedup to one queued wake.

        Multiple defers within one session (user message + tool
        completion + sweep) collapse into a single wake; the queued
        wake handles all events that arrived since it was deferred.
        """
        from aios.harness.wake import defer_wake

        pool = MagicMock()

        with patch("aios.harness.wake.sessions_service.append_event", AsyncMock()):
            await defer_wake(pool, "sess_x", cause="message")
            await defer_wake(pool, "sess_x", cause="sweep")
            await defer_wake(pool, "sess_x", cause="tool_confirmation")

        rows = _job_rows(in_memory_app)
        assert len(rows) == 1
        assert rows[0]["queueing_lock"] == "sess_x"
