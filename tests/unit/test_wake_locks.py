"""Regression tests for issue #192 — ``defer_wake`` passes per-session locks.

Lives at the unit layer because the contract being pinned is purely about
the *values* ``defer_wake`` writes into procrastinate's ``lock`` and
``queueing_lock`` columns. The :class:`InMemoryConnector` from
procrastinate's test kit gives direct access to those columns without
spinning up a database or a worker.

Background: the original bug — fixed alongside these tests — was that
``src/aios/harness/tasks.py`` declared
``@app.task(lock="{session_id}", queueing_lock="{session_id}")``. The
braces look like a kwarg-template placeholder, but procrastinate stores
the string literally — there is no template substitution at any layer.
Every wake job ended up sharing the lock value ``"{session_id}"`` and
procrastinate's per-lock serialization (the unique partial index plus
the ``NOT EXISTS`` clause in ``procrastinate_fetch_job_v2``) gated them
all to one at a time, regardless of session.

Fix: ``src/aios/harness/wake.py`` now passes per-session
``lock=session_id, queueing_lock=session_id`` through
``configure_task``. These tests pin the contract so the same regression
can't recur:

* :meth:`TestDeferWakeLockValues.test_lock_is_session_id` — single
  defer assigns ``lock = session_id``.
* :meth:`TestDeferWakeLockValues.test_queueing_lock_is_session_id` —
  same for ``queueing_lock``. Together with the index, this is what
  prevents cross-session ``AlreadyEnqueued``.
* :meth:`TestDeferWakeCrossSessionIsolation.test_distinct_sessions_do_not_coalesce`
  — bursty defers across N sessions all enqueue, no
  ``AlreadyEnqueued``. This is the user-visible symptom of #192.
* :meth:`TestDeferWakeCrossSessionIsolation.test_same_session_still_coalesces`
  — guards against an over-zealous regression that drops the dedup.
  Two defers for one session must still produce one queued job.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from procrastinate import App
from procrastinate.testing import InMemoryConnector


@pytest.fixture
async def in_memory_app() -> AsyncIterator[App]:
    """Patch the aios procrastinate app to use an in-memory connector.

    Mirrors the fixture in ``tests/unit/test_wake_instrumentation.py``.
    The InMemoryConnector implements the same defer/fetch interface
    against a Python dict, so we can read job rows directly via
    ``connector.jobs``.
    """
    from aios.harness.procrastinate_app import app

    with app.replace_connector(InMemoryConnector()) as patched:
        yield patched


def _job_rows(app: App) -> list[dict]:
    """Return the in-memory job rows. Helper so the cast lives in one place."""
    connector = app.connector
    assert isinstance(connector, InMemoryConnector)
    return list(connector.jobs.values())


class TestDeferWakeLockValues:
    """``defer_wake`` must populate ``lock`` and ``queueing_lock`` with the
    session_id value, not the literal template string."""

    async def test_lock_is_session_id(self, in_memory_app: App) -> None:
        """One defer ⇒ one job row with ``lock = session_id``.

        The ``lock`` column drives procrastinate's ``WHERE
        status='doing'`` partial unique index — distinct values let
        the worker run distinct sessions concurrently up to its
        configured concurrency.
        """
        from aios.harness.wake import defer_wake

        pool = MagicMock()
        with patch("aios.harness.wake.sessions_service.append_event", AsyncMock()):
            await defer_wake(pool, "sess_alpha", cause="message")

        rows = _job_rows(in_memory_app)
        assert len(rows) == 1
        assert rows[0]["lock"] == "sess_alpha", (
            f"job row's lock = {rows[0]['lock']!r}; expected 'sess_alpha'. "
            f"defer_wake must pass lock=session_id to configure_task — "
            f"the @app.task decorator's '{{session_id}}' is a literal "
            f"string in procrastinate, not a kwarg template."
        )

    async def test_queueing_lock_is_session_id(self, in_memory_app: App) -> None:
        """Same contract for ``queueing_lock``.

        Without this, the partial unique index on
        ``queueing_lock WHERE status='todo'`` coalesces defers across
        unrelated sessions, hiding new wake events behind already-queued
        jobs for different sessions.
        """
        from aios.harness.wake import defer_wake

        pool = MagicMock()
        with patch("aios.harness.wake.sessions_service.append_event", AsyncMock()):
            await defer_wake(pool, "sess_beta", cause="message")

        rows = _job_rows(in_memory_app)
        assert len(rows) == 1
        assert rows[0]["queueing_lock"] == "sess_beta", (
            f"job row's queueing_lock = {rows[0]['queueing_lock']!r}; "
            f"expected 'sess_beta'. defer_wake must pass "
            f"queueing_lock=session_id to configure_task."
        )

    async def test_defer_retry_wake_uses_session_id_lock(self, in_memory_app: App) -> None:
        """The retry path has the same contract as the main defer path."""
        from aios.harness.wake import defer_retry_wake

        pool = MagicMock()
        with patch("aios.harness.wake.sessions_service.append_event", AsyncMock()):
            await defer_retry_wake(pool, "sess_gamma", delay_seconds=2)

        rows = _job_rows(in_memory_app)
        assert len(rows) == 1
        assert rows[0]["lock"] == "sess_gamma", (
            f"defer_retry_wake's job row lock = {rows[0]['lock']!r}; "
            f"expected 'sess_gamma'. The retry path has the same "
            f"per-session lock contract."
        )
        assert rows[0]["queueing_lock"] == "sess_gamma"


class TestDeferWakeCrossSessionIsolation:
    """The user-visible symptom of #192: bursty defers across distinct
    sessions must not coalesce, but bursty defers within ONE session
    must still coalesce."""

    async def test_distinct_sessions_do_not_coalesce(self, in_memory_app: App) -> None:
        """N defers across N sessions ⇒ N queued jobs.

        Mirrors issue #192's "Repro" scenario: many sessions POST
        messages back-to-back. Each ``session_id`` owns its own
        ``queueing_lock`` slot, so all N jobs queue up and the worker
        can run them concurrently.

        ``defer_wake`` swallows ``AlreadyEnqueued``, so a regression
        of the original bug wouldn't raise — it would surface as
        ``connector.jobs`` holding fewer rows than expected.
        """
        from aios.harness.wake import defer_wake

        pool = MagicMock()
        n = 5

        with patch("aios.harness.wake.sessions_service.append_event", AsyncMock()):
            for i in range(n):
                await defer_wake(pool, f"sess_{i}", cause="message")

        rows = _job_rows(in_memory_app)
        assert len(rows) == n, (
            f"only {len(rows)}/{n} wake jobs queued — the rest were "
            f"coalesced via queueing_lock because every defer used the "
            f"literal lock value '{{session_id}}'. This is exactly the "
            f"#192 symptom: under bursty load across distinct sessions, "
            f"only one wake survives the queueing_lock dedup."
        )

        locks = sorted(row["lock"] for row in rows)
        assert locks == [f"sess_{i}" for i in range(n)], (
            f"lock values = {locks}; expected one per session. "
            f"Each session must own its own lock slot for the worker "
            f"to be able to run them concurrently."
        )

    async def test_same_session_still_coalesces(self, in_memory_app: App) -> None:
        """Two defers for the same session ⇒ one queued job.

        The whole point of the ``queueing_lock`` is that within ONE
        session, multiple defers (user message + tool completion +
        sweep firing in the same window) collapse into a single wake
        — the existing queued job sees all the new events when it
        runs. Dropping this dedup would multiply queue and step
        traffic without adding throughput.
        """
        from aios.harness.wake import defer_wake

        pool = MagicMock()

        with patch("aios.harness.wake.sessions_service.append_event", AsyncMock()):
            await defer_wake(pool, "sess_x", cause="message")
            await defer_wake(pool, "sess_x", cause="sweep")
            await defer_wake(pool, "sess_x", cause="tool_confirmation")

        rows = _job_rows(in_memory_app)
        assert len(rows) == 1, (
            f"got {len(rows)} jobs for one session_id; expected 1. "
            f"Same-session defers must coalesce via queueing_lock — "
            f"the queued wake handles all events that arrived since "
            f"it was deferred."
        )
        assert rows[0]["queueing_lock"] == "sess_x"
