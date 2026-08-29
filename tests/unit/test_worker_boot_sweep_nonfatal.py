"""The startup orphan-attachment sweep must never stop the worker booting.

Regression for 2026-08-28: ``sweep_orphan_attachments`` runs in ``worker_main``
*before* the maintenance-loop tasks are created, and any exception propagated
out of it killed the process. A journal drain loaded Postgres enough that
``list_attachment_paths_for_sessions`` hit the statement timeout, so the worker
crash-looped three times before catching a quiet moment — i.e. **the worker
could not start while the database was busy, which is precisely when its
maintenance loops (journal prune, sandbox GC, reapers) were most needed.**

The asymmetry that makes failing open correct: a skipped sweep defers orphan
reclamation to the next successful boot (disk hygiene); a dead worker halts
every session, run, trigger and reaper on the instance.

These tests pin the CONSTRAINT (never raises), not just the capability — the
incident's failure mode was an exception escaping, so the load-bearing
assertions are the ones that let a raising sweep through and demand a return.
"""

from __future__ import annotations

from unittest import mock

import asyncpg
import pytest

from aios.harness.worker import sweep_orphan_attachments_best_effort

_TARGET = "aios.harness.worker.sweep_orphan_attachments"


@pytest.mark.asyncio
async def test_returns_count_on_success() -> None:
    """The happy path is unchanged: the reclaimed count is passed through."""
    pool = mock.Mock(spec=asyncpg.Pool)
    with mock.patch(_TARGET, new=mock.AsyncMock(return_value=7)):
        assert await sweep_orphan_attachments_best_effort(pool) == 7


@pytest.mark.asyncio
async def test_zero_is_preserved_not_coerced() -> None:
    """0 must stay 0, distinct from the None that signals failure."""
    pool = mock.Mock(spec=asyncpg.Pool)
    with mock.patch(_TARGET, new=mock.AsyncMock(return_value=0)):
        result = await sweep_orphan_attachments_best_effort(pool)
    assert result == 0
    assert result is not None


@pytest.mark.asyncio
async def test_statement_timeout_does_not_propagate() -> None:
    """The exact incident shape: QueryCanceledError must be swallowed."""
    pool = mock.Mock(spec=asyncpg.Pool)
    boom = asyncpg.exceptions.QueryCanceledError("canceling statement due to statement timeout")
    with mock.patch(_TARGET, new=mock.AsyncMock(side_effect=boom)):
        assert await sweep_orphan_attachments_best_effort(pool) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        asyncpg.exceptions.QueryCanceledError("statement timeout"),
        asyncpg.exceptions.TooManyConnectionsError("pool exhausted"),
        OSError("workspace root gone read-only"),
        RuntimeError("anything at all"),
    ],
    ids=["timeout", "pool-exhausted", "fs-error", "arbitrary"],
)
async def test_no_exception_type_escapes(exc: BaseException) -> None:
    """Fail open for ANY failure, not just the one class we happened to see.

    Pinning only ``QueryCanceledError`` would re-open the hole for the next
    DB-pressure symptom (pool exhaustion, a read-only FS during the walk).
    """
    pool = mock.Mock(spec=asyncpg.Pool)
    with mock.patch(_TARGET, new=mock.AsyncMock(side_effect=exc)):
        assert await sweep_orphan_attachments_best_effort(pool) is None


@pytest.mark.asyncio
async def test_failure_is_logged_at_exception_level() -> None:
    """A skipped sweep must be loud — silence would let orphans accumulate
    with nothing in the log to point at."""
    pool = mock.Mock(spec=asyncpg.Pool)
    logger = mock.Mock()
    with (
        mock.patch(_TARGET, new=mock.AsyncMock(side_effect=RuntimeError("boom"))),
        mock.patch("aios.harness.worker.get_logger", return_value=logger),
    ):
        await sweep_orphan_attachments_best_effort(pool)

    logger.exception.assert_called_once()
    assert logger.exception.call_args.args[0] == "worker.orphan_attachment_sweep_failed"
    # never downgraded to a warning that an operator would scroll past
    logger.warning.assert_not_called()


@pytest.mark.asyncio
async def test_worker_main_calls_the_guarded_helper() -> None:
    """Guard against the fix being undone by a future edit reverting the call
    site to the raw sweep — the helper is worthless if worker_main stops using
    it.  Asserted on the source, since booting worker_main is not unit-scale.
    """
    import inspect

    from aios.harness import worker

    src = inspect.getsource(worker.worker_main)
    assert "sweep_orphan_attachments_best_effort(pool)" in src
    # the raw, raising call must not be reachable from worker_main
    assert "await sweep_orphan_attachments(pool)" not in src


def test_helper_signature_is_optional_int() -> None:
    """``None`` is the failure signal and callers must treat it as falsy."""
    import inspect

    sig = inspect.signature(sweep_orphan_attachments_best_effort)
    assert sig.return_annotation in ("int | None", int | None)
