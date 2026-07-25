"""Durable-outcome coverage for the ``harness.recycle_sandbox`` job (#2022).

Review finding 2: the destructive execution used to be one-shot (``retry=False``)
with no durable failure outcome — a transient container-removal / proxy-teardown /
pointer-clear / provision failure permanently consumed the admitted, rate-limited
request while the journal held only ``sandbox_recycle_requested`` and the caller
had already been told 202.

These tests pin the replacement contract:

* the task carries a bounded retry strategy (transient failures converge),
* a transient failure on an early attempt RE-RAISES for procrastinate to
  re-drive and writes no terminal event,
* an exhausted budget writes the typed ``sandbox_recycle_failed`` terminal
  event (so the request has an observable, redrivable outcome),
* success writes exactly one ``sandbox_recycled`` event.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from procrastinate import RetryStrategy
from procrastinate.jobs import Job

from aios.harness import tasks
from aios.models.events import SANDBOX_RECYCLE_FAILED_EVENT, SANDBOX_RECYCLED_EVENT


def _job(attempts: int) -> Job:
    """A REAL ``procrastinate.jobs.Job`` — the retry strategy reads it directly."""
    return Job(
        id=1,
        status="doing",
        queue="sessions",
        lock="sess_1",
        queueing_lock="recycle:evt_1",
        task_name="harness.recycle_sandbox",
        task_kwargs={"session_id": "sess_1", "requested_by": "self"},
        attempts=attempts,
    )


def _context_for(job: Job) -> Any:
    context = MagicMock()
    context.job = job
    return context


def _job_context(attempts: int) -> Any:
    return _context_for(_job(attempts))


def _patches(
    events: list[dict[str, Any]],
    *,
    recycle: Any,
    provision: Any = None,
) -> Any:
    registry = MagicMock()
    registry.recycle = recycle
    registry.get_or_provision = provision or AsyncMock()
    inflight = MagicMock()

    async def _append(
        _pool: Any, _session_id: str, _kind: str, data: dict[str, Any], **_k: Any
    ) -> None:
        events.append(data)

    return (
        patch("aios.harness.runtime.require_pool", return_value=MagicMock()),
        patch("aios.harness.runtime.require_sandbox_registry", return_value=registry),
        patch("aios.harness.runtime.require_inflight_tool_registry", return_value=inflight),
        patch(
            "aios.services.sessions.load_session_account_id",
            AsyncMock(return_value="acct_1"),
        ),
        patch("aios.services.sessions.append_event", _append),
    )


def test_recycle_task_is_not_one_shot() -> None:
    """The task must retry — a transient failure may not consume the request."""
    strategy = tasks.recycle_sandbox.retry_strategy
    assert strategy, "recycle_sandbox must not be registered with retry=False"
    assert isinstance(strategy, RetryStrategy)
    assert strategy.max_attempts == tasks._RECYCLE_MAX_ATTEMPTS > 1
    assert strategy.exponential_wait > 0, "retries must back off, not hammer the daemon"


@pytest.mark.asyncio
async def test_transient_failure_retries_without_terminal_event() -> None:
    """An early-attempt backend failure re-raises (re-drive) and stays non-terminal."""
    events: list[dict[str, Any]] = []
    boom = AsyncMock(side_effect=RuntimeError("docker daemon hiccup"))
    p = _patches(events, recycle=boom)
    with p[0], p[1], p[2], p[3], p[4], pytest.raises(RuntimeError):
        await tasks.recycle_sandbox(_job_context(attempts=0), "sess_1", "self")

    assert events == [], "a retryable attempt must not write a terminal outcome"


@pytest.mark.asyncio
async def test_exhausted_retries_record_typed_terminal_failure() -> None:
    """Finding 2: the last attempt records a durable, redrivable failure event."""
    events: list[dict[str, Any]] = []
    boom = AsyncMock(side_effect=RuntimeError("container removal failed"))
    p = _patches(events, recycle=boom)
    with p[0], p[1], p[2], p[3], p[4], pytest.raises(RuntimeError):
        await tasks.recycle_sandbox(
            _job_context(attempts=tasks._RECYCLE_MAX_ATTEMPTS), "sess_1", "self"
        )

    assert len(events) == 1
    assert events[0]["event"] == SANDBOX_RECYCLE_FAILED_EVENT
    assert events[0]["requested_by"] == "self"
    assert events[0]["attempts"] == tasks._RECYCLE_MAX_RUNS
    assert "container removal failed" in events[0]["error"]


@pytest.mark.asyncio
async def test_provision_failure_after_teardown_is_also_terminal() -> None:
    """A FRESH-PROVISION failure (teardown already done) gets the same treatment.

    This is the reported worst case: the writable layer is already discarded,
    so silently dropping the job would leave the session with no sandbox and
    no journal record of why.
    """
    events: list[dict[str, Any]] = []
    p = _patches(
        events,
        recycle=AsyncMock(),
        provision=AsyncMock(side_effect=RuntimeError("provision failed: image pull")),
    )
    with p[0], p[1], p[2], p[3], p[4], pytest.raises(RuntimeError):
        await tasks.recycle_sandbox(
            _job_context(attempts=tasks._RECYCLE_MAX_ATTEMPTS), "sess_1", "operator"
        )

    assert [e["event"] for e in events] == [SANDBOX_RECYCLE_FAILED_EVENT]
    assert "provision failed" in events[0]["error"]


@pytest.mark.asyncio
async def test_success_records_single_recycled_event() -> None:
    events: list[dict[str, Any]] = []
    p = _patches(events, recycle=AsyncMock())
    with p[0], p[1], p[2], p[3], p[4]:
        await tasks.recycle_sandbox(_job_context(attempts=0), "sess_1", "operator")

    assert [e["event"] for e in events] == [SANDBOX_RECYCLED_EVENT]
    assert events[0]["requested_by"] == "operator"


@pytest.mark.asyncio
async def test_retry_ladder_emits_exactly_one_terminal_failure() -> None:
    """Finding 1: drive the REAL strategy to exhaustion; exactly one terminal event.

    This is the test the previous rounds lacked: rather than fabricating a
    single context and asserting on it, it simulates the whole ladder the way
    procrastinate 3.8.1 actually runs it, and asserts the invariant across it.

    Attempt accounting under procrastinate 3.8.1 (the bug the previous code
    had):

    * ``procrastinate_fetch_job_v2`` does not touch ``attempts``, so the first
      run of a job observes ``job.attempts == 0``.
    * On failure the worker calls ``RetryStrategy.get_retry_decision()``, which
      returns ``None`` (no retry) iff ``job.attempts >= max_attempts`` — i.e. it
      keeps retrying while ``job.attempts < max_attempts``.
    * ``procrastinate_retry_job_v*`` (and ``InMemoryConnector.retry_job``) do
      ``attempts = attempts + 1`` when scheduling that retry.

    So the run sequence is ``attempts`` = 0, 1, ..., max_attempts, and only the
    LAST one is terminal. The old predicate (``attempts + 1 >= max_attempts``,
    i.e. ``attempts == max_attempts - 1``) fired one run EARLY on a run that was
    still retried, then fired again on the real final run — two terminal events,
    the first premature.
    """
    events: list[dict[str, Any]] = []
    boom = AsyncMock(side_effect=RuntimeError("docker daemon down"))
    p = _patches(events, recycle=boom)

    runs = 0
    attempts = 0  # a fresh job's ``attempts`` column
    while True:
        runs += 1
        job = _job(attempts)
        with p[0], p[1], p[2], p[3], p[4]:
            try:
                await tasks.recycle_sandbox(_context_for(job), "sess_1", "self", "evt_1")
            except RuntimeError as exc:
                # Exactly what procrastinate's worker does with the raised exception.
                decision = tasks.recycle_retry.get_retry_decision(exception=exc, job=job)
            else:  # pragma: no cover - the backend always fails here
                raise AssertionError("recycle unexpectedly succeeded")
        if decision is None:
            break  # job goes to the failed table; no further run
        attempts += 1  # the connector's ``attempts = attempts + 1`` on retry
        assert runs < 20, "retry ladder did not terminate"

    terminal = [e for e in events if e["event"] == SANDBOX_RECYCLE_FAILED_EVENT]
    assert len(terminal) == 1, (
        f"expected exactly one {SANDBOX_RECYCLE_FAILED_EVENT} across the whole "
        f"retry ladder, got {len(terminal)}: {events}"
    )
    assert events == terminal, "no other lifecycle event may be written on the failure path"
    # ...and it is emitted on the genuinely LAST run, not one early.
    assert runs == tasks._RECYCLE_MAX_RUNS == tasks._RECYCLE_MAX_ATTEMPTS + 1
    assert terminal[0]["attempts"] == runs
    assert terminal[0]["request_id"] == "evt_1"
    assert "docker daemon down" in terminal[0]["error"]


@pytest.mark.asyncio
async def test_penultimate_attempt_is_still_retried_and_silent() -> None:
    """The run the old code called terminal is in fact retried — so it must be quiet."""
    job = _job(attempts=tasks._RECYCLE_MAX_ATTEMPTS - 1)
    events: list[dict[str, Any]] = []
    exc = RuntimeError("still transient")
    p = _patches(events, recycle=AsyncMock(side_effect=exc))
    with p[0], p[1], p[2], p[3], p[4], pytest.raises(RuntimeError):
        await tasks.recycle_sandbox(_context_for(job), "sess_1", "self", "evt_1")

    assert tasks.recycle_retry.get_retry_decision(exception=exc, job=job) is not None, (
        "procrastinate DOES retry this run (attempts < max_attempts), so the task "
        "must not have declared it terminal"
    )
    assert events == [], "a run that will be retried must not write a terminal outcome"


def test_final_attempt_predicate_matches_the_library_decision() -> None:
    """``_is_final_attempt`` is exactly ``get_retry_decision() is None``, for all attempts."""
    exc = RuntimeError("boom")
    for attempts in range(0, tasks._RECYCLE_MAX_ATTEMPTS + 3):
        job = _job(attempts)
        expected = tasks.recycle_retry.get_retry_decision(exception=exc, job=job) is None
        assert tasks._is_final_attempt(job, exc) is expected, attempts
    # The concrete boundary, spelled out so a regression is legible:
    assert not tasks._is_final_attempt(_job(tasks._RECYCLE_MAX_ATTEMPTS - 1), exc)
    assert tasks._is_final_attempt(_job(tasks._RECYCLE_MAX_ATTEMPTS), exc)


@pytest.mark.asyncio
async def test_success_stamps_request_identity() -> None:
    events: list[dict[str, Any]] = []
    p = _patches(events, recycle=AsyncMock())
    with p[0], p[1], p[2], p[3], p[4]:
        await tasks.recycle_sandbox(_job_context(attempts=0), "sess_1", "operator", "evt_9")

    assert [e["event"] for e in events] == [SANDBOX_RECYCLED_EVENT]
    assert events[0]["request_id"] == "evt_9"
    assert events[0]["attempts"] == 1
