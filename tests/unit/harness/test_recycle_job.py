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

from aios.harness import tasks
from aios.models.events import SANDBOX_RECYCLE_FAILED_EVENT, SANDBOX_RECYCLED_EVENT


def _job_context(attempts: int) -> Any:
    context = MagicMock()
    context.job.attempts = attempts
    return context


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
            _job_context(attempts=tasks._RECYCLE_MAX_ATTEMPTS - 1), "sess_1", "self"
        )

    assert len(events) == 1
    assert events[0]["event"] == SANDBOX_RECYCLE_FAILED_EVENT
    assert events[0]["requested_by"] == "self"
    assert events[0]["attempts"] == tasks._RECYCLE_MAX_ATTEMPTS
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
            _job_context(attempts=tasks._RECYCLE_MAX_ATTEMPTS - 1), "sess_1", "operator"
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
