"""Writer↔reader coupling for the errored-session park (#1084).

The error latch (``harness/loop.py:_latch_errored_turn``) writes a
``turn_ended`` lifecycle event whose ``status``/``stop_reason`` the sweep's
errored-park predicate reads back to bump ``last_error_seq`` and skip the
session. That value round-trips writer → ``json.dumps`` → Postgres JSONB →
asyncpg → ``dict[str, Any]``: the read is type ``Any``, so the type checker
*cannot* bind the write literal to the read literal. Two free strings on either
side of that boundary type-check and pass CI even when they DIVERGE — flip the
writer to ``stop_reason='errored'`` (the tempting transposition matching
``status``) and the park silently breaks, busy-waking the session forever
(#155 class).

These tests are the FLOOR: they assert the latch's actual write is exactly what
the park predicate accepts, so a future literal flip on either side fails
deterministically — the coupling the ``Any`` read would otherwise evaporate.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.harness import loop
from aios.models.events import (
    ERRORED_LIFECYCLE_STATUS,
    ERRORED_LIFECYCLE_STOP_REASON,
    is_errored_lifecycle_event,
)
from aios.services import sessions as sessions_service


async def _capture_latch_lifecycle_event() -> dict[str, Any]:
    """Run the real error latch with its DB deps mocked and return the exact
    lifecycle ``data`` payload it appends.

    Nothing here re-states the literal: the payload is whatever
    ``_latch_errored_turn`` → ``_append_lifecycle`` → ``append_event`` actually
    writes, so a flip of the writer literal flows straight into the assertions.
    """
    appended: list[dict[str, Any]] = []

    async def _record_append(
        pool: Any, session_id: str, kind: str, data: dict[str, Any], **kwargs: Any
    ) -> Any:
        if kind == "lifecycle":
            appended.append(data)
        return None

    with (
        patch.object(loop, "fail_all_open_requests", new=AsyncMock(return_value=None)),
        patch.object(sessions_service, "set_session_stop_reason", new=AsyncMock(return_value=None)),
        patch.object(sessions_service, "append_event", side_effect=_record_append),
    ):
        await loop._latch_errored_turn(
            pool=object(),
            session_id="ses_x",
            error_kind="test_error",
            account_id="acc_test",
        )

    lifecycle_events = [d for d in appended if d.get("event") == "turn_ended"]
    assert len(lifecycle_events) == 1, (
        "the error latch must append exactly one turn_ended lifecycle event"
    )
    return lifecycle_events[0]


@pytest.mark.asyncio
async def test_error_latch_write_is_read_by_errored_park_predicate() -> None:
    """The CORE coupling: the lifecycle event the latch writes is recognized by
    the sweep's errored-park predicate.

    If a future edit flips the writer literal (e.g.
    ``stop_reason='errored'``) WITHOUT flipping the reader, this fails — the
    regression the bug describes (park silently breaks) is caught here even
    though both literals would type-check across the JSONB ``Any`` boundary.
    """
    data = await _capture_latch_lifecycle_event()

    assert is_errored_lifecycle_event("lifecycle", data), (
        "the lifecycle event the error latch writes must be recognized by the "
        "errored-park predicate (is_errored_lifecycle_event); a divergence "
        "between the writer and reader literals silently busy-wakes the session "
        "forever (#1084 / #155 class)"
    )


@pytest.mark.asyncio
async def test_error_latch_uses_shared_lifecycle_constants() -> None:
    """The latch must write the SHARED constants, not re-stated free strings —
    so the write and the read have one source of truth.
    """
    data = await _capture_latch_lifecycle_event()

    assert data["stop_reason"] == ERRORED_LIFECYCLE_STOP_REASON
    assert data["status"] == ERRORED_LIFECYCLE_STATUS


def test_errored_park_predicate_rejects_non_error_stop_reasons() -> None:
    """Guard the predicate's discrimination: only the error stop_reason latches.

    Without this, a predicate that returned True for every lifecycle event would
    satisfy the coupling test above yet wrongly park (e.g.) rescheduling turns.
    """
    assert not is_errored_lifecycle_event(
        "lifecycle", {"event": "turn_ended", "stop_reason": "rescheduling"}
    )
    assert not is_errored_lifecycle_event("lifecycle", {"event": "turn_started"})
    # A non-lifecycle event carrying the same string must NOT park.
    assert not is_errored_lifecycle_event("message", {"stop_reason": ERRORED_LIFECYCLE_STOP_REASON})


def test_status_and_stop_reason_constants_are_distinct() -> None:
    """The transposition the bug names: ``status='errored'`` and
    ``stop_reason='error'`` are DIFFERENT strings. Pinning them distinct keeps a
    careless "make them match" edit from collapsing the two fields.
    """
    # Compare via ``str`` so mypy does not narrow these distinct ``Literal``
    # types to a statically-known (non-overlapping) result and reject the
    # equality check — the assertion guards a runtime invariant, not a type one.
    assert str(ERRORED_LIFECYCLE_STOP_REASON) != str(ERRORED_LIFECYCLE_STATUS)
    assert ERRORED_LIFECYCLE_STOP_REASON == "error"
    assert ERRORED_LIFECYCLE_STATUS == "errored"
