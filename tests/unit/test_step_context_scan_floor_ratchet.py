"""Unit coverage for the step prelude's perf-only scan-floor ratchet (#1747).

``_advance_open_request_scan_floor_best_effort`` MUST NEVER let a failure in
the ratchet propagate out and fail the step — it's a perf-only optimization,
not a correctness path. These tests exercise that swallow behavior directly,
independent of the ``tests/unit/conftest.py`` autouse stub (which every other
unit test relies on to avoid driving this leaf over a bare ``MagicMock``
connection).

End-to-end "the floor actually advances" coverage lives in the integration
tier (``tests/integration/test_open_request_scan_floor.py::
test_step_prelude_ratchets_floor_end_to_end``).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.harness import step_context

pytestmark = pytest.mark.asyncio

# ``tests/unit/conftest.py`` installs an autouse ``mock.patch`` over
# ``aios.harness.step_context._advance_open_request_scan_floor_best_effort``
# itself (every other unit test needs that stub to avoid driving this leaf
# over a bare ``MagicMock`` connection). That patch replaces the *module
# attribute* for the duration of each test, so a same-named lookup
# (``step_context._advance_open_request_scan_floor_best_effort``) inside a
# test body would hit the autouse mock, not the function under test. Capture
# the real implementation once at collection time — before any test (and
# hence any per-test patch) has run — so these tests exercise the genuine
# swallow/savepoint logic instead of the no-op stand-in.
_real_advance = step_context._advance_open_request_scan_floor_best_effort


class _FakeTransactionCtx:
    """Async context manager standing in for ``conn.transaction()``."""

    def __init__(self, *, raise_on_enter: bool = False) -> None:
        self._raise_on_enter = raise_on_enter

    async def __aenter__(self) -> _FakeTransactionCtx:
        if self._raise_on_enter:
            raise RuntimeError("boom: savepoint could not start")
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


async def test_advance_swallows_query_failure_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = MagicMock()
    conn.transaction = MagicMock(return_value=_FakeTransactionCtx())

    failing_advance = AsyncMock(side_effect=RuntimeError("boom: UPDATE failed"))
    monkeypatch.setattr(
        "aios.db.queries.advance_open_request_scan_floor",
        failing_advance,
    )

    # Must not raise.
    await _real_advance(conn, "sess_x", account_id="acc_x")

    failing_advance.assert_awaited_once_with(conn, "sess_x", account_id="acc_x")


async def test_advance_swallows_transaction_enter_failure_without_raising() -> None:
    """Even a savepoint that fails to open (not just the UPDATE inside it)
    must not propagate — the whole point of the try/except wrapping the
    ``async with conn.transaction()`` block."""
    conn = MagicMock()
    conn.transaction = MagicMock(return_value=_FakeTransactionCtx(raise_on_enter=True))

    await _real_advance(conn, "sess_x", account_id="acc_x")


async def test_advance_increments_module_swallow_counter_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = MagicMock()
    conn.transaction = MagicMock(return_value=_FakeTransactionCtx())
    monkeypatch.setattr(
        "aios.db.queries.advance_open_request_scan_floor",
        AsyncMock(side_effect=RuntimeError("boom")),
    )
    monkeypatch.setattr(step_context, "_open_request_scan_floor_advance_swallow_count", 0)

    await _real_advance(conn, "sess_x", account_id="acc_x")
    assert step_context._open_request_scan_floor_advance_swallow_count == 1

    await _real_advance(conn, "sess_x", account_id="acc_x")
    assert step_context._open_request_scan_floor_advance_swallow_count == 2


async def test_advance_calls_query_on_success_and_does_not_swallow_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy path still calls through to the real query inside the
    savepoint — the try/except must not mask a working call as a failure."""
    conn = MagicMock()
    conn.transaction = MagicMock(return_value=_FakeTransactionCtx())

    ok_advance = AsyncMock(return_value=None)
    monkeypatch.setattr("aios.db.queries.advance_open_request_scan_floor", ok_advance)
    monkeypatch.setattr(step_context, "_open_request_scan_floor_advance_swallow_count", 0)

    await _real_advance(conn, "sess_x", account_id="acc_x")

    ok_advance.assert_awaited_once_with(conn, "sess_x", account_id="acc_x")
    assert step_context._open_request_scan_floor_advance_swallow_count == 0
