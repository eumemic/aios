"""Unit tests: ``_emit_pending_snapshot_reset_notice`` retires archived markers.

Pre-fix the production delivery path ignored the ``delivered`` result and
called ``queries.unscoped_deliver_pending_snapshot_reset_notice`` for its side
effect alone. When the session was archived between the durable outbox listing
and the claim, the claim's ``append_event`` raised ``NotFoundError`` (it fences
on ``archived_at IS NULL``); the delivery transaction rolled back without
clearing the marker, and ``_flush_pending_snapshot_reset_notices``'s broad
``except Exception`` swallowed the error while "leaving the marker intact" —
retrying the same doomed delivery every hourly GC tick forever and aborting
the synchronous emit's GC tick the one time the race landed there.

The fix captures the ``delivered`` result and, on a claim miss, retires the
marker when the session is archived (a permanent undeliverable) instead of
looping forever. A transient miss (claim lost to a concurrent worker; session
still live) is left for the next flush to retry. The lightweight unit seam
(``conn`` without ``transaction``) is unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from aios.harness import runtime
from aios.sandbox.registry import SANDBOX_FS_RESET_EVENT, SandboxRegistry
from tests.helpers.sandbox import FakeBackend, FakePool

_REASON = "snapshot_pool_pressure"
_NOW_RECLAIM = datetime(2026, 6, 10, tzinfo=UTC) - timedelta(days=30)


class _NoopTxn:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: Any) -> bool:
        return False


class _TxnConn:
    """asyncpg-conn stand-in exposing ``transaction`` (so the production
    delivery branch is taken) and a scripted ``fetchval`` for the archived
    check. ``transaction`` is never actually entered — the deliver/clear calls
    are mocked — it only needs to exist as an attribute."""

    def __init__(self, *, archived: bool | None) -> None:
        self.fetchval = AsyncMock(return_value=archived)

    def transaction(self) -> _NoopTxn:
        return _NoopTxn()


class _ScriptedAcquire:
    def __init__(self, conn: _TxnConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _TxnConn:
        return self._conn

    async def __aexit__(self, *args: Any) -> None:
        return None


class _ScriptedPool:
    def __init__(self, conn: _TxnConn) -> None:
        self.conn = conn
        self._acquire = _ScriptedAcquire(conn)

    def acquire(self) -> _ScriptedAcquire:
        return self._acquire


@pytest.fixture
def scripted_pool() -> Any:
    """Install a scripted pool whose conn takes the production delivery branch.

    ``runtime.pool`` is restored on teardown; a test rewrites the conn's
    archived verdict by mutating ``scripted_pool.conn.fetchval``'s return value.
    """
    prev = runtime.pool
    pool = _ScriptedPool(_TxnConn(archived=None))
    runtime.pool = cast(Any, pool)
    try:
        yield pool
    finally:
        runtime.pool = prev


def _patch_outbox(
    monkeypatch: pytest.MonkeyPatch,
    *,
    deliver: AsyncMock,
    clear: AsyncMock,
) -> None:
    monkeypatch.setattr(
        "aios.sandbox.registry.queries.unscoped_deliver_pending_snapshot_reset_notice",
        deliver,
    )
    monkeypatch.setattr(
        "aios.sandbox.registry.queries.unscoped_clear_pending_snapshot_reset_notice",
        clear,
    )


@pytest.mark.asyncio
async def test_emit_retires_marker_when_delivery_fails_on_archived(
    scripted_pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A claim miss on an archived session retires the marker so the outbox
    stops retrying a notice whose session can no longer accept the event."""
    scripted_pool.conn.fetchval = AsyncMock(return_value=True)
    deliver = AsyncMock(return_value=False)
    clear = AsyncMock(return_value=True)
    _patch_outbox(monkeypatch, deliver=deliver, clear=clear)

    registry = SandboxRegistry(backend=FakeBackend())
    registry._pending_snapshot_reset_notices["sess_arch"] = _REASON

    # Must not raise — the archived terminal state is retired, not propagated.
    await registry._emit_pending_snapshot_reset_notice("sess_arch", _REASON)

    deliver.assert_awaited_once()
    call = deliver.await_args
    assert call is not None
    assert call.args[0] is scripted_pool.conn
    assert call.args[1] == "sess_arch"
    assert call.kwargs == {"expected_reason": _REASON}
    # The archived check ran and the marker was retired in the same pool scope.
    assert scripted_pool.conn.fetchval.await_count == 1
    clear.assert_awaited_once()
    clear_call = clear.await_args
    assert clear_call is not None
    assert clear_call.args[1] == "sess_arch"
    assert clear_call.kwargs == {"expected_reason": _REASON}
    # The process-local mirror is popped regardless.
    assert "sess_arch" not in registry._pending_snapshot_reset_notices


@pytest.mark.asyncio
async def test_emit_does_not_retire_when_delivery_lost_to_concurrent_worker(
    scripted_pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A claim miss on a STILL-LIVE session is a transient race (another worker
    owns the row or already delivered it); the marker stays set for the next
    flush to retry and is NOT cleared (that would clobber a concurrent
    delivery's pending commit)."""
    scripted_pool.conn.fetchval = AsyncMock(return_value=False)
    deliver = AsyncMock(return_value=False)
    clear = AsyncMock(return_value=True)
    _patch_outbox(monkeypatch, deliver=deliver, clear=clear)

    registry = SandboxRegistry(backend=FakeBackend())
    registry._pending_snapshot_reset_notices["sess_live"] = _REASON

    await registry._emit_pending_snapshot_reset_notice("sess_live", _REASON)

    deliver.assert_awaited_once()
    assert scripted_pool.conn.fetchval.await_count == 1
    clear.assert_not_awaited()
    # The process-local mirror is still popped (another worker owns delivery).
    assert "sess_live" not in registry._pending_snapshot_reset_notices


@pytest.mark.asyncio
async def test_emit_does_not_check_archived_on_successful_delivery(
    scripted_pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a successful claim the deliver function clears the marker in its own
    transaction, so neither the archived check nor the retire runs."""
    deliver = AsyncMock(return_value=True)
    clear = AsyncMock(return_value=True)
    _patch_outbox(monkeypatch, deliver=deliver, clear=clear)

    registry = SandboxRegistry(backend=FakeBackend())
    registry._pending_snapshot_reset_notices["sess_ok"] = _REASON

    await registry._emit_pending_snapshot_reset_notice("sess_ok", _REASON)

    deliver.assert_awaited_once()
    assert scripted_pool.conn.fetchval.await_count == 0
    clear.assert_not_awaited()
    assert "sess_ok" not in registry._pending_snapshot_reset_notices


@pytest.mark.asyncio
async def test_emit_uses_event_writer_seam_when_conn_has_no_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lightweight unit callers (``FakePool`` conn has no ``transaction``)
    keep the event-writer seam: the production deliver/clear queries are NOT
    awaited and the seam appends the lifecycle event. Regression guard for the
    branch that the ``hasattr(conn, 'transaction')`` check selects."""
    prev = runtime.pool
    runtime.pool = cast(Any, FakePool())
    try:
        deliver = AsyncMock(return_value=True)
        clear = AsyncMock(return_value=True)
        _patch_outbox(monkeypatch, deliver=deliver, clear=clear)

        registry = SandboxRegistry(backend=FakeBackend())
        registry._append_fs_event = AsyncMock()  # type: ignore[method-assign]
        registry._pending_snapshot_reset_notices["sess_unit"] = _REASON

        await registry._emit_pending_snapshot_reset_notice("sess_unit", _REASON)

        deliver.assert_not_awaited()
        clear.assert_not_awaited()
        registry._append_fs_event.assert_awaited_once_with(
            "sess_unit", SANDBOX_FS_RESET_EVENT, {"reason": _REASON}
        )
        assert "sess_unit" not in registry._pending_snapshot_reset_notices
    finally:
        runtime.pool = prev


@pytest.mark.asyncio
async def test_reclaim_sync_emit_does_not_abort_tick_when_archived_after_fresh_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synchronous emit inside ``_reclaim_pool_candidate`` (run after a
    pointer is cleared) must not abort the GC tick when the session is archived
    between the fresh re-read and the claim. The fresh re-read sees a live
    session (so reclaim proceeds to remove the image), but the claim then misses
    on the now-archived row: the emit retires the marker instead of raising
    ``NotFoundError`` out of the tick handler.

    Pre-fix this raised ``NotFoundError`` → ``sandbox.gc_tick_failed`` and one
    hour of reclamation lost per pinned marker.
    """
    from aios.config import get_settings
    from aios.sandbox.backends.base import ManagedImage
    from aios.sandbox.registry import GcImageVerdict, SessionSnapshotState
    from aios.sandbox.spec import snapshot_tag

    instance_id = get_settings().instance_id
    session_id = "sess_race"
    tag = snapshot_tag(instance_id, session_id)
    # Fresh re-read sees a LIVE session; the archive happens only AFTER this
    # read, so the reclaim path proceeds to remove the image and arm the marker.
    live_state = SessionSnapshotState(
        session_id=session_id,
        account_id="acct",
        archived_at=None,
        last_event_at=_NOW_RECLAIM,
        snapshot_ref=tag,
        snapshot_host=instance_id,
        snapshot_bytes=1_000_000,
    )

    prev = runtime.pool
    pool = _ScriptedPool(_TxnConn(archived=True))  # archived claim miss
    runtime.pool = cast(Any, pool)
    try:
        deliver = AsyncMock(return_value=False)  # claim lost (archived guard)
        clear = AsyncMock(return_value=True)
        _patch_outbox(monkeypatch, deliver=deliver, clear=clear)

        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        registry._fresh_session_state = AsyncMock(return_value=live_state)  # type: ignore[method-assign]
        registry._remove_canonical_image_and_clear_pointer = AsyncMock(return_value=True)  # type: ignore[method-assign]
        registry._append_fs_event = AsyncMock()  # type: ignore[method-assign]
        # No pre-existing local mirror: the early re-emit arm is skipped, so
        # the only emit is the synchronous one after the pointer is cleared.
        assert session_id not in registry._pending_snapshot_reset_notices

        verdict = GcImageVerdict(
            image=ManagedImage(
                image_id=f"img-{session_id}",
                repo_tags=(tag,),
                parent_id=None,
                size_bytes=2_000_000,
                labels={
                    "aios.managed": "true",
                    "aios.instance_id": instance_id,
                    "aios.session_id": session_id,
                },
            ),
            session_id=session_id,
            is_canonical=True,
            removal_ref=tag,
            verdict="retain",
            reason="protected_live",
        )

        # Must not raise — the synchronous emit retires the archived marker.
        removed = await registry._reclaim_pool_candidate(
            verdict, {session_id: live_state}, instance_id
        )

        assert removed is True
        # The synchronous emit ran the production claim, detected the archive,
        # and retired the marker — all without raising into the tick handler.
        deliver.assert_awaited_once()
        assert pool.conn.fetchval.await_count == 1
        clear.assert_awaited_once()
        # The production branch (scripted conn has ``transaction``) was taken,
        # so the event-writer seam was NOT used.
        registry._append_fs_event.assert_not_awaited()
        assert session_id not in registry._pending_snapshot_reset_notices
    finally:
        runtime.pool = prev
