from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from aios.harness.production_watchdogs import (
    _CLEANUP_GRACE_SECONDS,
    HeldConnectionWatchdog,
    StandingSessionFilesystemProbe,
    ThroughputDeadMan,
    _build_filesystem_probe_command,
    _DeadlineExceeded,
)


class _Holder:
    def __init__(self, connection: object) -> None:
        self._in_use = object()
        self._con = connection


class _Pool:
    def __init__(self, holder: _Holder) -> None:
        self._holders = [holder]

    def get_size(self) -> int:
        return 1

    def get_idle_size(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_held_connection_watchdog_captures_parked_holder(tmp_path: Path) -> None:
    parked = asyncio.Event()
    connection = object()

    async def holder() -> None:
        conn = connection
        await parked.wait()
        assert conn

    task = asyncio.create_task(holder(), name="session:test-session")
    await asyncio.sleep(0)
    inspector = AsyncMock(return_value=[{"pid": 42, "state": "idle"}])
    journal = AsyncMock(return_value=[{"seq": 7, "kind": "span"}])
    alarm = MagicMock()
    watchdog = HeldConnectionWatchdog(
        _Pool(_Holder(connection)),
        threshold_seconds=0,
        rate_limit_seconds=60,
        specimen_dir=tmp_path,
        inspect_pg=inspector,
        load_journal=journal,
        alarm=alarm,
    )

    specimens = await watchdog.check_once()

    assert len(specimens) == 1
    assert specimens[0]["owner_id"] == "test-session"
    assert "await parked.wait()" in "\n".join(specimens[0]["task_stack"])
    assert specimens[0]["journal_events"] == [{"seq": 7, "kind": "span"}]
    path = next(
        iter(await asyncio.to_thread(lambda: list(tmp_path.glob("held-connection-*.json"))))
    )
    written = json.loads(await asyncio.to_thread(path.read_text))
    assert written["pg_stat_activity"] == [{"pid": 42, "state": "idle"}]
    alarm.assert_called_once()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_dead_man_alarms_only_when_claimed_work_stalls() -> None:
    alarm = MagicMock()
    monitor = ThroughputDeadMan(threshold_seconds=600, rate_limit_seconds=60, alarm=alarm)

    assert not monitor.observe(claimed=0, completed=0, now=0)
    assert not monitor.observe(claimed=1, completed=0, now=1)
    assert monitor.observe(claimed=1, completed=0, now=602)
    assert not monitor.observe(claimed=1, completed=0, now=603)
    assert not monitor.observe(claimed=1, completed=1, now=604)
    assert alarm.call_count == 1


@pytest.mark.asyncio
async def test_held_connection_watchdog_matches_asyncpg_proxy(tmp_path: Path) -> None:
    """A borrower frame holds a proxy while PoolConnectionHolder holds raw con."""
    parked = asyncio.Event()
    raw = object()
    holder_obj = _Holder(raw)

    # Use asyncpg's production proxy type. Bypass its constructor only because
    # the raw connection is deliberately inert; preserve its real slots/layout.
    proxy = asyncpg.pool.PoolConnectionProxy.__new__(asyncpg.pool.PoolConnectionProxy)
    proxy._con = raw
    proxy._holder = holder_obj

    async def borrower() -> None:
        session_id = "proxy-session"
        conn = proxy
        await parked.wait()
        assert conn and session_id

    task = asyncio.create_task(borrower())
    await asyncio.sleep(0)
    watchdog = HeldConnectionWatchdog(
        _Pool(holder_obj),
        threshold_seconds=0,
        rate_limit_seconds=60,
        specimen_dir=tmp_path,
        inspect_pg=AsyncMock(return_value=[]),
        load_journal=AsyncMock(return_value=[]),
        alarm=MagicMock(),
    )
    specimen = (await watchdog.check_once())[0]
    assert specimen["owner_id"] == "proxy-session"
    assert "await parked.wait()" in "\n".join(specimen["task_stack"])
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_runner_reconnects_after_failed_capture_tick(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Telemetry failure is isolated: a later tick reconnects and runs."""
    import aios.harness.production_watchdogs as module

    class Inspector:
        def __init__(self, *, fail: bool) -> None:
            self.fail = fail
            self.closed = False
            self.dead_man_seen = asyncio.Event()

        def is_closed(self) -> bool:
            return self.closed

        async def close(self) -> None:
            self.closed = True

        async def fetch(
            self, query: str, *args: object, **kwargs: object
        ) -> list[dict[str, object]]:
            if self.fail:
                raise OSError("telemetry database disconnected")
            return []

        async def fetchrow(self, query: str, *args: object, **kwargs: object) -> dict[str, int]:
            self.dead_man_seen.set()
            return {"claimed": 0, "completed": 0}

    first, second = Inspector(fail=True), Inspector(fail=False)
    connect = AsyncMock(side_effect=[first, second])
    monkeypatch.setattr(asyncpg, "connect", connect)
    raw = object()
    runner = asyncio.create_task(
        module.run_production_watchdogs(
            _Pool(_Holder(raw)),
            "postgresql://example/db",
            held_threshold_seconds=0,
            dead_man_threshold_seconds=10,
            interval_seconds=0.001,
            rate_limit_seconds=10,
            specimen_dir=tmp_path,
            journal_limit=10,
            operation_timeout_seconds=0.1,
        )
    )
    await asyncio.wait_for(second.dead_man_seen.wait(), 1)
    assert connect.await_count == 2
    assert first.closed
    assert not runner.done()
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner


@pytest.mark.asyncio
async def test_runner_counts_session_and_workflow_wake_completions_symmetrically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Healthy workflow wakes count as throughput even when the run parks."""
    import aios.harness.production_watchdogs as module

    class Inspector:
        def __init__(self) -> None:
            self.closed = False
            self.queries: list[str] = []
            self.observed = asyncio.Event()

        def is_closed(self) -> bool:
            return self.closed

        async def close(self) -> None:
            self.closed = True

        async def fetch(self, query: str, *args: object, **kwargs: object) -> list[object]:
            return []

        async def fetchrow(self, query: str, *args: object, **kwargs: object) -> dict[str, int]:
            self.queries.append(query)
            self.observed.set()
            return {"claimed": 1, "completed": 1}

    inspector = Inspector()
    monkeypatch.setattr(asyncpg, "connect", AsyncMock(return_value=inspector))
    runner = asyncio.create_task(
        module.run_production_watchdogs(
            _Pool(_Holder(object())),
            "postgresql://example/db",
            held_threshold_seconds=999,
            dead_man_threshold_seconds=0.001,
            interval_seconds=0.001,
            rate_limit_seconds=10,
            specimen_dir=tmp_path,
            journal_limit=10,
            operation_timeout_seconds=0.1,
        )
    )
    await asyncio.wait_for(inspector.observed.wait(), 1)
    query = inspector.queries[-1]
    assert "procrastinate_events" in query
    assert "harness.wake_session" in query
    assert "harness.wake_workflow" in query
    assert "run_completed" not in query
    assert not runner.done()
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner


def _cold_registry(return_value: object = None) -> MagicMock:
    """Build a mock registry where peek() returns None (cold) and get_or_provision succeeds.

    When ``return_value`` is None a fresh sentinel object is created so that
    generation-token identity checks work correctly.

    ``peek()`` behaviour: returns ``None`` on the first call (cold), then
    ``handle`` on every subsequent call.  This supports both single-call and
    multi-call-to-check_once tests without exhausting a fixed side_effect list.
    """
    handle = object() if return_value is None else return_value
    registry = MagicMock()

    # Track call count so the first peek returns None (cold) and all
    # subsequent peeks return the handle (post-provision state).
    _peek_calls = {"n": 0}

    def _peek_side_effect(session_id: str) -> object | None:
        _peek_calls["n"] += 1
        return None if _peek_calls["n"] == 1 else handle

    registry.peek = MagicMock(side_effect=_peek_side_effect)
    registry.get_or_provision = AsyncMock(return_value=handle)
    registry.release = AsyncMock()
    return registry


def _warm_registry(return_value: object = None) -> MagicMock:
    """Build a mock registry where peek() returns a handle (warm / already resident)."""
    handle = object() if return_value is None else return_value
    registry = MagicMock()
    registry.peek = MagicMock(return_value=handle)
    registry.get_or_provision = AsyncMock(return_value=handle)
    registry.release = AsyncMock()
    return registry


@pytest.mark.asyncio
async def test_standing_session_probe_uses_live_sandbox_write_read() -> None:
    """Core workspace write/read probe works without any sentinels configured."""
    from aios.sandbox.backends.base import CommandResult

    registry = _cold_registry()
    registry.exec = AsyncMock(
        return_value=CommandResult(0, "", "", timed_out=False, truncated=False)
    )
    alarm = MagicMock()
    pool = object()
    probe = StandingSessionFilesystemProbe(
        registry,
        pool,
        "sess_canonical",
        rate_limit_seconds=60,
        operation_timeout_seconds=5,
        alarm=alarm,
    )

    assert await probe.check_once(now=0)
    registry.get_or_provision.assert_awaited_once_with("sess_canonical", pool=pool)
    command = registry.exec.await_args.args[1]
    assert "mktemp /workspace/" in command
    assert 'cat "$probe"' in command
    # Without sentinels, no repo/memory checks
    assert ".git/HEAD" not in command
    assert "/mnt/memory" not in command
    alarm.assert_not_called()


@pytest.mark.asyncio
async def test_standing_session_probe_with_configured_sentinels() -> None:
    """When repo_sentinel and memory_sentinel are configured, the probe checks them."""
    from aios.sandbox.backends.base import CommandResult

    registry = _cold_registry()
    registry.exec = AsyncMock(
        return_value=CommandResult(0, "", "", timed_out=False, truncated=False)
    )
    alarm = MagicMock()
    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_canonical",
        rate_limit_seconds=60,
        operation_timeout_seconds=5,
        repo_sentinel=".git/HEAD",
        memory_sentinel="/mnt/memory/store/MEMORY.md",
        alarm=alarm,
    )

    assert await probe.check_once(now=0)
    command = registry.exec.await_args.args[1]
    assert ".git/HEAD" in command
    assert "/mnt/memory/store/MEMORY.md" in command
    alarm.assert_not_called()


@pytest.mark.asyncio
async def test_standing_session_probe_alarms_and_rate_limits_failures() -> None:
    registry = _cold_registry()
    registry.get_or_provision = AsyncMock(side_effect=ValueError("workspace rejected"))
    alarm = MagicMock()
    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_canonical",
        rate_limit_seconds=60,
        operation_timeout_seconds=5,
        alarm=alarm,
    )

    assert not await probe.check_once(now=0)
    assert not await probe.check_once(now=30)
    assert not await probe.check_once(now=61)
    assert alarm.call_count == 2
    assert alarm.call_args.args[0] == "standing_session_filesystem_probe"
    assert alarm.call_args.args[1]["session_id"] == "sess_canonical"


@pytest.mark.asyncio
async def test_standing_session_probe_alarms_on_nonzero_exec() -> None:
    from aios.sandbox.backends.base import CommandResult

    registry = _cold_registry()
    registry.exec = AsyncMock(
        return_value=CommandResult(1, "", "memory mount unreadable", False, False)
    )
    alarm = MagicMock()
    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_canonical",
        rate_limit_seconds=60,
        operation_timeout_seconds=5,
        alarm=alarm,
    )

    assert not await probe.check_once(now=0)
    assert alarm.call_args.args[1]["exit_code"] == 1
    assert "memory mount unreadable" in alarm.call_args.args[1]["stderr"]


@pytest.mark.asyncio
async def test_standing_session_probe_uses_overall_deadline() -> None:
    """The probe must use one overall deadline, not additive per-op waits."""
    from aios.sandbox.backends.base import CommandResult

    provision_time = 0.0

    async def slow_provision(session_id: str, pool: object = None) -> str:
        nonlocal provision_time
        provision_time = asyncio.get_event_loop().time()
        return "handle"

    handle = "handle"
    registry = MagicMock()
    registry.peek = MagicMock(side_effect=[None, handle])
    registry.get_or_provision = AsyncMock(side_effect=slow_provision)
    registry.exec = AsyncMock(
        return_value=CommandResult(0, "", "", timed_out=False, truncated=False)
    )
    registry.release = AsyncMock()
    alarm = MagicMock()
    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_canonical",
        rate_limit_seconds=60,
        operation_timeout_seconds=5,
        alarm=alarm,
    )

    result = await probe.check_once(now=0)
    assert result is True
    # The exec call should have received a timeout <= the overall budget
    exec_call = registry.exec.await_args
    exec_timeout = exec_call.kwargs.get(
        "timeout_seconds", exec_call.args[2] if len(exec_call.args) > 2 else None
    )
    # Timeout should be positive and <= 5 (the overall budget)
    assert exec_timeout is not None
    assert 0 < exec_timeout <= 5


def test_build_filesystem_probe_command_no_sentinels() -> None:
    """Without sentinels, only core workspace write/read assertions."""
    cmd = _build_filesystem_probe_command()
    assert "mktemp" in cmd
    assert "aios-fs-probe" in cmd
    assert ".git" not in cmd
    assert "/mnt/memory" not in cmd


def test_build_filesystem_probe_command_with_sentinels() -> None:
    """Configured sentinels add their specific checks."""
    cmd = _build_filesystem_probe_command(
        repo_sentinel=".git/HEAD",
        memory_sentinel="/mnt/memory/store/MEMORY.md",
    )
    assert "mktemp" in cmd
    assert ".git/HEAD" in cmd
    assert "/mnt/memory/store/MEMORY.md" in cmd


def test_build_filesystem_probe_command_handles_git_worktree() -> None:
    """The repo sentinel check uses head -c 64, handling worktree .git files.

    Sentinel clarification: ``.git/HEAD`` for a normal repo (regular file
    inside the ``.git/`` directory); ``.git`` alone for a worktree checkout
    (where ``.git`` is itself a regular file containing ``gitdir: <path>``).
    """
    cmd = _build_filesystem_probe_command(repo_sentinel="/workspace/repo/.git")
    # Should use test -e (works for both file and directory) and head -c 64
    assert "test -e" in cmd
    assert "head -c 64" in cmd


def test_build_filesystem_probe_command_git_head_sentinel() -> None:
    """Sentinel ``.git/HEAD`` — the normal-repo shape (file inside .git/ dir)."""
    cmd = _build_filesystem_probe_command(repo_sentinel=".git/HEAD")
    assert "test -e" in cmd
    assert ".git/HEAD" in cmd
    assert "head -c 64" in cmd


# ── Ownership-aware lifecycle tests (generation-token based) ────────────


@pytest.mark.asyncio
async def test_warm_preservation_no_release() -> None:
    """When the sandbox is already warm (peek returns a handle), the probe
    must NOT release it — the original consumer owns it."""
    from aios.sandbox.backends.base import CommandResult

    registry = _warm_registry()
    registry.exec = AsyncMock(
        return_value=CommandResult(0, "", "", timed_out=False, truncated=False)
    )

    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_warm",
        rate_limit_seconds=60,
        operation_timeout_seconds=5,
    )

    assert await probe.check_once(now=0)
    # The probe must NOT have called release — the sandbox was warm.
    registry.release.assert_not_awaited()


@pytest.mark.asyncio
async def test_cold_success_cleanup() -> None:
    """When the probe cold-provisions solely for monitoring and succeeds,
    it must release the sandbox after the check."""
    from aios.sandbox.backends.base import CommandResult

    handle = object()
    registry = MagicMock()
    # peek: None before provision, handle after provision (for ownership check)
    registry.peek = MagicMock(side_effect=[None, handle])
    registry.get_or_provision = AsyncMock(return_value=handle)
    registry.exec = AsyncMock(
        return_value=CommandResult(0, "", "", timed_out=False, truncated=False)
    )
    registry.release = AsyncMock()

    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_cold_ok",
        rate_limit_seconds=60,
        operation_timeout_seconds=5,
    )

    assert await probe.check_once(now=0)
    registry.release.assert_awaited_once_with("sess_cold_ok")


@pytest.mark.asyncio
async def test_cold_failure_cleanup() -> None:
    """When the probe cold-provisions and the exec fails (nonzero exit),
    it must still release the sandbox it provisioned."""
    from aios.sandbox.backends.base import CommandResult

    handle = object()
    registry = MagicMock()
    registry.peek = MagicMock(side_effect=[None, handle])
    registry.get_or_provision = AsyncMock(return_value=handle)
    registry.exec = AsyncMock(
        return_value=CommandResult(1, "", "oops", timed_out=False, truncated=False)
    )
    registry.release = AsyncMock()

    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_cold_fail",
        rate_limit_seconds=60,
        operation_timeout_seconds=5,
    )

    assert not await probe.check_once(now=0)
    registry.release.assert_awaited_once_with("sess_cold_fail")


@pytest.mark.asyncio
async def test_cold_provision_error_no_release() -> None:
    """When get_or_provision raises on a cold probe, provision_handle is
    None (never assigned) so _release_owned correctly no-ops.  The probe
    must NOT release because it never received a handle — the generation-
    token check (provision_handle is None → skip) prevents it."""
    handle = object()
    registry = MagicMock()
    # peek returns None initially (cold); after failed provision peek
    # returns handle (simulating a partial cache), but the probe's
    # provision_handle is still None because provision raised.
    registry.peek = MagicMock(side_effect=[None, handle])
    registry.get_or_provision = AsyncMock(side_effect=RuntimeError("provision boom"))
    registry.release = AsyncMock()

    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_provision_err",
        rate_limit_seconds=60,
        operation_timeout_seconds=5,
    )

    assert not await probe.check_once(now=0)
    # Must NOT release — provision_handle is None (provision raised before
    # returning a handle), so _release_owned(None) correctly no-ops.
    registry.release.assert_not_awaited()


@pytest.mark.asyncio
async def test_provisioning_timeout_cancellation_cleanup() -> None:
    """When provisioning times out, the probe must await/settle the in-flight
    provision before cleanup so no overlap/orphan."""
    settled = {"settled": False}

    async def slow_provision(session_id: str, pool: object = None) -> str:
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            settled["settled"] = True
            raise
        return "handle"  # pragma: no cover

    registry = MagicMock()
    # peek returns None (cold); after timeout peek returns None (provision never completed)
    registry.peek = MagicMock(return_value=None)
    registry.get_or_provision = AsyncMock(side_effect=slow_provision)
    registry.release = AsyncMock()

    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_timeout",
        rate_limit_seconds=60,
        operation_timeout_seconds=0.05,
    )

    assert not await probe.check_once(now=0)
    # The in-flight provision must have been settled (cancelled + awaited)
    assert settled["settled"], "provision was not settled before cleanup"


@pytest.mark.asyncio
async def test_exec_timeout_cleanup() -> None:
    """When exec times out on a cold-provisioned sandbox, the probe must
    still release it and settle the exec task first."""
    from aios.sandbox.backends.base import CommandResult

    exec_settled = {"settled": False}

    async def slow_exec(*args: object, **kwargs: object) -> CommandResult:
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            exec_settled["settled"] = True
            raise
        return CommandResult(0, "", "", timed_out=False, truncated=False)  # pragma: no cover

    handle = object()
    registry = MagicMock()
    registry.peek = MagicMock(side_effect=[None, handle])
    registry.get_or_provision = AsyncMock(return_value=handle)
    registry.exec = AsyncMock(side_effect=slow_exec)
    registry.release = AsyncMock()

    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_exec_timeout",
        rate_limit_seconds=60,
        operation_timeout_seconds=0.05,
    )

    assert not await probe.check_once(now=0)
    # Exec must have been settled before release
    assert exec_settled["settled"], "exec task was not settled before release"
    registry.release.assert_awaited_once_with("sess_exec_timeout")


@pytest.mark.asyncio
async def test_no_overlap_concurrent_warm() -> None:
    """When another consumer warmed the sandbox between peek and
    get_or_provision, the probe must not release it (warm preservation)."""
    from aios.sandbox.backends.base import CommandResult

    handle = object()
    registry = MagicMock()
    # peek returns a handle on the first call — sandbox is warm
    registry.peek = MagicMock(return_value=handle)
    registry.get_or_provision = AsyncMock(return_value=handle)
    registry.exec = AsyncMock(
        return_value=CommandResult(0, "", "", timed_out=False, truncated=False)
    )
    registry.release = AsyncMock()

    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_concurrent",
        rate_limit_seconds=60,
        operation_timeout_seconds=5,
    )

    assert await probe.check_once(now=0)
    # Must NOT release — the sandbox was warm (owned by someone else)
    registry.release.assert_not_awaited()


@pytest.mark.asyncio
async def test_warm_failure_no_release() -> None:
    """When the sandbox was warm but exec fails, the probe must NOT release
    it — the original consumer owns it regardless of probe outcome."""
    from aios.sandbox.backends.base import CommandResult

    handle = object()
    registry = _warm_registry(return_value=handle)
    registry.exec = AsyncMock(
        return_value=CommandResult(1, "", "fail", timed_out=False, truncated=False)
    )

    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_warm_fail",
        rate_limit_seconds=60,
        operation_timeout_seconds=5,
    )

    assert not await probe.check_once(now=0)
    # Must NOT release — the sandbox was warm
    registry.release.assert_not_awaited()


# ── Finding 1: Strict deadline — fail immediately at remaining<=0 ────────


@pytest.mark.asyncio
async def test_zero_budget_fails_immediately() -> None:
    """With operation_timeout_seconds=0 the probe must fail immediately,
    not grant any implicit minimum grace (no max(0.1, remaining))."""
    registry = _cold_registry()
    alarm = MagicMock()
    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_zero",
        rate_limit_seconds=60,
        operation_timeout_seconds=0,
        alarm=alarm,
    )

    result = await probe.check_once()
    assert result is False
    # get_or_provision should NOT have been called — budget was zero
    registry.get_or_provision.assert_not_awaited()
    # Alarm should report DeadlineExceeded
    assert alarm.call_count == 1
    assert alarm.call_args.args[1]["error_type"] == "DeadlineExceeded"


@pytest.mark.asyncio
async def test_cleanup_grace_is_bounded() -> None:
    """The _CLEANUP_GRACE_SECONDS constant must be defined, positive, and
    bounded (not infinite)."""
    assert isinstance(_CLEANUP_GRACE_SECONDS, (int, float))
    assert 0 < _CLEANUP_GRACE_SECONDS < 60


@pytest.mark.asyncio
async def test_deadline_exceeded_type_is_distinct() -> None:
    """_DeadlineExceeded is a distinct exception type, not a bare TimeoutError."""
    assert not issubclass(_DeadlineExceeded, TimeoutError)
    assert issubclass(_DeadlineExceeded, Exception)


# ── Finding 2: Atomic probe ownership via generation token ───────────────


@pytest.mark.asyncio
async def test_cold_peek_concurrent_provision_race_no_release() -> None:
    """Exact race: probe peeks (cold), concurrent consumer provisions before
    probe's get_or_provision returns.  The probe must NOT release because
    the current handle (from peek post-provision) is a DIFFERENT object
    than what the probe received — the concurrent consumer owns it.

    Simulates: probe peek()→None, concurrent consumer provisions handle_A,
    probe get_or_provision returns handle_A (serialized), but by the time
    the probe tries to release, the concurrent consumer has re-provisioned
    handle_B.  peek() returns handle_B ≠ handle_A → no release.
    """
    from aios.sandbox.backends.base import CommandResult

    probe_handle = object()  # what the probe got from get_or_provision
    concurrent_handle = object()  # what a concurrent consumer re-provisioned

    registry = MagicMock()
    # peek calls: 1st=None (cold), 2nd=concurrent_handle (different identity!)
    registry.peek = MagicMock(side_effect=[None, concurrent_handle])
    registry.get_or_provision = AsyncMock(return_value=probe_handle)
    registry.exec = AsyncMock(
        return_value=CommandResult(0, "", "", timed_out=False, truncated=False)
    )
    registry.release = AsyncMock()

    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_race",
        rate_limit_seconds=60,
        operation_timeout_seconds=5,
    )

    assert await probe.check_once(now=0)
    # Must NOT release — peek() returned concurrent_handle which is not
    # the same object as probe_handle.  The concurrent consumer owns it.
    registry.release.assert_not_awaited()


@pytest.mark.asyncio
async def test_cold_provision_same_handle_releases() -> None:
    """When peek() post-provision returns the SAME handle object that the
    probe received from get_or_provision, the probe owns it and must release."""
    from aios.sandbox.backends.base import CommandResult

    handle = object()
    registry = MagicMock()
    # peek: None (cold), then same handle (probe owns it)
    registry.peek = MagicMock(side_effect=[None, handle])
    registry.get_or_provision = AsyncMock(return_value=handle)
    registry.exec = AsyncMock(
        return_value=CommandResult(0, "", "", timed_out=False, truncated=False)
    )
    registry.release = AsyncMock()

    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_own",
        rate_limit_seconds=60,
        operation_timeout_seconds=5,
    )

    assert await probe.check_once(now=0)
    registry.release.assert_awaited_once_with("sess_own")


# ── Finding 3: Shield+bound cleanup, exec settled before release ─────────


@pytest.mark.asyncio
async def test_external_cancellation_settles_exec_before_release() -> None:
    """On external cancellation, the exec task must be settled (cancelled +
    awaited) before the release call.  No probe-owned cold handle must
    remain in the registry after cancellation."""
    from aios.sandbox.backends.base import CommandResult

    exec_settled = asyncio.Event()
    release_called = asyncio.Event()

    handle = object()

    async def slow_exec(*args: object, **kwargs: object) -> CommandResult:
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            exec_settled.set()
            raise
        return CommandResult(0, "", "", timed_out=False, truncated=False)  # pragma: no cover

    registry = MagicMock()
    registry.peek = MagicMock(side_effect=[None, handle])
    registry.get_or_provision = AsyncMock(return_value=handle)
    registry.exec = AsyncMock(side_effect=slow_exec)

    async def mock_release(session_id: str) -> None:
        release_called.set()

    registry.release = AsyncMock(side_effect=mock_release)

    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_cancel",
        rate_limit_seconds=60,
        operation_timeout_seconds=100,
    )

    task = asyncio.create_task(probe.check_once(now=0))
    # Wait for exec to start
    await asyncio.sleep(0.05)
    # Cancel externally
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Exec must have been settled before release was called
    assert exec_settled.is_set(), "exec was not settled before release"
    # Release must have been called (probe owned the cold-provisioned handle)
    assert release_called.is_set(), "release was not called after cancellation"


@pytest.mark.asyncio
async def test_no_cold_residency_after_probe_cancellation() -> None:
    """After probe cancellation on a cold-provisioned sandbox, the handle
    must not remain in the registry (release must have been called)."""
    handle = object()

    async def slow_provision(session_id: str, pool: object = None) -> object:
        await asyncio.sleep(0.01)
        return handle

    registry = MagicMock()
    # Pre-provision: cold.  Post-provision: handle exists.
    registry.peek = MagicMock(side_effect=[None, handle])
    registry.get_or_provision = AsyncMock(side_effect=slow_provision)
    registry.exec = AsyncMock(side_effect=asyncio.CancelledError)
    registry.release = AsyncMock()

    probe = StandingSessionFilesystemProbe(
        registry,
        object(),
        "sess_residency",
        rate_limit_seconds=60,
        operation_timeout_seconds=100,
    )

    task = asyncio.create_task(probe.check_once(now=0))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    registry.release.assert_awaited_once_with("sess_residency")
