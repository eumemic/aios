"""Startup validation for API/worker workspace-root agreement (#2064)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.config import get_settings
from aios.sandbox.workspace_root_startup import (
    WorkspaceScanTimeoutError,
    validate_workspace_root_against_sessions,
)


@pytest.fixture
def workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
    return tmp_path


def _pool_releasing_conn(rows: list[dict[str, str]]) -> MagicMock:
    """Build a mock pool that releases its connection between pages.

    Each ``pool.acquire()`` context-manager call returns a fresh mock
    connection whose ``fetch`` returns the next page of rows. This
    mirrors the production code which now acquires/releases a pooled
    connection per page.
    """
    pages = [rows, []]
    call_index = {"i": 0}

    def _make_acquired() -> AsyncMock:
        conn = AsyncMock()
        idx = call_index["i"]
        call_index["i"] += 1
        conn.fetch.return_value = pages[idx] if idx < len(pages) else []
        acquired = AsyncMock()
        acquired.__aenter__.return_value = conn
        return acquired

    pool = MagicMock()
    pool.acquire.side_effect = _make_acquired
    return pool


def _pool_multi_page(page_list: list[list[dict[str, str]]]) -> MagicMock:
    """Build a mock pool that returns multiple pages then empty."""
    all_pages = [*list(page_list), []]
    call_index = {"i": 0}

    def _make_acquired() -> AsyncMock:
        conn = AsyncMock()
        idx = call_index["i"]
        call_index["i"] += 1
        conn.fetch.return_value = all_pages[idx] if idx < len(all_pages) else []
        acquired = AsyncMock()
        acquired.__aenter__.return_value = conn
        return acquired

    pool = MagicMock()
    pool.acquire.side_effect = _make_acquired
    return pool


@pytest.mark.asyncio
async def test_accepts_canonical_account_scoped_rows(workspace_root: Path) -> None:
    row = {
        "id": "sess_ok",
        "account_id": "acc_a",
        "workspace_volume_path": str(workspace_root / "acc_a" / "sess_ok"),
    }
    await validate_workspace_root_against_sessions(_pool_releasing_conn([row]), service="worker")


@pytest.mark.asyncio
async def test_rejects_root_drift_at_startup_with_full_diagnostic(workspace_root: Path) -> None:
    raw = "/srv/aios/workspaces/acc_a/sess_bad"
    row = {"id": "sess_bad", "account_id": "acc_a", "workspace_volume_path": raw}

    with pytest.raises(RuntimeError) as exc_info:
        await validate_workspace_root_against_sessions(
            _pool_releasing_conn([row]), service="worker"
        )

    message = str(exc_info.value)
    assert "workspace-root startup validation failed" in message
    assert "service='worker'" in message
    assert f"workspace_root={str(workspace_root)!r}" in message
    assert f"account_root={str(workspace_root / 'acc_a')!r}" in message
    assert f"raw_path={raw!r}" in message
    assert f"resolved_path={raw!r}" in message
    assert "account_id='acc_a'" in message
    assert "session_id='sess_bad'" in message


@pytest.mark.asyncio
async def test_cross_tenant_row_still_fails_closed(workspace_root: Path) -> None:
    row = {
        "id": "sess_a",
        "account_id": "acc_a",
        "workspace_volume_path": str(workspace_root / "acc_b" / "sess_b"),
    }
    with pytest.raises(RuntimeError):
        await validate_workspace_root_against_sessions(_pool_releasing_conn([row]), service="api")


@pytest.mark.asyncio
async def test_absolute_legacy_row_remains_accepted(workspace_root: Path) -> None:
    row = {
        "id": "sess_legacy",
        "account_id": "acc_a",
        "workspace_volume_path": str(workspace_root / "sess_legacy"),
    }
    await validate_workspace_root_against_sessions(_pool_releasing_conn([row]), service="worker")


@pytest.mark.asyncio
async def test_scans_live_rows_in_bounded_keyset_pages(
    workspace_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aios.sandbox.workspace_root_startup as module

    monkeypatch.setattr(module, "_WORKSPACE_SCAN_PAGE_SIZE", 2)
    rows = [
        {
            "id": f"sess_{index}",
            "account_id": "acc_a",
            "workspace_volume_path": str(workspace_root / "acc_a" / f"sess_{index}"),
        }
        for index in range(3)
    ]
    pool = _pool_multi_page([rows[:2], rows[2:]])

    await validate_workspace_root_against_sessions(pool, service="api")

    # Three acquire calls: page1, page2, empty-page sentinel
    assert pool.acquire.call_count == 3


@pytest.mark.asyncio
async def test_releases_connection_between_pages(
    workspace_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each page acquires and releases a pooled connection."""
    import aios.sandbox.workspace_root_startup as module

    monkeypatch.setattr(module, "_WORKSPACE_SCAN_PAGE_SIZE", 1)
    rows = [
        {
            "id": f"sess_{i}",
            "account_id": "acc_a",
            "workspace_volume_path": str(workspace_root / "acc_a" / f"sess_{i}"),
        }
        for i in range(2)
    ]
    pool = _pool_multi_page([[rows[0]], [rows[1]]])

    await validate_workspace_root_against_sessions(pool, service="api")

    # 3 acquires: 2 data pages + 1 empty sentinel
    assert pool.acquire.call_count == 3


@pytest.mark.asyncio
async def test_scan_timeout_raises(workspace_root: Path) -> None:
    """The scan must raise WorkspaceScanTimeoutError when its budget is exceeded."""

    async def _slow_fetch(*args: object, **kwargs: object) -> list[dict[str, str]]:
        return [
            {
                "id": "sess_slow",
                "account_id": "acc_a",
                "workspace_volume_path": str(workspace_root / "acc_a" / "sess_slow"),
            }
        ]

    # Build a pool that always returns data (never empty → never terminates)
    def _make_acquired() -> AsyncMock:
        conn = AsyncMock()
        conn.fetch.side_effect = _slow_fetch
        acquired = AsyncMock()
        acquired.__aenter__.return_value = conn
        return acquired

    pool = MagicMock()
    pool.acquire.side_effect = _make_acquired

    with pytest.raises(WorkspaceScanTimeoutError) as exc_info:
        await validate_workspace_root_against_sessions(
            pool, service="api", scan_timeout_seconds=0.0
        )
    assert "exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_scan_passes_query_timeout(workspace_root: Path) -> None:
    """Each DB fetch must receive a timeout <= the configured query_timeout_seconds."""
    row = {
        "id": "sess_qt",
        "account_id": "acc_a",
        "workspace_volume_path": str(workspace_root / "acc_a" / "sess_qt"),
    }

    conns: list[AsyncMock] = []

    def _make_acquired() -> AsyncMock:
        conn = AsyncMock()
        conn.fetch.return_value = [row] if len(conns) == 0 else []
        conns.append(conn)
        acquired = AsyncMock()
        acquired.__aenter__.return_value = conn
        return acquired

    pool = MagicMock()
    pool.acquire.side_effect = _make_acquired

    await validate_workspace_root_against_sessions(
        pool, service="api", query_timeout_seconds=7.5, scan_timeout_seconds=30.0
    )

    # Each conn.fetch should have been called with timeout <= 7.5
    # (clamped to min(query_timeout, remaining_budget))
    for conn in conns:
        if conn.fetch.await_count > 0:
            call_kwargs = conn.fetch.await_args
            timeout = call_kwargs.kwargs.get("timeout")
            assert timeout is not None
            assert 0 < timeout <= 7.5


@pytest.mark.asyncio
async def test_high_cardinality_with_budget(
    workspace_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Many pages complete within budget without timeout."""
    import aios.sandbox.workspace_root_startup as module

    monkeypatch.setattr(module, "_WORKSPACE_SCAN_PAGE_SIZE", 2)
    rows = [
        {
            "id": f"sess_{i:04d}",
            "account_id": "acc_a",
            "workspace_volume_path": str(workspace_root / "acc_a" / f"sess_{i:04d}"),
        }
        for i in range(20)
    ]
    pages = [rows[i : i + 2] for i in range(0, len(rows), 2)]
    pool = _pool_multi_page(pages)

    await validate_workspace_root_against_sessions(
        pool, service="worker", scan_timeout_seconds=30.0
    )

    # 10 data pages + 1 empty sentinel = 11 acquires
    assert pool.acquire.call_count == 11


# ── Focused deadline-enforcement tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_blocked_acquire_triggers_scan_timeout(workspace_root: Path) -> None:
    """A pool whose acquire blocks indefinitely must be interrupted by the
    overall scan deadline, not allowed to hang startup forever."""

    async def _block_forever() -> object:
        await asyncio.sleep(3600)
        return object()  # pragma: no cover

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(side_effect=_block_forever)
    ctx.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire.return_value = ctx

    with pytest.raises(WorkspaceScanTimeoutError, match="pool acquire"):
        await validate_workspace_root_against_sessions(
            pool, service="api", scan_timeout_seconds=0.05, query_timeout_seconds=10.0
        )


@pytest.mark.asyncio
async def test_slow_fetch_near_deadline_triggers_scan_timeout(
    workspace_root: Path,
) -> None:
    """A fetch that takes longer than the remaining budget must be clamped so
    the overall deadline is still enforced even when the DB is slow."""
    fetch_received_timeout: list[float] = []

    async def _slow_fetch(*args: object, **kwargs: object) -> list[dict[str, str]]:
        timeout = kwargs.get("timeout", 999.0)
        assert isinstance(timeout, (int, float))
        fetch_received_timeout.append(timeout)
        # Simulate a DB that takes longer than the remaining budget
        await asyncio.sleep(timeout + 0.5)
        return [
            {
                "id": "sess_never",
                "account_id": "acc_a",
                "workspace_volume_path": str(workspace_root / "acc_a" / "sess_never"),
            }
        ]

    def _make_acquired() -> AsyncMock:
        conn = AsyncMock()
        conn.fetch.side_effect = _slow_fetch
        acquired = AsyncMock()
        acquired.__aenter__.return_value = conn
        acquired.__aexit__.return_value = None
        return acquired

    pool = MagicMock()
    pool.acquire.side_effect = _make_acquired

    # Give a tight budget; the query_timeout is larger than the scan budget
    # so the effective timeout must be clamped to remaining budget.
    with pytest.raises((WorkspaceScanTimeoutError, asyncio.TimeoutError)):
        await validate_workspace_root_against_sessions(
            pool,
            service="api",
            scan_timeout_seconds=0.1,
            query_timeout_seconds=60.0,
        )

    # Verify the fetch timeout was clamped to remaining budget (≤ scan_timeout)
    assert len(fetch_received_timeout) >= 1
    assert fetch_received_timeout[0] <= 0.1 + 0.01  # small epsilon


@pytest.mark.asyncio
async def test_slow_validation_triggers_scan_timeout(
    workspace_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If path validation (e.g. symlink-heavy resolve) is slow, the per-row
    deadline check must fire before the validation loop runs away."""
    import aios.sandbox.workspace_root_startup as module

    call_count = {"n": 0}

    original_validate = module.validate_workspace_path

    def _slow_validate(raw_path: str, account_id: str, **kw: object) -> None:
        call_count["n"] += 1
        # Burn wall-clock time to push past deadline
        time.sleep(0.03)
        return original_validate(raw_path, account_id, **kw)

    monkeypatch.setattr(module, "validate_workspace_path", _slow_validate)
    monkeypatch.setattr(module, "_WORKSPACE_SCAN_PAGE_SIZE", 1000)

    # Many rows that would individually pass but collectively exceed deadline
    rows = [
        {
            "id": f"sess_{i:04d}",
            "account_id": "acc_a",
            "workspace_volume_path": str(workspace_root / "acc_a" / f"sess_{i:04d}"),
        }
        for i in range(200)
    ]

    def _make_acquired() -> AsyncMock:
        conn = AsyncMock()
        conn.fetch.return_value = rows
        acquired = AsyncMock()
        acquired.__aenter__.return_value = conn
        acquired.__aexit__.return_value = None
        return acquired

    pool = MagicMock()
    pool.acquire.side_effect = _make_acquired

    with pytest.raises(WorkspaceScanTimeoutError, match="exceeded"):
        await validate_workspace_root_against_sessions(
            pool, service="worker", scan_timeout_seconds=0.05, query_timeout_seconds=10.0
        )

    # Should have validated some rows before the deadline killed it
    assert 0 < call_count["n"] < 200


@pytest.mark.asyncio
async def test_high_cardinality_pagination_releases_connections(
    workspace_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With many pages, each page acquires and releases its own connection,
    and the scan completes within budget without leaking connections."""
    import aios.sandbox.workspace_root_startup as module

    monkeypatch.setattr(module, "_WORKSPACE_SCAN_PAGE_SIZE", 3)

    rows = [
        {
            "id": f"sess_{i:04d}",
            "account_id": "acc_a",
            "workspace_volume_path": str(workspace_root / "acc_a" / f"sess_{i:04d}"),
        }
        for i in range(30)
    ]
    pages = [rows[i : i + 3] for i in range(0, len(rows), 3)]

    # Track acquire/release calls to verify connection lifecycle
    acquired_count = {"n": 0}
    released_count = {"n": 0}

    def _make_acquired() -> MagicMock:
        idx = acquired_count["n"]
        acquired_count["n"] += 1
        conn = AsyncMock()
        conn.fetch.return_value = pages[idx] if idx < len(pages) else []

        ctx = MagicMock()

        async def _enter() -> AsyncMock:
            return conn

        async def _exit(*args: object) -> None:
            released_count["n"] += 1

        ctx.__aenter__ = AsyncMock(side_effect=_enter)
        ctx.__aexit__ = AsyncMock(side_effect=_exit)
        return ctx

    pool = MagicMock()
    pool.acquire.side_effect = _make_acquired

    await validate_workspace_root_against_sessions(
        pool, service="worker", scan_timeout_seconds=30.0
    )

    # 10 data pages + 1 empty sentinel = 11 acquires
    assert acquired_count["n"] == 11
    # Every acquired connection must have been released
    assert released_count["n"] == acquired_count["n"]
