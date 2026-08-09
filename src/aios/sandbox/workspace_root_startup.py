"""Fail-fast validation that persisted workspaces agree with this process's root."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.errors import ForbiddenError
from aios.sandbox.volumes import validate_workspace_path

_WORKSPACE_SCAN_PAGE_SIZE = 1000

# Upper bound on a single Path.resolve() offloaded to a thread.  Symlink-heavy
# trees can stall resolve() for seconds; this cap ensures the per-row cost is
# bounded and the overall scan deadline remains honest.
_PATH_RESOLVE_TIMEOUT_SECONDS: float = 2.0

# Upper bound on the connection-release (__aexit__) call.  Normally instant,
# but a misbehaving pool could block; bounding it keeps the startup budget
# honest.
_CONN_RELEASE_TIMEOUT_SECONDS: float = 5.0


async def _resolve_in_thread(
    p: Path, *, resolve_timeout: float = _PATH_RESOLVE_TIMEOUT_SECONDS
) -> Path:
    """Offload ``Path.resolve()`` to a thread with a bounded timeout.

    ``Path.resolve()`` is synchronous and can block on symlink-heavy or
    NFS-backed trees.  Offloading to a thread and capping the wall-clock
    cost keeps the startup scan budget honest.
    """
    return await asyncio.wait_for(asyncio.to_thread(p.resolve), timeout=resolve_timeout)


def _remaining(deadline: float) -> float:
    """Seconds left until *deadline*; always non-negative."""
    return max(0.0, deadline - time.monotonic())


def _check_deadline(
    deadline: float,
    scan_timeout_seconds: float,
    service: str,
    last_id: str | None,
) -> None:
    """Raise ``WorkspaceScanTimeoutError`` if the deadline has passed."""
    if time.monotonic() > deadline:
        raise WorkspaceScanTimeoutError(
            f"workspace-root startup scan exceeded {scan_timeout_seconds}s deadline "
            f"(service={service!r}, last_id={last_id!r})"
        )


class WorkspaceScanTimeoutError(RuntimeError):
    """The startup workspace-root scan exceeded its overall deadline."""


async def validate_workspace_root_against_sessions(
    pool: asyncpg.Pool[Any],
    *,
    service: str,
    scan_timeout_seconds: float | None = None,
    query_timeout_seconds: float | None = None,
) -> None:
    """Reject API/worker root drift before the process accepts any work.

    Session rows are shared by the API and worker, so validating every live
    row against each process's configured root turns divergent deployment
    configuration into a startup failure instead of disabling filesystem tools
    only when a standing session next provisions its sandbox.

    Resource discipline:
    - Each page acquires and releases a pooled connection so the scan never
      holds a connection across the full row set.
    - Pool acquisition is wrapped in ``asyncio.wait_for`` with the remaining
      budget so a blocked/contended pool cannot exceed the overall deadline.
    - Connection release (``__aexit__``) is bounded by
      ``_CONN_RELEASE_TIMEOUT_SECONDS`` so a misbehaving pool cannot stall
      the scan past its deadline.
    - Each DB fetch honours ``min(query_timeout_seconds, remaining_budget)``
      so a slow query cannot exceed the overall deadline.
    - ``Path.resolve()`` is offloaded to a thread and bounded by
      ``_PATH_RESOLVE_TIMEOUT_SECONDS`` so symlink-heavy trees cannot
      accumulate past the contract.
    - The deadline is checked between rows during validation so slow path
      resolution cannot accumulate past the contract.
    - The overall scan honours ``scan_timeout_seconds`` (defaults from config)
      so high-cardinality deployments can't block startup indefinitely.
    """
    settings = get_settings()
    if scan_timeout_seconds is None:
        scan_timeout_seconds = settings.workspace_scan_timeout_seconds
    if query_timeout_seconds is None:
        query_timeout_seconds = settings.workspace_scan_query_timeout_seconds

    deadline = time.monotonic() + scan_timeout_seconds
    last_id: str | None = None

    while True:
        _check_deadline(deadline, scan_timeout_seconds, service, last_id)

        # Acquire a pooled connection within the remaining budget so a
        # contended / blocked pool cannot exceed the overall deadline.
        remaining = _remaining(deadline)
        try:
            ctx = pool.acquire()
            conn = await asyncio.wait_for(ctx.__aenter__(), timeout=remaining)
        except TimeoutError as exc:
            raise WorkspaceScanTimeoutError(
                f"workspace-root startup scan exceeded {scan_timeout_seconds}s deadline "
                f"during pool acquire (service={service!r}, last_id={last_id!r})"
            ) from exc

        try:
            # Each fetch timeout is clamped to the remaining budget so a slow
            # query page cannot push the scan past its overall deadline.
            effective_query_timeout = min(query_timeout_seconds, _remaining(deadline))
            rows = await conn.fetch(
                """
                SELECT id, account_id, workspace_volume_path
                  FROM sessions
                 WHERE archived_at IS NULL
                   AND ($1::text IS NULL OR id > $1)
                 ORDER BY id
                 LIMIT $2
                """,
                last_id,
                _WORKSPACE_SCAN_PAGE_SIZE,
                timeout=effective_query_timeout,
            )
        finally:
            # Always release the connection, even on fetch failure.
            # Bounded so a misbehaving pool cannot stall the scan.
            try:
                await asyncio.wait_for(
                    ctx.__aexit__(None, None, None),
                    timeout=_CONN_RELEASE_TIMEOUT_SECONDS,
                )
            except TimeoutError as release_exc:
                raise WorkspaceScanTimeoutError(
                    f"workspace-root startup scan: connection release timed out "
                    f"after {_CONN_RELEASE_TIMEOUT_SECONDS}s "
                    f"(service={service!r}, last_id={last_id!r})"
                ) from release_exc

        if not rows:
            return
        for row in rows:
            # Check deadline between rows so slow path validation (e.g.
            # symlink-heavy resolve()) cannot accumulate past the contract.
            _check_deadline(deadline, scan_timeout_seconds, service, last_id)

            session_id = row["id"]
            account_id = row["account_id"]
            raw_path = row["workspace_volume_path"]
            try:
                validate_workspace_path(raw_path, account_id, session_id=session_id)
            except ForbiddenError as exc:
                try:
                    resolved_root = await _resolve_in_thread(
                        get_settings().workspace_root,
                        resolve_timeout=min(_PATH_RESOLVE_TIMEOUT_SECONDS, _remaining(deadline)),
                    )
                    resolved_account = await _resolve_in_thread(
                        get_settings().workspace_root / account_id,
                        resolve_timeout=min(_PATH_RESOLVE_TIMEOUT_SECONDS, _remaining(deadline)),
                    )
                    resolved_path = await _resolve_in_thread(
                        Path(raw_path),
                        resolve_timeout=min(_PATH_RESOLVE_TIMEOUT_SECONDS, _remaining(deadline)),
                    )
                except (TimeoutError, OSError):
                    # Diagnostic resolve failed — use un-resolved strings.
                    resolved_root = get_settings().workspace_root
                    resolved_account = get_settings().workspace_root / account_id
                    resolved_path = Path(raw_path)
                raise RuntimeError(
                    "workspace-root startup validation failed: "
                    f"service={service!r}, workspace_root={str(resolved_root)!r}, "
                    f"account_root={str(resolved_account)!r}, raw_path={raw_path!r}, "
                    f"resolved_path={str(resolved_path)!r}, account_id={account_id!r}, "
                    f"session_id={session_id!r}"
                ) from exc
        last_id = rows[-1]["id"]
