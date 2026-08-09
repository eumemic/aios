"""Fail-fast validation that persisted workspaces agree with this process's root."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.errors import ForbiddenError
from aios.sandbox.volumes import validate_workspace_path

_WORKSPACE_SCAN_PAGE_SIZE = 1000


def _workspace_diagnostic(raw_path: str, account_id: str) -> tuple[str, str, str]:
    workspace_root = get_settings().workspace_root.resolve()
    return (
        str(workspace_root),
        str((workspace_root / account_id).resolve()),
        str(Path(raw_path).resolve()),
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
    - Each DB fetch honours ``query_timeout_seconds`` (defaults from config).
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
        if time.monotonic() > deadline:
            raise WorkspaceScanTimeoutError(
                f"workspace-root startup scan exceeded {scan_timeout_seconds}s deadline "
                f"(service={service!r}, last_id={last_id!r})"
            )

        async with pool.acquire() as conn:
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
                timeout=query_timeout_seconds,
            )

        if not rows:
            return
        for row in rows:
            session_id = row["id"]
            account_id = row["account_id"]
            raw_path = row["workspace_volume_path"]
            try:
                validate_workspace_path(raw_path, account_id, session_id=session_id)
            except ForbiddenError as exc:
                workspace_root, account_root, resolved_path = _workspace_diagnostic(
                    raw_path, account_id
                )
                raise RuntimeError(
                    "workspace-root startup validation failed: "
                    f"service={service!r}, workspace_root={workspace_root!r}, "
                    f"account_root={account_root!r}, raw_path={raw_path!r}, "
                    f"resolved_path={resolved_path!r}, account_id={account_id!r}, "
                    f"session_id={session_id!r}"
                ) from exc
        last_id = rows[-1]["id"]
