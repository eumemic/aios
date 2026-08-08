"""Fail-fast validation that persisted workspaces agree with this process's root."""

from __future__ import annotations

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


async def validate_workspace_root_against_sessions(
    pool: asyncpg.Pool[Any], *, service: str
) -> None:
    """Reject API/worker root drift before the process accepts any work.

    Session rows are shared by the API and worker, so validating every live
    row against each process's configured root turns divergent deployment
    configuration into a startup failure instead of disabling filesystem tools
    only when a standing session next provisions its sandbox.
    """
    last_id: str | None = None
    async with pool.acquire() as conn:
        while True:
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
