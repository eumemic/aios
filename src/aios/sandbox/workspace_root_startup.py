"""Validate persisted live-session workspaces against this process's root."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Literal

import asyncpg

from aios.config import get_settings
from aios.errors import ForbiddenError
from aios.logging import get_logger
from aios.sandbox.volumes import validate_workspace_path

WorkspaceRootValidationMode = Literal["enforce", "warn", "off"]
_WORKSPACE_SCAN_PAGE_SIZE = 1000
_VIOLATION_SAMPLE_SIZE = 10
_VALIDATE_TIMEOUT_SECONDS = 2.0
_CONN_RELEASE_TIMEOUT_SECONDS = 5.0


class WorkspaceScanTimeoutError(RuntimeError):
    """The workspace-root scan exceeded its configured deadline."""


@dataclass(frozen=True, slots=True)
class WorkspaceRootViolation:
    session_id: str
    account_id: str
    raw_path: str
    reason: str


@dataclass(frozen=True, slots=True)
class WorkspaceRootValidationResult:
    violation_count: int


class WorkspaceRootValidationError(RuntimeError):
    """One or more live session rows violate the workspace jail."""

    def __init__(
        self,
        violation_count: int,
        sample: list[WorkspaceRootViolation],
        *,
        service: str,
    ) -> None:
        self.violation_count = violation_count
        rendered = "; ".join(
            f"session_id={row.session_id!r}, account_id={row.account_id!r}, "
            f"workspace_volume_path={row.raw_path!r}, reason={row.reason!r}"
            for row in sample
        )
        omitted = violation_count - len(sample)
        suffix = f"; {omitted} additional violation(s) omitted" if omitted else ""
        super().__init__(
            f"workspace-root validation found {violation_count} violation(s) "
            f"(service={service!r}); sample: {rendered}{suffix}"
        )


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _timeout(scan_timeout_seconds: float, service: str, last_id: str | None, phase: str) -> None:
    raise WorkspaceScanTimeoutError(
        f"workspace-root validation exceeded {scan_timeout_seconds}s deadline during {phase} "
        f"(service={service!r}, last_id={last_id!r})"
    )


async def validate_workspace_root_against_sessions(
    pool: asyncpg.Pool[Any],
    *,
    service: str,
    mode: WorkspaceRootValidationMode | None = None,
    scan_timeout_seconds: float | None = None,
    query_timeout_seconds: float | None = None,
) -> WorkspaceRootValidationResult:
    """Scan every unarchived session and enforce or report workspace-jail failures.

    ``warn`` logs each violating row and returns normally. ``enforce`` performs
    the same full scan and logging, then raises one aggregate error. ``off``
    performs no database work. The pre-flight command invokes this exact
    function in ``warn`` mode and maps the returned count to its exit status.
    """
    settings = get_settings()
    mode = mode or settings.workspace_root_validation
    if mode == "off":
        return WorkspaceRootValidationResult(violation_count=0)
    scan_timeout_seconds = scan_timeout_seconds or settings.workspace_scan_timeout_seconds
    query_timeout_seconds = query_timeout_seconds or settings.workspace_scan_query_timeout_seconds
    deadline = time.monotonic() + scan_timeout_seconds
    last_id: str | None = None
    violation_count = 0
    sample: list[WorkspaceRootViolation] = []
    log = get_logger("aios.workspace_root_validation")

    while True:
        remaining = _remaining(deadline)
        if remaining <= 0:
            _timeout(scan_timeout_seconds, service, last_id, "pool acquire")
        ctx = pool.acquire()
        try:
            conn = await asyncio.wait_for(ctx.__aenter__(), timeout=remaining)
        except TimeoutError:
            _timeout(scan_timeout_seconds, service, last_id, "pool acquire")

        try:
            remaining = _remaining(deadline)
            if remaining <= 0:
                _timeout(scan_timeout_seconds, service, last_id, "query")
            try:
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
                    timeout=min(query_timeout_seconds, remaining),
                )
            except TimeoutError:
                _timeout(scan_timeout_seconds, service, last_id, "query")
        finally:
            try:
                await asyncio.wait_for(
                    ctx.__aexit__(None, None, None), timeout=_CONN_RELEASE_TIMEOUT_SECONDS
                )
            except TimeoutError:
                _timeout(scan_timeout_seconds, service, last_id, "connection release")

        if not rows:
            break
        for row in rows:
            remaining = _remaining(deadline)
            if remaining <= 0:
                _timeout(scan_timeout_seconds, service, last_id, "row validation")
            session_id = row["id"]
            account_id = row["account_id"]
            raw_path = row["workspace_volume_path"]
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        validate_workspace_path,
                        raw_path,
                        account_id,
                        session_id=session_id,
                    ),
                    timeout=min(_VALIDATE_TIMEOUT_SECONDS, remaining),
                )
            except TimeoutError:
                _timeout(scan_timeout_seconds, service, session_id, "row validation")
            except (ForbiddenError, OSError, ValueError) as exc:
                violation = WorkspaceRootViolation(
                    session_id=session_id,
                    account_id=account_id,
                    raw_path=raw_path,
                    reason=str(exc),
                )
                violation_count += 1
                if len(sample) < _VIOLATION_SAMPLE_SIZE:
                    sample.append(violation)
                log.warning(
                    "workspace_root_validation.violation",
                    service=service,
                    mode=mode,
                    workspace_root=str(settings.workspace_root),
                    session_id=session_id,
                    account_id=account_id,
                    workspace_volume_path=raw_path,
                    reason=str(exc),
                )
        last_id = rows[-1]["id"]

    result = WorkspaceRootValidationResult(violation_count=violation_count)
    log.info(
        "workspace_root_validation.complete",
        service=service,
        mode=mode,
        violation_count=result.violation_count,
    )
    if violation_count and mode == "enforce":
        raise WorkspaceRootValidationError(violation_count, sample, service=service)
    return result
