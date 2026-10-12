"""In-process migration helpers.

Replaces ``subprocess.call(["alembic", "upgrade", ...])`` so callers (the
``aios migrate`` CLI command and the e2e test fixtures) skip the
``uv run`` cold-start tax. ``migrations/env.py`` reads ``AIOS_DB_URL``
from the environment, so we temporarily set it for the duration of the
upgrade.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO
from unittest import mock

if TYPE_CHECKING:
    import asyncpg
    from alembic.config import Config

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MIGRATION_LOCK_ID = 0x41494F534D494752  # "AIOSMIGR" as a signed-bigint-safe key


def _known_revisions() -> set[str]:
    """Return every revision understood by this image's Alembic graph.

    Let Alembic parse the migration modules rather than maintaining a partial
    parser for its Python declarations.  In particular, repository migrations
    legitimately use both ``revision = ...`` and ``revision: str = ...``.
    """
    from alembic.script import ScriptDirectory

    scripts = ScriptDirectory.from_config(alembic_config())
    return {revision.revision for revision in scripts.walk_revisions()}


def _sync_db_url(db_url: str) -> str:
    if db_url.startswith("postgresql+asyncpg://"):
        return db_url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    if db_url.startswith("postgresql://"):
        return db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return db_url


@contextmanager
def _migration_admission(db_url: str) -> Iterator[bool]:
    """Serialize sibling migrators and admit rollback images without migrating.

    An older, otherwise compatible image cannot ask Alembic to interpret a
    candidate-only revision: Alembic rejects that revision before the service
    gets a chance to prove runtime compatibility.  Once serialized, a revision
    unknown to this image therefore means "database is newer" and migration is
    skipped.  Normal startup/readiness remains responsible for compatibility.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(_sync_db_url(db_url), pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": _MIGRATION_LOCK_ID})
            try:
                table = conn.execute(text("SELECT to_regclass('alembic_version')")).scalar()
                current = (
                    {
                        row[0]
                        for row in conn.execute(text("SELECT version_num FROM alembic_version"))
                    }
                    if table is not None
                    else set()
                )
                known = _known_revisions()
                unknown = current - known
                if unknown:
                    # Revisions are monotonically numbered in this repository.
                    # Only a strictly newer revision can represent a rollback
                    # image observing a successful forward migration.  An
                    # unparseable or divergent stamp remains fail-closed.
                    newest_known = max(int(revision) for revision in known)
                    if not all(
                        revision.isdigit() and int(revision) > newest_known for revision in unknown
                    ):
                        raise RuntimeError(
                            "database has an unknown non-forward migration revision: "
                            + ", ".join(sorted(unknown))
                        )
                yield not unknown
            finally:
                conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _MIGRATION_LOCK_ID})
    finally:
        engine.dispose()


def alembic_config(*, stdout: TextIO | None = None) -> Config:
    """Alembic ``Config`` bound to the repo's ``alembic.ini`` + ``migrations/``.

    The one home for where the migration ladder lives on disk — shared by
    :func:`upgrade_to_head` and the test-side runner
    (``tests/helpers/alembic.py``) so the two cannot drift.
    """
    from alembic.config import Config

    cfg = Config(
        str(_REPO_ROOT / "alembic.ini"), stdout=stdout if stdout is not None else sys.stdout
    )
    cfg.set_main_option("script_location", str(_REPO_ROOT / "migrations"))
    return cfg


def upgrade_to_head(db_url: str) -> None:
    """In-process equivalent of ``alembic upgrade head`` against ``db_url``."""
    from alembic import command

    with mock.patch.dict(os.environ, {"AIOS_DB_URL": db_url}):
        command.upgrade(alembic_config(), "head")


_TRIGGER_LOCK_TIMEOUT = "5s"
_TRIGGER_MAX_ATTEMPTS = 10
_TRIGGER_RETRY_SLEEP_SECONDS = 1.0


async def _ensure_lock_release_trigger(
    conn: asyncpg.Connection[Any],
    *,
    verbose: bool,
    max_attempts: int,
    retry_sleep_seconds: float,
    lock_timeout: str,
) -> None:
    """Install/refresh the aios lock-release trigger only when it has drifted.

    The DDL takes an AccessExclusive lock on ``procrastinate_jobs``; while that
    request waits (e.g. behind ``pg_dump``'s AccessShare) every worker
    defer/fetch queues behind it and the job queue stalls (#2539).  So:

    1. Compare the installed definitions (catalog read, no table lock) and skip
       entirely when they already match.
    2. Otherwise apply under ``SET LOCAL lock_timeout`` with the same bounded
       retry ``aios migrate`` uses for Alembic; exhaustion re-raises so the
       migrate fails rather than stalling the queue.
    """
    import asyncpg

    from aios.db.procrastinate_extensions import (
        LOCK_RELEASE_STATE_SQL,
        LOCK_RELEASE_TRIGGER_DDL,
        lock_release_trigger_is_current,
    )
    from aios.logging import get_logger

    logger = get_logger("aios.migrate")
    for attempt in range(1, max_attempts + 1):
        try:
            async with conn.transaction():
                state = await conn.fetchrow(LOCK_RELEASE_STATE_SQL)
                if state is not None and lock_release_trigger_is_current(
                    functiondef=state["functiondef"],
                    triggerdef=state["triggerdef"],
                    function_schema=state["function_schema"],
                    table_schema=state["table_schema"],
                ):
                    if verbose:
                        print(
                            "aios lock-release trigger already current, skipping", file=sys.stderr
                        )
                    return
                await conn.execute("SELECT set_config('lock_timeout', $1, true)", lock_timeout)
                await conn.execute(LOCK_RELEASE_TRIGGER_DDL)
            if verbose:
                print("aios lock-release trigger ensured", file=sys.stderr)
            return
        except asyncpg.exceptions.LockNotAvailableError:
            if attempt == max_attempts:
                raise
            logger.warning(
                "migration.lock_retry",
                step="lock_release_trigger",
                attempt=attempt,
                max_attempts=max_attempts,
                retry_in_seconds=retry_sleep_seconds,
            )
            await asyncio.sleep(retry_sleep_seconds)


async def apply_procrastinate_schema(
    db_url: str,
    *,
    verbose: bool = False,
    max_attempts: int = _TRIGGER_MAX_ATTEMPTS,
    retry_sleep_seconds: float = _TRIGGER_RETRY_SLEEP_SECONDS,
    lock_timeout: str = _TRIGGER_LOCK_TIMEOUT,
) -> None:
    """Apply procrastinate's schema (if missing) and the aios lock-release
    trigger (if missing or drifted) against ``db_url``. Idempotent — and on an
    already-current DB takes no lock on ``procrastinate_jobs`` at all.

    ``apply_schema_async`` isn't idempotent (no ``IF NOT EXISTS``), hence the
    ``to_regclass`` guard.  The trigger DDL is gated on a catalog comparison
    and bounded by ``lock_timeout`` + retry (see
    :func:`_ensure_lock_release_trigger`); a lock it cannot get within
    ``max_attempts`` raises ``asyncpg.exceptions.LockNotAvailableError``.

    Pass ``verbose=True`` from CLI contexts to surface user-facing status.
    """
    import asyncpg
    from procrastinate import App, PsycopgConnector

    from aios.db.pool import normalize_dsn

    # ``db_url`` arrives verbatim from ``settings.db_url`` (``aios migrate``) and
    # from test fixtures. Strip any SQLAlchemy/alembic driver suffix so both
    # asyncpg (``postgresql://`` only) and procrastinate's psycopg connector
    # (bare scheme) accept it — mirrors ``aios.db.pool.normalize_dsn`` used by
    # ``create_pool`` and ``_sync_dsn`` used by the worker's connector.
    db_url = normalize_dsn(db_url)
    conn = await asyncpg.connect(db_url)
    try:
        present = await conn.fetchval("SELECT to_regclass('procrastinate_jobs')")
        if present is None:
            if verbose:
                print("applying procrastinate schema...", file=sys.stderr)
            tmp_app = App(connector=PsycopgConnector(conninfo=db_url))
            await tmp_app.open_async()
            try:
                await tmp_app.schema_manager.apply_schema_async()
            finally:
                await tmp_app.close_async()
            if verbose:
                print("procrastinate schema applied", file=sys.stderr)
        elif verbose:
            print("procrastinate schema already present, skipping", file=sys.stderr)
        await _ensure_lock_release_trigger(
            conn,
            verbose=verbose,
            max_attempts=max_attempts,
            retry_sleep_seconds=retry_sleep_seconds,
            lock_timeout=lock_timeout,
        )
    finally:
        await conn.close()
