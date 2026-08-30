"""In-process migration helpers.

Replaces ``subprocess.call(["alembic", "upgrade", ...])`` so callers (the
``aios migrate`` CLI command and the e2e test fixtures) skip the
``uv run`` cold-start tax. ``migrations/env.py`` reads ``AIOS_DB_URL``
from the environment, so we temporarily set it for the duration of the
upgrade.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, TextIO
from unittest import mock

if TYPE_CHECKING:
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


async def apply_procrastinate_schema(db_url: str, *, verbose: bool = False) -> None:
    """Apply procrastinate's schema (if missing) and the aios lock-release
    trigger against ``db_url``. Idempotent — safe on an already-migrated DB.

    ``apply_schema_async`` isn't idempotent (no ``IF NOT EXISTS``), hence the
    ``to_regclass`` guard; the trigger DDL is.

    Pass ``verbose=True`` from CLI contexts to surface user-facing status.
    """
    import asyncpg
    from procrastinate import App, PsycopgConnector

    from aios.db.procrastinate_extensions import LOCK_RELEASE_TRIGGER_DDL

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
        await conn.execute(LOCK_RELEASE_TRIGGER_DDL)
        if verbose:
            print("aios lock-release trigger ensured", file=sys.stderr)
    finally:
        await conn.close()
