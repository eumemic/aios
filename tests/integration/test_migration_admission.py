"""Real-Postgres controls for candidate migration admission."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import psycopg
import pytest

from aios.db import migrations


def _current_revision(db_url: str) -> str:
    # psycopg consumes a libpq URL, not SQLAlchemy's ``postgresql+psycopg``
    # dialect URL.  The shared fixture supplies the former directly.
    with psycopg.connect(db_url) as conn:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    assert row is not None
    return str(row[0])


def test_candidate_admits_real_previous_in_image_revision(migrated_db_url: str) -> None:
    """A database one revision behind this image proceeds to migration."""
    current = _current_revision(migrated_db_url)
    script = migrations.alembic_config()
    from alembic.script import ScriptDirectory

    down_revision = ScriptDirectory.from_config(script).get_revision(current).down_revision
    assert isinstance(down_revision, str)
    assert down_revision in migrations._known_revisions()

    with psycopg.connect(migrated_db_url, autocommit=True) as conn:
        conn.execute("UPDATE alembic_version SET version_num = %s", (down_revision,))
    try:
        with migrations._migration_admission(migrated_db_url) as should_migrate:
            assert should_migrate is True
    finally:
        with psycopg.connect(migrated_db_url, autocommit=True) as conn:
            conn.execute("UPDATE alembic_version SET version_num = %s", (current,))


def test_rollback_image_admitted_after_forward_revision(
    migrated_db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An adjacent older image does not ask Alembic to resolve the new stamp."""
    current = _current_revision(migrated_db_url)
    known = migrations._known_revisions()
    assert current in known
    monkeypatch.setattr(migrations, "_known_revisions", lambda: known - {current})

    with migrations._migration_admission(migrated_db_url) as should_migrate:
        assert should_migrate is False


def test_sibling_candidates_share_serialized_migration_gate(migrated_db_url: str) -> None:
    """API and worker cannot independently race the same migration boundary."""
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with migrations._migration_admission(migrated_db_url) as should_migrate:
            assert should_migrate is True
            first_entered.set()
            assert release_first.wait(timeout=10)

    def second() -> None:
        with migrations._migration_admission(migrated_db_url) as should_migrate:
            assert should_migrate is True
            second_entered.set()

    with ThreadPoolExecutor(max_workers=2) as executor, ExitStack():
        first_future = executor.submit(first)
        assert first_entered.wait(timeout=10)
        second_future = executor.submit(second)
        assert not second_entered.wait(timeout=0.25)
        release_first.set()
        first_future.result(timeout=10)
        second_future.result(timeout=10)

    assert second_entered.is_set()
