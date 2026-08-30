"""Real-Postgres controls for candidate migration admission."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import psycopg
import pytest

from aios.db import migrations


def _current_revision(db_url: str) -> str:
    with psycopg.connect(migrations._sync_db_url(db_url)) as conn:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    assert row is not None
    return str(row[0])


def test_candidate_admits_real_previous_in_image_revision(migrated_db_url: str) -> None:
    """A database one revision behind this image proceeds to migration."""
    current = _current_revision(migrated_db_url)
    previous = str(int(current) - 1).zfill(len(current))
    assert previous in migrations._known_revisions()

    sync_url = migrations._sync_db_url(migrated_db_url)
    with psycopg.connect(sync_url, autocommit=True) as conn:
        conn.execute("UPDATE alembic_version SET version_num = %s", (previous,))
    try:
        with migrations._migration_admission(migrated_db_url) as should_migrate:
            assert should_migrate is True
    finally:
        with psycopg.connect(sync_url, autocommit=True) as conn:
            conn.execute("UPDATE alembic_version SET version_num = %s", (current,))


def test_rollback_image_admitted_after_forward_revision(
    migrated_db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An adjacent older image does not ask Alembic to resolve the new stamp."""
    current = _current_revision(migrated_db_url)
    known = migrations._known_revisions()
    previous = str(int(current) - 1).zfill(len(current))
    assert previous in known
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
