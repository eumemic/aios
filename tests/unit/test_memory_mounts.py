"""Tests for ``materialize_store_to_host`` race-window behavior.

The first session to provision a store materializes the host-mirror
directory from DB. ``_mirror_to_host`` (called after API/tool writes
commit to DB) gates on ``host_dir.exists()`` — which becomes True the
moment ``materialize_store_to_host`` calls ``host_dir.mkdir`` at the
top of its body, BEFORE the DB snapshot read. That opens a window
where a concurrent mirror writes the new content to disk and then
materialize's atomic-write loop clobbers it with the stale snapshot
value.

Single-process async-land: ``atomic_write``'s temp-create plus
``os.replace`` happen inside one sync block (no scheduler yield), so a
``target.exists()`` skip in the materialize loop closes the race
without needing a lock.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aios.sandbox.atomic_mirror import atomic_write
from aios.sandbox.memory_mounts import materialize_store_to_host


@pytest.fixture
def store_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Per-test host_dir + lock_path under the test's tmp_path."""
    return tmp_path / "host", tmp_path / "lock"


async def test_materialize_defers_to_concurrent_disk_write(
    store_paths: tuple[Path, Path],
) -> None:
    """A concurrent write to the host dir during materialize's snapshot
    read must not be clobbered by the materialize's stale snapshot.

    Pre-fix the materialize loop ran ``atomic_write`` for every
    snapshot entry unconditionally — so a fresher write that landed
    during the snapshot-read yield got overwritten with the stale value.
    Post-fix the loop checks ``target.exists()`` before writing and
    defers to whoever wrote first; since ``materialize_store_to_host``
    creates the host dir empty before reading the snapshot, any
    existing file at the target path was placed there by a concurrent
    writer who had a strictly fresher DB view.
    """
    host_dir, lock_path = store_paths

    snapshot_returned = asyncio.Event()
    materialize_can_continue = asyncio.Event()

    async def slow_snapshot(*_args: Any, **_kwargs: Any) -> list[tuple[str, str]]:
        snapshot_returned.set()
        await materialize_can_continue.wait()
        return [("/foo.md", "v1")]  # stale snapshot value

    with (
        patch("aios.sandbox.memory_mounts.memory_store_host_dir", return_value=host_dir),
        patch("aios.sandbox.memory_mounts.memory_store_lock_path", return_value=lock_path),
        patch(
            "aios.db.queries.list_active_memory_paths_and_content",
            slow_snapshot,
        ),
    ):
        mat_task = asyncio.create_task(
            materialize_store_to_host(
                MagicMock(),
                store_id="memstore_test",
                account_id="acc_test_stub",
            )
        )
        await snapshot_returned.wait()

        # Concurrent write — production shape: ``_mirror_to_host`` after a
        # committed API update with content="v2". host_dir already exists
        # (materialize just made it), so the mirror's exists() gate fires
        # and the atomic_write lands.
        atomic_write(host_dir / "foo.md", "v2")

        materialize_can_continue.set()
        await mat_task

    actual = (host_dir / "foo.md").read_text()
    assert actual == "v2", (
        f"materialize must defer to the concurrent writer; got {actual!r}. "
        f"Pre-fix symptom: the stale snapshot value 'v1' clobbered the "
        f"freshly-mirrored 'v2', leaving DB and disk permanently inconsistent."
    )


async def test_materialize_writes_when_target_absent(
    store_paths: tuple[Path, Path],
) -> None:
    """No concurrent writer → materialize writes its snapshot as
    before. Verifies the skip-if-exists fix doesn't break the happy
    path (empty store → populated from DB)."""
    host_dir, lock_path = store_paths

    async def quick_snapshot(*_args: Any, **_kwargs: Any) -> list[tuple[str, str]]:
        return [("/foo.md", "v1"), ("/sub/bar.md", "B")]

    with (
        patch("aios.sandbox.memory_mounts.memory_store_host_dir", return_value=host_dir),
        patch("aios.sandbox.memory_mounts.memory_store_lock_path", return_value=lock_path),
        patch(
            "aios.db.queries.list_active_memory_paths_and_content",
            quick_snapshot,
        ),
    ):
        await materialize_store_to_host(
            MagicMock(),
            store_id="memstore_test",
            account_id="acc_test_stub",
        )

    assert (host_dir / "foo.md").read_text() == "v1"
    assert (host_dir / "sub" / "bar.md").read_text() == "B"
    assert (host_dir / ".materialized").exists()
