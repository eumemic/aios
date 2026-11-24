"""F4 end-to-end: ``rm gone.md; chmod -R u+rw .`` in session A, then a peer
provision before A's reconcile lands. The delete must stay on disk and reach
the DB through A's reconcile."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from tests.unit.test_memory_mounts import _materialize


async def test_in_flight_delete_with_chmod_survives_peer_provision(tmp_path: Path) -> None:
    from aios.harness import runtime
    from aios.models.memory_stores import MemoryStoreResourceEcho
    from aios.tools.bash_memory_reconcile import (
        reconcile_memory_mounts,
        snapshot_memory_mounts,
    )

    host_dir, lock_path = tmp_path / "host", tmp_path / "lock"
    store_id = "memstore_test"
    session_id = "sess_in_flight_delete_chmod"
    rows = [("/keep.md", "k"), ("/gone.md", "g")]
    await _materialize(host_dir, lock_path, rows)

    echo = MemoryStoreResourceEcho(
        memory_store_id=store_id,
        access="read_write",
        instructions="",
        name="notes",
        description="",
        mount_path="/mnt/memory/notes",
    )
    runtime.set_session_memory_mounts(session_id, [echo])
    fake_memory = MagicMock()
    fake_memory.id = "mem_gone"
    delete = AsyncMock()
    prev_pool = runtime.pool
    runtime.pool = MagicMock()
    try:
        with patch("aios.tools.bash_memory_reconcile.memory_store_host_dir", return_value=host_dir):
            before, snapshot_ns = snapshot_memory_mounts(session_id)
            (host_dir / "gone.md").unlink()
            await asyncio.to_thread(
                subprocess.run, ["chmod", "-R", "u+rw", str(host_dir)], check=True
            )
            restored = await _materialize(host_dir, lock_path, rows)
            with (
                patch(
                    "aios.tools.bash_memory_reconcile.memory_service.get_memory_by_path",
                    AsyncMock(return_value=fake_memory),
                ),
                patch("aios.tools.bash_memory_reconcile.memory_service.delete_memory", delete),
                patch("aios.tools.bash_memory_reconcile.memory_service.update_memory", AsyncMock()),
                patch("aios.tools.bash_memory_reconcile.memory_service.create_memory", AsyncMock()),
                patch(
                    "aios.tools.bash_memory_reconcile.sessions_service.load_session_account_id",
                    AsyncMock(return_value="acc_test_stub"),
                ),
            ):
                await reconcile_memory_mounts(session_id, before, snapshot_ns)
    finally:
        runtime.pool = prev_pool
        runtime.clear_session_memory_mounts(session_id)

    assert not restored, "peer provision reported a rollback for an in-flight delete"
    assert not (host_dir / "gone.md").exists(), "peer provision resurrected a deleted file"
    delete.assert_awaited_once()
