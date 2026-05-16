"""Unit tests for ``aios.harness.attachment_gc``.

The sweep must NOT delete files that were just staged by an
in-flight inbound transaction. ``handle_inbound`` writes attachment
bytes to disk BEFORE its ``append_event`` + ``try_record_inbound_ack``
transaction commits. If a worker reboot (laptop sleep, redeploy, OOM)
fires ``sweep_orphan_attachments`` between the rename and the commit,
the sweep sees a file with no committed event row referencing it and
classifies it as orphan. Without an age filter, it unlinks. The
API's commit then succeeds but the persisted ``in_sandbox_path``
points at a deleted file — the renderer's later read raises
``FileNotFoundError`` and the attachment is silently lost.

Same defect class as PR #517 (cleanup scope: don't unlink files this
invocation doesn't own), one layer up: the sweep doesn't own ANY
file it didn't write, so the conservative rule is to skip recently
created files entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness.attachment_gc import sweep_orphan_attachments


@pytest.fixture
def attachments_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test attachments root pointed at a tmpdir."""
    root = tmp_path / "_attachments"
    root.mkdir()
    monkeypatch.setattr(
        "aios.harness.attachment_gc.attachments_root",
        lambda: root,
    )
    return root


def _seed_attachment(root: Path, session_id: str, connector: str, filename: str) -> Path:
    """Create ``_attachments/<sid>/<conn>/<filename>`` and return its path."""
    target = root / session_id / connector / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\x89PNG\r\n\x1a\nstaged-bytes-just-now")
    return target


@pytest.fixture
def fake_pool() -> MagicMock:
    """Pool whose ``acquire()`` returns a context-managed MagicMock conn."""
    pool = MagicMock()

    class _AsyncCm:
        async def __aenter__(self) -> Any:
            return MagicMock()

        async def __aexit__(self, *_args: Any) -> None:
            return None

    pool.acquire.return_value = _AsyncCm()
    return pool


async def test_sweep_skips_recently_staged_file_with_no_event_yet(
    attachments_root: Path,
    fake_pool: MagicMock,
) -> None:
    """A file just placed on disk by ``stage_inbound_attachments`` —
    whose ``append_event`` + dedup txn hasn't committed yet — must
    NOT be unlinked by the sweep. Pre-fix the sweep had no age
    filter and would race-delete the file before the API's commit;
    post-fix it should skip files newer than the staging-txn
    latency budget."""
    session_id = "sess_01STAGING0000000000000001"
    connector = "echo"
    filename = "evt_inflight-photo.png"
    file_path = _seed_attachment(attachments_root, session_id, connector, filename)
    assert file_path.exists()

    # The events table has NO row referencing this path — the txn is
    # mid-flight on a different process. Mock the query to return empty.
    with patch(
        "aios.harness.attachment_gc.queries.list_attachment_paths_for_sessions",
        AsyncMock(return_value={}),
    ):
        await sweep_orphan_attachments(fake_pool)

    assert file_path.exists(), (
        "sweep must skip recently-staged files (mtime within the staging-"
        "txn latency window); pre-fix the sweep race-deletes any file "
        "whose referencing event hasn't committed yet, which happens "
        "deterministically on every worker restart that lands between "
        "the stage_inbound_attachments rename and the _append_with_dedup "
        "commit. Renderer would later FileNotFoundError on the in-sandbox "
        "path the committed event ends up referencing."
    )


async def test_sweep_still_unlinks_genuinely_old_orphans(
    attachments_root: Path,
    fake_pool: MagicMock,
) -> None:
    """The fix mustn't disable the sweep entirely. A file old enough
    that its inbound txn definitely either committed (and the events
    table now references it) or rolled back (and it should be reaped)
    must still be unlinked when no event references it. Set mtime
    well in the past to simulate a long-completed-rolled-back txn."""
    import os
    import time

    session_id = "sess_01STAGING0000000000000002"
    file_path = _seed_attachment(attachments_root, session_id, "echo", "evt_old-img.png")
    # Mtime 1 hour ago — far past any sane staging-txn latency.
    old = time.time() - 3600
    os.utime(file_path, (old, old))
    assert file_path.exists()

    with patch(
        "aios.harness.attachment_gc.queries.list_attachment_paths_for_sessions",
        AsyncMock(return_value={}),
    ):
        deleted = await sweep_orphan_attachments(fake_pool)

    assert not file_path.exists(), (
        "sweep must still reap genuinely-orphaned old files; the age "
        "filter should be a small protect-in-flight-txn window, not a "
        "disable-the-sweep refactor."
    )
    assert deleted == 1
