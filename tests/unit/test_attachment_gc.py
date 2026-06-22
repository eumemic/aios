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
call doesn't own), one layer up: the sweep doesn't own ANY
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


async def test_sweep_retains_inline_sibling_when_query_returns_it(
    attachments_root: Path,
    fake_pool: MagicMock,
) -> None:
    """When ``list_attachment_paths_for_sessions`` returns both the
    original ``in_sandbox_path`` and the ``inline.in_sandbox_path``
    from a staged attachment record, the sweep must retain both
    files. Pre-feature the query only returned ``in_sandbox_path`` —
    the inline sibling would have been reaped on every restart,
    erasing the downsampled bytes the renderer relies on.
    """
    import os
    import time

    session_id = "sess_01STAGING0000000000000003"
    original = _seed_attachment(attachments_root, session_id, "echo", "evt-1-big.jpg")
    inline = _seed_attachment(attachments_root, session_id, "echo", "evt-1-big.jpg.inline.jpg")
    # Both files older than the staging-txn window so the recent-file
    # protection isn't what's saving them.
    old = time.time() - 3600
    os.utime(original, (old, old))
    os.utime(inline, (old, old))

    referenced = {
        session_id: {
            "/mnt/attachments/echo/evt-1-big.jpg",
            "/mnt/attachments/echo/evt-1-big.jpg.inline.jpg",
        }
    }
    with patch(
        "aios.harness.attachment_gc.queries.list_attachment_paths_for_sessions",
        AsyncMock(return_value=referenced),
    ):
        await sweep_orphan_attachments(fake_pool)

    assert original.exists()
    assert inline.exists()


async def test_sweep_reaps_inline_sibling_when_event_predates_feature(
    attachments_root: Path,
    fake_pool: MagicMock,
) -> None:
    """Reverse case: an inline sibling on disk whose containing event's
    record carries no ``inline`` sub-key (e.g. orphaned by a partial
    write or a prior deploy state) must still be reaped. The sweep
    treats every on-disk file as a candidate; the inline retention
    above hinges entirely on the query returning the path.
    """
    import os
    import time

    session_id = "sess_01STAGING0000000000000004"
    original = _seed_attachment(attachments_root, session_id, "echo", "evt-1-big.jpg")
    inline = _seed_attachment(attachments_root, session_id, "echo", "evt-1-big.jpg.inline.jpg")
    old = time.time() - 3600
    os.utime(original, (old, old))
    os.utime(inline, (old, old))

    # Event referenced original only (no `inline` sub-record).
    referenced = {
        session_id: {"/mnt/attachments/echo/evt-1-big.jpg"},
    }
    with patch(
        "aios.harness.attachment_gc.queries.list_attachment_paths_for_sessions",
        AsyncMock(return_value=referenced),
    ):
        deleted = await sweep_orphan_attachments(fake_pool)

    assert original.exists()
    assert not inline.exists()
    assert deleted == 1


def _seed_spill(root: Path, session_id: str, filename: str) -> Path:
    """Create a tool-result spill file at ``_attachments/<sid>/tool_results/<filename>``.

    Mirrors ``cap_tool_result_content`` (#735): oversized tool output spills
    to the ``tool_results`` subdir of the session's attachments dir, recorded
    (post-#1093) in the tool-role event's ``metadata.attachments`` like any
    staged inbound so the GC's referenced-set query protects it.
    """
    target = root / session_id / "tool_results" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x" * 4096, encoding="utf-8")
    return target


async def test_sweep_retains_spill_file_recorded_in_metadata_attachments(
    attachments_root: Path,
    fake_pool: MagicMock,
) -> None:
    """#1093: a tool-result spill file whose reference is recorded in the
    tool event's ``metadata.attachments`` (the single, unified convention)
    must be retained by the sweep.

    Pre-fix the spill writer recorded its path ONLY in the result-content
    stub, never in ``metadata.attachments``; ``list_attachment_paths_for_sessions``
    is built exclusively from ``data->'metadata'->'attachments'``, so the
    spill file was invisible to the referenced-set and reaped on the next
    worker boot — after which the event log still told the model to ``read``
    a now-missing file. With the spill recorded under the same convention
    every staged inbound uses, the EXISTING query returns its sandbox path
    and the sweep keeps it.
    """
    import os
    import time

    session_id = "sess_01SPILL00000000000000001"
    spill = _seed_spill(attachments_root, session_id, "tc_spill_1.txt")
    # Older than the in-flight window so the recent-file protection isn't
    # what's saving it — only the referenced-set membership is.
    old = time.time() - 3600
    os.utime(spill, (old, old))
    assert spill.exists()

    # The fixed spill writer records this path in metadata.attachments, so
    # the referenced-set query surfaces it exactly as the GC walk reconstructs
    # it: /mnt/attachments/tool_results/<file>.
    referenced = {session_id: {"/mnt/attachments/tool_results/tc_spill_1.txt"}}
    with patch(
        "aios.harness.attachment_gc.queries.list_attachment_paths_for_sessions",
        AsyncMock(return_value=referenced),
    ):
        deleted = await sweep_orphan_attachments(fake_pool)

    assert spill.exists(), (
        "a spill file recorded in metadata.attachments must be retained by "
        "the sweep — pre-#1093 the spill reference lived only in the result "
        "content stub, invisible to the referenced-set, and was reaped on "
        "the next worker boot, leaving the model pointed at a deleted file "
        "it was told to read."
    )
    assert deleted == 0


async def test_sweep_reaps_orphaned_spill_file_with_no_reference(
    attachments_root: Path,
    fake_pool: MagicMock,
) -> None:
    """The complement of the retain case (and the self-heal path in #1093's
    migration plan): a spill file under ``tool_results/`` that NO event
    references — e.g. an already-orphaned file written before the fix, or a
    spill whose tool-result append rolled back — must still be reaped. The
    sweep treats the ``tool_results`` subdir like any connector dir; only
    membership in the referenced-set keeps a file alive.
    """
    import os
    import time

    session_id = "sess_01SPILL00000000000000002"
    spill = _seed_spill(attachments_root, session_id, "tc_orphan.txt")
    old = time.time() - 3600
    os.utime(spill, (old, old))
    assert spill.exists()

    with patch(
        "aios.harness.attachment_gc.queries.list_attachment_paths_for_sessions",
        AsyncMock(return_value={}),
    ):
        deleted = await sweep_orphan_attachments(fake_pool)

    assert not spill.exists()
    assert deleted == 1
