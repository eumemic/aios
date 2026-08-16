from __future__ import annotations

import ast
import inspect
import os
import textwrap
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aios.harness import worker
from aios.harness.attachment_gc import sweep_orphan_uploads


def test_worker_startup_does_not_call_upload_reconciliation() -> None:
    """Pin the call graph, not a coincidentally empty filesystem outcome."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(worker.worker_main)))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "sweep_orphan_uploads" not in called_names


class _AsyncContext:
    async def __aenter__(self) -> Any:
        return MagicMock()

    async def __aexit__(self, *_args: object) -> None:
        return None


def _write_old(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("bytes")
    old = time.time() - 3600
    os.utime(path, (old, old))


async def test_sweep_skips_live_session_with_no_file_rows(tmp_path: Path) -> None:
    root = tmp_path / "_uploads"
    operator_placed = root / "sess_live" / "credentials" / "operator.pem"
    _write_old(operator_placed)

    pool = MagicMock()
    pool.acquire.return_value = _AsyncContext()
    with (
        patch("aios.harness.attachment_gc.uploads_root", return_value=root),
        patch(
            "aios.harness.attachment_gc.queries.list_upload_paths_for_sessions",
            AsyncMock(return_value={"sess_live": None}),
        ),
    ):
        deleted = await sweep_orphan_uploads(pool)

    assert deleted == 0
    assert operator_placed.exists()


async def test_sweep_reaps_orphan_upload_and_gone_session_directory(tmp_path: Path) -> None:
    root = tmp_path / "_uploads"
    orphan = root / "sess_live" / "file_orphan" / "orphan.txt"
    retained = root / "sess_live" / "file_kept" / "kept.txt"
    gone = root / "sess_gone" / "file_old" / "old.txt"
    for path in (orphan, retained, gone):
        _write_old(path)

    pool = MagicMock()
    pool.acquire.return_value = _AsyncContext()
    referenced = {"sess_live": {str(retained)}}
    with (
        patch("aios.harness.attachment_gc.uploads_root", return_value=root),
        patch(
            "aios.harness.attachment_gc.queries.list_upload_paths_for_sessions",
            AsyncMock(return_value=referenced),
        ),
    ):
        deleted = await sweep_orphan_uploads(pool)

    assert deleted == 2
    assert retained.exists()
    assert not orphan.exists()
    assert not (root / "sess_gone").exists()
