"""F4: rollback detection must not trip on in-place sandbox metadata changes.

The store host dir is a writable bind mount, so a sandbox can chmod/chown/
touch/link the in-tree ``.materialized`` marker or the root itself, or even
delete the marker. None of that replaces the directory, so a provision that
runs after an in-place delete must leave the deleted file absent and return
False.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.unit.test_memory_mounts import _materialize


def _chmod_r(d: Path) -> None:
    subprocess.run(["chmod", "-R", "u+rw", str(d)], check=True)


def _touch_marker(d: Path) -> None:
    subprocess.run(["touch", "-c", str(d / ".materialized")], check=True)


def _utime_root(d: Path) -> None:
    os.utime(d, (0, 0))


def _link_marker(d: Path) -> None:
    os.link(d / ".materialized", d / "hardlink")
    (d / "hardlink").unlink()


def _chmod_marker(d: Path) -> None:
    os.chmod(d / ".materialized", 0o600)


def _rm_marker(d: Path) -> None:
    (d / ".materialized").unlink()


@pytest.mark.parametrize(
    "touch_cmd",
    [_chmod_r, _touch_marker, _utime_root, _link_marker, _chmod_marker, _rm_marker],
    ids=[f"touch_cmd{i}" for i in range(6)],
)
async def test_in_place_delete_after_marker_metadata_change_not_restored(
    tmp_path: Path, touch_cmd: Callable[[Path], None]
) -> None:
    host_dir, lock_path = tmp_path / "host", tmp_path / "lock"
    rows = [("/keep.md", "k"), ("/gone.md", "g")]
    await _materialize(host_dir, lock_path, rows)

    (host_dir / "gone.md").unlink()
    touch_cmd(host_dir)

    restored = await _materialize(host_dir, lock_path, rows)

    assert restored is False
    assert not (host_dir / "gone.md").exists()
    assert (host_dir / ".materialized").exists()
