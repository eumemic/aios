"""SDK boundary validation for the :class:`Attachment` dataclass.

The 5 MiB cap and path-readable check live at the SDK layer (not in the
supervisor) so connector authors get a clean ``AttachmentError`` at
``emit_inbound`` time rather than an opaque crash deep in the harness
pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aios_connector import Attachment, AttachmentError


def test_as_params_returns_wire_dict(tmp_path: Path) -> None:
    blob = tmp_path / "photo.jpg"
    blob.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")

    att = Attachment(
        host_path=str(blob),
        filename="photo.jpg",
        content_type="image/jpeg",
    )

    assert att.as_params() == {
        "host_path": str(blob),
        "filename": "photo.jpg",
        "content_type": "image/jpeg",
        "size": len(b"\xff\xd8\xff\xe0fake-jpeg-bytes"),
    }


def test_as_params_rejects_missing_path(tmp_path: Path) -> None:
    att = Attachment(
        host_path=str(tmp_path / "does-not-exist.jpg"),
        filename="x.jpg",
        content_type="image/jpeg",
    )
    with pytest.raises(AttachmentError, match="does not exist"):
        att.as_params()


def test_as_params_rejects_directory(tmp_path: Path) -> None:
    """A directory is not a regular file — same error path as missing.

    Catches the common bug of passing the parent of the temp file by
    accident.
    """
    att = Attachment(
        host_path=str(tmp_path),
        filename="x.jpg",
        content_type="image/jpeg",
    )
    with pytest.raises(AttachmentError, match="not a regular file"):
        att.as_params()


def test_as_params_rejects_oversize(tmp_path: Path) -> None:
    """5 MiB + 1 byte trips the cap — connector decides how to handle.

    The cap lives at the SDK boundary so the connector sees the rejection
    synchronously rather than discovering it via a supervisor drop.
    """
    big = tmp_path / "huge.bin"
    big.write_bytes(b"\0" * (5 * 1024 * 1024 + 1))
    att = Attachment(
        host_path=str(big),
        filename="huge.bin",
        content_type="application/octet-stream",
    )
    with pytest.raises(AttachmentError, match="5 MiB"):
        att.as_params()


def test_as_params_accepts_exactly_at_cap(tmp_path: Path) -> None:
    blob = tmp_path / "edge.bin"
    blob.write_bytes(b"\0" * (5 * 1024 * 1024))
    att = Attachment(
        host_path=str(blob),
        filename="edge.bin",
        content_type="application/octet-stream",
    )
    assert att.as_params()["size"] == 5 * 1024 * 1024


def test_attachment_is_frozen() -> None:
    """frozen=True on the dataclass — connector code can't mutate after build."""
    att = Attachment(host_path="/x", filename="y", content_type="z")
    with pytest.raises((AttributeError, TypeError)):
        att.host_path = "/elsewhere"  # type: ignore[misc]
