"""Magic-byte image-mime detector.

Anthropic's ``/v1/messages`` validates declared mime against actual
bytes and 400s on mismatch; inbound platform metadata occasionally
disagrees with the bytes, so callers re-detect rather than trust the
declaration.
"""

from __future__ import annotations

_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def sniff_image_mime(data: bytes) -> str | None:
    """Return the detected mime, or ``None`` for short or unrecognised bytes."""
    for sig, mime in _IMAGE_MAGIC:
        if data.startswith(sig):
            return mime
    return None
