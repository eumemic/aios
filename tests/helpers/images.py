"""Shared image fixtures for vision / attachment rendering tests."""

from __future__ import annotations

from io import BytesIO


def valid_jpeg_bytes() -> bytes:
    """A genuinely-decodable 1x1 JPEG.

    The render boundary decode-gates inline images (``context._apply_attachments``
    degrades an undecodable attachment to a text marker rather than 400-ing the
    provider on every wake), so any test asserting the *inline* path must stage
    bytes Pillow can actually decode — not the ``b"...bytes"`` literals that
    passed the old magic-sniff-only gate.
    """
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (1, 1), (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()
