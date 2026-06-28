"""Shared image fixtures for vision / attachment rendering tests."""

from __future__ import annotations

from io import BytesIO


def _valid_image(fmt: str) -> bytes:
    """A genuinely-decodable 1x1 image in ``fmt`` (e.g. ``"JPEG"``, ``"PNG"``).

    The render boundary decode-gates inline images (``context._apply_attachments``
    degrades an undecodable attachment to a text marker rather than 400-ing the
    provider on every wake), so any test asserting the *inline* path must stage
    bytes Pillow can actually decode — not the ``b"...bytes"`` literals that
    passed the old magic-sniff-only gate.
    """
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (1, 1), (10, 20, 30)).save(buf, format=fmt)
    return buf.getvalue()


def valid_jpeg_bytes() -> bytes:
    """A genuinely-decodable 1x1 JPEG. See :func:`_valid_image`."""
    return _valid_image("JPEG")


def valid_png_bytes() -> bytes:
    """A genuinely-decodable 1x1 PNG. See :func:`_valid_image`."""
    return _valid_image("PNG")


def valid_gif_bytes() -> bytes:
    """A genuinely-decodable 1x1 GIF. See :func:`_valid_image`."""
    return _valid_image("GIF")


def valid_webp_bytes() -> bytes:
    """A genuinely-decodable 1x1 WEBP. See :func:`_valid_image`."""
    return _valid_image("WEBP")


def valid_tiff_bytes() -> bytes:
    """A genuinely-decodable 1x1 TIFF — a format Pillow decodes but no vision
    provider accepts for inlining, used to exercise the render boundary's
    provider-format gate. See :func:`_valid_image`."""
    return _valid_image("TIFF")


def large_png_bytes(side: int = 4000, *, noisy: bool = False) -> bytes:
    """A genuinely-decodable opaque RGB PNG ``side``x``side`` px.

    Used by the read()-inline and replay-clamp tests to exercise the
    dimension-downscale path: an opaque image >2000px must come back
    re-encoded as JPEG and <=2000px on each side. ``noisy`` fills the
    image with high-entropy pixels so the JPEG/PNG re-encode can't trivially
    compress it to nothing (useful for cap-pressure tests).
    """
    from PIL import Image

    if noisy:
        import os

        img = Image.frombytes("RGB", (side, side), os.urandom(side * side * 3))
    else:
        img = Image.new("RGB", (side, side), (10, 20, 30))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def large_transparent_png_bytes(side: int = 4000) -> bytes:
    """A genuinely-decodable RGBA PNG ``side``x``side`` px with transparency.

    Used to assert the read()-inline / replay-clamp transparency path keeps
    a PNG (or palette-PNG) content type rather than flattening to JPEG.
    """
    from PIL import Image

    img = Image.new("RGBA", (side, side), (10, 20, 30, 128))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
