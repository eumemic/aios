"""Downsample inbound images to fit the inline cap.

Called once per upload from :mod:`aios.services.attachment_staging`; the
result is written to a sibling ``.inline.<ext>`` file and referenced
from a new ``inline`` sub-record on the attachment metadata.

Pillow is CPU-bound — :func:`maybe_downsample` wraps the decode +
re-encode in :func:`asyncio.to_thread` so other sessions handled by the
same worker don't stall while we re-encode a 5 MB JPEG.

Resize policy: dimension downscale first (preserves quality and is
often enough on its own), then JPEG re-encode at quality
80 → 60 → 40 → 25.  Transparency-bearing images stay in the PNG path
and fall back to palette PNG when the cap is tight.  Mirrors
claude-code's ``imageResizer.ts`` ladder.
"""

from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING, Any, NamedTuple

from aios.harness.vision import (
    INLINE_MAX_DIMENSION,
    INLINE_SIZE_CAP_BYTES,
    PRE_RESIZE_CEILING_BYTES,
)
from aios.logging import get_logger

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = get_logger("aios.harness.image_resize")

# Decompression-bomb ceiling.  Pillow's default is 89 MP; we tighten to
# 100 MP (~10000² inputs) — well above any realistic phone/screenshot
# upload but well below the multi-100-MP pathological inputs that would
# decode to a 600 MB+ RGB matrix and OOM the worker.  Applied once at
# first decode (idempotent assignment to the PIL module attribute).
_PIXEL_LIMIT = 100_000_000


def _pillow() -> Any:
    """Deferred Pillow accessor.

    Avoids pulling Pillow (200-400ms cold) into every API/worker boot,
    matching the deferred-import pattern this codebase uses for heavy
    third-party modules (see ``vision.py:67`` for litellm).  Idempotently
    applies ``MAX_IMAGE_PIXELS`` on every call so the ceiling is in
    effect by the time the caller invokes ``Image.open``.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = _PIXEL_LIMIT
    return Image


class ImageDownsampleError(Exception):
    """Raised when an oversize image can't be downsampled to fit the cap.

    Caller (staging) logs and falls through to the existing text-marker
    degradation: the model still sees ``[image: foo.jpg (..., at /mnt/...)]``
    and can ``read`` the original from the sandbox if it wants the pixels.
    """


class ImageDownsampleResult(NamedTuple):
    """Bytes + metadata for a successful downsample.

    ``content_type`` may differ from the original (PNG → JPEG when no
    transparency is present), so the caller must persist what we return
    rather than reusing the input mime.
    """

    data: bytes
    content_type: str
    width: int
    height: int


_JPEG_QUALITY_LADDER = (80, 60, 40, 25)


async def maybe_downsample(
    data: bytes,
    content_type: str,
    *,
    cap_bytes: int = INLINE_SIZE_CAP_BYTES,
    max_dim: int = INLINE_MAX_DIMENSION,
    pre_resize_ceiling: int = PRE_RESIZE_CEILING_BYTES,
) -> ImageDownsampleResult | None:
    """Downsample ``data`` to fit ``cap_bytes`` raw and ``max_dim`` px.

    Returns ``None`` when the original already fits both caps and no
    work is needed.  Raises :class:`ImageDownsampleError` when the
    input exceeds ``pre_resize_ceiling``, when Pillow can't decode the
    bytes, or when the final step of the quality ladder still
    overshoots ``cap_bytes``.

    The ``content_type`` argument is accepted for symmetry with the
    caller's record shape; the actual format chosen for the output is
    decoded from the bytes (Pillow's format detection is more reliable
    than the upstream content-type header, which sometimes lies).
    """
    if len(data) > pre_resize_ceiling:
        raise ImageDownsampleError(
            f"image size {len(data)}B exceeds pre-resize ceiling {pre_resize_ceiling}B"
        )
    return await asyncio.to_thread(_blocking_downsample, data, cap_bytes, max_dim)


def _blocking_downsample(
    data: bytes,
    cap_bytes: int,
    max_dim: int,
) -> ImageDownsampleResult | None:
    pil = _pillow()
    try:
        img = pil.open(io.BytesIO(data))
        # ``Image.open`` parses the header (.size, .format, .mode populated)
        # but does NOT decode pixel data — ``img.load()`` does.  If the
        # image already fits both caps, skip the decode entirely; this
        # keeps the common-case "small image goes through staging" path
        # cheap even though the caller now always invokes us (so that
        # high-resolution-but-well-compressed inputs can't bypass the
        # dimension cap via the byte-size guard).
        if len(data) <= cap_bytes and img.width <= max_dim and img.height <= max_dim:
            return None
        img.load()
    except Exception as err:
        # Pillow throws a mix of UnidentifiedImageError (subclass of
        # OSError), DecompressionBombError, ValueError, and provider-
        # specific OSErrors for truncated streams.  Collapse to our own
        # exception type — the caller falls through to the text marker
        # either way; the inner message is preserved for the warn log.
        raise ImageDownsampleError(f"pillow decode failed: {err}") from err

    has_transparency = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info

    if img.width > max_dim or img.height > max_dim:
        img.thumbnail((max_dim, max_dim), pil.Resampling.LANCZOS)

    if has_transparency:
        return _encode_with_transparency(img, cap_bytes)
    return _encode_as_jpeg(img, cap_bytes)


def _encode_as_jpeg(img: PILImage.Image, cap_bytes: int) -> ImageDownsampleResult:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    final_size = 0
    for quality in _JPEG_QUALITY_LADDER:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        final_size = buf.tell()
        if final_size <= cap_bytes:
            return ImageDownsampleResult(
                data=buf.getvalue(),
                content_type="image/jpeg",
                width=img.width,
                height=img.height,
            )
    raise ImageDownsampleError(
        f"jpeg re-encoding overshoots cap {cap_bytes}B even at quality "
        f"{_JPEG_QUALITY_LADDER[-1]} (final size {final_size}B)"
    )


def _encode_with_transparency(img: PILImage.Image, cap_bytes: int) -> ImageDownsampleResult:
    """Encode a transparency-bearing image, preferring PNG fidelity but
    falling back to palette PNG when the cap is tight."""
    pil = _pillow()
    # Both encode steps below only accept some modes: PNG cannot write ``PA``/``La``,
    # and ``convert("P", ADAPTIVE)`` rejects ``LA`` (a grayscale+alpha PNG — the common
    # transparent-logo case — decodes to ``LA``). Normalize any non-canonical mode to
    # ``RGBA`` (the full-alpha mode both steps accept) so the encode is total; ``P`` is
    # already PNG-saveable and the palette target, so it is left as-is. Without this the
    # raw ValueError/OSError escaped ImageDownsampleError → 500'd the connector inbound.
    if img.mode not in ("RGBA", "P"):
        img = img.convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    if buf.tell() <= cap_bytes:
        return ImageDownsampleResult(
            data=buf.getvalue(),
            content_type="image/png",
            width=img.width,
            height=img.height,
        )
    palette = img.convert("P", palette=pil.Palette.ADAPTIVE, colors=256)
    buf = io.BytesIO()
    palette.save(buf, format="PNG", optimize=True)
    final_size = buf.tell()
    if final_size <= cap_bytes:
        return ImageDownsampleResult(
            data=buf.getvalue(),
            content_type="image/png",
            width=palette.width,
            height=palette.height,
        )
    raise ImageDownsampleError(
        f"transparent PNG encoding overshoots cap {cap_bytes}B even at palette "
        f"PNG (final size {final_size}B)"
    )
