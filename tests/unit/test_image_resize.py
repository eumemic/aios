"""Unit coverage for :mod:`aios.harness.image_resize`.

Pure pixel work — no I/O beyond Pillow's in-memory encoders.  We
build synthetic images at known dimensions/sizes so each branch of
the resize ladder (no-op, dimension downscale, quality step, palette
fallback, ceiling rejection, decode failure) is exercised
deterministically.
"""

from __future__ import annotations

import functools
import io
import random

import pytest
from PIL import Image

from aios.harness.image_resize import (
    ImageDownsampleError,
    ImageDownsampleResult,
    maybe_downsample,
)
from aios.harness.vision import (
    INLINE_MAX_DIMENSION,
    INLINE_SIZE_CAP_BYTES,
    PRE_RESIZE_CEILING_BYTES,
)


def _encode(img: Image.Image, fmt: str, **save_kwargs: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


@functools.cache
def _noisy_rgb(width: int, height: int) -> Image.Image:
    """Return an RGB image whose pixel data resists JPEG compression.

    Solid colors compress to a few bytes regardless of quality, which
    makes it impossible to test the quality ladder.  Seeded random
    bytes keep the encoded size proportional to dimensions.

    The old per-pixel Python loop wrote every channel by hand (12.25M
    ``PixelAccess`` writes for the 3500x3500 case, regenerated per
    test).  ``Random(seed).randbytes`` + :meth:`Image.frombytes` builds
    the same noise in one buffer, and ``functools.cache`` memoizes the
    result so identical (width, height) requests reuse it.  Callers
    must treat the returned image as read-only — copy before mutating.
    """
    data = random.Random(f"{width}x{height}").randbytes(width * height * 3)
    return Image.frombytes("RGB", (width, height), data)


@functools.cache
def _noisy_la(width: int, height: int) -> Image.Image:
    """Return a grayscale+alpha (mode ``LA``) noise image.

    A grayscale logo/screenshot with a transparent background decodes to mode
    ``LA`` via ``Image.open`` (PNG color type 4).  Noise keeps the encoded PNG
    above the byte cap so the transparency encoder must reach its palette
    fallback — the site that historically rejected ``LA``.
    """
    data = random.Random(f"LA{width}x{height}").randbytes(width * height * 2)
    return Image.frombytes("LA", (width, height), data)


class TestNoOp:
    async def test_returns_none_when_image_fits_both_caps(self) -> None:
        small = _encode(_noisy_rgb(50, 50), "JPEG", quality=90)
        assert len(small) < INLINE_SIZE_CAP_BYTES
        result = await maybe_downsample(small, "image/jpeg")
        assert result is None


class TestDimensionDownscale:
    async def test_oversize_dimensions_get_thumbnailed(self) -> None:
        # Small file size but oversized dimensions — the dimension
        # cap should trigger downscale even though the byte cap is met.
        oversize_dims = _encode(_noisy_rgb(INLINE_MAX_DIMENSION + 500, 100), "JPEG", quality=70)
        result = await maybe_downsample(oversize_dims, "image/jpeg")
        assert result is not None
        assert result.width <= INLINE_MAX_DIMENSION
        assert result.height <= INLINE_MAX_DIMENSION
        assert result.content_type == "image/jpeg"


class TestJpegQualityLadder:
    async def test_oversize_jpeg_shrinks_under_cap(self) -> None:
        # Big enough that the quality ladder must kick in to fit the cap.
        big = _encode(_noisy_rgb(3000, 3000), "JPEG", quality=95)
        assert len(big) > INLINE_SIZE_CAP_BYTES
        result = await maybe_downsample(big, "image/jpeg")
        assert result is not None
        assert len(result.data) <= INLINE_SIZE_CAP_BYTES
        assert result.content_type == "image/jpeg"
        # Dimension cap enforced before quality ladder.
        assert result.width <= INLINE_MAX_DIMENSION
        assert result.height <= INLINE_MAX_DIMENSION


class TestPngTransparency:
    async def test_transparent_png_stays_png(self) -> None:
        # Oversize dimensions trigger resize; the transparency check
        # forces the PNG-preserving path regardless of byte savings.
        rgba = Image.new(
            "RGBA",
            (INLINE_MAX_DIMENSION + 200, INLINE_MAX_DIMENSION + 200),
            (255, 0, 0, 128),
        )
        data = _encode(rgba, "PNG")
        result = await maybe_downsample(data, "image/png")
        assert result is not None
        assert result.content_type == "image/png"

    async def test_opaque_png_converts_to_jpeg(self) -> None:
        # Oversize dimensions trigger resize; opaque images take the
        # JPEG path because it produces smaller bytes for photographic
        # content.  No transparency to preserve.
        opaque = _noisy_rgb(INLINE_MAX_DIMENSION + 200, INLINE_MAX_DIMENSION + 200)
        data = _encode(opaque, "PNG")
        result = await maybe_downsample(data, "image/png")
        assert result is not None
        assert result.content_type == "image/jpeg"

    async def test_grayscale_alpha_la_downsamples_without_crashing(self) -> None:
        # A grayscale+alpha PNG decodes to mode LA. Oversized + noisy, so the
        # first transparency PNG overshoots the cap and the encoder reaches its
        # palette fallback — convert("P", ADAPTIVE) rejects LA, which previously
        # raised a bare ValueError that escaped ImageDownsampleError, 500'd the
        # connector inbound, and poisoned its retry loop. The encoder must
        # normalize LA to RGBA and produce a valid downsampled PNG.
        la = _noisy_la(INLINE_MAX_DIMENSION + 400, INLINE_MAX_DIMENSION + 400)
        data = _encode(la, "PNG")
        result = await maybe_downsample(data, "image/png")
        assert result is not None
        assert result.content_type == "image/png"
        assert len(result.data) <= INLINE_SIZE_CAP_BYTES
        assert result.width <= INLINE_MAX_DIMENSION
        assert result.height <= INLINE_MAX_DIMENSION


class TestCeiling:
    async def test_above_ceiling_raises(self) -> None:
        # Build a buffer that exceeds the ceiling without actually
        # allocating 50 MB — pass a tiny ceiling instead.
        data = _encode(_noisy_rgb(500, 500), "JPEG", quality=90)
        with pytest.raises(ImageDownsampleError, match="ceiling"):
            await maybe_downsample(data, "image/jpeg", pre_resize_ceiling=len(data) // 2)

    async def test_default_ceiling_documented(self) -> None:
        # Smoke: confirm the default ceiling is what we documented to ops
        # (changes here are not free and warrant a deliberate bump).
        assert PRE_RESIZE_CEILING_BYTES == 50 * 1024 * 1024


class TestDecodeFailure:
    async def test_garbage_bytes_raise(self) -> None:
        with pytest.raises(ImageDownsampleError, match="decode failed"):
            await maybe_downsample(b"not an image at all", "image/jpeg")

    async def test_truncated_jpeg_raises(self) -> None:
        full = _encode(_noisy_rgb(200, 200), "JPEG", quality=80)
        with pytest.raises(ImageDownsampleError, match="decode failed"):
            await maybe_downsample(full[:100], "image/jpeg")


class TestResultShape:
    async def test_named_tuple_fields(self) -> None:
        big = _encode(_noisy_rgb(2500, 2500), "JPEG", quality=95)
        result = await maybe_downsample(big, "image/jpeg")
        assert isinstance(result, ImageDownsampleResult)
        assert isinstance(result.data, bytes)
        assert isinstance(result.content_type, str)
        assert isinstance(result.width, int)
        assert isinstance(result.height, int)
