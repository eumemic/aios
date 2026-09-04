"""Vision-policy helper: single source of truth for image-into-vision decisions.

LiteLLM's :func:`token_counter` returns a flat ~89 tokens per
``image_url`` part regardless of provider AND regardless of payload
size (measured 2026-08-15 on litellm 1.97.0: 89 against 9,600 /
191,447 / 1,912,670 as text for 10 KB / 200 KB / 2 MB of image
bytes).  An earlier version of this note said the under-counting
"only matters near the window boundary" and that "provider rejection
there is recoverable".  BOTH claims are false:

* it is not boundary-local -- a constant against a linear truth is an
  unbounded error, so a single inlined image can put the built request
  arbitrarily far over the ceiling; and
* it is not recoverable -- aios#2050 records the consequence as a
  4.5h HARD-DOWN, because every wake rebuilds the same oversized
  context, so the rejection repeats forever rather than clearing.

Calibration cannot correct it either: the scaling layer corrects a
RATIO, and no coefficient multiplied by zero reaches a positive number
(see the ``tokens.py`` invariant).  Fix is explicit image-mass
tracking, issue #2050 / aios#2073.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

from aios.logging import get_logger
from aios_connector_http.mime import sniff_image_mime

log = get_logger("aios.harness.vision")


def correct_image_mime_b64(declared: str, data_b64: str) -> str:
    """Return the magic-byte-detected mime for a base64-encoded image,
    or ``declared`` unchanged when sniffing yields nothing.  Warns on
    substitution so operators see when a persisted event's declared
    mime disagreed with its bytes.
    """
    head_b64 = data_b64[:24]
    pad = (-len(head_b64)) % 4
    try:
        head = base64.b64decode(head_b64 + "=" * pad)
    except binascii.Error:
        return declared
    sniffed = sniff_image_mime(head)
    if sniffed is None or sniffed == declared:
        return declared
    log.warning("vision.image_mime_corrected", declared=declared, actual=sniffed)
    return sniffed


INLINE_SIZE_CAP_BYTES = 3_932_160  # 3.75 MiB — matches Anthropic's 5 MB base64 API ceiling.

# Largest edge (px) for downsampled inline copies.  See
# :mod:`aios.harness.image_resize` for the resize implementation.
INLINE_MAX_DIMENSION = 2000

# Upper bound on raw image size that we'll feed to Pillow.  Above this,
# downsampling is skipped and the renderer falls through to the marker —
# the worker would otherwise spend seconds decoding pathological inputs
# (uploaded TIFFs, multi-100MP camera RAWs, etc.).
PRE_RESIZE_CEILING_BYTES = 50 * 1024 * 1024

# Explicit per-model escape hatch. This is the only name-based assertion:
# catalog values remain authoritative, while uncatalogued models are unknown
# and consumers optimistically attempt safe image inputs.
_VISION_OVERRIDES: dict[str, bool] = {}


def supports_vision(model: str) -> bool | None:
    """True when ``model`` accepts ``image_url`` content parts.

    Resolution order:

    1. :data:`_VISION_OVERRIDES` — intentional per-model force ``True`` or
       ``False``.
    2. LiteLLM's actual boolean ``supports_vision`` catalog value.
    3. ``None`` when lookup fails or the capability is absent/non-boolean.

    ``None`` means unknown, not unsupported. Image consumers default unknown
    models to allowed, while retaining all size, mime, decode, and provider
    format gates. This avoids a model-name allowlist that needs patching for
    every release and ensures an explicit catalog ``False`` is never silently
    overridden by a family-name guess.
    """
    if model in _VISION_OVERRIDES:
        return _VISION_OVERRIDES[model]
    # Defer the heavy ``litellm`` import: most harness paths do not ask for
    # image capability, so they should not pay its ~1.18s import cost.
    import litellm

    try:
        info = litellm.get_model_info(model)
    except Exception as err:
        # ``get_model_info`` raises a mix of ``BadRequestError`` (unknown
        # model), KeyError, and import/network errors depending on the
        # failure mode. Preserve that uncertainty rather than claiming the
        # model has no vision; callers surface it in their visible marker.
        log.warning("vision.litellm_lookup_failed", model=model, error=str(err))
        return None
    capability = info.get("supports_vision")
    return capability if isinstance(capability, bool) else None


# Image formats every vision-capable provider in aios's routing set accepts as
# an inline ``image_url`` — the intersection of Anthropic (jpeg/png/gif/webp)
# and OpenAI (png/jpeg/webp/gif). Values are Pillow ``Image.format`` names
# (uppercase). Other formats (TIFF, BMP, ICO, HEIC, AVIF, SVG) decode in Pillow
# but the providers 400 on them, so the renderer degrades them to a text marker
# the model can still ``read``. The check runs on the DECODED format, not the
# declared mime — a sender can mislabel either way (a JPEG sent as image/jpg, a
# TIFF sent as image/png), so only the decoded format is trustworthy.
PROVIDER_INLINE_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "GIF", "WEBP"})


def inline_image_format(data: bytes) -> str | None:
    """Return Pillow's format name ("JPEG"/"PNG"/"TIFF"/...) if ``data`` fully
    decodes as an image, else ``None``.

    Every inline path (the attachment render boundary in
    ``context._apply_attachments`` and the ``read`` tool's ``_read_image``)
    must apply the SAME verdict the provider will: it full-decodes the bytes
    and 400s on anything it can't, so neither the declared mime
    (:func:`can_inline_image`) nor the 24-byte magic sniff
    (:func:`make_image_url_part`) is enough. ``img.load()`` forces the full
    decode (``verify()`` passes a truncated body); the caller then checks the
    returned name against :data:`PROVIDER_INLINE_IMAGE_FORMATS`. A TIFF/BMP
    decodes fine yet no provider accepts it, and a declared mime can lie either
    way (a JPEG sent as image/jpg, a TIFF sent as image/png), so only the
    decoded format is trustworthy. Both failure modes — undecodable, or
    decodable-but-unsupported — degrade to a text marker rather than re-sending
    a rejected part on every replay wake (a permanent brick the model can't
    see). This helper is shared so the two inline paths cannot drift apart —
    which is exactly how the ``read`` path missed the gate the render path got.

    Returns ``None`` on ANY decode failure (total): Pillow raises a wide,
    format-plugin-dependent set on hostile bytes (UnidentifiedImageError/
    OSError, DecompressionBombError, ValueError/SyntaxError/struct.error), so
    catch ``Exception`` (never ``BaseException``).
    """
    if not data:
        return None
    from io import BytesIO

    from PIL import Image

    try:
        with Image.open(BytesIO(data)) as img:
            img.load()
            return img.format
    except Exception:
        return None


def can_inline_image(*, model: str, content_type: str, size_bytes: int) -> bool:
    """True when ``model`` can see image bytes inlined as ``image_url``.

    Returns ``False`` for non-image content types, oversize files
    (over :data:`INLINE_SIZE_CAP_BYTES`), and models explicitly known not to
    support vision. Unknown capability is allowed so newly released and custom
    gateway models can receive images without a model-list patch. Callers still
    apply decoded-format and resize safety gates before constructing a part.
    """
    if not content_type.startswith("image/"):
        return False
    if size_bytes > INLINE_SIZE_CAP_BYTES:
        return False
    return supports_vision(model) is not False


def make_image_url_part(*, content_type: str, data_b64: str) -> dict[str, Any]:
    """Build a chat-completions ``image_url`` content part.

    Reconciles the declared ``content_type`` against the magic bytes —
    inbound platform metadata and extension-based guesses both
    occasionally lie, and Anthropic rejects mime-vs-magic mismatches.
    Centralising the sniff here means every caller is covered without
    having to remember to wire correction at the call site.

    Also strips RFC-7231 parameters (anything after ``;``) from the mime
    before building the data URI. A connector posting an attachment with
    ``Content-Type: image/svg+xml; charset=utf-8`` would otherwise produce
    ``data:image/svg+xml; charset=utf-8;base64,...``, which Anthropic and
    most providers reject as malformed — bricking every wake of any
    session whose context now includes that part. ``correct_image_mime_b64``
    only rewrites for PNG/JPEG/GIF/WebP magic, so SVG/HEIC/AVIF/BMP
    declared values flow through unchanged unless stripped here.
    """
    content_type = correct_image_mime_b64(content_type, data_b64)
    bare_mime = content_type.split(";", 1)[0].strip()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{bare_mime};base64,{data_b64}"},
    }


def text_marker(record: dict[str, Any]) -> str:
    """Inert text marker for an attachment that won't be inlined.

    Used when the model can't see the pixels (non-vision model, oversize
    image, non-image attachment, legacy stub without ``in_sandbox_path``).
    The marker carries enough info for the model to ``read`` the path
    if the file is in fact reachable.
    """
    filename = record.get("filename") or "unnamed"
    content_type = record.get("content_type") or "application/octet-stream"
    size = record.get("size")
    path = record.get("in_sandbox_path")

    size_str = human_size(size) if isinstance(size, int) else "unknown size"
    kind = (
        "image"
        if isinstance(content_type, str) and content_type.startswith("image/")
        else "attachment"
    )
    if path:
        return f"[{kind}: {filename} ({content_type}, {size_str}) at {path}]"
    return f"[{kind}: {filename} ({content_type}, {size_str})]"


def human_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"
