"""Vision-policy helper: single source of truth for image-into-vision decisions.

LiteLLM's :func:`token_counter` returns a flat ~85 tokens per
``image_url`` part regardless of provider; under-counting only
matters near the window boundary and provider rejection there is
recoverable.
"""

from __future__ import annotations

import base64
from typing import Any

import litellm

from aios.logging import get_logger

log = get_logger("aios.harness.vision")

INLINE_SIZE_CAP_BYTES = 2 * 1024 * 1024

_VISION_OVERRIDES: dict[str, bool] = {}

# Magic-byte signatures for the image formats Anthropic accepts.  Used to
# correct mismatched ``Content-Type`` declarations that creep in from
# inbound platform metadata or extension-based guesses (Anthropic's
# ``/v1/messages`` strictly validates declared mime against actual bytes
# and 400s on mismatch — see #294 incident).
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def sniff_image_mime(data_b64: str) -> str | None:
    """Return the actual mime type of the base64-encoded image, or None.

    Decodes only the first ~16 bytes (24 base64 chars) — enough to
    discriminate the four formats Anthropic supports.  WebP is not in
    the magic table because it requires reading offset 8-11; if we hit
    a real WebP we add it.  Unknown signatures return ``None`` so the
    caller can leave the declared mime untouched.
    """
    head_b64 = data_b64[:24]
    pad = (-len(head_b64)) % 4
    try:
        head = base64.b64decode(head_b64 + "=" * pad)
    except Exception:
        return None
    for sig, mime in _IMAGE_MAGIC:
        if head.startswith(sig):
            return mime
    return None


def supports_vision(model: str) -> bool:
    """True when ``model`` accepts ``image_url`` content parts.

    Consults :data:`_VISION_OVERRIDES` first for cases LiteLLM is wrong
    or behind on (empty initially; populated as we hit them), then
    falls back to ``litellm.get_model_info``.
    """
    if model in _VISION_OVERRIDES:
        return _VISION_OVERRIDES[model]
    try:
        info = litellm.get_model_info(model)
    except Exception as err:
        # ``get_model_info`` raises a mix of ``BadRequestError`` (unknown
        # model), KeyError, and import/network errors depending on the
        # failure mode.  Collapsing to "no vision" is the safe fallback
        # (we degrade to a text marker the model can still ``read``), but
        # the silence makes a transient outage look identical to "unknown
        # model" — log warn-level so operators have a grep target when
        # vision unexpectedly degrades across a deploy or provider blip.
        log.warning("vision.litellm_lookup_failed", model=model, error=str(err))
        return False
    return bool(info.get("supports_vision"))


def can_inline_image(*, model: str, content_type: str, size_bytes: int) -> bool:
    """True when ``model`` can see image bytes inlined as ``image_url``.

    Returns ``False`` for non-image content_types, oversize files
    (over :data:`INLINE_SIZE_CAP_BYTES`), and models without vision
    support.  Callers fall back to a text marker referencing the
    in-sandbox path so the model can still ``read`` the file later.
    """
    if not content_type.startswith("image/"):
        return False
    if size_bytes > INLINE_SIZE_CAP_BYTES:
        return False
    return supports_vision(model)


def make_image_url_part(*, content_type: str, data_b64: str) -> dict[str, Any]:
    """Build a chat-completions ``image_url`` content part.

    The declared ``content_type`` is reconciled against the actual magic
    bytes — inbound platform metadata and extension-based guesses both
    occasionally lie ("png" extension on a JPEG, Telegram reporting
    image/jpeg for a PNG photo, etc.).  Anthropic's ``/v1/messages``
    rejects mismatches outright and that error round-trips into a tight
    retry loop because the bad mime is baked into the persisted event.
    Sniffing here means new events carry the correct mime from the
    start; ``_correct_image_data_url_mimes`` in context.py handles
    historical events whose mime was already wrong.
    """
    actual = sniff_image_mime(data_b64)
    if actual is not None and actual != content_type:
        log.warning(
            "vision.image_mime_corrected_at_write",
            declared=content_type,
            actual=actual,
        )
        content_type = actual
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{content_type};base64,{data_b64}"},
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
