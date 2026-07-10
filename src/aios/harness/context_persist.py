"""Async companion to :mod:`aios.harness.context`'s render-time image clamp.

:func:`aios.harness.context._clamp_oversize_image_data_urls` runs on EVERY
``build_messages`` call and, for a persisted ``tool_result``/message event
that still carries a pre-#1616 oversize ``image_url`` data-url part,
downsamples it IN MEMORY so the current render fits the provider's caps.
Because that result is never written back, every future build re-decodes
and re-downsamples the same backlog part forever (issue #1745 Part B/C).

:func:`persist_clamped_image_parts` closes the loop: called once per step
BEFORE ``build_messages`` (worker path only — see ``persist_image_rewrites``
at the call site in :func:`aios.harness.step_context.compose_step_context`),
it performs the identical downsample, writes the shrunk part back to
``events.data`` via :func:`aios.db.queries.replace_event_data`, and updates
the in-memory :class:`~aios.models.events.Event` so THIS step's render
already sees the persisted (small) bytes. Once persisted, the render-time
clamp pass's fit-verdict cache converges to a steady-state dict lookup —
the whole point of this module.
"""

from __future__ import annotations

import asyncio
import base64
from typing import TYPE_CHECKING, Any

from aios.harness.image_resize import (
    ImageDownsampleError,
    _blocking_downsample,
    is_oversize_image,
)
from aios.harness.vision import INLINE_MAX_DIMENSION, INLINE_SIZE_CAP_BYTES
from aios.logging import get_logger

if TYPE_CHECKING:
    import asyncpg

    from aios.models.events import Event

log = get_logger("aios.harness.context_persist")


async def persist_clamped_image_parts(
    pool: asyncpg.Pool[Any],
    events: list[Event],
    *,
    session_id: str,
    account_id: str,
) -> None:
    """Downsample + persist any oversize persisted ``image_url`` part once.

    Mirrors :func:`aios.harness.context._clamp_oversize_image_data_urls`'s
    per-part decode/downsample logic exactly, sharing its module-level
    fit-verdict cache (function-local import below avoids a module-load
    cycle: ``context.py`` stays free of any DB/async import). For each
    ``kind == "message"`` event whose ``data["content"]`` is a list of
    parts, an oversize ``image_url`` part is downsampled via
    :func:`asyncio.to_thread` (off the event loop) and the shrunk part is
    written back with :func:`aios.db.queries.replace_event_data` — a
    single-row, account-scoped ``UPDATE`` — then ``e.data`` is replaced
    (copy-on-write; never mutated in place) so this step's subsequent
    ``build_messages`` call renders the persisted bytes directly.

    An :class:`~aios.harness.image_resize.ImageDownsampleError` (the part
    is un-shrinkable — above the pre-resize ceiling, or the byte cap is
    unreachable even at the bottom of the quality ladder) is NEVER
    persisted: the render-time placeholder stays render-only so the
    original log bytes are never destroyed. The DEGRADE verdict is cached
    so neither this pass nor the render pass re-attempts the decode.

    Crash-safe idempotency: ``_blocking_downsample`` is a deterministic
    function of the immutable persisted bytes, so a crash before the
    ``UPDATE`` commits just means the next call re-derives byte-identical
    output and retries; a crash after means the persisted part already
    fits, so this pass no-ops on it next time (cached FITS or the size/
    dimension gate short-circuits before any decode). Concurrent writers
    hitting the same event compute byte-equal output (determinism), so
    last-writer-wins on the ``UPDATE`` is safe.
    """
    # Function-local import: these are context.py's module-level caches and
    # verdict constants. Importing them here (not at context_persist.py's
    # module top) means context.py itself never has to import anything
    # DB/async-shaped, preserving its "no DB access, no async" contract
    # (module docstring) — context_persist.py is the async composer-side
    # half of this feature, context.py stays the pure sync renderer.
    from aios.db import queries
    from aios.harness.context import (
        _CLAMP_VERDICT_DEGRADE,
        _CLAMP_VERDICT_FITS,
        _clamp_cache_get,
        _clamp_cache_key,
        _clamp_cache_put,
    )

    for e in events:
        if e.kind != "message":
            continue
        content = e.data.get("content")
        if not isinstance(content, list):
            continue
        new_content: list[Any] | None = None
        pending_logs: list[dict[str, Any]] = []
        for i, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url")
            if not isinstance(url, str) or not url.startswith("data:"):
                continue
            head, sep, data_b64 = url.partition(",")
            if not sep or ";base64" not in head:
                continue
            # Same fit-verdict cache the render pass consults — a hit
            # here means either a prior render or a prior persist call
            # already classified this exact part: FITS needs no write,
            # DEGRADE must never be written.
            cache_key = _clamp_cache_key(data_b64)
            cached_verdict = _clamp_cache_get(cache_key)
            if cached_verdict in (_CLAMP_VERDICT_FITS, _CLAMP_VERDICT_DEGRADE):
                continue
            byte_oversize = (len(data_b64) * 3 // 4) > INLINE_SIZE_CAP_BYTES
            try:
                raw = base64.b64decode(data_b64, validate=True)
            except Exception:
                # Malformed base64 — not this pass's concern (mirrors
                # the render pass's stance); leave it for the provider.
                continue
            if not byte_oversize and not is_oversize_image(raw):
                _clamp_cache_put(cache_key, _CLAMP_VERDICT_FITS)
                continue
            try:
                resized = await asyncio.to_thread(
                    _blocking_downsample,
                    raw,
                    INLINE_SIZE_CAP_BYTES,
                    INLINE_MAX_DIMENSION,
                )
            except ImageDownsampleError:
                _clamp_cache_put(cache_key, _CLAMP_VERDICT_DEGRADE)
                continue
            if resized is None:
                # Header check said oversize but the downsample
                # disagreed (race on the cap boundary) — nothing to
                # persist this round.
                continue
            encoded = base64.b64encode(resized.data).decode("ascii")
            if new_content is None:
                new_content = list(content)
            old_size = len(data_b64)
            new_content[i] = {
                **part,
                "image_url": {
                    **image_url,
                    "url": f"data:{resized.content_type};base64,{encoded}",
                },
            }
            pending_logs.append(
                {
                    "event_id": e.id,
                    "part_idx": i,
                    "old_size": old_size,
                    "new_size": len(encoded),
                }
            )
        if new_content is not None:
            # Acquire only for the UPDATE: decode/downsample can take seconds and
            # must not occupy a scarce pool connection while running in a thread.
            new_data = {**e.data, "content": new_content}
            async with pool.acquire() as conn:
                updated = await queries.replace_event_data(
                    conn, session_id, e.id, new_data, account_id=account_id
                )
            if updated:
                e.data = new_data
                for fields in pending_logs:
                    log.info("context.image_part_persisted_clamp", **fields)
