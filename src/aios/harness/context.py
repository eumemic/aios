"""Context builder for the step function.

:func:`build_messages` assembles the chat-completions message list
from the event log, synthesizing ``"pending"`` results for in-flight
tool calls and reordering tool results so they appear immediately
after their requesting assistant message.

It is a pure function of the event log + caller-supplied vision policy
inputs (``model``, ``session_id``).  It DOES read host bytes when an
image attachment can be inlined for the bound model — that I/O is the
cost of producing an ``image_url`` content part.  No DB access, no async.

The ``reacting_to`` field on assistant messages is the key coordination
mechanism. Each assistant message records the seq of the latest user or
tool_result event that was in its context.
:func:`~aios.harness.sweep.find_sessions_needing_inference` uses this
to define "new" as events after ``reacting_to``, not after the assistant's
own seq. This correctly handles the race where a tool result arrives
during inference — the model's response has a ``reacting_to`` that
predates the tool result, so the result is "new" and triggers a
follow-up step.

Because this replay runs on EVERY wake over the immutable log, a per-event
render failure is a structural brick risk: the model is never called, so
the "model sees the error and retries" recovery cannot engage. A quarantine
backstop guards against it — any per-event render that raises is caught and
replaced by a deterministic placeholder that is a function of ``e.seq`` ONLY
(see :func:`_quarantine_placeholder`), preserving the monotonicity invariant,
and the failure is signalled via the ``context.poison_event_quarantined``
structlog event. One poison event degrades exactly one position; every other
event in the window still renders.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from aios.harness.image_resize import (
    ImageDownsampleError,
    _blocking_downsample,
    is_oversize_image,
)
from aios.harness.vision import (
    INLINE_MAX_DIMENSION,
    INLINE_SIZE_CAP_BYTES,
    PROVIDER_INLINE_IMAGE_FORMATS,
    can_inline_image,
    correct_image_mime_b64,
    inline_image_format,
    make_image_url_part,
    text_marker,
)
from aios.harness.window import WindowOmission
from aios.logging import get_logger
from aios.models.events import MODEL_VISIBLE_LIFECYCLE_EVENTS, Event
from aios.sandbox.volumes import resolve_to_host_path

log = get_logger("aios.harness.context")

# ── Attachment render cache (issue #1745 Part A) ─────────────────────────
#
# ``_apply_attachments`` re-reads + re-sniffs + re-base64-encodes every
# inlinable staged image on EVERY ``build_messages`` call (every wake), even
# though ``/mnt/attachments/`` staged bytes are immutable in steady state.
# This module-level LRU memoizes the read+sniff+encode verdict keyed on full
# file identity (path, mtime, size) so a steady-state replay over a window
# of previously-seen attachments performs zero file reads.
#
# Value shapes:
#   ("inline", content_type, data_b64)  — the file inlines cleanly.
#   ("marker",)                          — undecodable / unsupported-format
#                                           verdict (the model degrades to a
#                                           text marker either way).
# A "marker" entry does NOT record the size for the byte-cap accounting
# below since it holds no encoded payload.
_ATTACHMENT_CACHE_LOCK = threading.Lock()
_ATTACHMENT_CACHE: OrderedDict[tuple[str, int, int], tuple[Any, ...]] = OrderedDict()
_ATTACHMENT_CACHE_BYTES = 0
_ATTACHMENT_CACHE_MAX_ENTRIES = 256


def _attachment_cache_max_bytes() -> int:
    """Read the byte-cap env override lazily (not at import time) so tests
    can monkeypatch ``os.environ`` per-case without needing a process
    restart."""
    raw = os.environ.get("AIOS_CONTEXT_IMAGE_CACHE_MAX_BYTES")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            pass
    return 64 * 1024 * 1024


def _attachment_cache_entry_bytes(value: tuple[Any, ...]) -> int:
    if value and value[0] == "inline":
        return len(value[2])
    return 0


def _attachment_cache_get(key: tuple[str, int, int]) -> tuple[Any, ...] | None:
    with _ATTACHMENT_CACHE_LOCK:
        value = _ATTACHMENT_CACHE.get(key)
        if value is not None:
            _ATTACHMENT_CACHE.move_to_end(key)
        return value


def _attachment_cache_put(key: tuple[str, int, int], value: tuple[Any, ...]) -> None:
    global _ATTACHMENT_CACHE_BYTES
    with _ATTACHMENT_CACHE_LOCK:
        # A concurrent racer may have inserted the same key already —
        # duplicate compute under a race is harmless (deterministic), so
        # just overwrite and re-account.
        existing = _ATTACHMENT_CACHE.pop(key, None)
        if existing is not None:
            _ATTACHMENT_CACHE_BYTES -= _attachment_cache_entry_bytes(existing)
        _ATTACHMENT_CACHE[key] = value
        _ATTACHMENT_CACHE_BYTES += _attachment_cache_entry_bytes(value)
        max_bytes = _attachment_cache_max_bytes()
        while _ATTACHMENT_CACHE and (
            len(_ATTACHMENT_CACHE) > _ATTACHMENT_CACHE_MAX_ENTRIES
            or max_bytes < _ATTACHMENT_CACHE_BYTES
        ):
            _, evicted = _ATTACHMENT_CACHE.popitem(last=False)
            _ATTACHMENT_CACHE_BYTES -= _attachment_cache_entry_bytes(evicted)


def _clear_attachment_cache() -> None:
    """Test-only: reset the module-level LRU between cases."""
    global _ATTACHMENT_CACHE_BYTES
    with _ATTACHMENT_CACHE_LOCK:
        _ATTACHMENT_CACHE.clear()
        _ATTACHMENT_CACHE_BYTES = 0


# ── Clamp-pass fit-verdict cache (issue #1745 Part B) ────────────────────
#
# ``_clamp_oversize_image_data_urls`` full-``base64.b64decode``s EVERY
# persisted ``image_url`` data-url part on every build to check whether it
# needs downsampling. This LRU memoizes the FITS/DEGRADE verdict keyed on
# ``(len(data_b64), blake2b-128(data_b64))`` so steady state is a dict lookup,
# not a decode + Pillow header parse. The stable cryptographic digest avoids
# sticky false verdicts from same-length collisions in Python's 64-bit hash.
_CLAMP_VERDICT_FITS = "FITS"
_CLAMP_VERDICT_DEGRADE = "DEGRADE"
_CLAMP_CACHE_LOCK = threading.Lock()
_CLAMP_CACHE: OrderedDict[tuple[int, bytes], str] = OrderedDict()
_CLAMP_CACHE_MAX_ENTRIES = 8192


def _clamp_cache_key(data_b64: str) -> tuple[int, bytes]:
    digest = hashlib.blake2b(data_b64.encode("ascii"), digest_size=16).digest()
    return (len(data_b64), digest)


def _clamp_cache_get(key: tuple[int, bytes]) -> str | None:
    with _CLAMP_CACHE_LOCK:
        value = _CLAMP_CACHE.get(key)
        if value is not None:
            _CLAMP_CACHE.move_to_end(key)
        return value


def _clamp_cache_put(key: tuple[int, bytes], value: str) -> None:
    with _CLAMP_CACHE_LOCK:
        _CLAMP_CACHE[key] = value
        _CLAMP_CACHE.move_to_end(key)
        while len(_CLAMP_CACHE) > _CLAMP_CACHE_MAX_ENTRIES:
            _CLAMP_CACHE.popitem(last=False)


def _clear_clamp_cache() -> None:
    """Test-only: reset the module-level LRU between cases."""
    with _CLAMP_CACHE_LOCK:
        _CLAMP_CACHE.clear()


# Chat-completions spec fields per role.  Provider-specific extensions
# (reasoning, reasoning_details, internal reacting_to, etc.) stay in the
# event log but are excluded from the message list.
_ALLOWED_FIELDS: dict[str, frozenset[str]] = {
    "assistant": frozenset({"role", "content", "tool_calls"}),
    "tool": frozenset({"role", "tool_call_id", "content", "name"}),
    "user": frozenset({"role", "content", "name"}),
    "system": frozenset({"role", "content", "name"}),
}

# Anthropic's contract: thinking blocks must be preserved across turns
# for the model to use them as continuation context.
_THINKING_FIELDS: frozenset[str] = frozenset({"thinking_blocks", "reasoning_content"})

# Unaffected-resources tail shared by the reset/expired notices.
_FS_UNAFFECTED = (
    " /workspace and mounted directories (memory stores, uploads, attachments, "
    "github working trees) are unaffected."
)

# Fresh-base advisory shared by the reset/expired notices (the discard/reset
# preamble differs; this tail is identical).
_FS_FRESH_BASE = (
    "The next command runs on a fresh base filesystem; previously installed "
    f"packages and tools are gone, reinstall as needed.{_FS_UNAFFECTED}"
)


def _render_fs_lifecycle_notice(data: dict[str, Any]) -> str:
    """Render an FS-loss lifecycle event as a bracketed user-role notice.

    Total by contract — a pure function of ``data`` that never raises (an
    unknown event/reason falls back to a generic line), since it runs inside
    the per-wake replay where a raise would brick the session.
    """
    event = data.get("event")
    reason = data.get("reason")
    if event == "sandbox_fs_over_limit":
        return (
            "[The persisted sandbox filesystem for this session exceeded its size budget. "
            "Installed packages and files are retained, but older write history may be "
            "collapsed to reclaim space; keep the persistent footprint small — large data "
            "belongs in /workspace, which is unaffected.]"
        )
    if event == "sandbox_fs_expired":
        if reason == "disk_pressure":
            cause = "to reclaim disk space"
        elif reason == "account_cap":
            cause = "because the account exceeded its snapshot storage quota"
        else:
            cause = "after a period of inactivity (retention limit)"
        return (
            f"[The persisted sandbox filesystem for this session was discarded {cause}. "
            f"{_FS_FRESH_BASE}]"
        )
    # sandbox_fs_reset (or any other allowlisted reset-shaped event).
    if reason == "environment_image_changed":
        detail = "the environment's base image was changed"
    elif reason == "snapshot_missing":
        detail = "the persisted filesystem could no longer be found"
    else:
        detail = "the persisted filesystem was reset"
    return f"[The sandbox filesystem for this session was reset because {detail}. {_FS_FRESH_BASE}]"


def _render_delivery_failure_notice(data: dict[str, Any]) -> str:
    """Render a ``connector_delivery_failed`` lifecycle event (#1261) as a
    bracketed user-role notice.

    A connector appends this when an outbound the model consciously sent did
    not arrive (carrier block / delivery failure). Like the FS renderer it is
    total by contract — a pure function of ``data`` that never raises, since it
    runs inside the per-wake replay where a raise would brick the session. The
    connector-specific particulars ride in ``data`` (``connector`` names the
    transport, ``detail`` carries the carrier reason, ``peer`` the recipient)
    so core renders the fact without knowing about any specific transport.
    """
    connector = data.get("connector") or data.get("connection_id") or "a connector"
    nested = data.get("data")
    peer = data.get("peer")
    detail = data.get("detail") or data.get("reason")
    if isinstance(nested, dict):
        # Producers (e.g. the SMS status-callback handler) carry carrier
        # specifics under ``data`` per the broadcast-route payload shape.
        peer = peer or nested.get("peer")
        detail = detail or nested.get("detail")
        connector = nested.get("connector") or connector
    peer_clause = f" to {peer}" if peer else ""
    detail_clause = f": {detail}" if detail else ""
    return (
        f"[A message you sent via {connector}{peer_clause} was not delivered"
        f"{detail_clause}. The recipient did not receive it.]"
    )


def _render_ack_notice(data: dict[str, Any]) -> str:
    """Render a ``connector_message_delivered`` / ``connector_message_edited``
    lifecycle event (#1341) as a bracketed user-role notice.

    The success-path complement to ``_render_delivery_failure_notice``: a
    connector appends one of these when the platform confirmed an outbound the
    model consciously sent was delivered, or that an edit landed. Like the
    delivery-failure renderer it is total by contract — a pure function of
    ``data`` that never raises, since it runs inside the per-wake replay where a
    raise would brick the session. Connector specifics
    (``platform_message_id``/``tool_call_id``) ride in ``data`` so core renders
    the fact without knowing about any specific transport.
    """
    event = data.get("event")
    connector = data.get("connector") or data.get("connection_id") or "a connector"
    if event == "connector_message_edited":
        return f"[Your edit via {connector} landed.]"
    return f"[Your message via {connector} was delivered.]"


# Notification markers truncate the source content to this many chars
# (plus an ellipsis when truncated) so a busy non-focal channel
# contributes O(tens-of-tokens) per inbound to the context — cheap
# enough to keep in-log at its seq position for episodic chronology.
_NOTIFICATION_PREVIEW_CHARS = 80


def _extract_reaction_emoji(reaction: dict[str, Any]) -> str | None:
    """Extract a human-readable emoji string from reaction metadata.

    Two producer shapes are in the wild:
    - Signal: ``{"emoji": "👍", ...}`` (the single active emoji)
    - Telegram: ``{"new_emojis": ["👍", ...], "old_emojis": [...], ...}``
      (the post-reaction state; ``new_emojis=[]`` means the user
      cleared their reaction)

    Returns ``None`` only if neither shape is recognizable. The
    ``"cleared"`` literal is returned for an explicit empty Telegram
    ``new_emojis`` so the model can tell a removal apart from a
    missing field.
    """
    emoji = reaction.get("emoji")
    if isinstance(emoji, str) and emoji:
        return emoji
    new_emojis = reaction.get("new_emojis")
    if isinstance(new_emojis, list):
        emojis = [e for e in new_emojis if isinstance(e, str) and e]
        if emojis:
            return " ".join(emojis)
        return "cleared"
    return None


def _format_received(created_at: datetime, tz_name: str) -> str:
    """Absolute receipt timestamp for the message envelope, in the account's
    effective timezone — UTC offset plus IANA name, e.g.
    ``2026-06-06T09:00:00-07:00 (America/Los_Angeles)``.

    Sourced from the event's immutable ``created_at``; ``tz_name`` is resolved
    once per step (a static function of account config) and pre-validated (see
    ``services.accounts.resolve_effective_timezone``), so this renders
    byte-identical on every rebuild — deterministic and prompt-cache-stable.
    """
    stamp = created_at.astimezone(ZoneInfo(tz_name)).isoformat(timespec="seconds")
    return f"{stamp} ({tz_name})"


def _prepend_header(msg: dict[str, Any], header: str) -> None:
    """Prepend a bracketed header line above the message's existing content."""
    existing = msg.get("content") or ""
    msg["content"] = f"{header}\n{existing}" if existing else header


def _wake_header(metadata: dict[str, Any] | None) -> str | None:
    """System-derived wake-provenance header, or None when not a wake.

    Reads only the keys wake_session stamps (wake_source_session_id,
    wake_depth) — never caller-suppliable free text.
    """
    if not metadata:
        return None
    src = metadata.get("wake_source_session_id")
    if not isinstance(src, str) or not src:
        return None
    depth = metadata.get("wake_depth")
    depth_str = str(depth) if isinstance(depth, int) else "?"
    return f"[wake from session={src} · depth={depth_str}]"


def _format_channel_header(metadata: dict[str, Any], received: str) -> str:
    """Render a one-line header describing the origin of an inbound message.

    When a user message carries channel metadata, the raw fields are
    whitelisted out of the chat-completions message before the model
    ever sees them (see ``_ALLOWED_FIELDS``).  That leaves the model
    with no way to know the sender or timestamp — values the connector
    tools need as arguments.  Inline the salient fields into the
    visible ``content`` so the model can read them natively.  The
    ``received`` envelope field is always appended (see
    :func:`_format_received`); the connector fields are added when present.
    """
    parts: list[str] = []
    if "channel" in metadata:
        parts.append(f"channel={metadata['channel']}")
    chat_type = metadata.get("chat_type")
    if isinstance(chat_type, str) and chat_type:
        parts.append(f"chat_type={chat_type}")
    chat_name = metadata.get("chat_name")
    if isinstance(chat_name, str) and chat_name:
        parts.append(f"chat_name={chat_name!r}")
    sender_name = metadata.get("sender_name")
    if isinstance(sender_name, str) and sender_name:
        parts.append(f"from={sender_name}")
    sender_uuid = metadata.get("sender_uuid")
    if isinstance(sender_uuid, str) and sender_uuid:
        parts.append(f"sender_uuid={sender_uuid}")
    sender_id = metadata.get("sender_id")
    if isinstance(sender_id, int):
        parts.append(f"sender_id={sender_id}")
    timestamp_ms = metadata.get("timestamp_ms")
    if isinstance(timestamp_ms, int):
        # Raw origin-time int — the model copies it verbatim into signal_send /
        # react / edit / delete tool args. The human-readable ISO now lives in
        # the envelope's `received=` field (receipt time), so no parenthetical.
        parts.append(f"timestamp_ms={timestamp_ms}")
    message_id = metadata.get("message_id")
    # Telegram surfaces ints; WhatsApp's whatsmeow IDs are hex strings
    # like "3EB0E03B46303C22D750E2".  Both render the same — the model
    # only needs the value verbatim to pass into react/edit/delete tools.
    if isinstance(message_id, int):
        parts.append(f"message_id={message_id}")
    elif isinstance(message_id, str) and message_id:
        parts.append(f"message_id={message_id!r}")
    if metadata.get("edited") is True:
        parts.append("edited=true")
    edit_target = metadata.get("edit_target_timestamp_ms")
    if isinstance(edit_target, int):
        parts.append(f"edit_target_timestamp_ms={edit_target}")
    edit_target_message_id = metadata.get("edit_target_message_id")
    if isinstance(edit_target_message_id, str) and edit_target_message_id:
        # WhatsApp identifies edit targets by string message_id, not
        # by Signal's timestamp_ms.  Render verbatim so the model can
        # match the target against the message_id of a prior event.
        parts.append(f"edit_target_message_id={edit_target_message_id!r}")
    if metadata.get("revoked") is True:
        parts.append("revoked=true")
    revoke_target_message_id = metadata.get("revoke_target_message_id")
    if isinstance(revoke_target_message_id, str) and revoke_target_message_id:
        # The peer revoked their own message; the original event is
        # still in the log (monotonicity), but the model should treat
        # it as retracted going forward.
        parts.append(f"revoke_target_message_id={revoke_target_message_id!r}")
    quoted_message_id = metadata.get("quoted_message_id")
    if isinstance(quoted_message_id, str) and quoted_message_id:
        # The peer replied to a previous message in the chat (WhatsApp's
        # reply gesture).  Renders the same shape as the edit/revoke
        # target ids so the model can match it against the message_id
        # header on a prior event.
        parts.append(f"quoted_message_id={quoted_message_id!r}")
    if metadata.get("self_mentioned") is True:
        # Hoist ahead of the structured ``mentions`` list — for
        # group chats the model often only needs to know "was I
        # tagged?" and substring-matching ``content`` is the
        # alternative that #5 set out to eliminate.
        parts.append("self_mentioned=true")
    sticker_emoji = metadata.get("sticker_emoji")
    if isinstance(sticker_emoji, str):
        # Empty string means the connector saw a StickerMessage but
        # the sender's WhatsApp client didn't pick an emoji label
        # (custom stickers from the sticker maker land this way).
        # Always render under the ``sticker_emoji=`` field name so
        # the model's prompt contract (per the connector prompts.py
        # mentioning ``metadata.sticker_emoji`` verbatim) stays
        # consistent across labeled and unlabeled stickers — empty
        # value still signals "a sticker arrived" without breaking
        # the established field-name pattern.
        parts.append(f"sticker_emoji={sticker_emoji!r}")
    parts.append(f"received={received}")
    header = "[" + " · ".join(parts) + "]"
    mentions = metadata.get("mentions")
    if isinstance(mentions, list) and mentions:
        # One entry per line; uuid is the platform-stable identity the
        # model needs for outbound mention encoding via signal_send.
        mention_lines: list[str] = []
        for m in mentions:
            if not isinstance(m, dict):
                continue
            uuid = m.get("uuid")
            if not isinstance(uuid, str) or not uuid:
                continue
            name = m.get("name")
            if isinstance(name, str) and name:
                mention_lines.append(f"[mention: name={name!r} · uuid={uuid}]")
            else:
                mention_lines.append(f"[mention: uuid={uuid}]")
        if mention_lines:
            header += "\n" + "\n".join(mention_lines)
    reaction = metadata.get("reaction")
    if isinstance(reaction, dict):
        emoji = _extract_reaction_emoji(reaction) or "?"
        r_parts: list[str] = [f"reaction={emoji!r}"]
        target_author = reaction.get("target_author_uuid")
        if isinstance(target_author, str) and target_author:
            r_parts.append(f"target_author_uuid={target_author}")
        target_ts = reaction.get("target_timestamp_ms")
        if isinstance(target_ts, int):
            r_parts.append(f"target_timestamp_ms={target_ts}")
        target_message_id = reaction.get("target_message_id")
        if isinstance(target_message_id, str) and target_message_id:
            # WhatsApp reactions identify their target by string
            # message_id (no equivalent of Signal's author+timestamp
            # pair).  Render verbatim so the model can match against
            # the message_id of a prior event.
            r_parts.append(f"target_message_id={target_message_id!r}")
        header += "\n[" + " · ".join(r_parts) + "]"
    reply_to = metadata.get("reply_to")
    if isinstance(reply_to, dict):
        text = reply_to.get("text")
        quoted = text.replace("\n", " ").strip() if isinstance(text, str) else ""
        quote_parts: list[str] = []
        author = reply_to.get("author_uuid")
        if isinstance(author, str) and author:
            quote_parts.append(f"author_uuid={author}")
        ts = reply_to.get("timestamp_ms")
        if isinstance(ts, int):
            quote_parts.append(f"timestamp_ms={ts}")
        quote_meta = " · ".join(quote_parts) if quote_parts else "?"
        if quoted:
            snippet = quoted if len(quoted) <= 200 else quoted[:200] + "…"
            header += f"\n[reply_to: {quote_meta}] > {snippet}"
        else:
            header += f"\n[reply_to: {quote_meta}]"
    return header


def _notification_preview(content: str, metadata: dict[str, Any] | None) -> str:
    """Short preview string for a notification marker.

    Prefers truncated text content.  Falls back to the reaction emoji
    for reaction events (which arrive with empty content — without the
    fallback the marker would be blank).
    """
    if isinstance(content, str) and content:
        if len(content) <= _NOTIFICATION_PREVIEW_CHARS:
            return content
        return content[:_NOTIFICATION_PREVIEW_CHARS] + "…"
    if isinstance(metadata, dict):
        reaction = metadata.get("reaction")
        if isinstance(reaction, dict):
            emoji = _extract_reaction_emoji(reaction)
            if emoji:
                return f"reacted {emoji}"
    return ""


_NOTIFICATION_MARKER_PREFIX = "🔔"


def _format_notification_marker(
    orig_channel: str,
    metadata: dict[str, Any] | None,
    content: str,
) -> str:
    """Render a concise, non-focal-channel notification marker.

    Shape::

        🔔 channel_id=<orig_channel> · from=<sender_name> · <preview>
        (to respond, call switch_channel(channel_id=<orig_channel>) first)

    The ``from`` clause is omitted when ``sender_name`` is absent from
    metadata.  The preview clause is omitted when content is empty and
    there's no reaction to surface.  The trailing hint line is always
    emitted — it tells the reader how to turn this notification into
    full-content context.
    """
    parts = [f"{_NOTIFICATION_MARKER_PREFIX} channel_id={orig_channel}"]
    if isinstance(metadata, dict):
        sender_name = metadata.get("sender_name")
        if isinstance(sender_name, str) and sender_name:
            parts.append(f"from={sender_name}")
    preview = _notification_preview(content, metadata)
    if preview:
        parts.append(preview)
    header = " · ".join(parts)
    hint = f"(to respond, call switch_channel(channel_id={orig_channel!r}) first)"
    return f"{header}\n{hint}"


def message_is_notification_marker(msg: dict[str, Any]) -> bool:
    """True if *msg* is a non-focal-channel notification marker (``🔔 …``).

    Mirrors the shape produced by :func:`_format_notification_marker`: a
    user-role message whose (text) content begins with the bell prefix. The
    composer uses this to tell a *direct* trailing stimulus the agent must
    answer (a focal inbound or tool result — keep it last, suppress the
    channels tail block so a literal model doesn't anchor on the tail) apart
    from a *navigation* prompt (a non-focal notification, whose companion is
    the tail listing — keep the tail).
    """
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return content.startswith(_NOTIFICATION_MARKER_PREFIX)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                return isinstance(text, str) and text.startswith(_NOTIFICATION_MARKER_PREFIX)
    return False


def render_user_event(
    event_data: dict[str, Any],
    orig_channel: str | None,
    focal_channel_at_arrival: str | None,
    created_at: datetime,
    *,
    tz_name: str = "UTC",
    model: str | None = None,
    session_id: str | None = None,
    workspace_path: Path | None = None,
) -> dict[str, Any]:
    """Render a user event into its chat-completions message form.

    Rendering is a deterministic function of the event's stamped
    ``orig_channel``, ``focal_channel_at_arrival``, ``created_at``, and
    the optional vision policy inputs ``model`` / ``session_id``.
    ``build_messages`` threads vision policy through; the append-time
    token counter in ``queries.append_event`` does not (and pays a small
    under-count per inlined image, absorbed by ``model_token_ratio``
    calibration).

    Every user message carries a ``received`` envelope field (the absolute
    receipt timestamp, from the immutable ``created_at``; see
    :func:`_format_received`) — so even the metadata-less ``channel=None``
    path is structured, not raw.

    Branches:

    * ``orig_channel`` is ``None`` → non-connector event; the only header
      is the ``received`` envelope.  Covers direct
      ``POST /sessions/{id}/messages`` traffic, workflow children, and
      pre-migration rows.
    * ``orig == focal_at_arrival`` (non-NULL) → full content with a
      metadata header inlined (connector fields + ``received``).  When the
      focal event's metadata carries ``attachments`` and the bound model
      supports vision, inlinable images are emitted as ``image_url``
      content parts; everything else gets a text marker referencing the
      in-sandbox path.
    * Otherwise (focal is NULL, or focal differs from orig) →
      notification marker: a short, emoji-prefixed one-liner surfacing
      the origin channel, sender, and a truncated content preview.  Kept
      deliberately terse — no ``received`` envelope; the per-message
      timestamp surfaces when the model ``switch_channel``s in and reads
      the full-content recap.  Any attachments are appended as
      ``text_marker`` lines (images → ``[image: … at <path>]``, others →
      ``[attachment: …]``) — never inlined off-channel — so the model can
      ``read`` the in-sandbox path now, or ``switch_channel`` to recover
      pixels via the reorient recap.  Before #718 they were silently
      dropped here.
    """
    msg = {k: v for k, v in event_data.items() if k != "metadata"}
    metadata = event_data.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else None
    received = _format_received(created_at, tz_name)

    # A request injected via invoke_session (e.g. a workflow agent child's first
    # message) carries metadata.request.request_id. return/error require that id, so
    # surface it on the rendered message — this is where the model reads what to echo
    # back. Deterministic function of the event, so the context stays monotonic.
    request = metadata.get("request") if metadata else None
    if isinstance(request, dict) and isinstance(request.get("request_id"), str):
        rid = request["request_id"]
        marker = f"[request_id: {rid} — reply with return/error using this request_id]"
        output_schema = request.get("output_schema")
        if output_schema is not None:
            # The request demands a typed `value`; show the shape so the model
            # conforms first-try (return enforces it regardless). Per-request — each
            # request carries its own schema.
            marker += (
                "\nYour return `value` must match this JSON Schema:\n"
                f"```json\n{json.dumps(output_schema, indent=2)}\n```"
            )
        _prepend_header(msg, marker)

    if orig_channel is None:
        _prepend_header(msg, f"[received={received}]")
        wake = _wake_header(metadata)
        if wake is not None:
            _prepend_header(msg, wake)
        return msg

    if orig_channel == focal_channel_at_arrival:
        header = (
            _format_channel_header(metadata, received) if metadata else f"[received={received}]"
        )
        _prepend_header(msg, header)
        wake = _wake_header(metadata)
        if wake is not None:
            _prepend_header(msg, wake)
        if metadata:
            attachments = metadata.get("attachments")
            if isinstance(attachments, list) and attachments:
                _apply_attachments(
                    msg,
                    attachments,
                    model=model,
                    session_id=session_id,
                    workspace_path=workspace_path,
                )
        return msg

    content = event_data.get("content", "")
    if not isinstance(content, str):
        content = ""
    marker = _format_notification_marker(orig_channel, metadata, content)
    attachments = metadata.get("attachments") if metadata else None
    if isinstance(attachments, list) and attachments:
        lines = [text_marker(r) for r in attachments if isinstance(r, dict)]
        marker = "\n".join([marker, *lines])
    return {"role": "user", "content": marker}


def _apply_attachments(
    msg: dict[str, Any],
    attachments: list[Any],
    *,
    model: str | None,
    session_id: str | None,
    workspace_path: Path | None = None,
) -> None:
    leading_text = msg.get("content") if isinstance(msg.get("content"), str) else ""
    marker_lines: list[str] = []
    image_parts: list[dict[str, Any]] = []

    for record in attachments:
        if not isinstance(record, dict):
            # ``metadata.attachments`` is ``list[Any]`` at the wire boundary
            # (``SessionUserMessage.metadata: dict[str, Any]`` doesn't drill
            # into the value, and connector inbound payloads can be
            # malformed). A non-dict record here would crash
            # ``record.get(...)`` and brick the session permanently, since
            # the renderer is called on every wake.
            log.warning(
                "context.attachment_record_not_dict",
                session_id=session_id,
                record_type=type(record).__name__,
            )
            continue
        # When staging produced a downsampled sibling for an oversize
        # image, prefer it: the original may exceed the inline cap but
        # the sibling is purpose-built to fit.  If the inline path
        # itself fails to resolve or fails the inline gate (e.g. GC
        # raced and deleted the sibling), the renderer drops straight
        # to the marker — the original is NOT re-attempted as a
        # fallback here.  ``text_marker`` always describes the
        # ORIGINAL record because that's the path the model would
        # ``read`` if it wants the bytes.
        inline = record.get("inline")
        effective = inline if isinstance(inline, dict) else record

        host_path = _resolve_attachment_host_path(effective, session_id, workspace_path)
        size = effective.get("size")
        content_type = effective.get("content_type")
        if (
            host_path is None
            or model is None
            or not isinstance(content_type, str)
            or not isinstance(size, int)
            or not can_inline_image(model=model, content_type=content_type, size_bytes=size)
        ):
            marker_lines.append(text_marker(record))
            continue
        # Attachment render cache (#1745 Part A): staged bytes under
        # ``/mnt/attachments/`` are immutable in steady state, so a cache
        # keyed on full file identity (path, mtime, size) lets a re-render
        # of a previously-seen attachment skip the read+sniff+encode
        # entirely. One ``stat()`` call resolves the key; an ``OSError``
        # here (file vanished) falls through to the SAME marker fallback
        # the old direct-read OSError guard used, so behavior is unchanged
        # for a missing file.
        try:
            st = host_path.stat()
        except OSError as err:
            log.warning(
                "context.attachment_read_failed",
                path=str(host_path),
                error=str(err),
            )
            marker_lines.append(text_marker(record))
            continue
        cache_key = (str(host_path), st.st_mtime_ns, st.st_size)
        cached = _attachment_cache_get(cache_key)
        if cached is not None:
            if cached[0] == "inline":
                _, cached_content_type, cached_data_b64 = cached
                # Never share the cached dict — each hit rebuilds the part
                # fresh via make_image_url_part.
                image_parts.append(
                    make_image_url_part(
                        content_type=cached_content_type,
                        data_b64=cached_data_b64,
                    )
                )
            else:
                marker_lines.append(text_marker(record))
            continue
        # Cache miss: compute (read + sniff + encode) OUTSIDE the lock —
        # only the insert is lock-guarded (see the cache helpers above).
        # A duplicate compute under a concurrent-render race is harmless:
        # the result is a deterministic function of the immutable bytes.
        try:
            payload = host_path.read_bytes()
        except OSError as err:
            # The staged file disappeared (manual cleanup, FS corruption,
            # GC race). Fall back to a text marker rather than raising
            # mid-render — losing one image shouldn't fail the whole step.
            log.warning(
                "context.attachment_read_failed",
                path=str(host_path),
                error=str(err),
            )
            marker_lines.append(text_marker(record))
            continue
        image_format = inline_image_format(payload)
        if image_format is None:
            # Bytes that pass the mime+size gate but don't actually decode (a
            # truncated/corrupt body behind a valid magic prefix, or a
            # zero-byte file). The provider full-decodes and 400s on these,
            # and the bytes are immutable in the log, so inlining them re-sends
            # the rejected part on every wake — terminally erroring the turn
            # the model can't see. Degrade to a marker the model can ``read``,
            # the same stance as the read-failure branch above and the
            # build_messages poison backstop.
            log.warning(
                "context.attachment_undecodable",
                path=str(host_path),
                filename=record.get("filename"),
            )
            _attachment_cache_put(cache_key, ("marker",))
            marker_lines.append(text_marker(record))
            continue
        if image_format not in PROVIDER_INLINE_IMAGE_FORMATS:
            # Decodes, but no vision provider accepts this format (TIFF, BMP,
            # …) — they take only jpeg/png/gif/webp and 400 on the rest, so
            # inlining it bricks the turn on every replay wake just like
            # undecodable bytes. Gate on the DECODED format, not the declared
            # mime (which can mislabel either way), and degrade to a marker.
            log.warning(
                "context.attachment_unsupported_image_format",
                image_format=image_format,
                path=str(host_path),
                filename=record.get("filename"),
            )
            _attachment_cache_put(cache_key, ("marker",))
            marker_lines.append(text_marker(record))
            continue
        data_b64 = base64.b64encode(payload).decode("ascii")
        _attachment_cache_put(cache_key, ("inline", content_type, data_b64))
        image_parts.append(
            make_image_url_part(
                content_type=content_type,
                data_b64=data_b64,
            )
        )

    text = leading_text
    if marker_lines:
        text = f"{text}\n{chr(10).join(marker_lines)}" if text else "\n".join(marker_lines)

    if image_parts:
        # Anthropic rejects empty text blocks (`text content blocks must be
        # non-empty`) so omit the leading text part when there is no caption,
        # no channel header, and no markers — the image_url parts stand on
        # their own.  OpenAI tolerates the empty part, but emitting it would
        # break any vision-capable Anthropic-routed model on a caption-less
        # image-only inbound.
        if text:
            msg["content"] = [{"type": "text", "text": text}, *image_parts]
        else:
            msg["content"] = list(image_parts)
    else:
        msg["content"] = text


def _resolve_attachment_host_path(
    record: dict[str, Any],
    session_id: str | None,
    workspace_path: Path | None = None,
) -> Any:
    sandbox_path = record.get("in_sandbox_path")
    if not isinstance(sandbox_path, str) or session_id is None:
        return None
    return resolve_to_host_path(session_id, sandbox_path, workspace_path=workspace_path)


def _sanitize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize tool_calls inner structure for cross-model replay.

    Some models produce malformed tool_calls (control characters in
    arguments, missing fields, extra provider keys) that break strict
    downstream providers.  Returns a cleaned copy with only spec fields
    and valid JSON arguments.

    Entries without a usable ``id`` are dropped: an id-less tool_call
    cannot be joined to any tool_result, so leaving it in the assistant
    message produces an assistant→user transition with no paired
    ``tool``-role message — invalid per the chat-completions schema.
    """
    sanitized = []
    for tc in tool_calls:
        tcid = tc.get("id")
        if not tcid:
            continue
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments", "{}")
        if isinstance(raw_args, dict):
            raw_args = json.dumps(raw_args)
        elif isinstance(raw_args, str):
            try:
                json.loads(raw_args)
            except (json.JSONDecodeError, ValueError):
                raw_args = "{}"
        else:
            raw_args = "{}"
        sanitized.append(
            {
                "id": tcid,
                "type": "function",
                "function": {"name": fn.get("name") or "", "arguments": raw_args},
            }
        )
    return sanitized


def _correct_image_data_url_mimes(messages: list[dict[str, Any]]) -> None:
    """Rewrite mismatched mime declarations on ``data:<mime>;base64,...``
    image URLs already in the assembled message list.

    The renderer cannot prevent persisted events from carrying a wrong
    mime — tool_result content arrives pre-formed and is appended
    verbatim, so historical events from before centralised sniffing
    will replay forever unless we re-check them every build.

    Mutates the message list's content entries by *replacing* the part
    dict (and the ``image_url`` sub-dict) with fresh copies — never
    mutates the original dicts in place. ``build_messages`` appends
    ``event.data`` to ``messages`` by reference (``context.py:560`` etc.)
    before this function runs, so an in-place mutation would corrupt the
    source-of-truth ``Event.data`` and violate the "pure function of the
    event log" contract documented at the top of this module.
    """
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content: list[Any] | None = None
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
            declared = head.removeprefix("data:").split(";", 1)[0]
            corrected = correct_image_mime_b64(declared, data_b64)
            if corrected != declared:
                if new_content is None:
                    new_content = list(content)
                new_content[i] = {
                    **part,
                    "image_url": {
                        **image_url,
                        "url": f"data:{corrected};base64,{data_b64}",
                    },
                }
        if new_content is not None:
            messages[msg_idx] = {**msg, "content": new_content}


def _clamp_oversize_image_data_urls(messages: list[dict[str, Any]]) -> None:
    """Downscale persisted ``data:<mime>;base64,...`` image parts whose
    decoded longest edge exceeds :data:`INLINE_MAX_DIMENSION` (2000px).

    Sibling pass to :func:`_correct_image_data_url_mimes`, called in the
    same place on every ``build_messages``. ``read()`` now downsamples on
    the inline path (issue #1616 Part A), but an image frozen at full
    resolution in a historical ``tool_result`` event from BEFORE that fix
    replays verbatim forever — and once >20 such parts accumulate and one
    is >2000px on a side, Anthropic HARD-REJECTS the many-image request
    (400, no server-side resize), wedging every wake. This pass re-checks
    persisted parts each build and rewrites any oversize one to a bounded
    copy, so an already-wedged session self-heals on its next build with
    no manual log surgery.

    Decode cost is bounded: ``_blocking_downsample`` does a header-only
    size check first and returns ``None`` (no decode, bytes untouched)
    for any part already <=2000px/side and <=cap — which, after Part A
    ships, is everything except the pre-fix backlog.

    Mutates by *replacing* part dicts with fresh copies — never in place —
    matching :func:`_correct_image_data_url_mimes`'s contract (the message
    list aliases the immutable ``Event.data``). A part that cannot be
    downscaled (undecodable / above the pre-resize ceiling) degrades to a
    short inert text placeholder rather than being dropped, preserving the
    surrounding content-list / message structure.
    """
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content: list[Any] | None = None
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
            # Fit-verdict cache (#1745 Part B): keyed on the encoded string
            # itself (length + hash), consulted BEFORE any decode. A hit
            # means this exact part was already classified on a prior
            # build — steady state is a dict lookup, zero decode.
            cache_key = _clamp_cache_key(data_b64)
            cached_verdict = _clamp_cache_get(cache_key)
            if cached_verdict == _CLAMP_VERDICT_FITS:
                continue
            if cached_verdict == _CLAMP_VERDICT_DEGRADE:
                if new_content is None:
                    new_content = list(content)
                new_content[i] = {
                    "type": "text",
                    "text": "[image omitted: too large to display inline]",
                }
                continue
            # Cache miss. O(1) size-gate on the ENCODED length classifies
            # byte-oversize with zero decode — base64 expands 4 bytes for
            # every 3 raw bytes, so ``len(data_b64) * 3 // 4`` approximates
            # the raw size without decoding a single byte. When this alone
            # already proves oversize, skip the redundant Pillow header
            # check below (``is_oversize_image``) — we already know the
            # verdict; decoding still happens once, right before, because
            # downsampling needs the actual bytes regardless.
            byte_oversize = (len(data_b64) * 3 // 4) > INLINE_SIZE_CAP_BYTES
            try:
                raw = base64.b64decode(data_b64, validate=True)
            except Exception:
                # Malformed base64 — leave it; the mime corrector and the
                # provider's own decode handle that orthogonal failure. Not
                # cached: this is a data defect, not a fit verdict.
                continue
            # Header-only oversize check: this pass is scoped to the DIMENSION
            # wedge only. A part that already fits both caps is left exactly
            # as the mime corrector produced it (an undecodable-but-small part
            # is the mime corrector's / provider's concern, not a dimension
            # wedge — degrading it here would clobber that contract).
            if not byte_oversize and not is_oversize_image(raw):
                _clamp_cache_put(cache_key, _CLAMP_VERDICT_FITS)
                continue
            try:
                # Blocking (not awaited): build_messages is sync. We only
                # reach here for genuine >2000px (or over-cap) parts — the
                # pre-fix backlog — so the re-encode cost is bounded.
                resized = _blocking_downsample(raw, INLINE_SIZE_CAP_BYTES, INLINE_MAX_DIMENSION)
            except ImageDownsampleError:
                # Oversize AND un-shrinkable (above the pre-resize ceiling, or
                # the byte cap is unreachable even at the bottom of the quality
                # ladder): a part we cannot bound is a wedge if left in place.
                # Replace it with an inert text placeholder, preserving list
                # shape rather than dropping the message.
                _clamp_cache_put(cache_key, _CLAMP_VERDICT_DEGRADE)
                if new_content is None:
                    new_content = list(content)
                new_content[i] = {
                    "type": "text",
                    "text": "[image omitted: too large to display inline]",
                }
                continue
            if resized is None:
                # Header check said oversize but the downsample disagreed
                # (e.g. a race on the cap boundary) — nothing to rewrite.
                # Not cached: this part's own persisted bytes are the same
                # every render, so re-deriving this (rare) race each time is
                # cheap and safer than caching a same-key/different-outcome
                # verdict.
                continue
            encoded = base64.b64encode(resized.data).decode("ascii")
            # NOT cached as FITS: this render successfully shrank the part in
            # MEMORY only — the persisted bytes are still oversize, so the
            # NEXT render (absent Part C's persist) must redo this same work.
            # Once Part C persists the shrunk bytes, the next build sees a
            # different (smaller) ``data_b64`` — a fresh cache key that hits
            # the byte-size-gate / header-check FITS path directly.
            if new_content is None:
                new_content = list(content)
            new_content[i] = {
                **part,
                "image_url": {
                    **image_url,
                    "url": f"data:{resized.content_type};base64,{encoded}",
                },
            }
        if new_content is not None:
            messages[msg_idx] = {**msg, "content": new_content}


def _is_replayable_thinking_block(block: Any) -> bool:
    """Whether a persisted thinking block is safe to replay to the provider.

    The read-path counterpart to ``completion._is_persistable_thinking_block``
    (issue #1588). Anthropic rejects a thinking block whose ``signature`` is
    empty or missing on EVERY replay (``400 Invalid signature in thinking
    block``); a block whose signature is intact replays fine. We gate on the
    signature alone here: the persist guard already drops empty-text blocks,
    and an in-window block that somehow carries a signature but no text is
    still replayable as long as the signature is complete. A non-dict block
    is never replayable.
    """
    if not isinstance(block, dict):
        return False
    return bool((block.get("signature") or "").strip())


def _strip_to_spec(
    msg: dict[str, Any],
    *,
    target_supports_thinking: bool,
) -> dict[str, Any]:
    """Return a copy of *msg* with only chat-completions spec fields.

    When ``target_supports_thinking``, also preserve ``thinking_blocks``
    and ``reasoning_content`` on assistant turns.
    """
    role = msg.get("role", "")
    allowed = _ALLOWED_FIELDS.get(role, frozenset())
    if role == "assistant" and target_supports_thinking:
        allowed = allowed | _THINKING_FIELDS
    out = {k: v for k, v in msg.items() if k in allowed}
    # Read-path mirror of the persist-time guard (see
    # ``completion._is_persistable_thinking_block`` / issue #1588). A poison
    # thinking block (non-empty thinking text, empty/missing ``signature``)
    # persisted BEFORE the persist-path guard shipped — or carried in from
    # another agent — would otherwise replay to Anthropic and 400 on EVERY
    # turn with "Invalid signature in thinking block", an unbreakable
    # terminal-error loop (clearing the errored state alone re-loops because
    # the poison block stays in the window). Dropping it here quarantines
    # the block by construction: a thinking-less turn always replays 200.
    if raw_blocks := out.get("thinking_blocks"):
        if isinstance(raw_blocks, list):
            safe_blocks = [b for b in raw_blocks if _is_replayable_thinking_block(b)]
            if safe_blocks:
                out["thinking_blocks"] = safe_blocks
            else:
                del out["thinking_blocks"]
        else:
            del out["thinking_blocks"]
    if raw_tcs := out.get("tool_calls"):
        sanitized = _sanitize_tool_calls(raw_tcs)
        if sanitized:
            out["tool_calls"] = sanitized
        else:
            # Strict providers reject an empty tool_calls array; drop
            # the key so the assistant is a plain text-only turn.
            del out["tool_calls"]
    return out


# ─── build_messages ──────────────────────────────────────────────────────────


_PENDING_BACKGROUND = json.dumps(
    {
        "status": "pending",
        "message": (
            "This tool is still executing in the background. "
            "Its result will arrive when ready. "
            "Do not re-request this tool."
        ),
    }
)
_PENDING_EXTERNAL = json.dumps(
    {
        "status": "pending",
        "message": (
            "This tool is awaiting external action (client execution or "
            "operator approval). Its result will arrive when that action "
            "completes. Continue handling other work; do not re-request "
            "this tool."
        ),
    }
)


@dataclass(slots=True)
class ContextResult:
    """Return value of :func:`build_messages`."""

    messages: list[dict[str, Any]]
    reacting_to: int  # max seq of user/tool events included in context


def _quarantine_placeholder(seq: int) -> dict[str, Any]:
    """Deterministic stand-in for an event whose render raised.

    A function of ``seq`` ONLY — no timestamps, no counters — so the
    context stays a monotonic function of the log (see module docstring):
    re-rendering the same window always produces byte-identical output at
    this position, and appending later events never rewrites it.
    """
    return {"role": "user", "content": f"[unrenderable event seq={seq} — quarantined]"}


# Upper bound (local approx_tokens units) reserved in the window budget for
# the omission marker, mirroring ``tail_block_upper_bound_local``: the marker
# is appended after windowing runs, so without a reserve it would push the
# send-time payload past ``window_max`` (the PR #165 full-payload invariant).
# Reserved unconditionally — like the tail block, which also may not render.
# ``TestOmissionMarker`` pins a worst-case render under this bound.
OMISSION_MARKER_UPPER_BOUND_LOCAL = 128


def _approx_count(n: int) -> str:
    """Round ``n`` down to two significant figures, comma-grouped.

    Deterministic (pure floor, no banker's rounding ambiguity) so the
    omission marker it feeds stays byte-stable across rebuilds.
    """
    if n >= 100:
        magnitude = 10 ** (len(str(n)) - 2)
        n -= n % magnitude
    return f"{n:,}"


def _omission_marker(omission: WindowOmission, boundary: datetime, tz_name: str) -> dict[str, Any]:
    """Head marker telling the model the window omits transcript (#738).

    Byte-stable within a snap chunk — see :class:`WindowOmission` for the
    cache-stability rationale.  The boundary timestamp reuses
    :func:`_format_received` so it renders identically to the
    ``received=`` envelope headers; the start date uses the same account
    timezone, date-only.  ``_prune_orphans`` may hide a few more
    rendered messages at/after the boundary — "everything before" remains
    true regardless, so the claim direction is safe.
    """
    began = omission.began_at.astimezone(ZoneInfo(tz_name)).date().isoformat()
    before = _format_received(boundary, tz_name)
    if omission.omitted_messages > 0:
        scrolled = (
            f"Everything before {before} — about "
            f"{_approx_count(omission.omitted_messages)} messages, including your own — "
            "has scrolled out of view."
        )
    else:
        # Degenerate: the omitted span holds only tool results.
        scrolled = f"Everything before {before} has scrolled out of view."
    return {
        "role": "user",
        "content": (
            f"[history: this conversation began {began}. {scrolled} "
            "Nothing is lost: the full transcript remains queryable with search_events. "
            "What seems unfamiliar is usually forgotten, not new: when in doubt about "
            "anything that's referred to, search first rather than fill the gap by "
            "assumption.]"
        ),
    }


def build_messages(
    events: list[Event],
    *,
    system_prompt: str | None,
    model: str | None = None,
    session_id: str | None = None,
    workspace_path: Path | None = None,
    in_flight_tool_call_ids: frozenset[str] = frozenset(),
    tz_name: str = "UTC",
    omission: WindowOmission | None = None,
) -> ContextResult:
    """Assemble a chat-completions message list from pre-windowed events.

    Callers are expected to pass events that have already been windowed
    (via :func:`~aios.db.queries.read_windowed_events`).  This function
    handles message assembly, pending-result synthesis, blind-spot
    injection, and leading-orphan pruning — but not windowing itself.

    **Monotonicity invariant:** the context is a monotonic function of
    the log — appending events only appends to the context, never
    rewrites earlier messages.

    Each assistant has a **visibility horizon** — the ``reacting_to``
    of the next assistant after it. A tool result with
    ``seq <= horizon`` was visible; one with ``seq > horizon`` arrived
    in the blind spot and is shown as pending in the paired position,
    then injected as a user message right after the horizon-setting
    assistant. This placement preserves monotonicity — new events
    append after the injection rather than before it.
    """
    # Index: tool_call_id → (data, seq).
    real_results: dict[str, dict[str, Any]] = {}
    real_result_seqs: dict[str, int] = {}
    for e in events:
        if e.kind == "message" and e.data.get("role") == "tool":
            tcid = e.data.get("tool_call_id")
            if tcid:
                real_results[tcid] = e.data
                real_result_seqs[tcid] = e.seq

    # Visibility horizon per assistant: the reacting_to of the NEXT
    # assistant. If there's no next assistant, horizon is infinite.
    _INF: int = 2**63
    asst_list = [
        (e.seq, e.data.get("reacting_to", e.seq))
        for e in events
        if e.kind == "message" and e.data.get("role") == "assistant"
    ]
    horizon_for: dict[int, int] = {}
    horizon_setter_for: dict[int, int] = {}
    for i, (seq, _rt) in enumerate(asst_list):
        if i + 1 < len(asst_list):
            horizon_for[seq] = asst_list[i + 1][1]
            horizon_setter_for[seq] = asst_list[i + 1][0]
        else:
            horizon_for[seq] = _INF

    # Blind-spot window for USER messages — the symmetric twin of the tool-result
    # injection below. A user message that arrived DURING the last assistant's
    # inference (committed before it, but after its visibility horizon
    # ``reacting_to``) is "stranded": in seq order it lands *before* that trailing
    # assistant turn, so the assembled context would END on an assistant message.
    # Current reasoning models reject a trailing assistant turn as an unsupported
    # prefill ("the conversation must end with a user message"), and the inference
    # gate fires for it anyway (it's genuinely unreacted). So defer a user message
    # in the window ``last_asst_rt < seq < last_asst_seq`` to the tail (flushed
    # after the walk). Only the LAST assistant can strand the tail — a user blind
    # to an earlier assistant is followed by a later one that reacted, so it never
    # ends the list.
    last_asst_seq, last_asst_rt = asst_list[-1] if asst_list else (0, 0)

    # Walk events in seq order.
    emitted_tcids: set[str] = set()
    messages: list[dict[str, Any]] = []
    inject_after: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    deferred_user_tail: list[dict[str, Any]] = []
    max_stimulus_seq: int = 0

    for e in events:
        # Durable-session-sandbox FS-loss notices (§5.9): the only non-message
        # events that render. Append-only at this seq position (monotonic) and
        # NOT stimulus-bearing — deliberately does not touch max_stimulus_seq,
        # so a GC/reset append never advances ``reacting_to`` or wakes the
        # session; the model reads the notice at its next genuine wake.
        if e.kind == "lifecycle" and e.data.get("event") in MODEL_VISIBLE_LIFECYCLE_EVENTS:
            # Dispatch on the lifecycle kind: FS-loss notices keep their
            # renderer; ``connector_delivery_failed`` (#1261) routes to its own.
            # Both are NON-stimulus-bearing here — the delivery-failure wake is
            # produced by the session-targeted lifecycle route, not by render.
            if e.data.get("event") == "connector_delivery_failed":
                content = _render_delivery_failure_notice(e.data)
            elif e.data.get("event") in (
                "connector_message_delivered",
                "connector_message_edited",
            ):
                # #1341: the success-path acks (delivered / edit-landed). Like
                # the delivery-failure arm they are NON-stimulus-bearing here;
                # the optional wake comes from the lifecycle route, not render.
                content = _render_ack_notice(e.data)
            else:
                content = _render_fs_lifecycle_notice(e.data)
            messages.append({"role": "user", "content": content})
            continue
        if e.kind != "message":
            continue
        # Quarantine backstop (#686): build_messages is a pure replay over
        # the immutable log, run on EVERY wake. A single event that makes the
        # render dispatch raise would brick the session permanently — the
        # model is never called, so the "model sees the error and retries"
        # recovery never engages. Degrade THAT ONE event to a deterministic
        # placeholder (a function of e.seq only, preserving monotonicity)
        # rather than failing the whole build. The inner isinstance/OSError
        # guards still pre-empt this for the shapes they cover; this catches
        # novel raisers they don't.
        mark = len(messages)
        try:
            role = e.data.get("role")

            if role == "user":
                msg = render_user_event(
                    e.data,
                    e.orig_channel,
                    e.focal_channel_at_arrival,
                    e.created_at,
                    tz_name=tz_name,
                    model=model,
                    session_id=session_id,
                    workspace_path=workspace_path,
                )
                max_stimulus_seq = max(max_stimulus_seq, e.seq)
                if last_asst_rt < e.seq < last_asst_seq:
                    # Blind to the last assistant — defer to the tail (below) so
                    # the context can't end on that assistant turn.
                    deferred_user_tail.append(msg)
                else:
                    messages.append(msg)

            elif role == "assistant":
                messages.append(e.data)
                horizon = horizon_for.get(e.seq, _INF)
                for tc in e.data.get("tool_calls") or []:
                    tcid = tc.get("id")
                    if not tcid or tcid in emitted_tcids:
                        continue
                    rseq = real_result_seqs.get(tcid)
                    if rseq is not None and rseq <= horizon:
                        messages.append(real_results[tcid])
                        max_stimulus_seq = max(max_stimulus_seq, rseq)
                    else:
                        placeholder = (
                            _PENDING_BACKGROUND
                            if tcid in in_flight_tool_call_ids
                            else _PENDING_EXTERNAL
                        )
                        messages.append(
                            {"role": "tool", "tool_call_id": tcid, "content": placeholder}
                        )
                        if tcid in real_results:
                            # Safe: last assistant has horizon=INF so rseq<=INF
                            # always takes the REAL branch above; only non-last
                            # assistants reach here, and they all have entries.
                            setter_seq = horizon_setter_for[e.seq]
                            inject_after.setdefault(setter_seq, []).append(
                                (tcid, real_results[tcid])
                            )
                    emitted_tcids.add(tcid)

                # Inline blind-spot injections anchored to this assistant
                # (preserves prefix monotonicity — see docstring).
                for inj_tcid, inj_data in inject_after.pop(e.seq, []):
                    name = inj_data.get("name", "tool")
                    header = f"[Tool result: {name} (call {inj_tcid}) completed]"
                    inj_content = inj_data.get("content", "")
                    if isinstance(inj_content, list):
                        # Multimodal tool result (e.g. image-aware read returning a
                        # text + image_url part list).  F-stringing would emit the
                        # Python repr of the list, losing the pixels — splice the
                        # parts into the synthetic user message instead so the model
                        # sees the image in the blind-spot signal too.  Spec'd text
                        # parts are concatenated under the header; non-text parts
                        # (image_url, etc.) follow as siblings.
                        text_chunks: list[str] = [header]
                        other_parts: list[dict[str, Any]] = []
                        for part in inj_content:
                            if not isinstance(part, dict):
                                continue
                            if part.get("type") == "text":
                                txt = part.get("text")
                                if isinstance(txt, str) and txt:
                                    text_chunks.append(txt)
                            else:
                                other_parts.append(part)
                        combined_text = "\n".join(text_chunks)
                        if other_parts:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": combined_text},
                                        *other_parts,
                                    ],
                                }
                            )
                        else:
                            messages.append({"role": "user", "content": combined_text})
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"{header}\n{inj_content}",
                            }
                        )
                    max_stimulus_seq = max(max_stimulus_seq, real_result_seqs[inj_tcid])

            elif role == "tool":
                # Reaching this branch means NO in-window assistant declared
                # this tcid — the assistant branch above records every id it
                # emits in ``emitted_tcids``, so a standalone tool event here
                # is always a structural orphan (its issuing assistant was
                # windowed out). We still advance ``max_stimulus_seq``: the
                # result is a real stimulus the watermark must count
                # (``reacting_to`` = f(log)), and under-counting it risks
                # re-waking on an already-consumed event. The structurally
                # invalid message itself is removed downstream by
                # ``_prune_orphans`` (message list = f(structural validity)) —
                # do NOT "correct-by-construction" this by skipping the append
                # here, which would entangle the two and regress the watermark.
                tcid = e.data.get("tool_call_id")
                if tcid and tcid not in emitted_tcids:
                    messages.append(e.data)
                    emitted_tcids.add(tcid)
                    max_stimulus_seq = max(max_stimulus_seq, e.seq)

        except Exception as exc:
            # Roll back any partial appends from THIS event so the quarantine
            # is atomic w.r.t. ``messages``: exactly one placeholder per
            # quarantined position, never a half-rendered assistant turn
            # (e.g. the assistant message appended before a corrupt
            # ``tool_calls`` raised, which would leave an orphan tool_calls
            # turn — an invalid chat-completions sequence that re-bricks).
            # Residual limitation: a quarantined ASSISTANT event's downstream
            # tool-result events (later in the window) may still render as
            # orphan ``tool`` messages — accepted, because assistant events are
            # harness-produced, not external connector poison (the realistic
            # source), and the alternative is the permanent brick this guard exists
            # to prevent.
            del messages[mark:]
            log.warning(
                "context.poison_event_quarantined",
                session_id=session_id,
                seq=e.seq,
                error_type=type(exc).__name__,
            )
            messages.append(_quarantine_placeholder(e.seq))
            max_stimulus_seq = max(max_stimulus_seq, e.seq)

    # Flush deferred blind-spot USER messages at the tail (see the window above):
    # they now follow the assistant turn that never saw them, so the context ends
    # on a user turn — the faithful chronology, since that turn's context never
    # contained them — and no accidental prefill is sent. (max_stimulus_seq was
    # already advanced for each at collection time, since max is order-independent.)
    messages.extend(deferred_user_tail)

    # Prune dangling messages at the start of the window.  DB-level
    # windowing can cut in the middle of an assistant+tool_result group,
    # leaving orphan tool results or an assistant with missing paired
    # results.
    messages = _prune_orphans(messages)

    # Head omission marker (#738): when the window omits transcript, tell
    # the model how much and how to recall it. After the prune (so it can't
    # be pruned), before the system insert (so it lands at messages[1]).
    # ``read_windowed_events`` guarantees a non-empty window whenever it
    # reports an omission (drop < total by construction), so ``events[0]``
    # is the first retained event — the boundary.
    if omission is not None:
        messages.insert(0, _omission_marker(omission, events[0].created_at, tz_name))

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    _correct_image_data_url_mimes(messages)
    _clamp_oversize_image_data_urls(messages)

    # Resolve the thinking-capability sniff through the shared provider-quirk
    # resolver in ``completion.py`` (consolidated from the prior inline
    # expression). The import is **function-local on purpose**: ``completion.py``
    # imports ``litellm`` at module top, so a module-top import here would
    # silently re-introduce the ~1.18s litellm bootstrap for every consumer of
    # this module — the very cost this lazy import defers. Most call sites either
    # exit before this point or run under tests that never reach it.
    #
    # Same stale-catalog short-circuit as ``supports_vision`` (see its docstring
    # for the full rationale): a Claude newer than this worker's catalog
    # snapshot — or a proxy-routed one litellm under-reports even when fresh —
    # must keep its ``thinking_blocks``, or stripping them across turns violates
    # Anthropic's preservation contract.  Extended thinking is Claude 4.x+;
    # over-broad for <= 3.5 (no thinking), which aios doesn't run.
    #
    # ``bool(model) and …`` preserves the empty-model short-circuit: the resolver
    # is not called for an empty model string.
    from aios.harness.completion import model_descriptor

    target_supports_thinking = bool(model) and model_descriptor(model).supports_thinking
    stripped = [
        _strip_to_spec(m, target_supports_thinking=target_supports_thinking) for m in messages
    ]
    # Drop degenerate empty assistant turns (no content, no tool_calls, no
    # thinking) before replay. They carry zero information, and replaying them
    # teaches literal-minded models — claude-fable-5 in particular — to imitate
    # silence: a run of empty assistant turns in the window makes fable
    # deterministically emit another empty turn (proven: 0% empty on a clean
    # window, 100% empty when the window holds such a run; opus-4.x is immune).
    # LiteLLM also rewrites their empty content to a "[System: Empty message
    # content sanitised…]" marker on the wire, so the model literally sees a wall
    # of its own malfunction markers. Excluding them is safe for every model
    # (nothing is lost) and breaks the imitation loop at the source.
    stripped = [m for m in stripped if not _is_degenerate_empty_assistant(m)]
    return ContextResult(messages=stripped, reacting_to=max_stimulus_seq)


def _is_degenerate_empty_assistant(msg: dict[str, Any]) -> bool:
    """True for an assistant turn carrying no content, no tool_calls, no thinking.

    Such a turn is a non-event — the model produced nothing actionable. It is
    distinct from a tool-call turn (``tool_calls`` present) or a thinking turn
    (``thinking_blocks`` present), both of which are load-bearing and kept.
    """
    if msg.get("role") != "assistant":
        return False
    if msg.get("tool_calls") or msg.get("thinking_blocks"):
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        return not any(isinstance(b, dict) and (b.get("text") or "").strip() for b in content)
    return content is None


# ─── helpers ─────────────────────────────────────────────────────────────────


def _prune_orphans(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop structurally-orphaned messages produced by window cuts.

    DB-level windowing slices the event log at a token budget and can cut
    through an assistant + tool_result group, leaving an invalid
    chat-completions sequence that strict providers (Anthropic / Bedrock)
    reject. Because :func:`build_messages` is pure replay over the
    immutable window, that rejection recurs on every wake and permanently
    wedges the session — so the structure must be repaired here. Two
    orphan shapes arise:

    * **Incomplete leading assistant** — an ``assistant`` at the front
      whose ``tool_calls`` ids are not all matched by following ``tool``
      results (a paired result was dropped by the boundary, or a
      ``tool_call`` carries no ``id``). The whole leading group is dropped.
    * **Orphan tool result** — a ``tool`` message whose ``tool_call_id``
      has no preceding ``assistant`` ``tool_calls`` declaring it (the
      issuing assistant was dropped). This sits at the FRONT of the
      window, or MID-list: the blind-spot race appends a late tool result
      after an interleaved user message (assistant < user < result in seq
      order), and a boundary that drops the assistant but keeps the user
      and the result strands the result *after* a clean ``user`` start.
      Such orphans are dropped at ANY position.

    The leading scan stops at the first clean start (a ``user`` or a
    complete ``assistant``), so it alone cannot reach a mid-list orphan;
    the position-independent sweep that follows is what catches those.
    """
    # Leading scan: establish a clean conversation start by dropping a
    # leading incomplete-assistant group (and any orphan tool results
    # ahead of it). Walk forward until a ``user`` or an ``assistant``
    # whose tool_calls are all paired — both are valid starts.
    start = 0
    while start < len(messages):
        msg = messages[start]
        role = msg.get("role")

        if role == "user":
            break  # clean start

        if role == "assistant":
            raw_tcs = msg.get("tool_calls") or []
            if not raw_tcs:
                break  # assistant with no tool_calls — clean start
            # An entry without an ``id`` is unjoinable to any tool result
            # — treat it like a tool_call whose pair was dropped by the
            # window, falling into the "incomplete group" path below.
            remaining_result_ids = {
                m.get("tool_call_id") for m in messages[start + 1 :] if m.get("role") == "tool"
            }
            if all(tc.get("id") in remaining_result_ids for tc in raw_tcs):
                break  # complete group — clean start
            # Incomplete group — drop the assistant and its partial results.
            start += 1
            while start < len(messages) and messages[start].get("role") == "tool":
                start += 1
            continue

        # tool or anything else at the front — orphan, drop.
        start += 1

    messages = messages[start:]

    # Position-independent sweep: drop any tool result whose tool_call_id
    # was never declared by a preceding in-window assistant. Catches the
    # mid-list orphan the leading scan stops short of (the interleaved-user
    # window cut above); a no-op on lists the leading scan already cleaned.
    declared: set[str] = set()
    pruned: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            declared.update(tc["id"] for tc in (msg.get("tool_calls") or []) if tc.get("id"))
            pruned.append(msg)
        elif role == "tool":
            if msg.get("tool_call_id") in declared:
                pruned.append(msg)
            # else: orphan result with no declaring assistant — drop
        else:
            pruned.append(msg)
    return pruned


def stub_missing_reasoning_content(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure every assistant message carries ``reasoning_content``.

    **Call this only for thinking-capable targets** (gate the call on
    ``model_descriptor(model).supports_thinking``).  Thinking-mode models
    (DeepSeek V4 Flash, etc.) reject transcripts whose assistant turns
    lack this field: ``The reasoning_content in the thinking mode must be
    passed back to the API``.

    Non-thinking targets must NOT receive this stub: ``_strip_to_spec``
    (in ``build_messages``) deliberately removes ``reasoning_content``
    from their assistant turns, and re-adding an empty one here would
    contradict that strip pass — reintroducing a field for providers that
    never asked for it.  Gating both passes on the single
    ``supports_thinking`` verdict keeps them in agreement.

    Mutates messages in place and returns the list for chaining.  Skips
    messages that already have a reasoning_content set (from a prior
    thinking-model turn whose output we preserved opaquely in the log).
    """
    for msg in messages:
        if msg.get("role") == "assistant" and "reasoning_content" not in msg:
            msg["reasoning_content"] = ""
    return messages


EPHEMERAL_TAIL_KEY = "_aios_ephemeral_tail"
"""Out-of-band marker key tagging a per-step-ephemeral tail message.

Set to ``True`` at construction on the render-only tail blocks
(:func:`~aios.harness.channels.build_channels_tail_block`,
:func:`~aios.harness.obligations.build_obligations_tail_block`) whose
content mutates every step (unread counts/previews; obligation ages/sets).

The cache-breakpoint recognizer in ``completion.py`` reads this marker —
never the rendered prose — to decide which message must NOT host the
conversation prefix ``cache_control`` breakpoint.  It is a *property*
("this message is per-step-ephemeral"), not a discriminated kind, so a
boolean is the honest shape.

The marker is non-standard (Anthropic rejects unknown message fields) and
is stripped from every message by ``inject_cache_breakpoints`` before any
provider call — on every route, including non-Anthropic early returns.
``_concat_user_messages`` propagates it under OR so a merge of any
ephemeral message with anything stays ephemeral.
"""


_USER_MESSAGE_SEPARATOR_CONTENT = "."
"""Single-byte placeholder for the role-transition separator.

Must survive every validator the messages array can traverse on the way
to a provider.  An earlier implementation used ``""`` and relied on
LiteLLM's ``modify_params = True`` Anthropic sanitizer to strip the
empty block — that only fires when LiteLLM is itself the Anthropic
provider.  When the model routes through a relay (``openrouter/*``,
``openai-compatible/*``), the placeholder reaches the upstream provider
unchanged.  Strict providers (Bedrock confirmed; Vertex likely) reject
non-final assistant messages with empty content, wedging the session.

``"."`` is the minimal change with maximum portability: one printable
byte that every text-content validator accepts.  Recognizer in
``completion.py`` (cache-breakpoint placement skips the placeholder)
imports this constant to stay in lock-step.
"""


def merge_adjacent_user_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge consecutive user-role messages into a single user turn.

    Anthropic requires alternating roles, so two adjacent user messages
    must become one. The earlier approach inserted a placeholder
    assistant turn (a ``"."``) between them to *force* them apart and
    stop LiteLLM concatenating a user inbound with the channels tail
    block (which made models narrate "your message included the channel
    state"). That placeholder is now both unnecessary and harmful:

    * Unnecessary — the channels tail block is no longer appended after
      an unanswered user inbound (see ``compose_step_context`` /
      ``_agent_owes_response``), so the tail-merge case it guarded
      against no longer arises; the only remaining adjacent-user case is
      genuine successive inbounds (feeder pings + user text that landed
      before the agent acted), which *should* read as one block.
    * Harmful — a ``"."`` is a degenerate assistant turn. Literal-minded
      models imitate it: ``claude-fable-5`` pattern-completes a run of
      ``"."`` placeholders into silence and emits empty turns
      deterministically (a single ``"."`` before the prompt flips fable
      from 0% to 100% empty turns in repro; opus-4.x is immune). The
      placeholders also accreted with every feeder ping, poisoning
      long-lived agent sessions.

    Merging here (rather than letting LiteLLM do it implicitly) keeps the
    behaviour deterministic and route-independent: no degenerate
    placeholder ever reaches the provider, and each merged inbound keeps
    its own channel header so the sequence stays legible.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if result and result[-1].get("role") == "user" and msg.get("role") == "user":
            result[-1] = _concat_user_messages(result[-1], msg)
        else:
            result.append(dict(msg))
    return result


def _concat_user_messages(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Concatenate two user messages, preserving content-block structure.

    String contents join with a blank line (each rendered inbound already
    carries its ``[channel=… received=…]`` header). List contents (vision
    parts) concatenate block-wise; mixed shapes normalise to a block list.
    """
    ca, cb = a.get("content"), b.get("content")
    if isinstance(ca, str) and isinstance(cb, str):
        merged: dict[str, Any] = {"role": "user", "content": f"{ca}\n\n{cb}"}
    else:
        la = ca if isinstance(ca, list) else [{"type": "text", "text": ca or ""}]
        lb = cb if isinstance(cb, list) else [{"type": "text", "text": cb or ""}]
        merged = {"role": "user", "content": [*la, *lb]}
    # Propagate the ephemeral-tail marker under OR: a dict that contains
    # *any* per-step-mutating content cannot host the stable-prefix cache
    # breakpoint, so the merge of any ephemeral message with anything is
    # ephemeral. This fixes the trailing-inbound + obligations merge case.
    if a.get(EPHEMERAL_TAIL_KEY) or b.get(EPHEMERAL_TAIL_KEY):
        merged[EPHEMERAL_TAIL_KEY] = True
    return merged
