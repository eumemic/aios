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
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import litellm

from aios.harness.vision import (
    can_inline_image,
    correct_image_mime_b64,
    make_image_url_part,
    text_marker,
)
from aios.logging import get_logger
from aios.models.events import Event
from aios.sandbox.volumes import resolve_to_host_path

log = get_logger("aios.harness.context")

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

# Notification markers truncate the source content to this many chars
# (plus an ellipsis when truncated) so a busy non-focal channel
# contributes O(tens-of-tokens) per inbound to the context — cheap
# enough to keep in-log at its seq position for episodic chronology.
_NOTIFICATION_PREVIEW_CHARS = 80


def _format_channel_header(metadata: dict[str, Any]) -> str:
    """Render a one-line header describing the origin of an inbound message.

    When a user message carries channel metadata, the raw fields are
    whitelisted out of the chat-completions message before the model
    ever sees them (see ``_ALLOWED_FIELDS``).  That leaves the model
    with no way to know the sender or timestamp — values the connector
    tools need as arguments.  Inline the salient fields into the
    visible ``content`` so the model can read them natively.
    """
    if not isinstance(metadata, dict) or "channel" not in metadata:
        return ""
    parts: list[str] = [f"channel={metadata['channel']}"]
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
        iso = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat(timespec="milliseconds")
        parts.append(f"timestamp_ms={timestamp_ms} ({iso})")
    message_id = metadata.get("message_id")
    if isinstance(message_id, int):
        parts.append(f"message_id={message_id}")
    if metadata.get("edited") is True:
        parts.append("edited=true")
    edit_target = metadata.get("edit_target_timestamp_ms")
    if isinstance(edit_target, int):
        parts.append(f"edit_target_timestamp_ms={edit_target}")
    if metadata.get("self_mentioned") is True:
        # Hoist ahead of the structured ``mentions`` list — for
        # group chats the model often only needs to know "was I
        # tagged?" and substring-matching ``content`` is the
        # alternative that #5 set out to eliminate.
        parts.append("self_mentioned=true")
    sticker_emoji = metadata.get("sticker_emoji")
    if isinstance(sticker_emoji, str) and sticker_emoji:
        parts.append(f"sticker_emoji={sticker_emoji!r}")
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
        emoji = reaction.get("emoji") or "?"
        r_parts: list[str] = [f"reaction={emoji!r}"]
        target_author = reaction.get("target_author_uuid")
        if isinstance(target_author, str) and target_author:
            r_parts.append(f"target_author_uuid={target_author}")
        target_ts = reaction.get("target_timestamp_ms")
        if isinstance(target_ts, int):
            r_parts.append(f"target_timestamp_ms={target_ts}")
        header += "\n[" + " · ".join(r_parts) + "]"
    reply_to = metadata.get("reply_to")
    if isinstance(reply_to, dict):
        quoted = (reply_to.get("text") or "").replace("\n", " ").strip()
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
            emoji = reaction.get("emoji")
            if isinstance(emoji, str) and emoji:
                return f"reacted {emoji}"
    return ""


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
    parts = [f"🔔 channel_id={orig_channel}"]
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


def render_user_event(
    event_data: dict[str, Any],
    orig_channel: str | None,
    focal_channel_at_arrival: str | None,
    *,
    model: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Render a user event into its chat-completions message form.

    Rendering is a deterministic function of the event's stamped
    ``orig_channel``, ``focal_channel_at_arrival``, and the optional
    vision policy inputs ``model`` / ``session_id``.  ``build_messages``
    threads vision policy through; the append-time token counter in
    ``queries.append_event`` does not (and pays a small under-count
    per inlined image, absorbed by ``model_token_ratio`` calibration).

    Branches:

    * ``orig_channel`` is ``None`` → legacy / non-connector event;
      metadata stripped, no header, no notification.  Covers direct
      ``POST /sessions/{id}/messages`` traffic and pre-migration rows.
    * ``orig == focal_at_arrival`` (non-NULL) → full content with a
      metadata header inlined.  When the focal event's metadata
      carries ``attachments`` and the bound model supports vision,
      inlinable images are emitted as ``image_url`` content parts;
      everything else gets a text marker referencing the in-sandbox
      path.
    * Otherwise (focal is NULL, or focal differs from orig) →
      notification marker: a short, emoji-prefixed one-liner surfacing
      the origin channel, sender, and a truncated content preview.
    """
    msg = {k: v for k, v in event_data.items() if k != "metadata"}
    metadata = event_data.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else None

    if orig_channel is None:
        return msg

    if orig_channel == focal_channel_at_arrival:
        if metadata:
            header = _format_channel_header(metadata)
            if header:
                existing = msg.get("content") or ""
                msg["content"] = f"{header}\n{existing}" if existing else header
            attachments = metadata.get("attachments")
            if isinstance(attachments, list) and attachments:
                _apply_attachments(
                    msg,
                    attachments,
                    model=model,
                    session_id=session_id,
                )
        return msg

    content = event_data.get("content", "")
    if not isinstance(content, str):
        content = ""
    marker = _format_notification_marker(orig_channel, metadata, content)
    return {"role": "user", "content": marker}


def _apply_attachments(
    msg: dict[str, Any],
    attachments: list[Any],
    *,
    model: str | None,
    session_id: str | None,
) -> None:
    leading_text = msg.get("content") if isinstance(msg.get("content"), str) else ""
    marker_lines: list[str] = []
    image_parts: list[dict[str, Any]] = []

    for record in attachments:
        host_path = _resolve_attachment_host_path(record, session_id)
        size = record.get("size")
        content_type = record.get("content_type")
        inline = (
            host_path is not None
            and model is not None
            and isinstance(content_type, str)
            and isinstance(size, int)
            and can_inline_image(model=model, content_type=content_type, size_bytes=size)
        )
        if inline:
            # v1 deliberately re-reads + re-base64-encodes per render: the
            # staged bytes are typically immutable (`/mnt/attachments/` is
            # read-only) so an LRU cache on (host_path, mtime, size) would
            # be a clean follow-up if profiling shows pressure.  See
            # PR #216 §14 for the discussion of cache vs. encode-at-staging
            # alternatives we deferred for v1.
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
            image_parts.append(
                make_image_url_part(
                    content_type=content_type,
                    data_b64=base64.b64encode(payload).decode("ascii"),
                )
            )
        else:
            marker_lines.append(text_marker(record))

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


def _resolve_attachment_host_path(record: dict[str, Any], session_id: str | None) -> Any:
    sandbox_path = record.get("in_sandbox_path")
    if not isinstance(sandbox_path, str) or session_id is None:
        return None
    return resolve_to_host_path(session_id, sandbox_path)


def _sanitize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize tool_calls inner structure for cross-model replay.

    Some models produce malformed tool_calls (control characters in
    arguments, missing fields, extra provider keys) that break strict
    downstream providers.  Returns a cleaned copy with only spec fields
    and valid JSON arguments.
    """
    sanitized = []
    for tc in tool_calls:
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
                "id": tc.get("id") or "",
                "type": "function",
                "function": {"name": fn.get("name") or "", "arguments": raw_args},
            }
        )
    return sanitized


def _correct_image_data_url_mimes(messages: list[dict[str, Any]]) -> None:
    """Rewrite mismatched mime declarations on ``data:<mime>;base64,...``
    image URLs already in the assembled message list.  Mutates in place.

    The renderer cannot prevent persisted events from carrying a wrong
    mime — tool_result content arrives pre-formed and is appended
    verbatim, so historical events from before centralised sniffing
    will replay forever unless we re-check them every build.
    """
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
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
                image_url["url"] = f"data:{corrected};base64,{data_b64}"


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
    if out.get("tool_calls"):
        out["tool_calls"] = _sanitize_tool_calls(out["tool_calls"])
    return out


# ─── build_messages ──────────────────────────────────────────────────────────


_PENDING_CONTENT = json.dumps(
    {
        "status": "pending",
        "message": (
            "This tool is still executing in the background. "
            "Its result will arrive when ready. "
            "Do not re-request this tool."
        ),
    }
)


@dataclass(slots=True)
class ContextResult:
    """Return value of :func:`build_messages`."""

    messages: list[dict[str, Any]]
    reacting_to: int  # max seq of user/tool events included in context


def build_messages(
    events: list[Event],
    *,
    system_prompt: str | None,
    model: str | None = None,
    session_id: str | None = None,
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

    # Walk events in seq order.
    emitted_tcids: set[str] = set()
    messages: list[dict[str, Any]] = []
    inject_after: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    max_stimulus_seq: int = 0

    for e in events:
        if e.kind != "message":
            continue
        role = e.data.get("role")

        if role == "user":
            msg = render_user_event(
                e.data,
                e.orig_channel,
                e.focal_channel_at_arrival,
                model=model,
                session_id=session_id,
            )
            messages.append(msg)
            max_stimulus_seq = max(max_stimulus_seq, e.seq)

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
                    messages.append(
                        {"role": "tool", "tool_call_id": tcid, "content": _PENDING_CONTENT}
                    )
                    if tcid in real_results:
                        # Safe: last assistant has horizon=INF so rseq<=INF
                        # always takes the REAL branch above; only non-last
                        # assistants reach here, and they all have entries.
                        setter_seq = horizon_setter_for[e.seq]
                        inject_after.setdefault(setter_seq, []).append((tcid, real_results[tcid]))
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
            tcid = e.data.get("tool_call_id")
            if tcid and tcid not in emitted_tcids:
                messages.append(e.data)
                emitted_tcids.add(tcid)
                max_stimulus_seq = max(max_stimulus_seq, e.seq)

    # Prune dangling messages at the start of the window.  DB-level
    # windowing can cut in the middle of an assistant+tool_result group,
    # leaving orphan tool results or an assistant with missing paired
    # results.
    messages = _prune_leading_orphans(messages)

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    _correct_image_data_url_mimes(messages)

    target_supports_thinking = bool(model) and litellm.supports_reasoning(model)
    return ContextResult(
        messages=[
            _strip_to_spec(m, target_supports_thinking=target_supports_thinking) for m in messages
        ],
        reacting_to=max_stimulus_seq,
    )


# ─── helpers ─────────────────────────────────────────────────────────────────


def _prune_leading_orphans(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop messages from the front until we reach a clean conversation start.

    Windowing can cut in the middle of an assistant + tool_result group,
    leaving orphan tool results at the start (no preceding assistant with
    matching tool_calls) or an assistant whose paired results were
    partially dropped.

    Walk forward and drop until we find a ``user`` message or an
    ``assistant`` without ``tool_calls`` — both are valid starts. An
    ``assistant`` with ``tool_calls`` is valid only if ALL its
    tool_call_ids have matching tool results later in the list.
    """
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

    return messages[start:]


def stub_missing_reasoning_content(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure every assistant message carries ``reasoning_content``.

    Thinking-mode models (DeepSeek V4 Flash, etc.) reject transcripts
    whose assistant turns lack this field: ``The reasoning_content in the
    thinking mode must be passed back to the API``.  Non-thinking models
    (Anthropic / OpenAI / Gemini / Llama / DeepSeek v3 — all probed)
    ignore the field entirely, so setting an empty stub unconditionally
    costs nothing and lets cross-model sessions use thinking models for
    a single turn without poisoning their replay on other providers.

    Mutates messages in place and returns the list for chaining.  Skips
    messages that already have a reasoning_content set (from a prior
    thinking-model turn whose output we preserved opaquely in the log).
    """
    for msg in messages:
        if msg.get("role") == "assistant" and "reasoning_content" not in msg:
            msg["reasoning_content"] = ""
    return messages


def separate_adjacent_user_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Insert an empty assistant message between any two adjacent user messages.

    LiteLLM's Anthropic translator enforces strict role alternation by
    merging adjacent same-role messages into a single multi-content-block
    payload.  When a user inbound is immediately followed by the
    per-step channels tail block (also user-role), Anthropic sees them
    as one message with two text blocks — and models narrate "your
    message included the channel state" about their own scaffolding.

    Inserting a no-content assistant turn between two consecutive user
    messages blocks the merge.  LiteLLM's ``modify_params = True`` (set
    in ``completion.py``) sanitizes the empty content block for
    Anthropic at request time, so no visible turn is added to the
    on-the-wire transcript — only the role transition remains.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if result and result[-1].get("role") == "user" and msg.get("role") == "user":
            result.append({"role": "assistant", "content": ""})
        result.append(msg)
    return result
