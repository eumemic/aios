"""Context builder for the step function.

Two public functions:

* :func:`should_call_model` — predicate: should the current wake result
  in an inference call, or should the step early-out?

* :func:`build_messages` — assemble the chat-completions message list
  from the event log, synthesizing ``"pending"`` results for in-flight
  tool calls and reordering tool results so they appear immediately
  after their requesting assistant message.

Both are pure functions: no DB access, no side effects, easy to test.

The ``reacting_to`` field on assistant messages is the key coordination
mechanism. Each assistant message records the seq of the latest user or
tool_result event that was in its context. ``should_call_model`` uses
this to define "new" as events after ``reacting_to``, not after the
assistant's own seq. This correctly handles the race where a tool result
arrives during inference — the model's response has a ``reacting_to``
that predates the tool result, so the result is "new" and triggers a
follow-up step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from aios.models.events import Event

# Chat-completions spec fields per role.  Only these are emitted in the
# context; provider-specific extensions (reasoning_content, etc.) stay
# in the event log but are excluded from the message list.
_ALLOWED_FIELDS: dict[str, frozenset[str]] = {
    "assistant": frozenset({"role", "content", "tool_calls"}),
    "tool": frozenset({"role", "tool_call_id", "content", "name"}),
    "user": frozenset({"role", "content", "name"}),
    "system": frozenset({"role", "content", "name"}),
}

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
    timestamp_ms = metadata.get("timestamp_ms")
    if isinstance(timestamp_ms, int):
        iso = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat(timespec="milliseconds")
        parts.append(f"timestamp_ms={timestamp_ms} ({iso})")
    header = "[" + " · ".join(parts) + "]"
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
) -> dict[str, Any]:
    """Render a user event into its chat-completions message form.

    Rendering is a deterministic function of the event's stamped
    ``orig_channel`` and ``focal_channel_at_arrival``.  ``build_messages``
    and the append-time token counter in ``queries.append_event`` share
    this helper so the context and the ``cumulative_tokens`` column
    stay in lock-step.

    Branches:

    * ``orig_channel`` is ``None`` → legacy / non-connector event;
      metadata stripped, no header, no notification.  Covers direct
      ``POST /sessions/{id}/messages`` traffic and pre-migration rows.
    * ``orig == focal_at_arrival`` (non-NULL) → full content with a
      metadata header inlined.
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
        return msg

    content = event_data.get("content", "")
    if not isinstance(content, str):
        content = ""
    marker = _format_notification_marker(orig_channel, metadata, content)
    return {"role": "user", "content": marker}


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


def _strip_to_spec(msg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *msg* with only chat-completions spec fields."""
    allowed = _ALLOWED_FIELDS.get(msg.get("role", ""), frozenset())
    out = {k: v for k, v in msg.items() if k in allowed}
    if out.get("tool_calls"):
        out["tool_calls"] = _sanitize_tool_calls(out["tool_calls"])
    return out


# ─── should_call_model ──────────────────────────────────────────────────────


def should_call_model(events: list[Event]) -> bool:
    """Decide whether this wake should produce an inference call.

    Returns ``True`` when:

    1. There are no assistant messages yet (first turn).
    2. A user message or completed tool batch exists that the model
       hasn't reacted to (determined via the ``reacting_to`` field).

    ``reacting_to`` is the seq of the latest user/tool event in the
    context the model was given. Events after ``reacting_to`` (excluding
    the assistant message itself) are "new." This handles the race where
    a tool result arrives during inference — the result has a seq after
    ``reacting_to`` and triggers a follow-up step.
    """
    if not events:
        return False

    # Find the most recent assistant message.
    last_asst: Event | None = None
    for e in reversed(events):
        if e.kind == "message" and e.data.get("role") == "assistant":
            last_asst = e
            break

    if last_asst is None:
        return True  # first turn — no assistant response yet

    # Use reacting_to if available; fall back to the assistant's own seq
    # (backward compat with events written before this field existed).
    reacting_to: int = last_asst.data.get("reacting_to", last_asst.seq)

    # "New" = events the model hasn't reacted to.
    new_events = [e for e in events if e.seq > reacting_to and e.seq != last_asst.seq]
    if not new_events:
        return False  # duplicate / stale wake

    # User injection → always proceed.
    if any(e.kind == "message" and e.data.get("role") == "user" for e in new_events):
        return True

    # Collect ALL real tool_result tool_call_ids from the full log.
    all_real_results: set[str] = set()
    for e in events:
        if e.kind == "message" and e.data.get("role") == "tool":
            tcid = e.data.get("tool_call_id")
            if tcid:
                all_real_results.add(tcid)

    # Check if any tool batch became fully resolved in the "new" window.
    new_tool_results = [
        e
        for e in new_events
        if e.kind == "message" and e.data.get("role") == "tool" and e.data.get("tool_call_id")
    ]
    for tr in new_tool_results:
        parent = _find_assistant_for_tool_call(events, tr.data["tool_call_id"])
        if parent is None:
            continue
        parent_call_ids = {tc["id"] for tc in (parent.data.get("tool_calls") or [])}
        if parent_call_ids <= all_real_results:
            return True  # this batch is complete

    return False


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
            msg = render_user_event(e.data, e.orig_channel, e.focal_channel_at_arrival)
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
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"[Tool result: {name} (call {inj_tcid}) completed]\n"
                            f"{inj_data.get('content', '')}"
                        ),
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

    return ContextResult(
        messages=[_strip_to_spec(m) for m in messages],
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
            tc_ids = {tc["id"] for tc in (msg.get("tool_calls") or [])}
            if not tc_ids:
                break  # assistant with no tool_calls — clean start
            # Check that all tool_calls have matching results in the rest.
            remaining_result_ids = {
                m.get("tool_call_id") for m in messages[start + 1 :] if m.get("role") == "tool"
            }
            if tc_ids <= remaining_result_ids:
                break  # complete group — clean start
            # Incomplete group — drop the assistant and its partial results.
            start += 1
            while start < len(messages) and messages[start].get("role") == "tool":
                start += 1
            continue

        # tool or anything else at the front — orphan, drop.
        start += 1

    return messages[start:]


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


def _find_assistant_for_tool_call(events: list[Event], tool_call_id: str) -> Event | None:
    """Find the assistant message that contains ``tool_call_id``."""
    for e in reversed(events):
        if e.kind != "message" or e.data.get("role") != "assistant":
            continue
        for tc in e.data.get("tool_calls") or []:
            if tc.get("id") == tool_call_id:
                return e
    return None
