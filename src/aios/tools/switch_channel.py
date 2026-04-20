"""The switch_channel tool — shift the agent's focal attention.

A many-to-one connector-aware session has at most one *focal* channel
at any given time.  Inbound events on the focal channel render in
full; inbound on non-focal channels render as truncated notification
markers.  The agent shifts its attention by calling ``switch_channel``
— the only way ``sessions.focal_channel`` changes after the session
exists.

The tool's tool_result event carries a ``metadata.switch_channel``
marker (``{target, success}``) that the unread-derivation helpers in
:mod:`aios.harness.channels` key off.  Successful real switches anchor
``last_seen_in_<target>``; failed switches and no-op calls (target
already equals current focal) do not — they're idempotent, nothing to
re-anchor.

Errors do not mutate ``focal_channel``.
"""

from __future__ import annotations

from typing import Any

from aios.db import queries
from aios.harness import runtime
from aios.harness.channels import (
    MONOLOGUE_PREFIX,
    SWITCH_CHANNEL_METADATA_KEY,
    derive_unread_counts,
)
from aios.harness.context import render_user_event
from aios.models.events import Event
from aios.tools.registry import ToolResult, registry

RE_ORIENT_FLOOR_N = 10
SWITCH_CHANNEL_TOOL_NAME = "switch_channel"


SWITCH_CHANNEL_DESCRIPTION = (
    "Shift your attention to a different bound channel, or put your phone "
    "down entirely. When focused on a channel, inbound messages on that "
    "channel render in full; inbound on other channels show up as short "
    "notification markers in your context. Call switch_channel(target=<address>) "
    "to focus on a bound channel, or switch_channel(target=null) to clear "
    "focal (no channel focused; all inbound renders as notifications). "
    "On a real switch the tool result includes a recap block quoting "
    "recent messages on the target channel (both peer and your own) so "
    "you can pick up where the conversation left off. A call whose "
    "target already equals your current focal is a no-op — no recap, "
    "no re-emit."
)

SWITCH_CHANNEL_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target": {
            "type": ["string", "null"],
            "description": (
                "The channel address to focus on (must be a bound channel "
                "on this session), or null to clear focal."
            ),
        },
    },
    "required": ["target"],
    "additionalProperties": False,
}


async def switch_channel_handler(session_id: str, arguments: dict[str, Any]) -> ToolResult:
    """Mutate ``sessions.focal_channel`` and build the recap block.

    Returns a :class:`ToolResult` carrying a ``switch_channel`` marker
    under ``metadata`` on real switches so the unread-derivation helpers
    can recognise them as anchors.  No-op calls (target already equals
    current focal) return a terse ack with empty metadata — they didn't
    actually change the agent's attention and shouldn't anchor anything.
    """
    target = arguments.get("target")
    if target is not None and not isinstance(target, str):
        return ToolResult(
            content=("Invalid target: must be a channel address string or null."),
            metadata={
                SWITCH_CHANNEL_METADATA_KEY: {"target": None, "success": False},
            },
            is_error=True,
        )

    pool = runtime.require_pool()

    # Hold a single connection across read-current-focal, validate,
    # update, and read-events-for-recap.  A no-op switch short-circuits
    # before any mutation; a real switch commits the UPDATE + reads the
    # event log on the same connection to render the recap.
    #
    # The no-op short-circuit (target already equals current focal) is
    # what issue #52 needs — re-emitting a recap on redundant switches
    # was misleading weak models into treating the quoted past as new
    # stimulus.
    async with pool.acquire() as conn:
        current_focal: str | None = await conn.fetchval(
            "SELECT focal_channel FROM sessions WHERE id = $1",
            session_id,
        )

        if target == current_focal:
            return ToolResult(
                content=f"Focal channel is already {target}.",
                metadata={},
            )

        if target is None:
            await conn.execute(
                "UPDATE sessions SET focal_channel = NULL WHERE id = $1",
                session_id,
            )
            return ToolResult(
                content="Focal cleared.",
                metadata={
                    SWITCH_CHANNEL_METADATA_KEY: {"target": None, "success": True},
                },
            )

        bindings = await queries.list_session_bindings(conn, session_id)
        valid_targets = {b.address for b in bindings if b.archived_at is None}
        if target not in valid_targets:
            return ToolResult(
                content=(
                    f"Cannot switch to {target!r}: not a bound channel on this session. "
                    f"Bound channels: {sorted(valid_targets) if valid_targets else '(none)'}."
                ),
                metadata={
                    SWITCH_CHANNEL_METADATA_KEY: {"target": target, "success": False},
                },
                is_error=True,
            )

        await conn.execute(
            "UPDATE sessions SET focal_channel = $1 WHERE id = $2",
            target,
            session_id,
        )
        all_events = await queries.read_message_events(conn, session_id)

    content = render_reorient_block(all_events, target)
    return ToolResult(
        content=content,
        metadata={
            SWITCH_CHANNEL_METADATA_KEY: {"target": target, "success": True},
        },
    )


def render_reorient_block(all_events: list[Event], target: str) -> str:
    """Render the recap block for ``target`` from the event log.

    Pure function over the session's message events — no database
    access, easy to unit-test.  The handler reads events on its own
    connection and passes them in.

    An event "belongs to" the target channel when its derived
    ``event.channel`` equals ``target`` — for user events that's
    ``orig_channel``, for assistant events that's
    ``focal_channel_at_arrival``, and for tool events it's the parent
    assistant's ``focal_channel_at_arrival`` (stamped at append time).
    A single ``e.channel == target`` predicate picks up both sides of
    the conversation and sidesteps the cross-channel leakage the naive
    "include all assistant events" fix produced.

    Tool results whose parent assistant's matching tool call is itself
    ``switch_channel`` are excluded to prevent the recursive embedding
    that made prior recaps quote their own predecessors.

    Size is ``max(unread_in_target, RE_ORIENT_FLOOR_N)``: the floor
    gives a quiet-channel context refresher, the unread-count term
    ensures no unread message gets silently skipped on switch-in.

    Output shape::

        ━━━ Recap: recent messages on <target> ━━━
        > <rendered event 1, line 1>
        > <rendered event 1, line 2>
        >
        > <rendered event 2, line 1>
        ━━━ End recap ━━━

    Each line is block-quoted with ``> `` so the historical content is
    visually distinct from fresh inbound, even on models that read
    through the role=tool framing.  Leading ``>`` inside any rendered
    line is escaped to ``\\>`` so peer messages that start with a
    quote character don't collide with the block-quote syntax.
    """
    switch_result_tcids = _switch_channel_tool_result_tcids(all_events)
    target_events = [
        e
        for e in all_events
        if e.channel == target
        and not (e.data.get("role") == "tool" and e.data.get("tool_call_id") in switch_result_tcids)
    ]
    if not target_events:
        return f"Switched to {target}. (no prior messages on this channel)"

    unread = derive_unread_counts(all_events, [target]).get(target, 0)
    n = max(unread, RE_ORIENT_FLOOR_N)
    recent = target_events[-n:] if n < len(target_events) else target_events

    rendered_blocks: list[str] = []
    for e in recent:
        block = _render_recap_event(e)
        if block:
            rendered_blocks.append(block)

    if not rendered_blocks:
        return f"Switched to {target}. (no prior messages on this channel)"

    quoted_body = _blockquote("\n\n".join(rendered_blocks))
    return f"━━━ Recap: recent messages on {target} ━━━\n{quoted_body}\n━━━ End recap ━━━"


def _switch_channel_tool_result_tcids(all_events: list[Event]) -> set[str]:
    """Collect tool_call_ids whose parent assistant requested ``switch_channel``.

    A single O(N) walk builds the set once; the recap filter then does
    O(1) membership checks per event — replacing the prior O(K*N)
    per-tool-result reverse scan into ``_find_assistant_for_tool_call``.

    Recap rendering excludes tool_result events whose id is in the
    returned set to prevent recursive embedding (each prior recap's
    tool_result content is itself a recap).
    """
    tcids: set[str] = set()
    for e in all_events:
        if e.kind != "message" or e.data.get("role") != "assistant":
            continue
        for tc in e.data.get("tool_calls") or []:
            if (tc.get("function") or {}).get("name") == SWITCH_CHANNEL_TOOL_NAME:
                tcid = tc.get("id")
                if isinstance(tcid, str) and tcid:
                    tcids.add(tcid)
    return tcids


def _render_recap_event(event: Event) -> str:
    """Render a single event into its body-of-the-recap text form.

    User events go through :func:`render_user_event` with
    ``focal_at_arrival=orig_channel`` to force the full-content branch
    (we want headers and full bodies inside the recap, never truncated
    notification markers).

    Assistant events render their text content with the
    ``INTERNAL_MONOLOGUE:`` prefix stripped — the prefix is a teaching
    signal for the author on replay, not useful framing when recapping
    "what was said on this channel."

    Tool events render as the tool output body, tagged with the tool
    call id so the agent can tie it back to its requesting assistant
    message.
    """
    role = event.data.get("role") if event.kind == "message" else None

    if role == "user":
        rendered = render_user_event(event.data, event.orig_channel, event.orig_channel)
        content = rendered.get("content")
        return content if isinstance(content, str) else ""

    if role == "assistant":
        text = _assistant_text(event.data)
        text = _strip_monologue_prefix(text)
        return f"[assistant] {text}".rstrip() if text else ""

    if role == "tool":
        content = event.data.get("content")
        if not isinstance(content, str):
            content = ""
        tcid = event.data.get("tool_call_id") or "?"
        return f"[tool result {tcid}] {content}".rstrip() if content else ""

    return ""


def _assistant_text(data: dict[str, Any]) -> str:
    """Extract the human-readable text from an assistant message.

    Assistant content may be a plain string or a list of typed blocks
    (``{"type": "text", "text": "..."}``).  Concatenate text blocks;
    ignore non-text blocks (tool_calls live on a sibling field, and
    reasoning blocks aren't useful in a recap).
    """
    content = data.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _strip_monologue_prefix(text: str) -> str:
    """Drop the ``INTERNAL_MONOLOGUE:`` prefix from assistant recap text."""
    return text[len(MONOLOGUE_PREFIX) :] if text.startswith(MONOLOGUE_PREFIX) else text


def _blockquote(text: str) -> str:
    """Prefix each line with ``> ``; escape leading ``>`` to ``\\>``.

    The escape prevents accidental nested-quote semantics when recapped
    content (peer messages, code, email replies) starts with a literal
    ``>``.  Blank lines render as a bare ``>`` so the block stays
    visually continuous.
    """
    out: list[str] = []
    for line in text.split("\n"):
        if line.startswith(">"):
            line = "\\" + line
        out.append(f"> {line}" if line else ">")
    return "\n".join(out)


def _register() -> None:
    registry.register(
        name=SWITCH_CHANNEL_TOOL_NAME,
        description=SWITCH_CHANNEL_DESCRIPTION,
        parameters_schema=SWITCH_CHANNEL_PARAMETERS_SCHEMA,
        handler=switch_channel_handler,
    )


_register()
