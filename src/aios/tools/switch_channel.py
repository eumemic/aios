"""The switch_channel tool — shift the agent's focal attention.

Slice 5 of the focal-channel redesign (issue #29). A many-to-one
connector-aware session has at most one *focal* channel at any given
time.  Inbound events on the focal channel render in full; inbound on
non-focal channels render as truncated notification markers.  The
agent shifts its attention by calling ``switch_channel`` — the only
way ``sessions.focal_channel`` changes post-session-creation.

The tool's tool_result event carries a ``metadata.switch_channel``
marker (``{target, success}``) that the unread-derivation helpers in
:mod:`aios.harness.channels` key off.  Successful switches anchor
``last_seen_in_<target>``; failed switches do not.

Re-orient block (non-NULL target):

    Switched to <target>. Recent messages:

    <last max(unread_in_target, FLOOR_N) events on target, rendered
     with full metadata header, newest last>

Phone down (``target=None``):

    Focal cleared.

Unknown-target (not a bound channel on this session):

    Cannot switch to <target>: not a bound channel on this session.

Errors do not mutate ``focal_channel``.
"""

from __future__ import annotations

from typing import Any

from aios.db import queries
from aios.harness import runtime
from aios.harness.channels import (
    SWITCH_CHANNEL_METADATA_KEY,
    derive_unread_counts,
)
from aios.harness.context import render_user_event
from aios.tools.registry import ToolResult, registry

# Floor on the number of events included in the re-orient block so a
# switch into a quiet channel still gives the agent a screen's worth of
# context to reorient on — matches how phone messaging apps show the
# last N messages when you tap into a conversation, regardless of
# whether any are "new."
RE_ORIENT_FLOOR_N = 10


SWITCH_CHANNEL_DESCRIPTION = (
    "Shift your attention to a different bound channel, or put your phone "
    "down entirely. When focused on a channel, inbound messages on that "
    "channel render in full; inbound on other channels show up as short "
    "notification markers in your context. Call switch_channel(target=<address>) "
    "to focus on a bound channel, or switch_channel(target=null) to clear "
    "focal (no channel focused; all inbound renders as notifications). "
    "The tool result includes a re-orient block quoting recent messages on "
    "the target channel so you can pick up the conversation in context."
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
    """Mutate ``sessions.focal_channel`` and build the re-orient block.

    Returns a :class:`ToolResult` carrying a ``switch_channel`` marker
    under ``metadata`` so the unread-derivation helpers can recognise
    successful switches as anchors.
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

    if target is None:
        async with pool.acquire() as conn:
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

    # Validate target against the session's active bindings.
    async with pool.acquire() as conn:
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

    # Atomically flip the focal pointer.  The UPDATE serialises against
    # append_event's row lock, so concurrent inbound stamping sees
    # either the pre-switch or post-switch focal, never a torn read.
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET focal_channel = $1 WHERE id = $2",
            target,
            session_id,
        )

    content = await _build_reorient_block(pool, session_id, target)
    return ToolResult(
        content=content,
        metadata={
            SWITCH_CHANNEL_METADATA_KEY: {"target": target, "success": True},
        },
    )


async def _build_reorient_block(pool: Any, session_id: str, target: str) -> str:
    """Pull recent events from ``target`` and render chronologically.

    Size is ``max(unread_in_target, RE_ORIENT_FLOOR_N)``: the floor
    gives a quiet-channel context refresher, the unread-count term
    ensures no unread message gets silently skipped on switch-in.
    """
    async with pool.acquire() as conn:
        all_events = await queries.read_message_events(conn, session_id)

    target_events = [e for e in all_events if e.orig_channel == target]
    if not target_events:
        return f"Switched to {target}. (no prior messages on this channel)"

    unread = derive_unread_counts(all_events, [target]).get(target, 0)
    n = max(unread, RE_ORIENT_FLOOR_N)
    recent = target_events[-n:] if n < len(target_events) else target_events

    lines = [f"Switched to {target}. Recent messages:", ""]
    for e in recent:
        # Render each quoted event as full content + metadata header
        # (as if orig==focal), regardless of its stamped focal_at_arrival.
        # The re-orient block is the "you've opened the app" view of
        # the channel's own chronology.
        rendered = render_user_event(e.data, e.orig_channel, e.orig_channel)
        content = rendered.get("content", "")
        if isinstance(content, str) and content:
            lines.append(content)
            lines.append("")
    return "\n".join(lines).rstrip("\n")


def _register() -> None:
    registry.register(
        name="switch_channel",
        description=SWITCH_CHANNEL_DESCRIPTION,
        parameters_schema=SWITCH_CHANNEL_PARAMETERS_SCHEMA,
        handler=switch_channel_handler,
    )


_register()
