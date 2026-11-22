"""Reply-or-decline: every inbound channel message gets a reply or a ``no_response`` (#2092).

A channel-bound session's assistant text is never delivered — replies go through
the connector's tools. The way the agent turns down a reply is to say so, by
calling ``no_response``. aios has no turn boundary to hang that rule on, so it is an
obligation the quiescence guard checks
(:func:`aios.services.sessions.append_assistant_and_guard_quiescence`):

* **Opened by** an inbound human message on a channel — a ``user`` row with an
  ``orig_channel``, not a reminder — that the step saw (``seq <= reacting_to``).
  Trigger and cron wakes, tool results, and channel-less operator messages open
  nothing.
* **Discharged by** a later assistant turn that called a connector tool while
  focused on that channel (a send, a reaction, …), or that called ``no_response``,
  which declines on every channel. A discharge covers every inbound up to that
  turn's ``reacting_to``: one reply answers a burst of messages. Under the old
  protocol a step declined by closing on stamped text (``LEGACY_MONOLOGUE_STAMP``);
  such a turn still reads as the decline it was, so history from before the change
  owes nothing.
* **Enforced** when a tool-call-free turn would idle the session: the guard
  appends one nudge naming the open channel(s). If the newest open inbound has
  already had its nudge, the session idles and a ``reply_lapsed`` span records it.
  A newer inbound gets its own nudge, so a nudge can't recur without new input.

Computed from the step's windowed slate: an open inbound is newer than every
discharge after it, and the window always keeps the tail.
"""

from __future__ import annotations

from collections.abc import Iterable, Set
from dataclasses import dataclass
from typing import Any, Final

from aios.harness.channels import LEGACY_MONOLOGUE_STAMP
from aios.models.events import Event, is_reminder_event

NO_RESPONSE_TOOL_NAME: Final[str] = "no_response"

# ``metadata`` key on the nudge row: ``{"through_seq": int, "channels": [str, ...]}``.
# ``through_seq`` is the newest inbound the nudge covered; a later quiescence that
# finds nothing newer open lapses instead of nudging again.
REPLY_NUDGE_METADATA_KEY: Final[str] = "reply_nudge"

# Span recorded when an inbound is still unanswered after its nudge.
REPLY_LAPSED_SPAN_EVENT: Final[str] = "reply_lapsed"


@dataclass(frozen=True, slots=True)
class ReplyDue:
    """The open part of the obligation at the end of a step."""

    through_seq: int
    """The newest unanswered inbound the step saw."""
    channels: tuple[str, ...]
    """Channels with an unanswered inbound, oldest first."""
    nudged: bool
    """A nudge already covered ``through_seq``: lapse, don't nudge again."""


def _called_names(data: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for tc in data.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function")
            name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(name, str):
                names.add(name)
    return names


def _is_legacy_decline(data: dict[str, Any]) -> bool:
    """A tool-call-free turn from before #2092 that closed its step on stamped
    text: the old protocol's way of saying "no reply"."""
    content = data.get("content")
    if isinstance(content, list):
        content = " ".join(
            b.get("text") or "" for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return isinstance(content, str) and LEGACY_MONOLOGUE_STAMP in content


def find_reply_due(
    events: Iterable[Event],
    *,
    reacting_to: int,
    connector_tool_names: Set[str],
) -> ReplyDue | None:
    """The unanswered inbound a step ending now would leave behind, or ``None``.

    ``events`` is the step's windowed slate in seq order and ``reacting_to`` the
    watermark of the assistant turn being appended (which carries no tool calls:
    the guard only acts on a turn that would idle the session).
    """
    declined = 0
    answered: dict[str, int] = {}
    nudged_through = 0
    inbound: list[tuple[int, str]] = []
    for e in events:
        if e.kind != "message":
            continue
        role = e.data.get("role")
        if role == "user":
            if is_reminder_event(e.kind, e.data):
                continue
            metadata = e.data.get("metadata")
            nudge = metadata.get(REPLY_NUDGE_METADATA_KEY) if isinstance(metadata, dict) else None
            if isinstance(nudge, dict):
                through = nudge.get("through_seq")
                if isinstance(through, int):
                    nudged_through = max(nudged_through, through)
                continue
            if e.orig_channel is not None and e.seq <= reacting_to:
                inbound.append((e.seq, e.orig_channel))
        elif role == "assistant":
            names = _called_names(e.data)
            raw_watermark = e.data.get("reacting_to")
            watermark = raw_watermark if isinstance(raw_watermark, int) else e.seq
            if not names:
                if _is_legacy_decline(e.data):
                    declined = max(declined, watermark)
                continue
            if NO_RESPONSE_TOOL_NAME in names:
                declined = max(declined, watermark)
            if e.channel is not None and not names.isdisjoint(connector_tool_names):
                answered[e.channel] = max(answered.get(e.channel, 0), watermark)
    unanswered = [
        (seq, channel) for seq, channel in inbound if seq > max(declined, answered.get(channel, 0))
    ]
    if not unanswered:
        return None
    through_seq = max(seq for seq, _ in unanswered)
    return ReplyDue(
        through_seq=through_seq,
        channels=tuple(dict.fromkeys(channel for _, channel in unanswered)),
        nudged=nudged_through >= through_seq,
    )


def reply_nudge_content(due: ReplyDue) -> str:
    """The nudge row's text: which channel(s) are owed, and the two ways out."""
    where = ", ".join(f"channel_id={channel}" for channel in due.channels)
    return (
        f"<system-reminder>The latest message on {where} has no reply. Your text "
        "is not sent. Reply with the connector's send tool (switch_channel first if "
        "it isn't your focal channel), or call no_response(reason) if no reply is "
        "needed.</system-reminder>"
    )
