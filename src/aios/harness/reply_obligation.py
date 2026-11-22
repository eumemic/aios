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
* **Discharged by** a later assistant turn that called ``no_response`` (which
  declines on every channel), or a connector tool on that channel (a send, a
  reaction, …) whose result came back without an error. The channel is the one
  the turn was focused on, or the target of a successful ``switch_channel`` in
  the same turn. A discharge covers every inbound up to that turn's
  ``reacting_to``: one reply answers a burst of messages.
* **Judged once**, at the first *closing* turn that saw it — an assistant turn
  without tool calls, which ends a step, whose ``reacting_to`` covers the
  inbound (a turn already in inference when the inbound landed is not it). If
  the inbound is still undischarged there, the guard appends one nudge naming
  its channel. At the first closing turn that saw that nudge it lapses: the
  session idles and a ``reply_lapsed`` span records it. After that it is closed.
  A closing turn that drew no nudge closes the inbound too — which is what makes
  a session's history from before this protocol (when every step closed on a
  line of text) owe nothing.

Computed from the step's windowed slate: an inbound's judgment points all follow
it, and the window always keeps the tail. The net errs toward silence, never
toward a second reply:

* When a closing turn can't idle the session (a new stimulus arrived during
  inference) the guard writes nothing, so the inbound it would have judged
  closes without a nudge.
* An inbound that scrolls out of the window before its first closing turn
  (a step that worked through more than the window of tool output first) is
  never judged — the model can no longer see that message either, so a nudge
  would point it at something outside its own context.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Set
from dataclasses import dataclass
from typing import Any, Final

from aios.harness.channels import SWITCH_CHANNEL_METADATA_KEY
from aios.models.events import Event, is_reminder_event

NO_RESPONSE_TOOL_NAME: Final[str] = "no_response"

# ``metadata`` key on the nudge row: ``{"through_seq": int, "channels": [str, ...]}``.
# ``through_seq`` is the newest inbound the nudge covered.
REPLY_NUDGE_METADATA_KEY: Final[str] = "reply_nudge"

# Span recorded when an inbound is still unanswered at the first closing turn
# after its nudge.
REPLY_LAPSED_SPAN_EVENT: Final[str] = "reply_lapsed"

# Key on a persisted assistant message: the ids of its tool calls that went to a
# connector tool, as offered at the step that made them (a harness field like
# ``reacting_to``; the render keeps only spec fields, so the model never sees it).
# Classifying a past call against the tools offered NOW would un-credit a reply
# whenever the connector set changes between the send and the judgment.
CONNECTOR_CALLS_KEY: Final[str] = "connector_calls"


def tag_connector_calls(assistant_msg: dict[str, Any], connector_tool_names: Set[str]) -> None:
    """Record on ``assistant_msg`` which of its tool calls are connector calls."""
    ids = [
        tc["id"]
        for tc in assistant_msg.get("tool_calls") or []
        if isinstance(tc, dict)
        and isinstance(tc.get("id"), str)
        and (tc.get("function") or {}).get("name") in connector_tool_names
    ]
    if ids:
        assistant_msg[CONNECTOR_CALLS_KEY] = ids


@dataclass(frozen=True, slots=True)
class ReplyOwed:
    """Unanswered inbound at one stage of the obligation."""

    through_seq: int
    """The newest of them."""
    channels: tuple[str, ...]
    """Their channels, oldest inbound first."""

    def detail(self) -> dict[str, Any]:
        """The nudge marker / lapse span payload."""
        return {"through_seq": self.through_seq, "channels": list(self.channels)}


@dataclass(frozen=True, slots=True)
class ReplyDue:
    """What a step ending now owes: inbound reaching their first closing turn
    unanswered (``nudge``), and inbound unanswered at the first closing turn after
    their nudge (``lapse``)."""

    nudge: ReplyOwed | None
    lapse: ReplyOwed | None


def _owed(inbound: list[tuple[int, str]]) -> ReplyOwed | None:
    if not inbound:
        return None
    return ReplyOwed(
        through_seq=max(seq for seq, _ in inbound),
        channels=tuple(dict.fromkeys(channel for _, channel in inbound)),
    )


def _discharges(
    turns: list[Event],
    results: dict[str, dict[str, Any]],
    connector_tool_names: Set[str],
) -> tuple[int, dict[str, int]]:
    """``(declined_through, answered_through_by_channel)`` over the tool-calling
    turns: the watermarks up to which ``no_response`` declined everything, and a
    connector call replied on each channel. A call counts once its result is back
    without an error — a failed send delivered nothing. Whether a call went to a
    connector is read from the turn's ``CONNECTOR_CALLS_KEY`` tag; a turn without
    one (written before the tag existed) falls back to this step's tool set."""
    declined = 0
    answered: dict[str, int] = {}
    for turn in turns:
        raw_watermark = turn.data.get("reacting_to")
        watermark = raw_watermark if isinstance(raw_watermark, int) else turn.seq
        tagged = turn.data.get(CONNECTOR_CALLS_KEY)
        replied = False
        switched_to: list[str] = []
        for tc in turn.data.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            result = results.get(str(tc.get("id")))
            if result is None or result.get("is_error"):
                continue
            fn = tc.get("function")
            name = fn.get("name") if isinstance(fn, dict) else None
            connector_call = (
                tc.get("id") in tagged if isinstance(tagged, list) else name in connector_tool_names
            )
            if name == NO_RESPONSE_TOOL_NAME:
                declined = max(declined, watermark)
            elif connector_call:
                replied = True
            metadata = result.get("metadata")
            marker = (
                metadata.get(SWITCH_CHANNEL_METADATA_KEY) if isinstance(metadata, dict) else None
            )
            if isinstance(marker, dict) and marker.get("success") is True:
                target = marker.get("target")
                if isinstance(target, str):
                    switched_to.append(target)
        if replied:
            # Focal is read when the connector picks the call up, so a send beside a
            # switch_channel may land on either channel: credit both.
            for channel in (turn.channel, *switched_to):
                if channel is not None:
                    answered[channel] = max(answered.get(channel, 0), watermark)
    return declined, answered


def find_reply_due(
    events: Iterable[Event],
    *,
    reacting_to: int,
    connector_tool_names: Set[str],
) -> ReplyDue | None:
    """What the closing turn being appended owes, or ``None``.

    ``events`` is the step's windowed slate in seq order, ``reacting_to`` the
    watermark of the turn being appended (a turn without tool calls: the guard
    only acts on a turn that ends the step), ``connector_tool_names`` the
    connector tools offered this step.
    """
    results: dict[str, dict[str, Any]] = {}
    turns: list[Event] = []
    closing: list[int] = []  # closing turns' seqs, ascending
    seen_through: list[int] = []  # running max of their ``reacting_to``
    nudges: list[tuple[int, int]] = []
    inbound: list[tuple[int, str]] = []
    for e in events:
        if e.kind != "message":
            continue
        role = e.data.get("role")
        if role == "tool":
            tcid = e.data.get("tool_call_id")
            if isinstance(tcid, str):
                results[tcid] = e.data
        elif role == "assistant":
            if e.data.get("tool_calls"):
                turns.append(e)
            else:
                raw = e.data.get("reacting_to")
                seen = raw if isinstance(raw, int) else e.seq
                closing.append(e.seq)
                seen_through.append(max(seen, seen_through[-1]) if seen_through else seen)
        elif role == "user" and not is_reminder_event(e.kind, e.data):
            metadata = e.data.get("metadata")
            nudge = metadata.get(REPLY_NUDGE_METADATA_KEY) if isinstance(metadata, dict) else None
            if isinstance(nudge, dict):
                through = nudge.get("through_seq")
                if isinstance(through, int):
                    nudges.append((e.seq, through))
            elif e.orig_channel is not None and e.seq <= reacting_to:
                inbound.append((e.seq, e.orig_channel))

    def first_closing_seeing(seq: int) -> int | None:
        """Index of the first closing turn whose ``reacting_to`` covers ``seq``."""
        i = bisect_left(seen_through, seq)
        return i if i < len(closing) else None

    declined, answered = _discharges(turns, results, connector_tool_names)
    to_nudge: list[tuple[int, str]] = []
    to_lapse: list[tuple[int, str]] = []
    for seq, channel in inbound:
        if seq <= max(declined, answered.get(channel, 0)):
            continue
        i = first_closing_seeing(seq)
        if i is None:
            to_nudge.append((seq, channel))  # this turn is its judgment point
            continue
        # Its nudge, if it drew one, is a nudge row covering it between that turn and
        # the next closing turn: the guard writes it in the same transaction.
        judged_at = closing[i]
        next_closing = closing[i + 1] if i + 1 < len(closing) else None
        nudge_seq = next(
            (
                n
                for n, through in nudges
                if n > judged_at and through >= seq and (next_closing is None or n < next_closing)
            ),
            None,
        )
        if nudge_seq is None:
            continue  # judged without a nudge: closed
        if first_closing_seeing(nudge_seq) is None and reacting_to >= nudge_seq:
            to_lapse.append((seq, channel))  # this turn is the first to see its nudge

    due = ReplyDue(nudge=_owed(to_nudge), lapse=_owed(to_lapse))
    return due if due.nudge or due.lapse else None


def reply_nudge_content(owed: ReplyOwed) -> str:
    """The nudge row's text: which channel(s) are owed, and the two ways out."""
    where = ", ".join(f"channel_id={channel}" for channel in owed.channels)
    return (
        f"<system-reminder>The latest message on {where} has no reply. Your text "
        "is not sent. Reply with the connector's send tool (switch_channel first if "
        "it isn't your focal channel), or call no_response(reason) if no reply is "
        "needed.</system-reminder>"
    )
