"""Durable, change-gated reminders — the harness's per-step reminders as rows.

The harness has four standing reminders it wants in front of the model: the
bound-channel listing with unread counts, the open-obligations listing, the
concise-style nag, and the trailing-stimulus notice. They used to be rendered
fresh and appended after ``build_messages`` on every step, tagged ephemeral so
the Anthropic cache breakpoint skipped them. OpenAI's Responses backend has no
such breakpoint: it caches through the END of the prompt, so a per-step tail
put the cached prefix's last matching byte at the tools block and every step
re-sent the whole conversation uncached (measured: 5% hit on a 490k session).

Here they are ordinary ``kind="message"`` / ``role="user"`` rows in the
session log, tagged ``data.metadata.aios_reminder``, written by the composer
ONLY when the reminder's content changed or its last row has scrolled out of
the window. The prompt is then a pure replay of the log — request N is a
byte-prefix of request N+1 through the last assistant item on every
provider — and the model still sees each reminder at least once per window.

This module is the pure planner: given the windowed slate and this step's
context it decides which rows to write. The composer executes the plan
(``persist_reminders=True`` on the worker's step path, via
:func:`aios.harness.context_persist.persist_reminder_rows`) or renders the
same rows as unpersisted stand-ins (the read-only ``/context`` preview), so
the two paths produce byte-identical message lists.

Rows are **non-stimulus** by construction (see
:func:`aios.models.events.is_reminder_event`): they never bump the session's
``last_stimulus_seq`` / ``last_user_seq`` / ``updated_at``, never wake a
session, and are excluded from every log-derived stimulus reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aios.harness.channels import max_channels_reminder_local, render_channels_reminder
from aios.harness.concise import (
    CONCISE_NAG_CONTENT,
    CONCISE_NAG_CONTENT_CHANNELS,
    CONCISE_NAG_UPPER_BOUND_LOCAL,
)
from aios.harness.context import (
    TRAILING_NOTICE_UPPER_BOUND_LOCAL,
    TRAILING_STIMULUS_NOTICE,
    TailOrigin,
)
from aios.harness.obligations import (
    OBLIGATIONS_EMPTY_CONTENT,
    OBLIGATIONS_EMPTY_UPPER_BOUND_LOCAL,
    max_obligations_reminder_local,
    render_obligations_reminder,
)
from aios.models.events import REMINDER_METADATA_KEY, ReminderSection, reminder_section

if TYPE_CHECKING:
    from aios.models.agents import OutputStyle
    from aios.models.events import Event
    from aios.models.sessions import Obligation


@dataclass(frozen=True, slots=True)
class PlannedReminder:
    section: ReminderSection
    content: str


@dataclass(frozen=True, slots=True)
class ReminderPlan:
    """What this step writes, in canonical order (channels → obligations →
    concise → trailing_stimulus), and how many applicable sections it did
    NOT write — because the in-window row already says the same thing, the
    tail owes a response (channels), or every open ask is still in the window
    (obligations)."""

    writes: tuple[PlannedReminder, ...]
    skipped: int


def reminder_event_data(section: ReminderSection, content: str) -> dict[str, Any]:
    """The ``data`` of a reminder row: a user message tagged for the readers."""
    return {
        "role": "user",
        "content": content,
        "metadata": {REMINDER_METADATA_KEY: {"section": section}},
    }


def latest_reminder_contents(events: list[Event]) -> dict[str, str]:
    """Content of the greatest-seq in-window reminder row per section — the
    change-gate's baseline (reminder contents are bounded by their reserves,
    so comparing them directly is as cheap as anything derived from them)."""
    latest: dict[str, str] = {}
    for e in events:  # seq-ascending: the last row per section wins
        section = reminder_section(e.kind, e.data)
        if section is None:
            continue
        content = e.data.get("content")
        latest[section] = content if isinstance(content, str) else ""
    return latest


def present_request_ids(events: list[Event]) -> frozenset[str]:
    """``request_id``s whose ORIGINAL request user message survived windowing.

    The obligations listing is presence-gated on this (#2221): an obligation
    whose ask is intact earlier in the same prompt needs no reminder — a
    second copy at the tail is a worse stimulus than the real one (a neutral
    pointer where the task sits above) and a write for nothing. Only an
    obligation whose ask has scrolled out of the window earns a listing.

    Reads the same ``metadata.request.request_id`` stamp
    :func:`~aios.harness.context.render_user_event` surfaces as the reply
    marker, off the POST-windowing slate. Defensive about shape throughout: a
    malformed ``metadata`` blob yields no id rather than raising inside
    context assembly.
    """
    present: set[str] = set()
    for event in events:
        if event.kind != "message":
            continue
        metadata = event.data.get("metadata")
        if not isinstance(metadata, dict):
            continue
        request = metadata.get("request")
        if not isinstance(request, dict):
            continue
        request_id = request.get("request_id")
        if isinstance(request_id, str) and request_id:
            present.add(request_id)
    return frozenset(present)


def tail_owes_response(tail_origin: TailOrigin, *, needs_trailing_notice: bool) -> bool:
    """True when the build ends on a *direct* stimulus the agent must answer.

    Gates the channels listing. A trailing focal **user** inbound or **tool**
    result is a direct stimulus: writing a "0 unread" status listing after it
    makes the listing the literal final message, and literal-minded models
    (claude-fable-5) anchor on it and emit an empty turn instead of answering.
    The trailing-stimulus notice takes the same arm — the missed events it
    points at ARE the stimulus — so the notice step keeps suppressing the
    listing exactly as the ephemeral tail did.

    Not a direct stimulus, so the listing may be written: an **assistant**
    turn (an idle/sweep re-check, where the channel status IS the useful
    signal), a non-focal **notification** marker (the listing is its
    navigation companion — how to ``switch_channel`` to it), a prior
    **reminder** row, or an empty/system-only build.
    """
    return needs_trailing_notice or tail_origin in ("user", "tool")


def max_reminders_local(channels: list[str], obligations: list[Obligation]) -> int:
    """Worst-case local-token cost of the rows ONE step may write, reserved
    from the window budget at windowing time (``prelude_overhead_local``).

    Rows written on earlier steps are ordinary log rows already priced into
    ``cumulative_tokens``; this covers only this step's possible writes: the
    channels listing at its fattest, the obligations listing for the fetched
    open set, the obligations-emptied one-liner (presence of a listing in the
    window is unknowable before the slate exists, so its reserve is
    unconditional), the concise nag, and the trailing-stimulus notice. Every
    reserve is unconditional — any may not be written, but the budget must
    hold when they are.
    """
    return (
        max_channels_reminder_local(channels)
        + max_obligations_reminder_local(obligations)
        + OBLIGATIONS_EMPTY_UPPER_BOUND_LOCAL
        + CONCISE_NAG_UPPER_BOUND_LOCAL
        + TRAILING_NOTICE_UPPER_BOUND_LOCAL
    )


def plan_reminders(
    *,
    events: list[Event],
    channels: list[str],
    focal_channel: str | None,
    obligations: list[Obligation],
    session_id: str,
    output_style: OutputStyle,
    tail_origin: TailOrigin,
    needs_trailing_notice: bool,
) -> ReminderPlan:
    """Decide this step's reminder rows from the windowed slate.

    Every section is gated on the greatest-seq in-window row of that section
    (:func:`latest_reminder_contents`): no row, or a row saying something
    else, means write. A row that scrolled out of the window is simply absent
    from the slate, so eviction re-emits without any extra bookkeeping.

    * **channels** — bound channels only; held back while the tail owes a
      response (:func:`tail_owes_response`), written on any content change
      (unread counts, previews, focal switch).
    * **obligations** — presence-gated on EVERY write
      (:func:`present_request_ids`): a listing goes in only when some open
      obligation's original ask is gone from the window and the listing
      differs from the in-window one. When the open set empties while an
      obligations row is still in the window, the one-line "(none)" row
      supersedes it, once; that one-liner is not a listing, so the next
      obligation is presence-gated afresh.
    * **concise** — once per window per variant (the channel-attached variant
      carries the delivery clause, #2262).
    * **trailing_stimulus** — whenever the build ends on an assistant turn
      while holding an unreacted stimulus; it cannot re-fire on the next build
      because the row itself becomes the tail.
    """
    latest = latest_reminder_contents(events)
    writes: list[PlannedReminder] = []
    skipped = 0

    def consider(section: ReminderSection, content: str) -> None:
        nonlocal skipped
        if latest.get(section) == content:
            skipped += 1
        else:
            writes.append(PlannedReminder(section, content))

    if channels:
        if tail_owes_response(tail_origin, needs_trailing_notice=needs_trailing_notice):
            skipped += 1
        else:
            listing = render_channels_reminder(channels, events, focal_channel)
            assert listing is not None  # channels is non-empty
            consider("channels", listing)

    if obligations:
        present = present_request_ids(events)
        if any(o.request_id not in present for o in obligations):
            consider("obligations", render_obligations_reminder(obligations, session_id=session_id))
        else:
            skipped += 1
    elif "obligations" in latest:
        consider("obligations", OBLIGATIONS_EMPTY_CONTENT)

    if output_style == "concise":
        consider("concise", CONCISE_NAG_CONTENT_CHANNELS if channels else CONCISE_NAG_CONTENT)

    if needs_trailing_notice:
        writes.append(PlannedReminder("trailing_stimulus", TRAILING_STIMULUS_NOTICE))

    return ReminderPlan(writes=tuple(writes), skipped=skipped)
