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
(``persist_reminders=True`` on the worker's step path) or renders the same
rows as unpersisted stand-ins (the read-only ``/context`` preview), so the two
paths produce byte-identical message lists.

Rows are **non-stimulus** by construction (see
:func:`aios.models.events.is_reminder_event`): they never bump the session's
``last_stimulus_seq`` / ``last_user_seq`` / ``updated_at``, never wake a
session, and are excluded from every log-derived stimulus reader.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from aios.harness.channels import render_channels_reminder
from aios.harness.concise import CONCISE_NAG_CONTENT, CONCISE_NAG_CONTENT_CHANNELS
from aios.harness.context import TRAILING_STIMULUS_NOTICE, TailOrigin
from aios.harness.obligations import OBLIGATIONS_EMPTY_CONTENT, render_obligations_reminder
from aios.models.events import REMINDER_METADATA_KEY, ReminderSection, reminder_section

if TYPE_CHECKING:
    from aios.models.agents import OutputStyle
    from aios.models.events import Event
    from aios.models.sessions import Obligation

# Bumped when the row's ``data`` shape changes; readers key on
# ``metadata.aios_reminder`` presence, not the version, so old rows keep
# rendering as plain user messages.
REMINDER_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True)
class PlannedReminder:
    section: ReminderSection
    content: str


@dataclass(frozen=True, slots=True)
class ReminderPlan:
    """What this step writes, in canonical order (channels → obligations →
    concise → trailing_stimulus), and how many applicable sections it left
    alone because their in-window row already says the same thing."""

    writes: tuple[PlannedReminder, ...]
    skipped: int


def reminder_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def reminder_event_data(section: ReminderSection, content: str) -> dict[str, Any]:
    """The ``data`` of a reminder row: a user message tagged for the readers."""
    return {
        "role": "user",
        "content": content,
        "metadata": {
            REMINDER_METADATA_KEY: {
                "section": section,
                "digest": reminder_digest(content),
                "v": REMINDER_SCHEMA_VERSION,
            }
        },
    }


def reminder_message(content: str) -> dict[str, Any]:
    """The rendered message for a reminder row — the same bare shape
    :func:`~aios.harness.context.render_user_event` gives a persisted row, so
    a stand-in (preview) and a replay (next step) are byte-identical."""
    return {"role": "user", "content": content}


def latest_reminder_digests(events: list[Event]) -> dict[ReminderSection, str]:
    """Digest of the greatest-seq in-window reminder row per section.

    The change-gate's baseline. Recomputed from the row's content rather than
    read from its metadata so a row can never claim a digest it doesn't have.
    """
    latest: dict[ReminderSection, str] = {}
    for e in events:  # seq-ascending: the last row per section wins
        section = reminder_section(e.kind, e.data)
        if section is None:
            continue
        content = e.data.get("content")
        latest[section] = reminder_digest(content if isinstance(content, str) else "")
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


_EMPTY_OBLIGATIONS_DIGEST: Final[str] = reminder_digest(OBLIGATIONS_EMPTY_CONTENT)


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
    (:func:`latest_reminder_digests`): no row, or a row saying something
    else, means write. A row that scrolled out of the window is simply absent
    from the slate, so eviction re-emits without any extra bookkeeping.

    * **channels** — bound channels only; suppressed while the tail owes a
      response (:func:`tail_owes_response`), written on any content change
      (unread counts, previews, focal switch).
    * **obligations** — presence-gated on EVERY write
      (:func:`present_request_ids`): a listing goes in only when some open
      obligation's original ask is gone from the window and the listing
      differs from the in-window one. When the open set empties while a
      listing is still in the window, a one-line "(none)" row supersedes it,
      once; that one-liner is not a listing baseline, so the next obligation
      is presence-gated afresh.
    * **concise** — once per window per variant (the channel-attached variant
      carries the delivery clause, #2262).
    * **trailing_stimulus** — whenever the build ends on an assistant turn
      while holding an unreacted stimulus; it cannot re-fire on the next build
      because the row itself becomes the tail.
    """
    latest = latest_reminder_digests(events)
    writes: list[PlannedReminder] = []
    skipped = 0

    def consider(section: ReminderSection, content: str) -> None:
        nonlocal skipped
        if latest.get(section) == reminder_digest(content):
            skipped += 1
        else:
            writes.append(PlannedReminder(section, content))

    owes = tail_owes_response(tail_origin, needs_trailing_notice=needs_trailing_notice)

    channels_content = render_channels_reminder(channels, events, focal_channel)
    if channels_content is not None:
        if owes:
            skipped += 1
        else:
            consider("channels", channels_content)

    latest_obligations = latest.get("obligations")
    if obligations:
        present = present_request_ids(events)
        if any(o.request_id not in present for o in obligations):
            consider("obligations", render_obligations_reminder(obligations, session_id=session_id))
        else:
            skipped += 1
    elif latest_obligations is not None and latest_obligations != _EMPTY_OBLIGATIONS_DIGEST:
        writes.append(PlannedReminder("obligations", OBLIGATIONS_EMPTY_CONTENT))

    if output_style == "concise":
        consider("concise", CONCISE_NAG_CONTENT_CHANNELS if channels else CONCISE_NAG_CONTENT)

    if needs_trailing_notice:
        writes.append(PlannedReminder("trailing_stimulus", TRAILING_STIMULUS_NOTICE))

    return ReminderPlan(writes=tuple(writes), skipped=skipped)
