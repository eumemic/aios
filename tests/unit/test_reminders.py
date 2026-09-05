"""The reminder planner (``aios.harness.reminders``).

Each section's write is gated on the greatest-seq in-window row of that
section: no row, or a row saying something else, means write; a matching row
means skip. Rows that scrolled out of the window are simply absent from the
slate, so eviction re-emits with no extra bookkeeping. The obligations section
is additionally presence-gated (a listing only when some open ask has left
the window) and the channels section waits while the tail owes a response.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aios.harness.channels import render_channels_reminder
from aios.harness.concise import (
    CONCISE_NAG_CONTENT,
    CONCISE_NAG_CONTENT_CHANNELS,
    CONCISE_NAG_OFF_CONTENT,
)
from aios.harness.context import TRAILING_STIMULUS_NOTICE, TailOrigin
from aios.harness.obligations import OBLIGATIONS_EMPTY_CONTENT, render_obligations_reminder
from aios.harness.reminders import (
    ReminderPlan,
    latest_reminder_contents,
    plan_reminders,
    present_request_ids,
    reminder_event_data,
)
from aios.models.agents import OutputStyle
from aios.models.events import REMINDER_METADATA_KEY, Event, ReminderSection
from aios.models.sessions import Obligation

_SESSION = "sess_plan"
_CHANNELS = ["signal/+1/chat-a", "signal/+1/chat-b"]
_FOCAL = "signal/+1/chat-a"


def _evt(
    seq: int,
    role: str,
    content: str = "hi",
    *,
    metadata: dict[str, Any] | None = None,
    orig_channel: str | None = None,
    reacting_to: int | None = None,
) -> Event:
    data: dict[str, Any] = {"role": role, "content": content}
    if metadata is not None:
        data["metadata"] = metadata
    if reacting_to is not None:
        data["reacting_to"] = reacting_to
    return Event(
        id=f"evt_{seq:04d}",
        session_id=_SESSION,
        seq=seq,
        kind="message",
        data=data,
        cumulative_tokens=None,
        created_at=datetime(2026, 9, 5, tzinfo=UTC),
        orig_channel=orig_channel,
    )


def _row(seq: int, section: ReminderSection, content: str) -> Event:
    """A reminder row exactly as the composer writes it."""
    return Event(
        id=f"evt_{seq:04d}",
        session_id=_SESSION,
        seq=seq,
        kind="message",
        data=reminder_event_data(section, content),
        cumulative_tokens=None,
        created_at=datetime(2026, 9, 5, tzinfo=UTC),
    )


def _ob(rid: str, *, summary: str = "do the thing") -> Obligation:
    return Obligation(
        request_id=rid,
        caller_kind="run",
        caller_id=None,
        opened_at=datetime(2026, 9, 5, 12, 0, tzinfo=UTC),
        summary=summary,
        output_schema=None,
    )


def _plan(
    events: list[Event],
    *,
    channels: list[str] | None = None,
    focal_channel: str | None = None,
    obligations: list[Obligation] | None = None,
    output_style: OutputStyle = "default",
    tail_origin: TailOrigin = "assistant",
    needs_trailing_notice: bool = False,
) -> ReminderPlan:
    return plan_reminders(
        events=events,
        channels=channels or [],
        focal_channel=focal_channel,
        obligations=obligations or [],
        session_id=_SESSION,
        output_style=output_style,
        tail_origin=tail_origin,
        needs_trailing_notice=needs_trailing_notice,
    )


def _sections(plan: ReminderPlan) -> list[str]:
    return [w.section for w in plan.writes]


_BASE = [_evt(1, "user", "hello"), _evt(2, "assistant", "hi", reacting_to=1)]


class TestRowShape:
    def test_event_data_is_a_tagged_user_message(self) -> None:
        data = reminder_event_data("concise", CONCISE_NAG_CONTENT)
        assert data == {
            "role": "user",
            "content": CONCISE_NAG_CONTENT,
            "metadata": {REMINDER_METADATA_KEY: {"section": "concise"}},
        }

    def test_latest_content_is_the_greatest_seq_row_per_section(self) -> None:
        events = [
            *_BASE,
            _row(3, "channels", "old listing"),
            _row(4, "concise", CONCISE_NAG_CONTENT),
            _row(5, "channels", "new listing"),
        ]
        assert latest_reminder_contents(events) == {
            "channels": "new listing",
            "concise": CONCISE_NAG_CONTENT,
        }

    def test_non_reminder_rows_are_not_a_baseline(self) -> None:
        # A plain user row whose text happens to equal a reminder is not one.
        events = [*_BASE, _evt(3, "user", CONCISE_NAG_CONTENT)]
        assert latest_reminder_contents(events) == {}


class TestNothingApplicable:
    def test_plain_session_writes_nothing(self) -> None:
        plan = _plan(_BASE)
        assert plan == ReminderPlan(writes=(), skipped=0)


class TestChannels:
    def _listing(self, events: list[Event], focal: str | None = _FOCAL) -> str:
        content = render_channels_reminder(_CHANNELS, events, focal)
        assert content is not None
        return content

    def test_first_build_writes_the_listing(self) -> None:
        plan = _plan(_BASE, channels=_CHANNELS, focal_channel=_FOCAL)
        assert _sections(plan) == ["channels"]
        assert plan.writes[0].content == self._listing(_BASE)
        assert plan.skipped == 0

    def test_in_window_row_with_same_content_skips(self) -> None:
        events = [*_BASE, _row(3, "channels", self._listing(_BASE))]
        plan = _plan(events, channels=_CHANNELS, focal_channel=_FOCAL)
        assert plan == ReminderPlan(writes=(), skipped=1)

    def test_unread_change_rewrites(self) -> None:
        events = [*_BASE, _row(3, "channels", self._listing(_BASE))]
        # A non-focal inbound changes the unread count on chat-b.
        events.append(_evt(4, "user", "psst", orig_channel="signal/+1/chat-b"))
        plan = _plan(events, channels=_CHANNELS, focal_channel=_FOCAL, tail_origin="notification")
        assert _sections(plan) == ["channels"]
        assert "1 unread" in plan.writes[0].content
        assert plan.writes[0].content != self._listing(_BASE)

    def test_focal_switch_rewrites(self) -> None:
        events = [*_BASE, _row(3, "channels", self._listing(_BASE))]
        plan = _plan(events, channels=_CHANNELS, focal_channel="signal/+1/chat-b")
        assert _sections(plan) == ["channels"]
        assert "channel_id=signal/+1/chat-b (focal)" in plan.writes[0].content

    def test_owed_tail_holds_the_listing_back(self) -> None:
        for origin in ("user", "tool", "notice"):
            plan = _plan(_BASE, channels=_CHANNELS, focal_channel=_FOCAL, tail_origin=origin)
            assert plan == ReminderPlan(writes=(), skipped=1), origin
        plan = _plan(
            _BASE,
            channels=_CHANNELS,
            focal_channel=_FOCAL,
            tail_origin="assistant",
            needs_trailing_notice=True,
        )
        assert _sections(plan) == ["trailing_stimulus"]
        assert plan.skipped == 1

    def test_not_owed_tails_allow_the_listing(self) -> None:
        for origin in ("assistant", "notification", "none"):
            plan = _plan(_BASE, channels=_CHANNELS, focal_channel=_FOCAL, tail_origin=origin)
            assert _sections(plan) == ["channels"], origin


class TestObligations:
    def test_never_owed_writes_nothing(self) -> None:
        assert _plan(_BASE) == ReminderPlan(writes=(), skipped=0)

    def test_ask_present_in_window_skips(self) -> None:
        events = [
            _evt(1, "user", "please do X", metadata={"request": {"request_id": "req_1"}}),
            _evt(2, "assistant", "on it", reacting_to=1),
        ]
        assert present_request_ids(events) == frozenset({"req_1"})
        plan = _plan(events, obligations=[_ob("req_1")])
        assert plan == ReminderPlan(writes=(), skipped=1)

    def test_ask_windowed_out_writes_the_listing(self) -> None:
        plan = _plan(_BASE, obligations=[_ob("req_1")])
        assert _sections(plan) == ["obligations"]
        assert plan.writes[0].content == render_obligations_reminder(
            [_ob("req_1")], session_id=_SESSION
        )

    def test_in_window_listing_with_same_set_skips(self) -> None:
        listing = render_obligations_reminder([_ob("req_1")], session_id=_SESSION)
        events = [*_BASE, _row(3, "obligations", listing)]
        assert _plan(events, obligations=[_ob("req_1")]) == ReminderPlan(writes=(), skipped=1)

    def test_set_change_rewrites(self) -> None:
        listing = render_obligations_reminder([_ob("req_1")], session_id=_SESSION)
        events = [*_BASE, _row(3, "obligations", listing)]
        plan = _plan(events, obligations=[_ob("req_1"), _ob("req_2")])
        assert _sections(plan) == ["obligations"]
        assert "req_2" in plan.writes[0].content

    def test_set_shrinks_with_remaining_asks_present_rewrites_the_listing(self) -> None:
        # An in-window listing is kept truthful regardless of presence: once
        # req_1 is answered, a row still naming it would be the model's last
        # word on what it owes for the rest of the window.
        listing = render_obligations_reminder([_ob("req_1"), _ob("req_2")], session_id=_SESSION)
        events = [
            *_BASE,
            _row(3, "obligations", listing),
            _evt(4, "user", "do req_2", metadata={"request": {"request_id": "req_2"}}),
        ]
        plan = _plan(events, obligations=[_ob("req_2")], tail_origin="user")
        assert _sections(plan) == ["obligations"]
        assert "req_1" not in plan.writes[0].content
        assert "req_2" in plan.writes[0].content
        # And once rewritten, nothing more.
        after = [*events, _row(5, "obligations", plan.writes[0].content)]
        assert _plan(after, obligations=[_ob("req_2")], tail_origin="user") == ReminderPlan(
            writes=(), skipped=1
        )

    def test_partially_present_set_still_lists_everything(self) -> None:
        # One ask in the window, one not: the listing is owed (for the absent
        # one) and lists the whole open set — the model's one place to look.
        events = [
            _evt(1, "user", "please do X", metadata={"request": {"request_id": "req_1"}}),
            _evt(2, "assistant", "on it", reacting_to=1),
        ]
        plan = _plan(events, obligations=[_ob("req_1"), _ob("req_2")])
        assert _sections(plan) == ["obligations"]
        assert "req_1" in plan.writes[0].content
        assert "req_2" in plan.writes[0].content

    def test_emptied_with_listing_in_window_writes_the_one_liner_once(self) -> None:
        listing = render_obligations_reminder([_ob("req_1")], session_id=_SESSION)
        events = [*_BASE, _row(3, "obligations", listing)]
        plan = _plan(events, obligations=[])
        assert _sections(plan) == ["obligations"]
        assert plan.writes[0].content == OBLIGATIONS_EMPTY_CONTENT
        # Once written, the one-liner is the in-window baseline: nothing more.
        after = [*events, _row(4, "obligations", OBLIGATIONS_EMPTY_CONTENT)]
        assert _plan(after, obligations=[]) == ReminderPlan(writes=(), skipped=1)

    def test_emptied_with_no_listing_in_window_writes_nothing(self) -> None:
        # The listing already scrolled out (or never existed): a "(none)" row
        # would supersede nothing the model can see.
        assert _plan(_BASE, obligations=[]) == ReminderPlan(writes=(), skipped=0)

    def test_owed_then_emptied_then_new_ask_present_writes_nothing(self) -> None:
        listing = render_obligations_reminder([_ob("req_1")], session_id=_SESSION)
        events = [
            *_BASE,
            _row(3, "obligations", listing),
            _row(4, "obligations", OBLIGATIONS_EMPTY_CONTENT),
            _evt(5, "user", "now do Y", metadata={"request": {"request_id": "req_2"}}),
        ]
        plan = _plan(events, obligations=[_ob("req_2")], tail_origin="user")
        assert plan == ReminderPlan(writes=(), skipped=1)

    def test_one_liner_is_not_a_listing_baseline(self) -> None:
        # After the one-liner, an obligation whose ask has left the window is
        # listed again (the one-liner's digest never equals a listing's).
        events = [*_BASE, _row(3, "obligations", OBLIGATIONS_EMPTY_CONTENT)]
        plan = _plan(events, obligations=[_ob("req_2")])
        assert _sections(plan) == ["obligations"]
        assert "req_2" in plan.writes[0].content


class TestConcise:
    def test_non_concise_writes_nothing(self) -> None:
        assert _plan(_BASE, output_style="default") == ReminderPlan(writes=(), skipped=0)

    def test_once_per_window(self) -> None:
        plan = _plan(_BASE, output_style="concise")
        assert _sections(plan) == ["concise"]
        assert plan.writes[0].content == CONCISE_NAG_CONTENT
        events = [*_BASE, _row(3, "concise", CONCISE_NAG_CONTENT)]
        assert _plan(events, output_style="concise") == ReminderPlan(writes=(), skipped=1)

    def test_written_even_while_the_tail_owes_a_response(self) -> None:
        # Unlike the channels listing, the nag is not a status block: the
        # counter-pressure has to be in the window whatever the tail is.
        plan = _plan(_BASE, output_style="concise", tail_origin="user")
        assert _sections(plan) == ["concise"]

    def test_variant_flip_rewrites(self) -> None:
        events = [*_BASE, _row(3, "concise", CONCISE_NAG_CONTENT)]
        plan = _plan(events, output_style="concise", channels=_CHANNELS, focal_channel=_FOCAL)
        concise = [w for w in plan.writes if w.section == "concise"]
        assert len(concise) == 1
        assert concise[0].content == CONCISE_NAG_CONTENT_CHANNELS

    def test_style_turned_off_supersedes_the_row_once(self) -> None:
        # The system-prompt rules block is gone the moment the style is off;
        # the stale nag must not remain the transcript's only steering.
        events = [*_BASE, _row(3, "concise", CONCISE_NAG_CONTENT)]
        plan = _plan(events, output_style="default")
        assert _sections(plan) == ["concise"]
        assert plan.writes[0].content == CONCISE_NAG_OFF_CONTENT
        after = [*events, _row(4, "concise", CONCISE_NAG_OFF_CONTENT)]
        assert _plan(after, output_style="default") == ReminderPlan(writes=(), skipped=1)
        # Turned back on: the nag is written again (differs from the OFF row).
        assert _sections(_plan(after, output_style="concise")) == ["concise"]


class TestTrailingStimulus:
    def test_written_whenever_the_build_reports_it(self) -> None:
        plan = _plan(_BASE, needs_trailing_notice=True)
        assert _sections(plan) == ["trailing_stimulus"]
        assert plan.writes[0].content == TRAILING_STIMULUS_NOTICE

    def test_not_digest_gated(self) -> None:
        # A prior notice row in the window never suppresses a new one: the
        # condition is about THIS build's tail, and a build whose tail is the
        # prior row cannot report it (its tail_origin is "notice").
        events = [*_BASE, _row(3, "trailing_stimulus", TRAILING_STIMULUS_NOTICE)]
        plan = _plan(events, needs_trailing_notice=True)
        assert _sections(plan) == ["trailing_stimulus"]


class TestCanonicalOrder:
    def test_idle_recheck_order(self) -> None:
        plan = _plan(
            _BASE,
            channels=_CHANNELS,
            focal_channel=_FOCAL,
            obligations=[_ob("req_1")],
            output_style="concise",
        )
        assert _sections(plan) == ["channels", "obligations", "concise"]

    def test_notice_step_order(self) -> None:
        plan = _plan(
            _BASE,
            channels=_CHANNELS,
            focal_channel=_FOCAL,
            obligations=[_ob("req_1")],
            output_style="concise",
            needs_trailing_notice=True,
        )
        assert _sections(plan) == ["obligations", "concise", "trailing_stimulus"]
        assert plan.skipped == 1
