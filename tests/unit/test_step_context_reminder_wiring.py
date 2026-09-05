"""The composer→planner wiring of the tail class, through the real composer.

``tail_owes_response`` and ``plan_reminders`` are pinned in isolation; this
file pins that ``compose_step_context`` feeds them the build's ACTUAL
``tail_origin`` / ``needs_trailing_notice`` (a wiring that passes a constant
survives every isolated test), and the retry shape the durable rows create:
a step that wrote rows behind an unanswered inbound and then failed must not
write the channels listing on the retry.
"""

from __future__ import annotations

from typing import Any

from aios.harness.concise import CONCISE_NAG_CONTENT_CHANNELS
from aios.harness.context import TRAILING_STIMULUS_NOTICE
from aios.harness.reminders import reminder_event_data
from aios.models.events import Event
from tests.unit.step_context_support import (
    FIXED_CREATED_AT,
    SESSION,
    compose_with_stubs,
    make_step_surface,
    message_event,
)

_CHANNELS = ["signal/+1/chat-a", "signal/+1/chat-b"]


def _row(seq: int, section: Any, content: str) -> Event:
    return Event(
        id=f"evt_{seq:04d}",
        session_id=SESSION,
        seq=seq,
        kind="message",
        data=reminder_event_data(section, content),
        cumulative_tokens=None,
        created_at=FIXED_CREATED_AT,
    )


async def _written(events: list[Event], **kwargs: Any) -> tuple[str, ...]:
    ctx = await compose_with_stubs(make_step_surface(**kwargs), events, channels=_CHANNELS)
    return ctx.reminders_written


class TestTailClassReachesThePlanner:
    async def test_focal_inbound_tail_holds_the_channels_listing(self) -> None:
        events = [
            message_event(1, "user", "hello"),
            message_event(2, "assistant", "hi", reacting_to=1),
            message_event(3, "user", "and now?"),
        ]
        assert await _written(events) == ()

    async def test_assistant_tail_writes_the_channels_listing(self) -> None:
        events = [
            message_event(1, "user", "hello"),
            message_event(2, "assistant", "hi", reacting_to=1),
        ]
        assert await _written(events) == ("channels",)

    async def test_pruned_orphan_build_writes_the_notice_and_holds_the_listing(self) -> None:
        events = [
            message_event(1, "user", "hello"),
            message_event(2, "assistant", "hi", reacting_to=1),
            Event(
                id="evt_0003",
                session_id=SESSION,
                seq=3,
                kind="message",
                data={"role": "tool", "tool_call_id": "ghost", "content": "late result"},
                cumulative_tokens=None,
                created_at=FIXED_CREATED_AT,
            ),
        ]
        ctx = await compose_with_stubs(make_step_surface(), events, channels=_CHANNELS)
        assert ctx.reminders_written == ("trailing_stimulus",)
        assert ctx.messages[-1] == {"role": "user", "content": TRAILING_STIMULUS_NOTICE}


class TestRetryAfterRowsWereWritten:
    async def test_rows_behind_an_unanswered_inbound_do_not_unlock_the_listing(self) -> None:
        # Step N wrote the concise row behind the inbound and then failed
        # (provider error / overflow / preemption). The retry's slate ends on
        # that row; the inbound is still the tail for the gate.
        events = [
            message_event(1, "user", "please handle this"),
            _row(2, "concise", CONCISE_NAG_CONTENT_CHANNELS),
        ]
        ctx = await compose_with_stubs(
            make_step_surface(output_style="concise"), events, channels=_CHANNELS
        )
        assert ctx.reminders_written == ()
        assert "━━━ Channels ━━━" not in str(ctx.messages[-1]["content"])

    async def test_notice_row_at_the_tail_is_not_written_twice(self) -> None:
        events = [
            message_event(1, "user", "hello"),
            message_event(2, "assistant", "hi", reacting_to=1),
            Event(
                id="evt_0003",
                session_id=SESSION,
                seq=3,
                kind="message",
                data={"role": "tool", "tool_call_id": "ghost", "content": "late result"},
                cumulative_tokens=None,
                created_at=FIXED_CREATED_AT,
            ),
            _row(4, "trailing_stimulus", TRAILING_STIMULUS_NOTICE),
        ]
        ctx = await compose_with_stubs(make_step_surface(), events, channels=_CHANNELS)
        assert ctx.reminders_written == ()
        assert ctx.messages[-1] == {"role": "user", "content": TRAILING_STIMULUS_NOTICE}
