"""``compose_step_context`` keeps request N a byte-prefix of request N+1.

The composer-level twin of ``test_context.py::TestMonotonicity``: the real
composer with ``persist_reminders=True``, a fake ``append_event`` that mints
the row into the next build's log exactly as the DB would, and the prefix
invariant asserted on consecutive builds THROUGH THE LAST ASSISTANT item.

The one permitted divergence is pinned too: when a reminder row is followed
by a real inbound, ``merge_adjacent_user_messages`` rewrites the trailing
user item (``R`` → ``R\\n\\nU``). The prefix through the previous assistant
still matches, so the loss is bounded to that one item.
"""

from __future__ import annotations

from typing import Any

from aios.harness.concise import CONCISE_NAG_CONTENT
from aios.harness.step_context import StepContext
from aios.models.events import Event, is_reminder_event
from tests.support import assert_message_prefix
from tests.unit.step_context_support import (
    ACCOUNT,
    FIXED_CREATED_AT,
    SESSION,
    compose_with_stubs,
    make_step_surface,
    message_event,
)


class _Log:
    """The session log as the composer's writer sees it: ``append_event``
    mints the next seq and the row is in the log for the NEXT build."""

    def __init__(self, events: list[Event]) -> None:
        self.events = list(events)
        self.written: list[Event] = []

    async def append_event(
        self,
        conn: Any,
        *,
        session_id: str,
        kind: str,
        data: dict[str, Any],
        account_id: str,
        **_: Any,
    ) -> Event:
        assert session_id == SESSION and account_id == ACCOUNT
        assert kind == "message" and is_reminder_event(kind, data)
        seq = max((e.seq for e in self.events), default=0) + 1
        row = Event(
            id=f"evt_r{seq:04d}",
            session_id=session_id,
            seq=seq,
            kind="message",
            data=data,
            cumulative_tokens=None,
            created_at=FIXED_CREATED_AT,
        )
        self.events.append(row)
        self.written.append(row)
        return row

    def reply(self, content: str, *, reacting_to: int) -> None:
        seq = max(e.seq for e in self.events) + 1
        self.events.append(message_event(seq, "assistant", content, reacting_to=reacting_to))

    def inbound(self, content: str) -> None:
        seq = max(e.seq for e in self.events) + 1
        self.events.append(message_event(seq, "user", content))


async def _compose(log: _Log, *, persist: bool = True) -> StepContext:
    slate = list(log.events)
    step_ctx = await compose_with_stubs(
        make_step_surface(output_style="concise"),
        slate,
        persist_reminders=persist,
        append_event=log.append_event,
    )
    assert slate == log.events[: len(slate)], "the composer must not mutate its events"
    return step_ctx


def _through_last_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    return messages[: last + 1]


class TestComposePrefix:
    async def test_consecutive_builds_are_prefixes_and_the_row_is_written_once(self) -> None:
        log = _Log([message_event(1, "user", "please respond")])

        b1 = await _compose(log)
        assert b1.reminders_written == ("concise",)
        assert b1.reminders_skipped == 0
        assert len(log.written) == 1
        # The trailing user item carries the nag (merged into the inbound).
        assert b1.messages[-1]["role"] == "user"
        assert b1.messages[-1]["content"].endswith(CONCISE_NAG_CONTENT)

        log.reply("sure", reacting_to=1)
        b2 = await _compose(log)
        assert b2.reminders_written == ()
        assert b2.reminders_skipped == 1
        assert len(log.written) == 1, "the in-window row must gate a second write"
        assert_message_prefix(b1.messages, b2.messages)

        log.inbound("and then?")
        b3 = await _compose(log)
        assert b3.reminders_written == ()
        assert len(log.written) == 1
        assert_message_prefix(b2.messages, b3.messages)
        assert b3.messages[-1]["role"] == "user"

    async def test_reminder_then_inbound_rewrites_only_the_trailing_user_item(self) -> None:
        # Idle re-check: the tail is an assistant, so the nag stands alone as
        # the last user item of build 1.
        log = _Log(
            [message_event(1, "user", "hello"), message_event(2, "assistant", "hi", reacting_to=1)]
        )
        b1 = await _compose(log)
        assert b1.reminders_written == ("concise",)
        assert b1.messages[-1] == {"role": "user", "content": CONCISE_NAG_CONTENT}

        # A real inbound lands after the row: the merge folds it into the
        # same user item, rewriting the tail — the bounded, pinned loss.
        log.inbound("new question")
        b2 = await _compose(log)
        assert b2.reminders_written == ()
        assert_message_prefix(_through_last_assistant(b1.messages), b2.messages)
        assert_message_prefix(b1.messages[:-1], b2.messages)
        assert b2.messages[-1]["role"] == "user"
        assert b2.messages[-1]["content"].startswith(CONCISE_NAG_CONTENT + "\n\n")
        assert "new question" in b2.messages[-1]["content"]
        assert len(b2.messages) == len(b1.messages)

    async def test_preview_path_renders_the_same_rows_without_writing(self) -> None:
        log = _Log(
            [message_event(1, "user", "hello"), message_event(2, "assistant", "hi", reacting_to=1)]
        )
        preview = await _compose(log, persist=False)
        assert log.written == []
        assert preview.reminders_written == ("concise",)
        sent = await _compose(log)
        assert len(log.written) == 1
        assert preview.messages == sent.messages
