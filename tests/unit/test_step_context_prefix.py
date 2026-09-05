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

from datetime import UTC, datetime
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

from aios.harness.concise import CONCISE_NAG_CONTENT
from aios.harness.step_context import StepContext, StepPrelude, compose_step_context
from aios.models.agents import AgentBinding, OutputStyle, StepSurface
from aios.models.events import Event, is_reminder_event
from tests.support import assert_message_prefix

_ACCOUNT = "acc_prefix"
_SESSION = "sess_prefix"


def _agent(*, output_style: OutputStyle = "concise") -> StepSurface:
    return StepSurface(
        model="gpt-test",
        system="you are a test agent",
        tools=[],
        skills=[],
        mcp_servers=[],
        http_servers=[],
        litellm_extra={},
        window_min=1,
        window_max=10,
        preempt_policy="wait",
        output_style=output_style,
        binding=AgentBinding(agent_id="agt_prefix", version=1),
    )


def _evt(seq: int, role: str, content: str, *, reacting_to: int | None = None) -> Event:
    data: dict[str, Any] = {"role": role, "content": content}
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
    )


def _prelude() -> StepPrelude:
    return StepPrelude(
        system_prompt="sys",
        tools=[],
        skill_versions=[],
        tail_block_upper_bound_local=0,
        obligations=[],
        obligations_block_upper_bound_local=0,
    )


class _Session:
    id = _SESSION
    focal_channel = None


class _Log:
    """The session log as the composer's writer sees it: ``append_event``
    mints the next seq and the row is in the log for the NEXT build."""

    def __init__(self, events: list[Event]) -> None:
        self.events = list(events)
        self.written: list[Event] = []

    async def append_event(
        self,
        pool: Any,
        session_id: str,
        kind: str,
        data: dict[str, Any],
        *,
        account_id: str,
        orig_channel: str | None = None,
    ) -> Event:
        assert session_id == _SESSION and account_id == _ACCOUNT
        assert kind == "message" and is_reminder_event(kind, data)
        seq = max((e.seq for e in self.events), default=0) + 1
        row = Event(
            id=f"evt_r{seq:04d}",
            session_id=session_id,
            seq=seq,
            kind="message",
            data=data,
            cumulative_tokens=None,
            created_at=datetime(2026, 9, 5, tzinfo=UTC),
        )
        self.events.append(row)
        self.written.append(row)
        return row

    def reply(self, content: str, *, reacting_to: int) -> None:
        seq = max(e.seq for e in self.events) + 1
        self.events.append(_evt(seq, "assistant", content, reacting_to=reacting_to))

    def inbound(self, content: str) -> None:
        seq = max(e.seq for e in self.events) + 1
        self.events.append(_evt(seq, "user", content))


async def _compose(log: _Log, agent: StepSurface) -> StepContext:
    slate = list(log.events)
    with (
        mock.patch(
            "aios.services.sessions.load_session_workspace_path",
            new=AsyncMock(return_value=None),
        ),
        mock.patch(
            "aios.services.accounts.resolve_effective_timezone",
            new=AsyncMock(return_value="UTC"),
        ),
        mock.patch("aios.services.sessions.append_event", new=log.append_event),
    ):
        step_ctx = await compose_step_context(
            pool=MagicMock(),
            session=_Session(),  # type: ignore[arg-type]
            account_id=_ACCOUNT,
            agent=agent,
            channels=[],
            prelude=_prelude(),
            events=slate,
            persist_image_rewrites=False,
            persist_reminders=True,
        )
    assert slate == log.events[: len(slate)], "the composer must not mutate its events"
    return step_ctx


def _through_last_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    return messages[: last + 1]


class TestComposePrefix:
    async def test_consecutive_builds_are_prefixes_and_the_row_is_written_once(self) -> None:
        log = _Log([_evt(1, "user", "please respond")])

        b1 = await _compose(log, _agent())
        assert b1.reminders_written == ("concise",)
        assert b1.reminders_skipped == 0
        assert len(log.written) == 1
        # The trailing user item carries the nag (merged into the inbound).
        assert b1.messages[-1]["role"] == "user"
        assert b1.messages[-1]["content"].endswith(CONCISE_NAG_CONTENT)

        log.reply("sure", reacting_to=1)
        b2 = await _compose(log, _agent())
        assert b2.reminders_written == ()
        assert b2.reminders_skipped == 1
        assert len(log.written) == 1, "the in-window row must gate a second write"
        assert_message_prefix(b1.messages, b2.messages)

        log.inbound("and then?")
        b3 = await _compose(log, _agent())
        assert b3.reminders_written == ()
        assert len(log.written) == 1
        assert_message_prefix(b2.messages, b3.messages)
        assert b3.messages[-1]["role"] == "user"

    async def test_reminder_then_inbound_rewrites_only_the_trailing_user_item(self) -> None:
        # Idle re-check: the tail is an assistant, so the nag stands alone as
        # the last user item of build 1.
        log = _Log([_evt(1, "user", "hello"), _evt(2, "assistant", "hi", reacting_to=1)])
        b1 = await _compose(log, _agent())
        assert b1.reminders_written == ("concise",)
        assert b1.messages[-1] == {"role": "user", "content": CONCISE_NAG_CONTENT}

        # A real inbound lands after the row: the merge folds it into the
        # same user item, rewriting the tail — the bounded, pinned loss.
        log.inbound("new question")
        b2 = await _compose(log, _agent())
        assert b2.reminders_written == ()
        assert_message_prefix(_through_last_assistant(b1.messages), b2.messages)
        assert_message_prefix(b1.messages[:-1], b2.messages)
        assert b2.messages[-1]["role"] == "user"
        assert b2.messages[-1]["content"].startswith(CONCISE_NAG_CONTENT + "\n\n")
        assert "new question" in b2.messages[-1]["content"]
        assert len(b2.messages) == len(b1.messages)

    async def test_preview_path_renders_the_same_rows_without_writing(self) -> None:
        log = _Log([_evt(1, "user", "hello"), _evt(2, "assistant", "hi", reacting_to=1)])
        slate = list(log.events)
        with (
            mock.patch(
                "aios.services.sessions.load_session_workspace_path",
                new=AsyncMock(return_value=None),
            ),
            mock.patch(
                "aios.services.accounts.resolve_effective_timezone",
                new=AsyncMock(return_value="UTC"),
            ),
            mock.patch("aios.services.sessions.append_event", new=log.append_event),
        ):
            preview = await compose_step_context(
                pool=MagicMock(),
                session=_Session(),  # type: ignore[arg-type]
                account_id=_ACCOUNT,
                agent=_agent(),
                channels=[],
                prelude=_prelude(),
                events=slate,
                persist_image_rewrites=False,
                persist_reminders=False,
            )
        assert log.written == []
        assert preview.reminders_written == ("concise",)
        sent = await _compose(log, _agent())
        assert preview.messages == sent.messages
