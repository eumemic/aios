"""Reply-or-decline on a channel session, through the real step loop (#2092).

The channel paradigm no longer labels the agent's text: bare text is simply not
sent. An inbound channel message is owed a reply (a connector tool) or a
deliberate ``no_response``. Pinned end to end:

* bare text is persisted as written, with no prefix;
* bare text in answer to an inbound draws exactly one nudge, then the session
  idles with a ``reply_lapsed`` span (no unbounded re-prompting);
* ``no_response`` ends the step: its quiet result wakes nothing and the session
  idles after a single model call;
* a connector send discharges the obligation, so the closing text draws no nudge;
* a refusal records the provider's ``stop_details`` on its span and stop message.
"""

from __future__ import annotations

from typing import Any

import pytest

from aios.db import queries
from aios.harness import runtime
from aios.harness.loop import REFUSAL_FINISH_REASON
from aios.harness.reply_obligation import (
    NO_RESPONSE_TOOL_NAME,
    REPLY_LAPSED_SPAN_EVENT,
    REPLY_NUDGE_METADATA_KEY,
)
from aios.models.events import QUIET_RESULT_METADATA_KEY
from aios.services import sessions as sessions_service
from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, tool_call

pytestmark = pytest.mark.docker

_ACCOUNT = "acc_test_stub"
_CHANNEL = "signal/+1/chat-a"


async def _inbound(harness: Harness, session_id: str, text: str = "are you around?") -> None:
    await sessions_service.append_user_message(
        harness._pool, session_id, text, metadata={"channel": _CHANNEL}, account_id=_ACCOUNT
    )


async def _nudges_and_lapses(harness: Harness, session_id: str) -> tuple[list[Any], list[Any]]:
    events = await harness.all_events(session_id)
    nudges = [
        e
        for e in events
        if e.kind == "message"
        and isinstance(e.data.get("metadata"), dict)
        and REPLY_NUDGE_METADATA_KEY in e.data["metadata"]
    ]
    lapses = [
        e for e in events if e.kind == "span" and e.data.get("event") == REPLY_LAPSED_SPAN_EVENT
    ]
    return nudges, lapses


@needs_docker
class TestReplyOrDecline:
    async def test_bare_text_is_unlabeled_and_draws_exactly_one_nudge(
        self, harness: Harness
    ) -> None:
        harness.script_model([assistant("Thinking it over."), assistant("Still thinking.")])
        session = await harness.start("hi")
        await _inbound(harness, session.id)
        await harness.run_until_idle(session.id)

        texts = [
            e.data["content"]
            for e in await harness.events(session.id)
            if e.data.get("role") == "assistant"
        ]
        assert texts == ["Thinking it over.", "Still thinking."]

        nudges, lapses = await _nudges_and_lapses(harness, session.id)
        assert len(nudges) == 1
        assert "no_response(reason)" in nudges[0].data["content"]
        assert f"channel_id={_CHANNEL}" in nudges[0].data["content"]
        assert len(lapses) == 1
        assert lapses[0].data["channels"] == [_CHANNEL]
        # Two scripted responses, two model calls: the second bare text ended it.
        assert len(harness.model_calls) == 2
        assert (await harness.session(session.id)).status == "idle"

    async def test_no_response_ends_the_step(self, harness: Harness) -> None:
        harness.script_model(
            [assistant(tool_calls=[tool_call(NO_RESPONSE_TOOL_NAME, {"reason": "not for me"})])]
        )
        session = await harness.start("hi")
        await _inbound(harness, session.id)
        await harness.run_until_idle(session.id)

        offered = {t["function"]["name"] for t in harness.model_calls[0]["tools"]}
        assert NO_RESPONSE_TOOL_NAME in offered
        assert len(harness.model_calls) == 1
        result = next(e for e in await harness.events(session.id) if e.data.get("role") == "tool")
        assert QUIET_RESULT_METADATA_KEY in result.data["metadata"]
        assert result.data.get("is_error") is not True
        nudges, lapses = await _nudges_and_lapses(harness, session.id)
        assert nudges == [] and lapses == []
        assert (await harness.session(session.id)).status == "idle"

    async def test_a_connector_send_discharges_the_inbound(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _SendToolProvider:
            async def list_tools_for_session(self, pool: Any, session_id: str) -> list[Any]:
                return [
                    {
                        "type": "custom",
                        "name": "chat_send",
                        "description": "Send a message to the focal chat.",
                        "input_schema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    }
                ]

            async def list_capabilities_for_session(
                self, pool: Any, session_id: str
            ) -> dict[str, Any]:
                return {}

        monkeypatch.setattr(runtime, "tool_provider", _SendToolProvider())
        send = tool_call("chat_send", {"text": "Yes, here."}, call_id="call_send")
        harness.script_model([assistant(tool_calls=[send]), assistant("Sent.")])
        session = await harness.start("hi")
        # Connector tools act on the focal channel: focus the chat first.
        async with harness._pool.acquire() as conn:
            await queries.set_session_focal_channel(conn, session.id, _CHANNEL, account_id=_ACCOUNT)
        await _inbound(harness, session.id)
        await harness.run_until_idle(session.id)  # parks on the unresolved custom call

        async with harness._pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id=_ACCOUNT,
                session_id=session.id,
                tool_call_id="call_send",
                content='{"sent": true}',
            )
        await harness.run_until_idle(session.id)

        assert len(harness.model_calls) == 2
        nudges, lapses = await _nudges_and_lapses(harness, session.id)
        assert nudges == [] and lapses == []
        assert (await harness.session(session.id)).status == "idle"

    async def test_trigger_style_wake_opens_nothing(self, harness: Harness) -> None:
        """A channel-less message (an operator post, a trigger wake) answered with
        bare text owes nothing, even on a session that has channels."""
        harness.script_model(
            [
                assistant(tool_calls=[tool_call(NO_RESPONSE_TOOL_NAME, {"reason": "handled"})]),
                assistant("Ran the nightly check."),
            ]
        )
        session = await harness.start("hi")
        await _inbound(harness, session.id)
        await harness.run_until_idle(session.id)
        await sessions_service.append_user_message(
            harness._pool, session.id, "nightly check", account_id=_ACCOUNT
        )
        await harness.run_until_idle(session.id)

        assert len(harness.model_calls) == 2
        nudges, lapses = await _nudges_and_lapses(harness, session.id)
        assert nudges == [] and lapses == []

    async def test_refusal_records_stop_details(self, harness: Harness) -> None:
        stop_details = {
            "type": "refusal",
            "category": "reasoning_extraction",
            "explanation": "Declined.",
        }
        refusal = assistant("", finish_reason=REFUSAL_FINISH_REASON)
        refusal["provider_specific_fields"] = {"stop_details": stop_details}
        harness.script_model([refusal])
        session = await harness.start("hi")
        await harness.run_step(session.id)

        spans = [
            e
            for e in await harness.all_events(session.id)
            if e.kind == "span" and e.data.get("event") == "model_refusal"
        ]
        assert len(spans) == 1
        assert spans[0].data["stop_details"] == stop_details
        s = await harness.session(session.id)
        assert s.stop_reason is not None
        assert "reasoning_extraction" in s.stop_reason["message"]
