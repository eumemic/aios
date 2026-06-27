"""Unit tests for the context builder (build_messages).

Uses lightweight FakeEvent objects to avoid touching the DB.
"""

from __future__ import annotations

import json
import logging
from datetime import MINYEAR, UTC, datetime
from typing import Any

import pytest

from aios.harness.channels import build_channels_tail_block
from aios.harness.context import (
    EPHEMERAL_TAIL_KEY,
    _approx_count,
    _concat_user_messages,
    _is_degenerate_empty_assistant,
    _quarantine_placeholder,
    build_messages,
    merge_adjacent_user_messages,
    render_user_event,
    stub_missing_reasoning_content,
)
from aios.harness.window import WindowOmission
from aios.models.events import Event


@pytest.fixture(autouse=True)
def _clear_model_descriptor_cache() -> None:
    """Reset the ``@cache``'d resolver before each test.

    ``build_messages``' thinking gate routes through the ``@cache``'d
    ``model_descriptor``. Several tests below monkeypatch
    ``litellm.supports_reasoning``; without clearing the cache the resolver's
    verdicts would stick across tests, producing order-dependent failures.
    """
    from aios.harness.completion import model_descriptor

    model_descriptor.cache_clear()


def _full_pipeline(
    events: list[Event],
    channels: list[str],
    focal_channel: str | None = None,
) -> list[dict[str, Any]]:
    """Compose ``build_messages`` → tail-block append → adjacent-user
    merge — the same sequence ``compose_step_context`` runs before
    handing the message list to LiteLLM."""
    ctx = build_messages(events, system_prompt=None)
    tail = build_channels_tail_block(channels, events, focal_channel)
    if tail is not None:
        ctx.messages.append(tail)
    return merge_adjacent_user_messages(ctx.messages)


def _full_pipeline_for_model(
    events: list[Event],
    model: str,
) -> list[dict[str, Any]]:
    """Compose the final message list for a concrete target model, exercising
    the SAME tail ``compose_step_context`` runs: ``build_messages`` (which
    strips reasoning fields for non-thinking targets) -> adjacent-user merge
    -> the production ``supports_thinking``-gated reasoning-stub helper.

    The stub step calls the real
    ``step_context._stub_reasoning_content_for_thinking_target`` — the exact
    seam the composer uses — so a regression that ungates it (re-adding the
    field for non-thinking targets) is caught here without a DB-backed
    ``compose_step_context`` round-trip."""
    from aios.harness.step_context import _stub_reasoning_content_for_thinking_target

    ctx = build_messages(events, system_prompt=None, model=model)
    messages = merge_adjacent_user_messages(ctx.messages)
    return _stub_reasoning_content_for_thinking_target(messages, model)


# Fixed receipt time so the per-message ``received=`` envelope field
# (see ``_format_received``) renders deterministically across assertions.
_FIXED_CREATED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
RECEIVED = "2026-01-02T03:04:05+00:00 (UTC)"  # _format_received(_FIXED_CREATED_AT, "UTC")


def _evt(
    seq: int,
    role: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
    content: str = "",
    metadata: dict[str, Any] | None = None,
    orig_channel: str | None = None,
    focal_channel_at_arrival: str | None = None,
    created_at: datetime | None = None,
) -> Event:
    """Build a minimal message Event for testing.

    When a user event is constructed with ``metadata["channel"]`` but
    no explicit ``orig_channel``, the helper auto-stamps it the same
    way :func:`aios.services.sessions.append_user_message` does at the
    real append site, so focal-aware rendering kicks in for these
    events just as it would in production.

    ``created_at`` defaults to :data:`_FIXED_CREATED_AT` so the
    ``received=`` envelope field is a stable :data:`RECEIVED` constant.
    """
    data: dict[str, Any] = {"role": role, "content": content}
    if tool_calls is not None:
        data["tool_calls"] = tool_calls
    if tool_call_id is not None:
        data["tool_call_id"] = tool_call_id
    if metadata is not None:
        data["metadata"] = metadata
    if orig_channel is None and role == "user" and isinstance(metadata, dict):
        ch = metadata.get("channel")
        if isinstance(ch, str):
            orig_channel = ch
    return Event(
        id=f"evt_{seq}",
        session_id="sess_01TEST",
        seq=seq,
        kind="message",
        data=data,
        created_at=created_at if created_at is not None else _FIXED_CREATED_AT,
        orig_channel=orig_channel,
        focal_channel_at_arrival=focal_channel_at_arrival,
    )


def _tc(call_id: str, name: str = "bash") -> dict[str, Any]:
    """Build a tool_call dict."""
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


# ─── build_messages ──────────────────────────────────────────────────────────


class TestBuildMessages:
    def test_simple_conversation(self) -> None:
        events = [
            _evt(1, "user", content="hello"),
            _evt(2, "assistant", content="hi there"),
        ]
        ctx = build_messages(
            events,
            system_prompt="you are helpful",
        )
        msgs = ctx.messages
        assert msgs[0] == {"role": "system", "content": "you are helpful"}
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_tool_results_grouped_with_assistant(self) -> None:
        events = [
            _evt(1, "user", content="do it"),
            _evt(2, "assistant", tool_calls=[_tc("a"), _tc("b")]),
            _evt(3, "tool", tool_call_id="a", content="result a"),
            _evt(4, "tool", tool_call_id="b", content="result b"),
        ]
        msgs = build_messages(
            events,
            system_prompt=None,
        ).messages
        # Order: user, assistant, tool_a, tool_b
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["tool_call_id"] == "a"
        assert msgs[3]["role"] == "tool"
        assert msgs[3]["tool_call_id"] == "b"

    def test_pending_tool_synthesized(self) -> None:
        events = [
            _evt(1, "user", content="do it"),
            _evt(2, "assistant", tool_calls=[_tc("a"), _tc("b")]),
            _evt(3, "tool", tool_call_id="a", content="result a"),
            # b is still pending
        ]
        msgs = build_messages(
            events,
            system_prompt=None,
        ).messages
        assert msgs[2]["tool_call_id"] == "a"
        assert "result a" in msgs[2]["content"]
        # b should be a synthetic pending result
        assert msgs[3]["tool_call_id"] == "b"
        assert "pending" in msgs[3]["content"]

    def test_out_of_order_tool_result_reordered(self) -> None:
        """Tool result arrives after a user message in seq order,
        but gets placed right after its assistant message in the prompt."""
        events = [
            _evt(1, "user", content="do X"),
            _evt(2, "assistant", tool_calls=[_tc("x")]),
            _evt(3, "user", content="how goes?"),  # user injection before tool completes
            _evt(4, "tool", tool_call_id="x", content="X done"),
        ]
        msgs = build_messages(
            events,
            system_prompt=None,
        ).messages
        # Should be: user, assistant+tool_calls, tool_result_x, user_injection
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == f"[received={RECEIVED}]\ndo X"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["tool_call_id"] == "x"
        assert msgs[3]["role"] == "user"
        assert msgs[3]["content"] == f"[received={RECEIVED}]\nhow goes?"

    def test_no_system_prompt_when_none(self) -> None:
        events = [_evt(1, "user", content="hi")]
        msgs = build_messages(
            events,
            system_prompt=None,
        ).messages
        assert msgs[0]["role"] == "user"

    def test_monotonic_blind_spot_shows_pending_then_injects_real(self) -> None:
        """When a tool result arrives during inference (seq between reacting_to
        and the assistant's own seq), the paired position shows PENDING (what the
        assistant actually saw) and the real result is injected as a user message
        after the horizon-setting assistant. This preserves prompt cache
        stability (monotonicity)."""
        events = [
            _evt(1, "user", content="run sleep 15"),
            _evt(2, "assistant", tool_calls=[_tc("bash_1")], content=""),
            # seq=3: user injection (before tool completes)
            _evt(3, "user", content="status?"),
            # seq=4: tool result (arrived DURING inference for assistant at seq=5)
            _evt(4, "tool", tool_call_id="bash_1", content='{"stdout": "DONE"}'),
            # seq=5: stale assistant response (reacting_to=3, saw pending)
            _evt(5, "assistant", content="still running"),
        ]
        # Simulate reacting_to on the assistant messages
        events[1].data["reacting_to"] = 1  # assistant at seq=2 reacted to user at seq=1
        events[4].data["reacting_to"] = 3  # assistant at seq=5 reacted to user at seq=3
        # (tool result at seq=4 has seq > reacting_to=3, so assistant saw pending)

        msgs = build_messages(
            events,
            system_prompt=None,
        ).messages

        # The paired position for bash_1 should show PENDING (not real)
        # because the assistant at seq=5 saw it as pending
        paired_tool = next(
            m for m in msgs if m.get("tool_call_id") == "bash_1" and m["role"] == "tool"
        )
        assert "pending" in paired_tool["content"], (
            "paired position should show pending (what the assistant actually saw)"
        )

        # The stale assistant should appear coherently after the pending result
        stale_asst = next(m for m in msgs if m.get("content") == "still running")
        assert stale_asst is not None

        # The real result should be injected as a user message after the stale assistant
        injected = next(
            (m for m in msgs if m["role"] == "user" and "DONE" in m.get("content", "")),
            None,
        )
        assert injected is not None, "real tool result should be injected as a user message"
        # The injected message should come AFTER the stale assistant
        injected_idx = msgs.index(injected)
        stale_idx = msgs.index(stale_asst)
        assert injected_idx > stale_idx, (
            "injected result should come after the stale assistant response"
        )

    def test_monotonic_no_rewrite_when_result_seen_as_real(self) -> None:
        """When the assistant saw the real result (result.seq <= reacting_to),
        the paired position shows the real result normally. No injection needed."""
        events = [
            _evt(1, "user", content="do it"),
            _evt(2, "assistant", tool_calls=[_tc("a")], content=""),
            _evt(3, "tool", tool_call_id="a", content="done"),
            _evt(4, "assistant", content="got it"),
        ]
        events[1].data["reacting_to"] = 1
        events[3].data["reacting_to"] = 3  # saw tool result at seq=3

        msgs = build_messages(
            events,
            system_prompt=None,
        ).messages
        # Paired position should show REAL result (not pending)
        paired_tool = next(m for m in msgs if m.get("tool_call_id") == "a")
        assert "done" in paired_tool["content"]
        assert "pending" not in paired_tool["content"]
        # No injected user messages for the tool result
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 1  # only the original user message

    def test_user_message_in_last_assistant_blind_spot_moves_to_tail(self) -> None:
        """A user message that arrives mid-inference (so the resulting assistant
        turn's ``reacting_to`` is behind it) must not be left stranded before that
        trailing assistant turn: the context would then END on an assistant
        message, which current reasoning models reject as an unsupported prefill
        ("the conversation must end with a user message"). It is deferred to the
        tail — symmetric with the blind-spot handling for late tool results."""
        events = [
            _evt(1, "user", content="summarize the news"),
            _evt(2, "assistant", tool_calls=[_tc("a", "web_search")]),
            _evt(3, "tool", tool_call_id="a", content="search results"),
            # arrives while the model is still composing its reply to the tool result
            _evt(4, "user", content="actually, clear everything"),
            # the reply only reacted to the tool result (seq 3); it never saw seq 4
            _evt(5, "assistant", content="Here is the news summary."),
        ]
        events[1].data["reacting_to"] = 1
        events[4].data["reacting_to"] = 3  # blind to the user message at seq 4

        ctx = build_messages(events, system_prompt=None)
        msgs = ctx.messages

        # The context must NOT end on an assistant turn (no accidental prefill).
        assert msgs[-1]["role"] == "user"
        assert "clear everything" in msgs[-1]["content"]
        # The trailing user message sits AFTER the assistant turn that never saw it.
        assert msgs[-2]["role"] == "assistant"
        # Surfaced exactly once — moved to the tail, not duplicated.
        clears = [
            m
            for m in msgs
            if m.get("role") == "user" and "clear everything" in str(m.get("content"))
        ]
        assert len(clears) == 1
        # reacting_to still advances to include the surfaced stimulus, so the
        # inference gate sees it as reacted and won't re-fire for it next step.
        assert ctx.reacting_to == 4

    def test_user_message_after_last_assistant_stays_at_tail(self) -> None:
        """The ordinary case — a user message that genuinely follows the last
        assistant turn — is untouched (already at the tail, surfaced once)."""
        events = [
            _evt(1, "user", content="hello"),
            _evt(2, "assistant", content="hi"),
            _evt(3, "user", content="how are you"),
        ]
        events[1].data["reacting_to"] = 1
        msgs = build_messages(events, system_prompt=None).messages
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert "how are you" in msgs[-1]["content"]

    def test_multiple_tool_batches(self) -> None:
        events = [
            _evt(1, "user", content="start"),
            _evt(2, "assistant", tool_calls=[_tc("a")]),
            _evt(3, "tool", tool_call_id="a", content="done a"),
            _evt(4, "assistant", tool_calls=[_tc("b")]),
            _evt(5, "tool", tool_call_id="b", content="done b"),
            _evt(6, "assistant", content="all done"),
        ]
        msgs = build_messages(
            events,
            system_prompt=None,
        ).messages
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]

    def test_tool_result_image_url_mime_corrected_at_replay(self) -> None:
        """Historical tool_result events (e.g. from the ``read`` tool) may
        carry an ``image_url`` data URL whose declared mime disagrees with
        the base64 bytes.  build_messages walks the assembled message list
        and substitutes the magic-byte-detected mime so Anthropic doesn't
        400 on replay — the original wedge incident was a tool_result
        image part.
        """
        import base64 as _b64

        jpeg_b64 = _b64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 32).decode("ascii")
        bad_url = f"data:image/png;base64,{jpeg_b64}"
        events = [
            _evt(1, "user", content="show photo"),
            _evt(2, "assistant", tool_calls=[_tc("a", name="read")]),
            Event(
                id="evt_3",
                session_id="sess_01TEST",
                seq=3,
                kind="message",
                data={
                    "role": "tool",
                    "tool_call_id": "a",
                    "content": [
                        {"type": "text", "text": "Image: photo.png"},
                        {"type": "image_url", "image_url": {"url": bad_url}},
                    ],
                },
                created_at=datetime.now(tz=UTC),
                orig_channel=None,
                focal_channel_at_arrival=None,
            ),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        tool_msg = msgs[2]
        assert tool_msg["role"] == "tool"
        url = tool_msg["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,"), (
            f"walker should have corrected image/png → image/jpeg, got {url[:40]}"
        )

    def test_prune_orphan_tool_results_at_start(self) -> None:
        """If DB windowing drops an assistant but keeps its tool results,
        those orphan tool results should be pruned from the start."""
        # Simulate pre-windowed events where the assistant at seq=2 was
        # dropped but its tool result at seq=3 was kept.
        events = [
            _evt(3, "tool", tool_call_id="a", content="result a"),
            _evt(4, "user", content="next question"),
            _evt(5, "assistant", content="answer"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == f"[received={RECEIVED}]\nnext question"
        assert msgs[1]["role"] == "assistant"

    def test_prune_orphan_tool_result_after_interleaved_user(self) -> None:
        """A window cut between an assistant's tool_call and a LATER
        interleaved user message leaves a MID-list orphan tool result.

        The blind-spot race (CLAUDE.md): an assistant emits a tool_call,
        a user message arrives DURING inference (higher seq), then the
        tool completes and appends its result (higher seq still) — log
        order assistant < user < result. When the token-budget window
        boundary drops the tool_call assistant but keeps the interleaved
        user message and the late result, the result is an orphan sitting
        AFTER a clean ``user`` start, which the leading-only prune stops
        before ever reaching. The emitted ``role=="tool"`` message then
        has no preceding ``tool_calls`` declaring it — an invalid
        chat-completions sequence that strict providers (Anthropic /
        Bedrock) reject, permanently wedging the session since
        ``build_messages`` is pure replay over the immutable window.
        """
        events = [
            _evt(11, "user", content="are you there?"),
            _evt(12, "tool", tool_call_id="bash_1", content="output"),
            _evt(13, "assistant", content="yes done"),
        ]
        events[2].data["reacting_to"] = 11
        msgs = build_messages(events, system_prompt=None).messages
        # Every tool result must be preceded by an assistant whose
        # tool_calls declare its id (same invariant asserted by
        # test_prune_partial_assistant_tool_group).
        for m in msgs:
            if m.get("role") == "tool":
                tc_id = m.get("tool_call_id")
                has_parent = any(
                    tc_id in {tc["id"] for tc in prior.get("tool_calls") or []}
                    for prior in msgs[: msgs.index(m)]
                    if prior.get("role") == "assistant"
                )
                assert has_parent, (
                    f"orphan tool result {tc_id!r}; roles={[mm.get('role') for mm in msgs]}"
                )

    def test_user_metadata_excluded_from_messages(self) -> None:
        """Metadata on user message events must not leak into the
        chat-completions message list sent to the model."""
        e = _evt(1, "user", content="hello")
        e.data["metadata"] = {"run_id": "abc123"}
        msgs = build_messages([e], system_prompt=None).messages
        assert msgs[0] == {"role": "user", "content": f"[received={RECEIVED}]\nhello"}

    def test_prune_partial_assistant_tool_group(self) -> None:
        """If DB windowing keeps an assistant with tool_calls but dropped
        one of its paired results, the incomplete group should be pruned."""
        # Simulate pre-windowed events where tool result "a" was dropped
        # by the window boundary but "b" was kept.
        events = [
            _evt(10, "assistant", tool_calls=[_tc("a"), _tc("b")]),
            _evt(12, "tool", tool_call_id="b", content="small"),
            _evt(13, "user", content="next"),
            _evt(14, "assistant", content="response"),
        ]
        events[0].data["reacting_to"] = 0
        events[3].data["reacting_to"] = 12
        msgs = build_messages(events, system_prompt=None).messages
        # The incomplete group (assistant + orphan tool "b") should be pruned.
        for m in msgs:
            if m.get("role") == "tool":
                tc_id = m.get("tool_call_id")
                has_parent = any(
                    tc_id in {tc["id"] for tc in prior.get("tool_calls") or []}
                    for prior in msgs[: msgs.index(m)]
                    if prior.get("role") == "assistant"
                )
                assert has_parent, f"orphan tool result for {tc_id}"

    def test_prune_handles_leading_assistant_with_malformed_tool_call(self) -> None:
        """Leading assistant with a tool_call missing ``id`` is unjoinable and
        must be dropped, not crash the prune."""
        events = [
            _evt(
                1,
                "assistant",
                tool_calls=[{"type": "function", "function": {"name": "bash", "arguments": "{}"}}],
            ),
            _evt(2, "user", content="next"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        assert [m["role"] for m in msgs] == ["user"]
        assert msgs[0]["content"] == f"[received={RECEIVED}]\nnext"

    def test_degenerate_empty_assistant_dropped_load_bearing_turns_kept(self) -> None:
        """A degenerate empty assistant turn (no content, no tool_calls, no
        thinking) is excluded from the composed messages — replaying it
        teaches literal-minded models to imitate silence. A tool-call turn
        and a real text turn are load-bearing and survive."""
        events = [
            _evt(1, "user", content="do it"),
            # A genuine empty turn the model emitted at some prior step.
            _evt(2, "assistant", content=""),
            _evt(3, "user", content="still there?"),
            _evt(4, "assistant", tool_calls=[_tc("a")]),
            _evt(5, "tool", tool_call_id="a", content="result a"),
            _evt(6, "assistant", content="here is the answer"),
        ]
        events[1].data["reacting_to"] = 1
        events[3].data["reacting_to"] = 3
        events[5].data["reacting_to"] = 5

        msgs = build_messages(events, system_prompt=None).messages

        # The degenerate empty assistant at seq=2 is gone.
        assert not any(
            m["role"] == "assistant" and m.get("content") == "" and "tool_calls" not in m
            for m in msgs
        )
        # The tool-call turn + its result and the text turn are kept.
        assert any(m["role"] == "assistant" and m.get("tool_calls") for m in msgs)
        assert any(m["role"] == "tool" and "result a" in m.get("content", "") for m in msgs)
        assert any(
            m["role"] == "assistant" and m.get("content") == "here is the answer" for m in msgs
        )


class TestTimezoneRendering:
    """The ``received=`` envelope renders in the account's effective timezone
    (threaded as ``tz_name``), with both the UTC offset and the IANA name."""

    def test_renders_in_account_timezone_with_name(self) -> None:
        ev = _evt(1, "user", content="hi", created_at=datetime(2026, 7, 1, 16, 0, tzinfo=UTC))
        content = build_messages([ev], system_prompt=None, tz_name="America/Los_Angeles").messages[
            0
        ]["content"]
        # July → PDT (-07:00); zone name appended. (DST-correct by stdlib:
        # an absolute instant formatted in a ZoneInfo gets that instant's
        # offset — not re-tested here to avoid coupling to calendar data.)
        assert content == "[received=2026-07-01T09:00:00-07:00 (America/Los_Angeles)]\nhi"


# ─── monotonicity ──────────────────────────────────────────────────────────


def _assert_prefix(short: list[dict[str, Any]], long: list[dict[str, Any]]) -> None:
    """Assert that *short* is a message-for-message prefix of *long*."""
    assert len(short) <= len(long), (
        f"short ({len(short)} msgs) is longer than long ({len(long)} msgs)"
    )
    for i, (a, b) in enumerate(zip(short, long, strict=False)):
        assert a == b, (
            f"monotonicity violation at index {i}:\n  short[{i}] = {a!r}\n  long[{i}]  = {b!r}"
        )


class TestMonotonicity:
    """build_messages(L1) must be a prefix of build_messages(L2) whenever L1
    is a prefix of L2.  This is the property that keeps the prompt prefix
    cache stable between successive inference calls."""

    @staticmethod
    def _build(events: list[Event]) -> list[dict[str, Any]]:
        return build_messages(
            events,
            system_prompt=None,
        ).messages

    def test_injection_stable_when_assistant_appended(self) -> None:
        """A blind-spot tool result is injected as a user message after
        the horizon-setter.  When the model responds (new assistant
        appended), the injection must not shift — the new assistant
        should appear AFTER the injection, preserving the prefix."""
        # L1: blind-spot result exists, model is about to be called.
        #   seq 4 (tool result) > reacting_to=3 of asst at seq 5
        #   → paired position shows PENDING, real result injected inline.
        l1 = [
            _evt(1, "user", content="run sleep 15"),
            _evt(2, "assistant", tool_calls=[_tc("bash_1")]),
            _evt(3, "user", content="status?"),
            _evt(4, "tool", tool_call_id="bash_1", content="DONE"),
            _evt(5, "assistant", content="still running"),
        ]
        l1[1].data["reacting_to"] = 1
        l1[4].data["reacting_to"] = 3  # blind to tool at seq 4

        # L2: model saw the injection and responded.
        l2 = [*l1, _evt(6, "assistant", content="ah it finished")]
        l2[5].data["reacting_to"] = 4

        ctx1 = self._build(l1)
        ctx2 = self._build(l2)

        # ctx1 should be a prefix of ctx2.
        _assert_prefix(ctx1, ctx2)

    def test_inline_injection_position(self) -> None:
        """Blind-spot injection appears right after the horizon-setter
        assistant, not at the absolute tail of the message list."""
        events = [
            _evt(1, "user", content="run it"),
            _evt(2, "assistant", tool_calls=[_tc("t1")]),
            _evt(3, "tool", tool_call_id="t1", content="RESULT"),
            # asst at seq=4 is the horizon-setter for seq=2.
            # Its reacting_to=1, so tool at seq=3 > horizon=1 → blind spot.
            _evt(4, "assistant", content="checking..."),
            _evt(5, "user", content="anything else?"),
            _evt(6, "assistant", content="nope"),
        ]
        events[1].data["reacting_to"] = 1
        events[3].data["reacting_to"] = 1  # didn't see tool at seq=3
        events[5].data["reacting_to"] = 5

        msgs = self._build(events)
        roles = [m["role"] for m in msgs]
        # Injection (user) sits between horizon-setter and the following user msg.
        assert roles == ["user", "assistant", "tool", "assistant", "user", "user", "assistant"]
        assert "RESULT" in msgs[4]["content"]

    def test_horizon_setter_with_tool_calls_injection_after(self) -> None:
        """When the horizon-setter itself has tool_calls, the blind-spot
        injection goes after the horizon-setter's own tool results."""
        events = [
            _evt(1, "user", content="do A and B"),
            _evt(2, "assistant", tool_calls=[_tc("a1")]),
            _evt(3, "tool", tool_call_id="a1", content="A done"),
            # Asst at seq=4: horizon-setter for seq=2, with its own tool_calls.
            _evt(4, "assistant", tool_calls=[_tc("b1")]),
            _evt(5, "tool", tool_call_id="b1", content="B done"),
            _evt(6, "assistant", content="all done"),
        ]
        events[1].data["reacting_to"] = 1
        events[3].data["reacting_to"] = 1  # blind to a1 at seq=3
        events[5].data["reacting_to"] = 5

        msgs = self._build(events)
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "tool", "assistant", "tool", "user", "assistant"]
        assert "pending" in msgs[2]["content"]
        assert msgs[4]["tool_call_id"] == "b1"
        assert "B done" in msgs[4]["content"]
        assert "A done" in msgs[5]["content"]

    def test_multiple_blind_spot_tools_same_assistant(self) -> None:
        """Multiple blind-spot tools from the same assistant are all injected
        inline after the same horizon-setter."""
        events = [
            _evt(1, "user", content="run two"),
            _evt(2, "assistant", tool_calls=[_tc("x"), _tc("y")]),
            _evt(3, "tool", tool_call_id="x", content="X done"),
            _evt(4, "tool", tool_call_id="y", content="Y done"),
            _evt(5, "assistant", content="both pending..."),
        ]
        events[1].data["reacting_to"] = 1
        events[4].data["reacting_to"] = 1  # blind to both tools

        msgs = self._build(events)
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "tool", "tool", "assistant", "user", "user"]
        assert "pending" in msgs[2]["content"]
        assert "pending" in msgs[3]["content"]
        assert "X done" in msgs[5]["content"]
        assert "Y done" in msgs[6]["content"]

    def test_multiple_assistants_with_blind_spots(self) -> None:
        """Two different assistants each with blind-spot tools inject after
        their respective horizon-setters."""
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[_tc("a")]),
            _evt(3, "tool", tool_call_id="a", content="A done"),
            # asst at seq=4 is horizon-setter for seq=2, and itself has tool_calls.
            _evt(4, "assistant", tool_calls=[_tc("b")]),
            _evt(5, "tool", tool_call_id="b", content="B done"),
            # asst at seq=6 is horizon-setter for seq=4.
            _evt(6, "assistant", content="wrapping up"),
        ]
        events[1].data["reacting_to"] = 1
        events[3].data["reacting_to"] = 1  # blind to a (seq=3 > horizon=1)
        events[5].data["reacting_to"] = 3  # blind to b (seq=5 > horizon=3)

        msgs = self._build(events)
        roles = [m["role"] for m in msgs]
        assert roles == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "user",
            "assistant",
            "user",
        ]
        assert "pending" in msgs[2]["content"]
        assert "pending" in msgs[4]["content"]
        assert "A done" in msgs[5]["content"]
        assert "B done" in msgs[7]["content"]

    def test_monotonicity_across_three_successive_appends(self) -> None:
        """L1 ⊂ L2 ⊂ L3: prefix preserved at each step."""
        base = [
            _evt(1, "user", content="run sleep 15"),
            _evt(2, "assistant", tool_calls=[_tc("bash_1")]),
            _evt(3, "user", content="status?"),
            _evt(4, "tool", tool_call_id="bash_1", content="DONE"),
            _evt(5, "assistant", content="still running"),
        ]
        base[1].data["reacting_to"] = 1
        base[4].data["reacting_to"] = 3

        l1 = list(base)
        l2 = [*l1, _evt(6, "assistant", content="ah it finished")]
        l2[5].data["reacting_to"] = 4
        l3 = [*l2, _evt(7, "user", content="great, now do Y")]

        ctx1, ctx2, ctx3 = self._build(l1), self._build(l2), self._build(l3)
        _assert_prefix(ctx1, ctx2)
        _assert_prefix(ctx2, ctx3)

    def test_merge_insertion_preserves_monotonicity(self) -> None:
        """Full pipeline (build_messages → tail-block → adjacent-user
        merge) must keep the prefix-stability invariant: output(L1) is a
        prefix of output(L2) when L1 ⊂ L2.  Pins the "mutations only at
        the volatile suffix" claim — a refactor that merged messages into
        the cache-stable prefix would fail this.

        The tail block lands as the trailing user-role message. When the
        preceding message is also user-role (L2: ``…, user "do B"``), the
        merge folds the tail *into* that user turn, so the tail header is
        no longer a standalone message — it's a substring of the final
        user content. ``_strip_tail`` drops any trailing user message
        carrying the header either way."""
        bindings = ["signal/test/1"]

        l1 = [
            _evt(1, "user", content="do A"),
            _evt(2, "assistant", content="done A"),
        ]
        l2 = [*l1, _evt(3, "user", content="do B")]
        l3 = [*l2, _evt(4, "assistant", content="done B")]

        out1 = _full_pipeline(l1, bindings)
        out2 = _full_pipeline(l2, bindings)
        out3 = _full_pipeline(l3, bindings)

        # No degenerate "." separator is ever produced now.
        for out in (out1, out2, out3):
            assert not any(m == {"role": "assistant", "content": "."} for m in out)

        # The tail block mutates per step, so compare prefixes only up to
        # (but not including) the trailing user turn that carries it —
        # whether standalone or merged into the preceding inbound.
        def _strip_tail(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
            for i in range(len(msgs) - 1, -1, -1):
                m = msgs[i]
                if m.get("role") == "user" and "━━━ Channels ━━━" in str(m.get("content", "")):
                    return msgs[:i]
            return msgs

        _assert_prefix(_strip_tail(out1), _strip_tail(out2))
        _assert_prefix(_strip_tail(out2), _strip_tail(out3))

    def test_reacting_to_includes_inline_injection_seq(self) -> None:
        """ContextResult.reacting_to must account for the seq of blind-spot
        tool results that are injected inline."""
        events = [
            _evt(1, "user", content="run it"),
            _evt(2, "assistant", tool_calls=[_tc("t1")]),
            _evt(3, "tool", tool_call_id="t1", content="RESULT"),
            _evt(4, "assistant", content="checking..."),
        ]
        events[1].data["reacting_to"] = 1
        events[3].data["reacting_to"] = 1

        ctx = build_messages(events, system_prompt=None)
        assert ctx.reacting_to >= 3

    def test_blind_spot_injection_multimodal_list_content(self) -> None:
        """When a tool returns ``list[dict]`` content (image-aware read,
        multi-part output) AND the result lands during inference, the
        synthetic user-message injection must splice the parts in
        properly — NOT f-string the list (which would emit Python repr
        and lose the image).  Regression test for PR #218."""
        image_part = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,AAAA"},
        }
        tool_content = [
            {"type": "text", "text": "screenshot below"},
            image_part,
        ]
        events = [
            _evt(1, "user", content="show me the screen"),
            _evt(2, "assistant", tool_calls=[_tc("read_1")]),
            _evt(3, "tool", tool_call_id="read_1", content=""),  # placeholder; overridden below
            _evt(4, "assistant", content="working on it"),
        ]
        # Set the tool result content to a list[dict] (the new shape PR2 added).
        events[2].data["content"] = tool_content
        events[2].data["name"] = "read"
        events[1].data["reacting_to"] = 1
        events[3].data["reacting_to"] = 1  # blind to tool at seq=3

        msgs = self._build(events)
        # Locate the injected user message — it follows the horizon-setter
        # assistant ("working on it") and carries the completion header.
        injected = next(
            (
                m
                for m in msgs
                if m["role"] == "user"
                and (
                    (isinstance(m.get("content"), str) and "completed]" in m["content"])
                    or (
                        isinstance(m.get("content"), list)
                        and any(
                            isinstance(p, dict)
                            and p.get("type") == "text"
                            and "completed]" in (p.get("text") or "")
                            for p in m["content"]
                        )
                    )
                )
            ),
            None,
        )
        assert injected is not None, "blind-spot injection must be emitted"
        # The injection MUST carry an image_url part — not the Python repr
        # of the list.
        content = injected["content"]
        assert isinstance(content, list), (
            f"multimodal injection must be content-parts, got {type(content).__name__}"
        )
        assert any(p.get("type") == "image_url" for p in content), (
            "image_url part must survive the injection"
        )
        # And no part should contain the literal repr fragment that the bug
        # would have produced.
        for p in content:
            text = p.get("text") if isinstance(p, dict) else None
            if isinstance(text, str):
                assert "image_url" not in text or text.startswith("[Tool result"), (
                    "f-stringed list repr leaked into the injection text"
                )

    def test_blind_spot_injection_string_content_unchanged(self) -> None:
        """Plain string tool results still produce a string-content injection
        (back-compat: PR #218's multimodal branch must not regress the
        common path)."""
        events = [
            _evt(1, "user", content="run it"),
            _evt(2, "assistant", tool_calls=[_tc("bash_1")]),
            _evt(3, "tool", tool_call_id="bash_1", content="DONE"),
            _evt(4, "assistant", content="still going"),
        ]
        events[2].data["name"] = "bash"
        events[1].data["reacting_to"] = 1
        events[3].data["reacting_to"] = 1

        msgs = self._build(events)
        injected = next(m for m in msgs if m["role"] == "user" and m != msgs[0])
        assert isinstance(injected["content"], str)
        assert "completed]" in injected["content"]
        assert "DONE" in injected["content"]


# ─── field stripping ────────────────────────────────────────────────────────


class TestFieldStripping:
    """build_messages strips provider-specific fields from the output,
    keeping only chat-completions spec fields per role."""

    def test_assistant_reasoning_content_stripped(self) -> None:
        """Provider-specific reasoning_content is excluded from context."""
        events = [
            _evt(1, "user", content="hello"),
            _evt(2, "assistant", content="hi"),
        ]
        events[1].data["reasoning_content"] = "I think the user wants..."
        msgs = build_messages(events, system_prompt=None).messages
        assert msgs[1]["content"] == "hi"
        assert "reasoning_content" not in msgs[1]

    def test_assistant_reacting_to_stripped(self) -> None:
        """Internal reacting_to field is excluded from context output."""
        events = [
            _evt(1, "user", content="hello"),
            _evt(2, "assistant", content="hi"),
        ]
        events[1].data["reacting_to"] = 1
        msgs = build_messages(events, system_prompt=None).messages
        assert "reacting_to" not in msgs[1]

    def test_assistant_tool_calls_preserved(self) -> None:
        """tool_calls is a spec field and must survive stripping."""
        events = [
            _evt(1, "user", content="do it"),
            _evt(2, "assistant", tool_calls=[_tc("a")]),
            _evt(3, "tool", tool_call_id="a", content="done"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        assert msgs[1]["tool_calls"] == [_tc("a")]

    def test_tool_message_extra_fields_stripped(self) -> None:
        """Unknown fields on tool messages are excluded."""
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[_tc("a")]),
            _evt(3, "tool", tool_call_id="a", content="done"),
        ]
        events[2].data["provider_metadata"] = {"some": "thing"}
        msgs = build_messages(events, system_prompt=None).messages
        tool_msg = next(m for m in msgs if m.get("role") == "tool")
        assert "provider_metadata" not in tool_msg
        assert tool_msg["tool_call_id"] == "a"
        assert tool_msg["content"] == "done"

    def test_multiple_provider_fields_all_stripped(self) -> None:
        """All provider-specific fields are excluded, only spec fields remain."""
        events = [
            _evt(1, "user", content="think hard"),
            _evt(2, "assistant", content="here's my answer"),
        ]
        events[1].data.update(
            {
                "reasoning_content": "deep thoughts...",
                "reasoning": "step by step...",
                "reasoning_details": [{"type": "thinking", "content": "hmm"}],
                "reacting_to": 1,
                "provider_specific_id": "abc123",
            }
        )
        msgs = build_messages(events, system_prompt=None).messages
        assert set(msgs[1].keys()) == {"role", "content"}

    def test_system_prompt_clean(self) -> None:
        """System prompt message contains only spec fields."""
        events = [_evt(1, "user", content="hi")]
        msgs = build_messages(events, system_prompt="You are helpful.").messages
        assert msgs[0] == {"role": "system", "content": "You are helpful."}

    def test_stripping_does_not_mutate_event_data(self) -> None:
        """Stripping produces new dicts; original event data is unchanged."""
        events = [
            _evt(1, "user", content="hello"),
            _evt(2, "assistant", content="hi"),
        ]
        events[1].data["reasoning_content"] = "thoughts"
        build_messages(events, system_prompt=None)
        assert "reasoning_content" in events[1].data


class TestThinkingBlockPreservation:
    def test_thinking_blocks_preserved_for_thinking_capable_target(self) -> None:
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hey"),
        ]
        events[1].data["thinking_blocks"] = [
            {"type": "thinking", "thinking": "user said hi", "signature": "abc"}
        ]
        msgs = build_messages(
            events, system_prompt=None, model="anthropic/claude-haiku-4-5"
        ).messages
        assert msgs[1]["thinking_blocks"] == [
            {"type": "thinking", "thinking": "user said hi", "signature": "abc"}
        ]

    def test_reasoning_content_preserved_for_thinking_capable_target(self) -> None:
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hey"),
        ]
        events[1].data["reasoning_content"] = "deep thoughts about hi"
        msgs = build_messages(
            events, system_prompt=None, model="anthropic/claude-haiku-4-5"
        ).messages
        assert msgs[1]["reasoning_content"] == "deep thoughts about hi"

    def test_thinking_blocks_stripped_for_non_thinking_target(self) -> None:
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hey"),
        ]
        events[1].data["thinking_blocks"] = [{"type": "thinking", "thinking": "user said hi"}]
        events[1].data["reasoning_content"] = "deep thoughts"
        msgs = build_messages(events, system_prompt=None, model="openai/gpt-4o-mini").messages
        assert "thinking_blocks" not in msgs[1]
        assert "reasoning_content" not in msgs[1]

    def test_thinking_blocks_stripped_when_model_arg_omitted(self) -> None:
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hey"),
        ]
        events[1].data["thinking_blocks"] = [{"type": "thinking", "thinking": "user said hi"}]
        msgs = build_messages(events, system_prompt=None).messages
        assert "thinking_blocks" not in msgs[1]

    def test_other_provider_fields_still_stripped_on_thinking_target(self) -> None:
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hey"),
        ]
        events[1].data.update(
            {
                # Signature present so the read-path poison guard (issue #1588)
                # keeps the block -- this test is about *other* provider fields.
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "...", "signature": "sig"}
                ],
                "reasoning": "step 1",
                "reasoning_details": [{"type": "thinking", "content": "..."}],
                "reacting_to": 1,
            }
        )
        msgs = build_messages(
            events, system_prompt=None, model="anthropic/claude-haiku-4-5"
        ).messages
        assert "thinking_blocks" in msgs[1]
        assert "reasoning" not in msgs[1]
        assert "reasoning_details" not in msgs[1]
        assert "reacting_to" not in msgs[1]

    def test_poison_empty_signature_block_dropped_on_replay(self) -> None:
        """A persisted poison thinking block -- real thinking text but an
        empty ``signature`` (the Ultron transcript-poison shape, issue
        #1588) -- is dropped on the read path so it never replays to
        Anthropic and 400s with "Invalid signature in thinking block". A
        thinking-less turn always replays 200, so dropping is safe."""
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hey"),
        ]
        events[1].data["thinking_blocks"] = [
            {"type": "thinking", "thinking": "real reasoning", "signature": ""}
        ]
        msgs = build_messages(
            events, system_prompt=None, model="anthropic/claude-haiku-4-5"
        ).messages
        assert "thinking_blocks" not in msgs[1]

    def test_poison_missing_signature_block_dropped_on_replay(self) -> None:
        """A block with thinking text but NO ``signature`` key at all is
        likewise quarantined on replay (Anthropic 400s ``Field
        required``)."""
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hey"),
        ]
        events[1].data["thinking_blocks"] = [
            {"type": "thinking", "thinking": "real reasoning"}
        ]
        msgs = build_messages(
            events, system_prompt=None, model="anthropic/claude-haiku-4-5"
        ).messages
        assert "thinking_blocks" not in msgs[1]

    def test_poison_block_dropped_but_signed_sibling_kept(self) -> None:
        """In a mixed list, only the poison (empty-signature) block is
        dropped; a fully-signed sibling still replays."""
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hey"),
        ]
        good = {"type": "thinking", "thinking": "kept", "signature": "sig"}
        events[1].data["thinking_blocks"] = [
            good,
            {"type": "thinking", "thinking": "lost-sig", "signature": ""},
        ]
        msgs = build_messages(
            events, system_prompt=None, model="anthropic/claude-haiku-4-5"
        ).messages
        assert msgs[1]["thinking_blocks"] == [good]

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-opus-4-8",  # stale catalog on a long-running worker
            "openrouter/anthropic/claude-opus-4-8",  # litellm under-reports proxy routes
        ],
    )
    def test_thinking_blocks_preserved_for_claude_despite_stale_catalog(
        self, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        """``litellm.supports_reasoning`` returns False for Claude models a
        long-running worker's catalog predates — and for proxy-routed Claude
        even when fresh.  Stripping ``thinking_blocks`` then violates
        Anthropic's cross-turn preservation contract; the Claude short-circuit
        keeps them regardless of the catalog.
        """
        monkeypatch.setattr("litellm.supports_reasoning", lambda *a, **k: False)
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hey"),
        ]
        events[1].data["thinking_blocks"] = [
            {"type": "thinking", "thinking": "user said hi", "signature": "abc"}
        ]
        msgs = build_messages(events, system_prompt=None, model=model).messages
        assert msgs[1]["thinking_blocks"] == [
            {"type": "thinking", "thinking": "user said hi", "signature": "abc"}
        ]

    def test_thinking_blocks_stripped_for_non_claude_when_catalog_says_no(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-Claude models still defer to litellm: a False there strips,
        confirming the Claude rule didn't over-broaden to every model."""
        monkeypatch.setattr("litellm.supports_reasoning", lambda *a, **k: False)
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hey"),
        ]
        events[1].data["thinking_blocks"] = [{"type": "thinking", "thinking": "x"}]
        msgs = build_messages(events, system_prompt=None, model="deepseek/deepseek-chat").messages
        assert "thinking_blocks" not in msgs[1]


class TestToolCallSanitization:
    """_strip_to_spec sanitizes the inner structure of tool_calls so
    malformed entries from one model don't break cross-model replay."""

    def test_valid_tool_calls_unchanged(self) -> None:
        """Well-formed tool_calls pass through identical."""
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[_tc("a", "bash")]),
            _evt(3, "tool", tool_call_id="a", content="done"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        assert msgs[1]["tool_calls"] == [_tc("a", "bash")]

    def test_malformed_arguments_replaced(self) -> None:
        """Control characters in function.arguments get replaced with '{}'."""
        bad_tc = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"cmd": "echo\nhello"}'},
        }
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[bad_tc]),
            _evt(3, "tool", tool_call_id="call_1", content="done"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        tc = msgs[1]["tool_calls"][0]
        assert tc["function"]["arguments"] == "{}"
        assert tc["function"]["name"] == "bash"
        assert tc["id"] == "call_1"

    def test_missing_function_name_defaults_empty(self) -> None:
        """Missing function.name defaults to empty string."""
        bad_tc = {"id": "call_1", "type": "function", "function": {"arguments": "{}"}}
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[bad_tc]),
            _evt(3, "tool", tool_call_id="call_1", content="done"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        assert msgs[1]["tool_calls"][0]["function"]["name"] == ""

    def test_missing_id_filtered_out(self) -> None:
        """A tool_call entry without an ``id`` is unjoinable — dropped, not
        preserved with id=''. Once its sole tool_call is stripped the
        assistant turn has no content and no tool_calls, so the
        degenerate-empty-assistant drop removes the turn entirely; only
        the user message survives."""
        bad_tc = {"type": "function", "function": {"name": "bash", "arguments": "{}"}}
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[bad_tc]),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        assert [m["role"] for m in msgs] == ["user"]
        assert not any(m.get("tool_calls") for m in msgs)

    def test_empty_id_string_filtered(self) -> None:
        """An explicit empty-string id is also unjoinable and filtered;
        the resulting degenerate empty assistant turn is then dropped."""
        bad_tc = {"id": "", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[bad_tc]),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        assert [m["role"] for m in msgs] == ["user"]
        assert not any(m.get("tool_calls") for m in msgs)

    def test_mid_window_malformed_filtered_among_valid(self) -> None:
        """Mid-window assistant with mixed valid + malformed tool_calls keeps only the valid entry."""
        malformed = {"type": "function", "function": {"name": "bash", "arguments": "{}"}}
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[_tc("call_a", "bash"), malformed]),
            _evt(3, "tool", tool_call_id="call_a", content="done"),
            _evt(4, "user", content="next"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        asst = next(m for m in msgs if m.get("role") == "assistant")
        assert asst["tool_calls"] == [_tc("call_a", "bash")]

    def test_extra_fields_stripped_from_tool_call(self) -> None:
        """Provider-specific fields inside tool_call dicts are excluded."""
        tc_with_extras = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
            "index": 0,
            "provider_id": "xyz",
        }
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[tc_with_extras]),
            _evt(3, "tool", tool_call_id="call_1", content="done"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        tc = msgs[1]["tool_calls"][0]
        assert set(tc.keys()) == {"id", "type", "function"}
        assert set(tc["function"].keys()) == {"name", "arguments"}

    def test_dict_arguments_serialized(self) -> None:
        """Arguments as a dict (from some providers) are serialized to JSON string."""
        bad_tc = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": {"command": "ls"}},
        }
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[bad_tc]),
            _evt(3, "tool", tool_call_id="call_1", content="done"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        args = msgs[1]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"command": "ls"}

    def test_missing_function_dict(self) -> None:
        """tool_call with no function dict at all gets safe defaults."""
        bad_tc = {"id": "call_1", "type": "function"}
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[bad_tc]),
            _evt(3, "tool", tool_call_id="call_1", content="done"),
        ]
        msgs = build_messages(events, system_prompt=None).messages
        tc = msgs[1]["tool_calls"][0]
        assert tc["function"]["name"] == ""
        assert tc["function"]["arguments"] == "{}"

    def test_sanitization_does_not_mutate_event_data(self) -> None:
        """Sanitization produces new dicts; original event data is unchanged."""
        bad_args = '{"cmd": "echo\nhello"}'
        bad_tc = {
            "id": "c1",
            "type": "function",
            "function": {"name": "bash", "arguments": bad_args},
        }
        events = [
            _evt(1, "user", content="go"),
            _evt(2, "assistant", tool_calls=[bad_tc]),
            _evt(3, "tool", tool_call_id="c1", content="done"),
        ]
        build_messages(events, system_prompt=None)
        assert events[1].data["tool_calls"][0]["function"]["arguments"] == bad_args


# ─── focal-channel rendering ────────────────────────────────────────────────


class TestFocalRendering:
    """Slice 4 of the focal-channel redesign (issue #29).

    User events are rendered differently based on ``orig_channel`` vs
    ``focal_channel_at_arrival``.  Three branches:

    * both NULL (legacy / direct non-connector message) → Phase 2
      rendering unchanged.
    * ``orig == focal_at_arrival`` → full content with the #46 metadata
      header inlined.
    * ``orig != focal_at_arrival`` OR focal NULL → short notification
      marker.
    """

    _CHAN_A = "signal/bot/alice"
    _CHAN_B = "signal/bot/bob"

    def test_legacy_null_event_phase2_rendering(self) -> None:
        """orig_channel=None: only the uniform ``received=`` envelope, no
        connector header, no notification marker."""
        events = [_evt(1, "user", content="hi")]
        msg = build_messages(events, system_prompt=None).messages[0]
        assert msg["content"] == f"[received={RECEIVED}]\nhi"
        assert "metadata" not in msg
        assert "🔔" not in msg["content"]

    def test_focal_match_renders_full_content_with_header(self) -> None:
        md = {
            "channel": self._CHAN_A,
            "sender_uuid": "peer-uuid",
            "timestamp_ms": 1776401210703,
        }
        events = [_evt(1, "user", content="hi", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        msg = build_messages(events, system_prompt=None).messages[0]
        assert msg["content"].startswith(f"[channel={self._CHAN_A}")
        assert "sender_uuid=peer-uuid" in msg["content"]
        assert msg["content"].endswith("\nhi")
        assert "metadata" not in msg

    def test_focal_match_header_includes_chat_type_name_and_sender(self) -> None:
        md = {
            "channel": self._CHAN_A,
            "chat_type": "group",
            "chat_name": "QA",
            "sender_uuid": "u1",
            "sender_name": "Tom",
            "timestamp_ms": 1776401210703,
        }
        events = [_evt(1, "user", content="yo", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "chat_type=group" in content
        assert "chat_name='QA'" in content
        assert "from=Tom" in content

    def test_focal_match_reply_to_surfaced(self) -> None:
        md = {
            "channel": self._CHAN_A,
            "reply_to": {
                "author_uuid": "bot",
                "timestamp_ms": 1776400000000,
                "text": "what I said before",
            },
        }
        events = [
            _evt(1, "user", content="reacting", metadata=md, focal_channel_at_arrival=self._CHAN_A)
        ]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "\n[reply_to: author_uuid=bot · timestamp_ms=1776400000000]" in content
        assert "> what I said before" in content

    def test_focal_match_reply_to_text_non_string_does_not_brick_session(self) -> None:
        """Non-string ``reply_to.text`` from a connector must not crash the
        renderer. ``_format_channel_header`` runs on every wake while the
        event is in the window, so a crash permanently bricks the session
        (same failure class as #446)."""
        md = {
            "channel": self._CHAN_A,
            "reply_to": {
                "author_uuid": "bot",
                "timestamp_ms": 1776400000000,
                "text": {"blocks": ["unexpected shape"]},
            },
        }
        events = [_evt(1, "user", content="ok", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        # Quote snippet omitted because the text isn't renderable as a string.
        assert "[reply_to: author_uuid=bot · timestamp_ms=1776400000000]" in content
        assert "blocks" not in content

    def test_focal_match_reaction_surfaced(self) -> None:
        md = {
            "channel": self._CHAN_A,
            "reaction": {
                "emoji": "👍",
                "target_author_uuid": "bot",
                "target_timestamp_ms": 1776400000000,
            },
        }
        events = [_evt(1, "user", content="", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "[reaction='👍'" in content
        assert "target_author_uuid=bot" in content

    def test_focal_match_telegram_reaction_surfaces_new_emojis(self) -> None:
        """Telegram reactions arrive as ``new_emojis: list[str]`` (current
        post-reaction state) rather than Signal's ``emoji: str``. The
        renderer must surface the emoji from either shape — pre-fix the
        Telegram shape rendered as ``[reaction='?']`` and the model could
        not tell what was reacted with."""
        md = {
            "channel": self._CHAN_A,
            "reaction": {
                "target_message_id": 42,
                "old_emojis": [],
                "new_emojis": ["👍"],
            },
        }
        events = [_evt(1, "user", content="", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "[reaction='👍'" in content

    def test_focal_match_telegram_reaction_cleared_rendered_distinctly(self) -> None:
        """Empty ``new_emojis`` means the user cleared their reaction.
        The model needs that signal distinctly — silently dropping it
        would leave the model believing the prior reaction is still
        active."""
        md = {
            "channel": self._CHAN_A,
            "reaction": {
                "target_message_id": 42,
                "old_emojis": ["👍"],
                "new_emojis": [],
            },
        }
        events = [_evt(1, "user", content="", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "reaction" in content
        assert "'?'" not in content

    def test_focal_match_reaction_renders_string_target_message_id(self) -> None:
        """WhatsApp reactions identify the target by string message_id
        (no equivalent of Signal's author+timestamp pair).  Pre-fix the
        renderer ignored the field, so the model saw `[reaction='👍']`
        with no way to match against the original send."""
        md = {
            "channel": self._CHAN_A,
            "reaction": {
                "emoji": "👍",
                "target_message_id": "3EB0ORIGINAL",
            },
        }
        events = [_evt(1, "user", content="", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "[reaction='👍'" in content
        assert "target_message_id='3EB0ORIGINAL'" in content

    def test_focal_match_edit_renders_target_message_id(self) -> None:
        """WhatsApp edits flag ``edited=True`` and identify the original
        by string ``edit_target_message_id`` (no equivalent of Signal's
        ``edit_target_timestamp_ms``).  Pre-fix the model saw
        `edited=true` with no anchor."""
        md = {
            "channel": self._CHAN_A,
            "edited": True,
            "edit_target_message_id": "3EB0ORIGINAL",
        }
        events = [
            _evt(1, "user", content="new body", metadata=md, focal_channel_at_arrival=self._CHAN_A)
        ]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "edited=true" in content
        assert "edit_target_message_id='3EB0ORIGINAL'" in content

    def test_focal_match_revoke_renders_target_message_id(self) -> None:
        """A peer revoking their message arrives with empty content and
        ``revoked=True`` + ``revoke_target_message_id``.  Pre-fix
        neither flag was rendered, so the revoke was invisible to the
        model."""
        md = {
            "channel": self._CHAN_A,
            "revoked": True,
            "revoke_target_message_id": "3EB0ORIGINAL",
        }
        events = [_evt(1, "user", content="", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "revoked=true" in content
        assert "revoke_target_message_id='3EB0ORIGINAL'" in content

    def test_focal_match_iso_timestamp(self) -> None:
        """The raw origin ``timestamp_ms`` int is surfaced for connector tool
        args; its human-readable ISO now lives in the uniform ``received=``
        envelope field (receipt time), not as a parenthetical."""
        md = {"channel": self._CHAN_A, "timestamp_ms": 1776401210703}
        events = [_evt(1, "user", content="hi", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "timestamp_ms=1776401210703" in content
        assert "timestamp_ms=1776401210703 (" not in content  # no parenthetical ISO
        assert f"received={RECEIVED}" in content

    def test_focal_match_message_id_inlined(self) -> None:
        """The ``message_id`` lets the model react/edit/delete the user
        message via the connector tools — without it, the model has to
        guess (or refuse).  See #245.
        """
        md = {"channel": self._CHAN_A, "message_id": 94}
        events = [
            _evt(1, "user", content="show me", metadata=md, focal_channel_at_arrival=self._CHAN_A)
        ]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "message_id=94" in content

    def test_focal_match_sender_id_inlined(self) -> None:
        """Telegram inbounds carry an integer ``sender_id`` (Telegram's
        platform-native user id); ``sender_uuid`` is the signal-shape.
        Both must surface so platform-aware models can use them.
        """
        md = {"channel": self._CHAN_A, "sender_id": 1595907265}
        events = [_evt(1, "user", content="hi", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "sender_id=1595907265" in content

    def test_focal_match_edited_flag(self) -> None:
        """Edited messages arrive with ``metadata.edited=True``; the model
        needs to know an edit happened so it can update / re-react / etc.
        """
        md = {"channel": self._CHAN_A, "message_id": 7, "edited": True}
        events = [
            _evt(
                1, "user", content="oops, fixed", metadata=md, focal_channel_at_arrival=self._CHAN_A
            )
        ]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "edited=true" in content

    def test_focal_match_edited_false_omitted(self) -> None:
        """Most messages aren't edits — don't clutter the header on the
        common path.  Only emit ``edited=true`` when the flag is actually
        set; absent / False stays off the header entirely.
        """
        md = {"channel": self._CHAN_A, "message_id": 7, "edited": False}
        events = [_evt(1, "user", content="hi", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "edited" not in content

    def test_focal_match_edit_target_timestamp(self) -> None:
        """When ``metadata.edited=True``, ``edit_target_timestamp_ms`` is
        emitted alongside so the model can correlate the edit back to the
        original (and react/delete/re-edit if needed).  Signal carries it
        natively via ``editMessage.targetSentTimestamp``; telegram doesn't
        but the field name stays platform-agnostic on the metadata.
        """
        md = {
            "channel": self._CHAN_A,
            "edited": True,
            "edit_target_timestamp_ms": 1776400000000,
        }
        events = [
            _evt(
                1, "user", content="oops, fixed", metadata=md, focal_channel_at_arrival=self._CHAN_A
            )
        ]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "edited=true" in content
        assert "edit_target_timestamp_ms=1776400000000" in content

    def test_focal_match_self_mentioned_surfaced(self) -> None:
        """Group chats use mentions to summon a response; ``self_mentioned``
        gives the model an unambiguous "the sender's client encoded a
        mention targeting my account" signal rather than substring-matching
        the (placeholder-substituted) text content.
        """
        md = {
            "channel": self._CHAN_A,
            "self_mentioned": True,
            "mentions": [
                {"uuid": "bot-uuid", "name": "SmokeBot"},
            ],
        }
        events = [
            _evt(
                1,
                "user",
                content="@SmokeBot hi",
                metadata=md,
                focal_channel_at_arrival=self._CHAN_A,
            )
        ]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "self_mentioned=true" in content
        # Structured mention entries render on their own line so the
        # uuid is available to the model for outbound mention encoding.
        assert "mention: name='SmokeBot' · uuid=bot-uuid" in content

    def test_focal_match_mentions_without_self_mention(self) -> None:
        """Mentions of someone other than the bot still render so the
        model has context (e.g. "Alice tagged Bob"), but
        ``self_mentioned=true`` doesn't appear when the bot wasn't
        the target.
        """
        md = {
            "channel": self._CHAN_A,
            "self_mentioned": False,
            "mentions": [{"uuid": "alice-uuid", "name": "Alice"}],
        }
        events = [
            _evt(
                1,
                "user",
                content="@Alice did it",
                metadata=md,
                focal_channel_at_arrival=self._CHAN_A,
            )
        ]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "self_mentioned" not in content
        assert "mention: name='Alice' · uuid=alice-uuid" in content

    def test_focal_match_sticker_emoji(self) -> None:
        """Stickers arrive with an empty body and a ``sticker_emoji`` in
        metadata — the model otherwise has no textual cue that something
        was sent.
        """
        md = {"channel": self._CHAN_A, "message_id": 12, "sticker_emoji": "🔥"}
        events = [_evt(1, "user", content="", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "sticker_emoji='🔥'" in content

    def test_focal_match_metadata_stripped_from_wire_message(self) -> None:
        md = {"channel": self._CHAN_A, "sender_uuid": "u1"}
        events = [_evt(1, "user", content="hi", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        msg = build_messages(events, system_prompt=None).messages[0]
        assert "metadata" not in msg
        assert set(msg.keys()) <= {"role", "content", "name"}

    def test_notification_when_orig_differs_from_focal(self) -> None:
        md = {"channel": self._CHAN_B, "sender_name": "Bob"}
        events = [
            _evt(
                1,
                "user",
                content="hey there",
                metadata=md,
                focal_channel_at_arrival=self._CHAN_A,
            )
        ]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert content.startswith(f"🔔 channel_id={self._CHAN_B}")
        assert "from=Bob" in content
        assert "hey there" in content

    def test_notification_with_attachment_appends_marker(self) -> None:
        """#718: an attachment arriving on a non-focal channel surfaces a
        ``read``-able marker appended to the notification — not silently
        dropped.  Exercised through the real ``build_messages`` path with no
        model threaded (how the append-time token counter calls it);
        ``text_marker`` needs none, so the breadcrumb appears regardless."""
        md = {
            "channel": self._CHAN_B,
            "sender_name": "Bob",
            "attachments": [
                {
                    "filename": "shot.png",
                    "content_type": "image/png",
                    "size": 200_000,
                    "in_sandbox_path": "/mnt/attachments/signal/evt-1-shot.png",
                }
            ],
        }
        events = [
            _evt(
                1,
                "user",
                content="eyeball this",
                metadata=md,
                focal_channel_at_arrival=self._CHAN_A,
            )
        ]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert isinstance(content, str)
        assert content.startswith(f"🔔 channel_id={self._CHAN_B}")
        assert "from=Bob" in content
        assert "[image: shot.png" in content
        assert "/mnt/attachments/signal/evt-1-shot.png" in content

    def test_notification_when_focal_null(self) -> None:
        """Phone-down state: all inbound renders as notifications."""
        md = {"channel": self._CHAN_B, "sender_name": "Bob"}
        events = [_evt(1, "user", content="hey", metadata=md, focal_channel_at_arrival=None)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert content.startswith(f"🔔 channel_id={self._CHAN_B}")

    def test_notification_omits_sender_when_absent(self) -> None:
        """No sender_name → no ``from=`` clause, just channel + preview
        on the header line.  Hint line follows unconditionally.
        """
        md = {"channel": self._CHAN_B}
        events = [
            _evt(1, "user", content="hey", metadata=md, focal_channel_at_arrival=self._CHAN_A)
        ]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "from=" not in content
        header, hint = content.split("\n", 1)
        assert header == f"🔔 channel_id={self._CHAN_B} · hey"
        assert hint.startswith("(to respond, call switch_channel(channel_id=")

    def test_notification_hint_names_the_channel_id(self) -> None:
        """The ``to respond...`` hint line includes the same channel_id
        as the marker, so weaker models can copy-paste it directly into
        a ``switch_channel`` call.
        """
        md = {"channel": self._CHAN_B, "sender_name": "Bob"}
        events = [
            _evt(1, "user", content="hey", metadata=md, focal_channel_at_arrival=self._CHAN_A)
        ]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert f"switch_channel(channel_id='{self._CHAN_B}')" in content

    def test_notification_truncation_at_80_chars(self) -> None:
        long = "x" * 200
        md = {"channel": self._CHAN_B, "sender_name": "Bob"}
        events = [_evt(1, "user", content=long, metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        # Exactly 80 x's followed by an ellipsis.
        assert "x" * 80 + "…" in content
        assert "x" * 81 not in content  # never more than 80 raw chars

    def test_notification_reaction_fallback_when_content_empty(self) -> None:
        md = {
            "channel": self._CHAN_B,
            "sender_name": "Bob",
            "reaction": {"emoji": "👍"},
        }
        events = [_evt(1, "user", content="", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "reacted 👍" in content

    def test_notification_telegram_reaction_shows_new_emoji(self) -> None:
        """Cross-channel Telegram reactions render via the notification
        marker; the preview must surface the emoji from ``new_emojis``,
        mirroring the Signal-shape ``reaction.emoji`` fallback. Pre-fix
        the model saw a bare ``🔔 channel_id=…`` with no hint anything
        was reacted."""
        md = {
            "channel": self._CHAN_B,
            "sender_name": "Bob",
            "reaction": {
                "target_message_id": 42,
                "old_emojis": [],
                "new_emojis": ["👍"],
            },
        }
        events = [_evt(1, "user", content="", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        content = build_messages(events, system_prompt=None).messages[0]["content"]
        assert "reacted 👍" in content

    def test_notification_metadata_stripped_from_wire_message(self) -> None:
        md = {"channel": self._CHAN_B, "sender_name": "Bob"}
        events = [_evt(1, "user", content="hi", metadata=md, focal_channel_at_arrival=self._CHAN_A)]
        msg = build_messages(events, system_prompt=None).messages[0]
        assert "metadata" not in msg
        assert set(msg.keys()) <= {"role", "content", "name"}

    def test_switch_channel_does_not_rewrite_past_events(self) -> None:
        """Monotonicity invariant: past events' rendering is pinned by
        their own stamped fields, regardless of subsequent focal changes.
        """
        md_b = {"channel": self._CHAN_B, "sender_name": "Bob"}
        # Event arrives on B while focal=A → notification at append time.
        ev_early = _evt(
            1, "user", content="early", metadata=md_b, focal_channel_at_arrival=self._CHAN_A
        )
        # Later event on A while focal=A (post-switch simulation) → full.
        ev_late = _evt(
            2,
            "user",
            content="later",
            metadata={"channel": self._CHAN_A, "sender_name": "Alice"},
            focal_channel_at_arrival=self._CHAN_A,
        )
        msgs = build_messages([ev_early, ev_late], system_prompt=None).messages
        assert msgs[0]["content"].startswith(f"🔔 channel_id={self._CHAN_B}")
        assert msgs[1]["content"].startswith(f"[channel={self._CHAN_A}")


def _has_dot_placeholder(msgs: list[dict[str, Any]]) -> bool:
    """True if any message is the old degenerate ``"."`` assistant separator."""
    return any(m == {"role": "assistant", "content": "."} for m in msgs)


class TestMergeAdjacentUserMessages:
    """``merge_adjacent_user_messages`` folds consecutive user-role turns
    into one, replacing the old ``"."`` placeholder-assistant separator
    (which literal-minded models imitated into silence)."""

    def test_merges_two_users_into_one(self) -> None:
        msgs = [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ]
        result = merge_adjacent_user_messages(msgs)
        assert result == [{"role": "user", "content": "one\n\ntwo"}]
        assert not _has_dot_placeholder(result)

    def test_preserves_existing_alternation(self) -> None:
        """A user/assistant/user sequence has no adjacency — no merge
        across the assistant turn."""
        msgs = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "two"},
        ]
        assert merge_adjacent_user_messages(msgs) == msgs
        assert not _has_dot_placeholder(merge_adjacent_user_messages(msgs))

    def test_tool_result_between_users_is_not_merged(self) -> None:
        """Adjacent means *consecutive same-role*. Tool results break the run."""
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_call_id": "a", "content": "r"},
            {"role": "user", "content": "two"},
        ]
        assert merge_adjacent_user_messages(msgs) == msgs

    def test_three_consecutive_users_merge_to_one(self) -> None:
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        result = merge_adjacent_user_messages(msgs)
        assert result == [{"role": "user", "content": "a\n\nb\n\nc"}]
        assert not _has_dot_placeholder(result)

    def test_list_content_users_concatenate_blocks(self) -> None:
        """Vision/multi-part list contents concatenate block-wise."""
        a_blocks = [{"type": "text", "text": "first"}]
        b_blocks = [
            {"type": "text", "text": "second"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": a_blocks},
            {"role": "user", "content": b_blocks},
        ]
        result = merge_adjacent_user_messages(msgs)
        assert result == [{"role": "user", "content": [*a_blocks, *b_blocks]}]
        assert not _has_dot_placeholder(result)

    def test_string_plus_list_normalizes_to_block_list(self) -> None:
        """A string user followed by a list user normalises the string to
        a text block and concatenates."""
        list_blocks = [
            {"type": "text", "text": "see this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "look:"},
            {"role": "user", "content": list_blocks},
        ]
        result = merge_adjacent_user_messages(msgs)
        assert result == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "look:"}, *list_blocks],
            }
        ]
        assert not _has_dot_placeholder(result)

    def test_empty_input_returns_empty(self) -> None:
        assert merge_adjacent_user_messages([]) == []

    def test_system_then_user_then_user_merges_only_users(self) -> None:
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ]
        result = merge_adjacent_user_messages(msgs)
        assert result == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "one\n\ntwo"},
        ]
        assert not _has_dot_placeholder(result)

    def test_does_not_mutate_input_messages(self) -> None:
        """Merge produces new dicts; the caller's input list is untouched."""
        original = [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ]
        snapshot = [dict(m) for m in original]
        merge_adjacent_user_messages(original)
        assert original == snapshot


class TestConcatUserMessages:
    """``_concat_user_messages`` is the block-aware joiner behind the merge."""

    def test_two_strings_join_with_blank_line(self) -> None:
        a = {"role": "user", "content": "alpha"}
        b = {"role": "user", "content": "beta"}
        assert _concat_user_messages(a, b) == {"role": "user", "content": "alpha\n\nbeta"}

    def test_two_lists_concatenate(self) -> None:
        a = {"role": "user", "content": [{"type": "text", "text": "x"}]}
        b = {"role": "user", "content": [{"type": "text", "text": "y"}]}
        assert _concat_user_messages(a, b) == {
            "role": "user",
            "content": [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}],
        }

    def test_string_then_list_normalizes_string(self) -> None:
        a = {"role": "user", "content": "hi"}
        b = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}
        assert _concat_user_messages(a, b) == {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "u"}},
            ],
        }

    def test_ephemeral_marker_sticky_under_or(self) -> None:
        """#1535: the ephemeral-tail marker propagates under OR — a merge of
        any ephemeral message with anything is ephemeral (a dict containing
        *any* per-step-mutating content cannot host the stable-prefix cache
        breakpoint). Covers the trailing real-inbound + obligations case."""
        inbound = {"role": "user", "content": "real peer text"}
        ephemeral = {"role": "user", "content": "tail", EPHEMERAL_TAIL_KEY: True}
        # ephemeral on either side -> merged dict is ephemeral.
        assert _concat_user_messages(inbound, ephemeral).get(EPHEMERAL_TAIL_KEY) is True
        assert _concat_user_messages(ephemeral, inbound).get(EPHEMERAL_TAIL_KEY) is True
        # both ephemeral -> still ephemeral.
        assert _concat_user_messages(ephemeral, ephemeral).get(EPHEMERAL_TAIL_KEY) is True

    def test_no_marker_when_neither_ephemeral(self) -> None:
        """Plain inbound + plain inbound stays unmarked - no spurious marker."""
        a = {"role": "user", "content": "one"}
        b = {"role": "user", "content": "two"}
        assert EPHEMERAL_TAIL_KEY not in _concat_user_messages(a, b)


class TestIsDegenerateEmptyAssistant:
    """``_is_degenerate_empty_assistant`` flags assistant turns carrying no
    content, no tool_calls, and no thinking — the non-events that
    ``build_messages`` drops to break the empty-turn imitation loop."""

    def test_empty_string_no_tools_no_thinking_is_degenerate(self) -> None:
        assert _is_degenerate_empty_assistant({"role": "assistant", "content": ""}) is True

    def test_whitespace_only_content_is_degenerate(self) -> None:
        assert _is_degenerate_empty_assistant({"role": "assistant", "content": "  \n\t"}) is True

    def test_content_none_is_degenerate(self) -> None:
        assert _is_degenerate_empty_assistant({"role": "assistant", "content": None}) is True

    def test_missing_content_is_degenerate(self) -> None:
        assert _is_degenerate_empty_assistant({"role": "assistant"}) is True

    def test_text_content_is_not_degenerate(self) -> None:
        assert _is_degenerate_empty_assistant({"role": "assistant", "content": "hi"}) is False

    def test_tool_calls_present_is_not_degenerate(self) -> None:
        msg = {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]}
        assert _is_degenerate_empty_assistant(msg) is False

    def test_thinking_blocks_present_is_not_degenerate(self) -> None:
        msg = {
            "role": "assistant",
            "content": "",
            "thinking_blocks": [{"type": "thinking", "thinking": "hmm"}],
        }
        assert _is_degenerate_empty_assistant(msg) is False

    def test_list_content_with_nonempty_text_block_is_not_degenerate(self) -> None:
        msg = {"role": "assistant", "content": [{"type": "text", "text": "real"}]}
        assert _is_degenerate_empty_assistant(msg) is False

    def test_list_content_with_only_empty_text_blocks_is_degenerate(self) -> None:
        msg = {"role": "assistant", "content": [{"type": "text", "text": "  "}]}
        assert _is_degenerate_empty_assistant(msg) is True

    def test_non_assistant_role_is_never_degenerate(self) -> None:
        assert _is_degenerate_empty_assistant({"role": "user", "content": ""}) is False
        assert _is_degenerate_empty_assistant({"role": "tool", "content": ""}) is False


class TestStubMissingReasoningContent:
    def test_adds_empty_stub_to_assistant_without_reasoning(self) -> None:
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        stub_missing_reasoning_content(msgs)
        assert msgs[1] == {"role": "assistant", "content": "hello", "reasoning_content": ""}

    def test_preserves_existing_reasoning_content(self) -> None:
        msgs = [
            {"role": "assistant", "content": "ok", "reasoning_content": "deep thoughts"},
        ]
        stub_missing_reasoning_content(msgs)
        assert msgs[0]["reasoning_content"] == "deep thoughts"

    def test_ignores_user_messages(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        stub_missing_reasoning_content(msgs)
        assert msgs[0] == {"role": "user", "content": "hi"}

    def test_ignores_tool_messages(self) -> None:
        msgs = [{"role": "tool", "tool_call_id": "x", "content": "result"}]
        stub_missing_reasoning_content(msgs)
        assert msgs[0] == {"role": "tool", "tool_call_id": "x", "content": "result"}

    def test_stubs_bare_dot_assistant(self) -> None:
        """Any assistant turn — including a stray single-byte one — must
        carry the empty reasoning stub so thinking-mode providers don't
        reject the transcript. (``merge_adjacent_user_messages`` no longer
        produces such a turn, but the stub must still cover one if present
        from any other source.)"""
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "."},
            {"role": "user", "content": "b"},
        ]
        stub_missing_reasoning_content(msgs)
        assert msgs[1] == {"role": "assistant", "content": ".", "reasoning_content": ""}

    def test_mutates_in_place_and_returns(self) -> None:
        msgs = [{"role": "assistant", "content": "x"}]
        returned = stub_missing_reasoning_content(msgs)
        assert returned is msgs

    def test_handles_tool_call_assistants(self) -> None:
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
        ]
        stub_missing_reasoning_content(msgs)
        assert msgs[0]["reasoning_content"] == ""
        assert msgs[0]["tool_calls"] == [{"id": "a"}]


class TestReasoningContentStubGate:
    """End-to-end checks that the ``reasoning_content`` stub pass is gated on
    the target's ``supports_thinking`` verdict — the same axis ``build_messages``
    uses to strip/keep reasoning fields. On a non-thinking target the strip
    pass removes ``reasoning_content`` and the (gated-off) stub pass must NOT
    re-add it; on a thinking target the stub fills the missing field."""

    def test_non_thinking_target_no_reasoning_content_stub(self) -> None:
        # Assistant turn lacks reasoning_content; non-thinking target must
        # ship it WITHOUT a synthesized empty stub. Fails on master, where
        # the unconditional stub adds ``reasoning_content: ""``.
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hey"),
        ]
        msgs = _full_pipeline_for_model(events, "openai/gpt-4o-mini")
        assistant = [m for m in msgs if m.get("role") == "assistant"]
        assert assistant, "expected at least one assistant turn"
        for msg in assistant:
            assert "reasoning_content" not in msg

    def test_thinking_target_gets_reasoning_content_stub(self) -> None:
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hey"),
        ]
        msgs = _full_pipeline_for_model(events, "anthropic/claude-haiku-4-5")
        assistant = [m for m in msgs if m.get("role") == "assistant"]
        assert assistant, "expected at least one assistant turn"
        for msg in assistant:
            assert msg["reasoning_content"] == ""

    def test_thinking_target_preserves_real_reasoning_content(self) -> None:
        # A thinking target whose assistant turn already carries
        # reasoning_content keeps it verbatim (strip preserves it, stub
        # skips already-set fields).
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hey"),
        ]
        events[1].data["reasoning_content"] = "deep thoughts about hi"
        msgs = _full_pipeline_for_model(events, "anthropic/claude-haiku-4-5")
        assistant = [m for m in msgs if m.get("role") == "assistant"]
        assert assistant[0]["reasoning_content"] == "deep thoughts about hi"

    def test_gate_seam_capability_verdicts(self) -> None:
        # The seam the call site gates on: non-thinking false, thinking true.
        from aios.harness.completion import model_descriptor

        assert model_descriptor("openai/gpt-4o-mini").supports_thinking is False
        assert model_descriptor("anthropic/claude-haiku-4-5").supports_thinking is True


class TestMergeAdjacentUserMessagesPipeline:
    """Exercise the merge against realistic ``build_messages`` output
    (rather than synthetic dicts) so a refactor that changes the output's
    role sequence can't silently break the fix."""

    def test_inbound_then_tail_block_merge_into_one_user(self) -> None:
        """An inbound followed by the user-role channels tail block is the
        canonical adjacency: the two fold into a single user turn, and no
        degenerate ``"."`` assistant placeholder is produced."""
        events = [_evt(1, "user", content="hello")]
        msgs = _full_pipeline(events, ["signal/test/1"])

        assert [m["role"] for m in msgs] == ["user"]
        assert not any(m == {"role": "assistant", "content": "."} for m in msgs)
        merged = msgs[0]["content"]
        assert merged.startswith(f"[received={RECEIVED}]\nhello")
        assert "\n\n━━━ Channels ━━━" in merged

    def test_blind_spot_injection_adjacent_user_merges(self) -> None:
        """``build_messages`` inlines a blind-spot tool result as a
        synthetic user message right after the horizon-setter.  When a
        real user event follows, the two land back-to-back and merge into
        one user turn (was: separated by a ``"."`` placeholder)."""
        events = [
            _evt(1, "user", content="run it"),
            _evt(2, "assistant", tool_calls=[_tc("t1")]),
            _evt(3, "tool", tool_call_id="t1", content="RESULT"),
            _evt(4, "assistant", content="checking..."),
            _evt(5, "user", content="anything else?"),
            _evt(6, "assistant", content="nope"),
        ]
        events[1].data["reacting_to"] = 1
        events[3].data["reacting_to"] = 1  # blind to tool at seq=3
        events[5].data["reacting_to"] = 5

        msgs = _full_pipeline(events, channels=[])

        assert not any(m == {"role": "assistant", "content": "."} for m in msgs)
        merged = next(
            m for m in msgs if m["role"] == "user" and "RESULT" in str(m.get("content", ""))
        )
        # The injected blind-spot result and the following real inbound are
        # one user turn now.
        assert "anything else?" in merged["content"]
        # The original inbound stays its own (earlier) user turn.
        assert merged["content"].count("RESULT") == 1

    def test_alternating_events_no_tail_block_no_merge(self) -> None:
        """No adjacency (empty bindings → tail block is ``None``) → the
        role sequence is untouched and no placeholder is produced."""
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hello"),
            _evt(3, "user", content="bye"),
            _evt(4, "assistant", content="later"),
        ]
        msgs = _full_pipeline(events, channels=[])

        assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
        assert not any(m == {"role": "assistant", "content": "."} for m in msgs)


class TestEventDataImmutability:
    """``build_messages`` is documented (``context.py:9-12``) as a pure
    function of the event log. Callers that hold an ``Event`` reference
    must see its ``data`` unchanged after the build runs.

    Pre-fix, ``_correct_image_data_url_mimes`` mutated ``image_url["url"]``
    in place on dicts that shared identity with ``event.data["content"]``
    (because ``build_messages`` appended ``e.data`` to its ``messages``
    list directly, before the ``_strip_to_spec`` copy boundary). So a
    tool-role event with a mis-declared image MIME got its source-of-truth
    rewritten on every render.
    """

    def test_build_messages_does_not_mutate_tool_event_data(self) -> None:
        import base64
        import copy

        # JPEG magic bytes (\xff\xd8\xff\xe0) declared as image/png — the
        # historical-event corruption pattern that motivated
        # ``_correct_image_data_url_mimes`` in the first place.
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 32
        bad_url = f"data:image/png;base64,{base64.b64encode(jpeg_bytes).decode()}"

        events = [
            _evt(1, "user", content="show"),
            _evt(2, "assistant", tool_calls=[_tc("a", name="read")]),
            Event(
                id="evt_3",
                session_id="sess_01TEST",
                seq=3,
                kind="message",
                data={
                    "role": "tool",
                    "tool_call_id": "a",
                    "name": "read",
                    "content": [{"type": "image_url", "image_url": {"url": bad_url}}],
                },
                created_at=datetime.now(tz=UTC),
                orig_channel=None,
                focal_channel_at_arrival=None,
            ),
        ]
        snapshot = copy.deepcopy([e.data for e in events])

        ctx = build_messages(events, system_prompt=None)

        # The build's output is allowed to contain the corrected MIME —
        # but the input events must remain pristine.
        assert [e.data for e in events] == snapshot, (
            "build_messages mutated event.data — the tool event's image_url URL "
            "was rewritten in place, violating the 'pure function of the event "
            "log' contract documented in context.py:9-12"
        )
        # Sanity: the correction did happen in the output messages (so the
        # test isn't passing because correction was skipped).
        tool_msg = next(m for m in ctx.messages if m.get("role") == "tool")
        assert isinstance(tool_msg["content"], list)
        out_url = tool_msg["content"][0]["image_url"]["url"]
        assert out_url.startswith("data:image/jpeg;base64,"), (
            f"renderer should have corrected png → jpeg in the output, got {out_url[:50]}"
        )


class TestPoisonEventQuarantine:
    """``build_messages`` is a pure replay over the immutable event log,
    run on EVERY wake. A single event that makes the renderer raise would
    permanently brick the session — the model is never called, so the
    "model sees the error and retries" recovery never engages (#686).

    The quarantine backstop catches any per-event render failure and
    degrades THAT ONE event to a deterministic placeholder (a function of
    ``e.seq`` only, preserving the monotonicity invariant) while every
    other event in the window still renders. The failure is signalled via
    the ``context.poison_event_quarantined`` structlog event.

    The pre-existing inner ``isinstance`` / ``OSError`` guards take
    precedence — those shapes render normally, NOT via quarantine. The
    outer quarantine is the last-resort backstop for novel raisers the
    inner guards don't cover.
    """

    def test_out_of_range_created_at_is_quarantined(self) -> None:
        """The GENUINE current brick: a ``created_at`` so small that
        ``_format_received``'s ``astimezone`` raises ``OverflowError``
        (``date value out of range``). No inner guard covers it, so it
        must degrade to the deterministic placeholder rather than propagate.
        """
        e = _evt(7, "user", content="hi", created_at=datetime(MINYEAR, 1, 1, tzinfo=UTC))
        ctx = build_messages([e], system_prompt=None, tz_name="America/Los_Angeles")
        assert ctx.messages == [_quarantine_placeholder(7)]
        assert ctx.messages[0]["role"] == "user"
        assert ctx.messages[0]["content"] == "[unrenderable event seq=7 — quarantined]"
        # The placeholder occupies seq 7's position; reacting_to must move
        # past it so find_sessions_needing_inference doesn't treat the
        # poison event as perpetually-new.
        assert ctx.reacting_to == 7

    def test_non_dict_attachment_renders_normally_not_quarantined(self) -> None:
        """The inner ``context.attachment_record_not_dict`` guard pre-empts
        the outer quarantine: non-dict attachment records are skipped
        in-place, so the event renders as a plain string with no
        ``[unrenderable`` marker."""
        md = {"channel": "echo/a/c", "attachments": ["oops", None, 42]}
        e = _evt(1, "user", content="hi", metadata=md, focal_channel_at_arrival="echo/a/c")
        ctx = build_messages([e], system_prompt=None, model="gpt-4o", session_id="s")
        content = ctx.messages[0]["content"]
        assert isinstance(content, str)
        assert "[unrenderable" not in content
        assert "hi" in content

    def test_render_user_event_raise_is_quarantined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A synthetic novel raiser inside ``render_user_event`` — the kind
        of new event shape that today would brick the session — degrades to
        the placeholder."""

        def _boom(*a: Any, **k: Any) -> dict[str, Any]:
            raise RuntimeError("boom")

        monkeypatch.setattr("aios.harness.context.render_user_event", _boom)
        ctx = build_messages([_evt(3, "user", content="x")], system_prompt=None)
        assert ctx.messages == [_quarantine_placeholder(3)]
        assert ctx.reacting_to == 3

    def test_quarantine_emits_structlog_signal(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Quarantines are observable: a ``context.poison_event_quarantined``
        warning carrying ``session_id``, ``seq``, and ``error_type``.

        The conftest routes structlog through stdlib's ConsoleRenderer, which
        flattens the event + bound fields into ``record.getMessage()`` rather
        than preserving them as record attributes — so we substring-match the
        rendered line (same convention as test_attachment_staging.py)."""

        def _boom(*a: Any, **k: Any) -> dict[str, Any]:
            raise RuntimeError("boom")

        monkeypatch.setattr("aios.harness.context.render_user_event", _boom)
        with caplog.at_level(logging.WARNING):
            build_messages(
                [_evt(3, "user", content="x")], system_prompt=None, session_id="sess_01TEST"
            )
        rec = next(
            r for r in caplog.records if "context.poison_event_quarantined" in r.getMessage()
        )
        rendered = rec.getMessage()
        assert "seq=3" in rendered
        assert "session_id=sess_01TEST" in rendered
        assert "error_type=RuntimeError" in rendered

    def test_only_poison_event_quarantined_others_render(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One poison event among good events: exactly its position is
        quarantined; the others (including assistant turns) render normally."""
        original = render_user_event

        def _selective(event_data: dict[str, Any], *a: Any, **k: Any) -> dict[str, Any]:
            if event_data.get("content") == "gamma":
                raise RuntimeError("gamma is poison")
            return original(event_data, *a, **k)

        monkeypatch.setattr("aios.harness.context.render_user_event", _selective)
        events = [
            _evt(1, "user", content="alpha"),
            _evt(2, "assistant", content="beta"),
            _evt(3, "user", content="gamma"),
            _evt(4, "assistant", content="delta"),
        ]
        ctx = build_messages(events, system_prompt=None)
        assert ctx.messages[0]["content"] == f"[received={RECEIVED}]\nalpha"
        assert ctx.messages[1]["role"] == "assistant"
        assert ctx.messages[1]["content"] == "beta"
        assert ctx.messages[2] == _quarantine_placeholder(3)
        assert ctx.messages[3]["role"] == "assistant"
        assert ctx.messages[3]["content"] == "delta"
        assert ctx.reacting_to == 3

    @pytest.mark.parametrize(
        "metadata",
        [
            {"channel": "c", "timestamp_ms": "not-an-int"},
            {"channel": "c", "timestamp_ms": 10**30},
            {"channel": "c", "mentions": "not-a-list"},
            {"channel": "c", "mentions": [None, 42, {"uuid": 5}, {"name": []}]},
            {"channel": "c", "reaction": "not-a-dict"},
            {"channel": "c", "reaction": {"emoji": {"nested": "obj"}}},
            {"channel": "c", "reaction": {"new_emojis": "not-a-list"}},
            {"channel": "c", "reply_to": {"text": {"blocks": [1, 2]}}},
            {"channel": "c", "reply_to": "not-a-dict"},
            {"channel": "c", "request": {"output_schema": {1: object()}}},
            {"channel": "c", "attachments": "not-a-list"},
            {"channel": "c", "attachments": [{"in_sandbox_path": 123}]},
            {"channel": 12345},
            {},
        ],
    )
    def test_build_messages_never_raises_on_adversarial_metadata(
        self, metadata: dict[str, Any]
    ) -> None:
        """Hand-rolled fuzz over adversarial metadata shapes. For each,
        ``build_messages`` must NOT raise, and the single resulting message
        is EITHER a normally-rendered user message OR the quarantine
        placeholder (union invariant — passes regardless of which path each
        shape takes). Most shapes are absorbed by inner ``isinstance``
        guards and render normally."""
        focal = metadata.get("channel") if isinstance(metadata, dict) else None
        focal = focal if isinstance(focal, str) else None
        e = _evt(
            1,
            "user",
            content="x",
            metadata=metadata,
            focal_channel_at_arrival=focal,
        )
        ctx = build_messages([e], system_prompt=None, model="gpt-4o", session_id="s")
        assert len(ctx.messages) == 1
        msg = ctx.messages[0]
        assert msg["role"] == "user"
        content = msg["content"]
        assert isinstance(content, (str, list))
        # A normal render must never accidentally produce a quarantine-shaped
        # message; the marker string appears iff this is the exact placeholder.
        if isinstance(content, str) and "unrenderable event seq=" in content:
            assert msg == _quarantine_placeholder(1)

    def test_unserializable_output_schema_reaches_outer_quarantine(self) -> None:
        """The one shape that genuinely reaches the OUTER quarantine: a
        request whose ``output_schema`` is un-JSON-serializable. The
        ``json.dumps(output_schema)`` line only executes when a string
        ``request_id`` is also present (the surrounding guard), so the
        adversarial fuzz shape #10 — which omits ``request_id`` — renders
        normally; this companion case includes it to exercise the outer
        backstop."""
        md = {
            "channel": "c",
            "request": {"request_id": "req_1", "output_schema": {1: object()}},
        }
        e = _evt(1, "user", content="x", metadata=md, focal_channel_at_arrival="c")
        ctx = build_messages([e], system_prompt=None, model="gpt-4o", session_id="s")
        assert ctx.messages == [_quarantine_placeholder(1)]
        assert ctx.reacting_to == 1

    def test_quarantined_assistant_event_is_atomic_single_message(self) -> None:
        """The assistant branch appends ``e.data`` BEFORE iterating
        ``tool_calls``; a raise mid-branch (here a truthy non-iterable
        ``tool_calls`` so ``for tc in 42`` raises ``TypeError`` after the
        append) must NOT leave both the orphan assistant turn AND the
        placeholder — that invalid chat-completions sequence (tool_calls not
        followed by results) re-bricks the session. The rollback makes the
        quarantine atomic: exactly the placeholder remains."""
        e = Event(
            id="evt_5",
            session_id="sess_01TEST",
            seq=5,
            kind="message",
            data={"role": "assistant", "content": "x", "tool_calls": 42},
            created_at=_FIXED_CREATED_AT,
            orig_channel=None,
            focal_channel_at_arrival=None,
        )
        ctx = build_messages([e], system_prompt=None)
        assert len(ctx.messages) == 1
        assert ctx.messages[0] == _quarantine_placeholder(5)


# ─── omission marker (#738) ──────────────────────────────────────────────────


_BEGAN_AT = datetime(2025, 11, 5, 14, 30, tzinfo=UTC)

_MARKER_TAIL = (
    "Nothing is lost: the full transcript remains queryable with search_events. "
    "What seems unfamiliar is usually forgotten, not new: when in doubt about "
    "anything that's referred to, search first rather than fill the gap by "
    "assumption.]"
)


class TestOmissionMarker:
    """Head marker rendered when the window omits transcript (#738).

    The load-bearing property is prompt-cache stability: the marker is a
    pure function of (omission, boundary event, tz) — byte-identical
    across rebuilds while the drop boundary is unchanged, changing only
    when the boundary moves (a snap, when the head changes anyway).
    """

    def _omission(self, *, omitted_messages: int = 9_837) -> WindowOmission:
        return WindowOmission(began_at=_BEGAN_AT, omitted_messages=omitted_messages)

    def test_exact_marker_text_and_placement(self) -> None:
        events = [_evt(50, "user", content="hi")]
        ctx = build_messages(events, system_prompt="SYS", omission=self._omission())
        assert ctx.messages[0] == {"role": "system", "content": "SYS"}
        assert ctx.messages[1] == {
            "role": "user",
            "content": (
                f"[history: this conversation began 2025-11-05. "
                f"Everything before {RECEIVED} — about 9,800 messages, "
                f"including your own — has scrolled out of view. " + _MARKER_TAIL
            ),
        }

    def test_no_omission_no_marker(self) -> None:
        events = [_evt(1, "user", content="hi")]
        ctx = build_messages(events, system_prompt="SYS")
        assert ctx.messages == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": f"[received={RECEIVED}]\nhi"},
        ]

    def test_marker_first_when_no_system_prompt(self) -> None:
        events = [_evt(50, "user", content="hi")]
        ctx = build_messages(events, system_prompt=None, omission=self._omission())
        assert ctx.messages[0]["content"].startswith("[history: ")

    def test_byte_stable_while_context_grows_within_chunk(self) -> None:
        """Appending tail events with an unchanged boundary must not perturb
        a single byte of the marker — the prompt-cache invariant."""
        omission = self._omission()
        head = [_evt(50, "user", content="hi"), _evt(51, "assistant", content="yo")]
        grown = [*head, _evt(52, "user", content="more"), _evt(53, "assistant", content="ok")]
        ctx_before = build_messages(head, system_prompt="SYS", omission=omission)
        ctx_after = build_messages(grown, system_prompt="SYS", omission=omission)
        _assert_prefix(ctx_before.messages, ctx_after.messages)

    def test_marker_changes_when_boundary_moves(self) -> None:
        """After a snap the head event differs → the marker re-renders with
        the new boundary timestamp (and changes exactly then, with the rest
        of the head)."""
        omission = self._omission()
        before_snap = [_evt(50, "user", content="a")]
        after_snap = [
            _evt(
                150,
                "user",
                content="b",
                created_at=datetime(2026, 1, 9, 12, 0, tzinfo=UTC),
            )
        ]
        m1 = build_messages(before_snap, system_prompt=None, omission=omission).messages[0]
        m2 = build_messages(after_snap, system_prompt=None, omission=omission).messages[0]
        assert "2026-01-02T03:04:05+00:00" in m1["content"]
        assert "2026-01-09T12:00:00+00:00" in m2["content"]

    def test_zero_message_count_drops_count_clause(self) -> None:
        """An omitted span holding only tool results renders without the
        'about N messages' clause rather than claiming 'about 0 messages'."""
        events = [_evt(50, "user", content="hi")]
        ctx = build_messages(
            events, system_prompt=None, omission=self._omission(omitted_messages=0)
        )
        content = ctx.messages[0]["content"]
        assert f"Everything before {RECEIVED} has scrolled out of view." in content
        assert "— about" not in content
        assert "including your own" not in content

    def test_account_timezone_applies_to_both_timestamps(self) -> None:
        """Boundary uses _format_received in the account tz; the began date
        is the account-tz calendar date (here UTC 2025-11-06 04:00 is still
        Nov 5 in Los Angeles)."""
        omission = WindowOmission(
            began_at=datetime(2025, 11, 6, 4, 0, tzinfo=UTC), omitted_messages=3
        )
        events = [_evt(50, "user", content="hi")]
        ctx = build_messages(
            events, system_prompt=None, omission=omission, tz_name="America/Los_Angeles"
        )
        content = ctx.messages[0]["content"]
        assert "began 2025-11-05." in content
        # November → PST (-08:00).
        assert "Everything before 2026-01-01T19:04:05-08:00 (America/Los_Angeles)" in content

    def test_upper_bound_covers_worst_case_render(self) -> None:
        """The window budget reserves OMISSION_MARKER_UPPER_BOUND_LOCAL for
        the marker (PR #165 full-payload invariant); a worst-case render —
        eight-digit count, longest-form IANA zone — must fit under it.
        Fails when someone fattens the marker text without bumping the
        reserve."""
        from aios.harness.context import OMISSION_MARKER_UPPER_BOUND_LOCAL
        from aios.harness.tokens import approx_tokens

        omission = WindowOmission(
            began_at=datetime(2025, 11, 5, 14, 30, tzinfo=UTC),
            omitted_messages=98_765_432,
        )
        marker = build_messages(
            [_evt(50, "user", content="x")],
            system_prompt=None,
            omission=omission,
            tz_name="America/Argentina/ComodRivadavia",
        ).messages[0]
        assert approx_tokens([marker]) <= OMISSION_MARKER_UPPER_BOUND_LOCAL

    def test_marker_merges_with_leading_user_inbound(self) -> None:
        """On the wire the user-role marker merges with a leading user
        inbound (Anthropic requires alternating roles); both parts are
        byte-stable so the merged turn is too."""
        events = [_evt(50, "user", content="hi")]
        ctx = build_messages(events, system_prompt=None, omission=self._omission())
        merged = merge_adjacent_user_messages(ctx.messages)
        assert len(merged) == 1
        content = merged[0]["content"]
        assert content.startswith("[history: ")
        assert content.endswith(f"[received={RECEIVED}]\nhi")


class TestApproxCount:
    """Floor-to-two-significant-figures rendering for the omission marker."""

    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (1, "1"),
            (7, "7"),
            (97, "97"),
            (99, "99"),
            (100, "100"),
            (123, "120"),
            (9_837, "9,800"),
            (10_000, "10,000"),
            (1_234_567, "1,200,000"),
        ],
    )
    def test_rounding(self, n: int, expected: str) -> None:
        assert _approx_count(n) == expected


class TestThinkingGateShortCircuit:
    """The thinking gate in ``build_messages`` preserves the empty-model
    short-circuit and matches the prior inline verdict.
    """

    def test_empty_model_does_not_call_resolver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``bool(model) and …`` short-circuits for an empty model string,
        so ``model_descriptor`` is never called (and a falsy thinking verdict
        results)."""
        import aios.harness.completion as completion

        called = False
        real = completion.model_descriptor

        def _tracking(model: str):  # type: ignore[no-untyped-def]
            nonlocal called
            called = True
            return real(model)

        monkeypatch.setattr(completion, "model_descriptor", _tracking)

        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hey"),
        ]
        events[1].data["thinking_blocks"] = [{"type": "thinking", "thinking": "x"}]
        msgs = build_messages(events, system_prompt=None, model="").messages
        assert called is False
        # supports_thinking falsy -> thinking_blocks stripped
        assert "thinking_blocks" not in msgs[1]

    def test_openai_model_strips_thinking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``openai/gpt-4o`` is non-Claude; with litellm reporting no reasoning
        support the old inline verdict was False, so thinking blocks strip."""
        monkeypatch.setattr("litellm.supports_reasoning", lambda *a, **k: False)
        events = [
            _evt(1, "user", content="hi"),
            _evt(2, "assistant", content="hey"),
        ]
        events[1].data["thinking_blocks"] = [{"type": "thinking", "thinking": "x"}]
        msgs = build_messages(events, system_prompt=None, model="openai/gpt-4o").messages
        assert "thinking_blocks" not in msgs[1]


class TestLazyLitellmBoundary:
    """Importing ``aios.harness.context`` must NOT pull in ``litellm`` at
    module import — the function-local import of ``model_descriptor`` keeps
    the ~1.18s litellm bootstrap deferred."""

    def test_context_import_does_not_import_litellm(self) -> None:
        import subprocess
        import sys

        code = (
            "import sys\n"
            "import aios.harness.context  # noqa: F401\n"
            "assert 'litellm' not in sys.modules, sorted(m for m in sys.modules if 'litellm' in m)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
