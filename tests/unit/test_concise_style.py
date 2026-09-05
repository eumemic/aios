"""Concise output style: the two step-time injections and the nag reserve.

Mechanism and rationale live in ``aios.harness.concise``.
"""

from __future__ import annotations

import itertools
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from aios.harness.concise import (
    CONCISE_NAG_CONTENT,
    CONCISE_NAG_CONTENT_CHANNELS,
    CONCISE_NAG_DELIVERY_CLAUSE,
    CONCISE_NAG_UPPER_BOUND_LOCAL,
    CONCISE_STYLE_BLOCK,
)
from aios.harness.step_context import (
    StepPrelude,
    compute_step_prelude,
    prelude_overhead_local,
)
from aios.harness.tokens import approx_tokens
from aios.models.agents import (
    AgentCreate,
    AgentUpdate,
    StepSurface,
)
from aios.models.events import Event
from tests.unit.conftest import fake_pool_yielding_conn
from tests.unit.step_context_support import (
    ACCOUNT,
    SESSION,
    compose_with_stubs,
    make_prelude,
    make_step_surface,
    message_event,
)

_ACCOUNT = ACCOUNT
_SESSION = SESSION
_agent = make_step_surface
_prelude = make_prelude


def _evt(seq: int, *, role: str, content: str = "hi") -> Event:
    return message_event(seq, role, content)


async def _compose(
    agent: StepSurface, events: list[Event], *, channels: list[str] | None = None
) -> list[dict[str, Any]]:
    return (await compose_with_stubs(agent, events, channels=channels)).messages


def _text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def _nag_count(messages: list[dict[str, Any]]) -> int:
    """Occurrences of EITHER nag variant.

    The channel-attached variant (#2262) is not a superstring of the plain one
    — the delivery clause lands inside the ``<system-reminder>`` element — so
    counting only ``CONCISE_NAG_CONTENT`` would silently score a channel
    session as zero nags and turn every exactly-once assertion green-for-the-
    wrong-reason.
    """
    return sum(
        _text(m).count(CONCISE_NAG_CONTENT) + _text(m).count(CONCISE_NAG_CONTENT_CHANNELS)
        for m in messages
    )


# ── the nag in the composed payload ──────────────────────────────────────────


@pytest.mark.asyncio
class TestConciseNag:
    async def test_nag_is_final_content_after_trailing_user_inbound(self) -> None:
        """A trailing user inbound merges with the nag (adjacent user turns are
        merged by design); the nag is the final content, exactly once."""
        events = [_evt(1, role="user", content="please respond")]
        messages = await _compose(_agent(output_style="concise"), events)
        assert messages[-1]["role"] == "user"
        assert _text(messages[-1]).endswith(CONCISE_NAG_CONTENT)
        assert "please respond" in _text(messages[-1])
        assert _nag_count(messages) == 1
        # The merge ran after the append: no adjacent user turns survive.
        for a, b in itertools.pairwise(messages):
            assert not (a.get("role") == "user" and b.get("role") == "user")

    async def test_nag_stands_alone_after_assistant_turn(self) -> None:
        """No channels -> no channels listing, but the nag still renders, standing as
        its own final user message after the trailing assistant turn."""
        events = [_evt(1, role="user"), _evt(2, role="assistant", content="done")]
        messages = await _compose(_agent(output_style="concise"), events)
        assert messages[-1]["role"] == "user"
        assert _text(messages[-1]) == CONCISE_NAG_CONTENT
        assert messages[-2]["role"] == "assistant"
        assert _nag_count(messages) == 1

    async def test_nag_lands_after_channels_tail_block(self) -> None:
        """With a channels tail rendered (trailing assistant turn), the nag is
        appended after it — the merged final user turn ends with the nag."""
        events = [_evt(1, role="user"), _evt(2, role="assistant", content="done")]
        messages = await _compose(
            _agent(output_style="concise"), events, channels=["signal/+1/chat-a"]
        )
        last = messages[-1]
        assert last["role"] == "user"
        text = _text(last)
        assert "━━━ Channels ━━━" in text
        assert text.endswith(CONCISE_NAG_CONTENT_CHANNELS)
        assert _nag_count(messages) == 1

    async def test_nag_carries_delivery_clause_when_channels_bound(self) -> None:
        """#2262: for a channel-attached session the nag must also say that
        being brief is not licence to stop posting.

        The mute this guards against was a per-step pressure that survived an
        explicit in-band correction from the user, so the counter-pressure has
        to ride the nag — the one injection rebuilt at maximum recency every
        step — rather than the system prompt alone.
        """
        events = [_evt(1, role="user"), _evt(2, role="assistant", content="done")]
        messages = await _compose(
            _agent(output_style="concise"), events, channels=["telegram/8622/chat-a"]
        )
        text = _text(messages[-1])
        assert CONCISE_NAG_DELIVERY_CLAUSE.strip() in text
        # The clause lands INSIDE the reminder element, after the steering
        # sentence — not as a second stray <system-reminder>.
        assert text.count("<system-reminder>") == 1
        assert _nag_count(messages) == 1

    async def test_nag_omits_delivery_clause_without_channels(self) -> None:
        """No bound channels -> no connector to send through, so the clause
        would be noise; the plain variant renders verbatim."""
        events = [_evt(1, role="user"), _evt(2, role="assistant", content="done")]
        messages = await _compose(_agent(output_style="concise"), events, channels=[])
        text = _text(messages[-1])
        assert text == CONCISE_NAG_CONTENT
        assert CONCISE_NAG_DELIVERY_CLAUSE.strip() not in text

    async def test_no_nag_when_not_concise(self) -> None:
        for channels in ([], ["signal/+1/chat-a"]):
            for events in (
                [_evt(1, role="user", content="please respond")],
                [_evt(1, role="user"), _evt(2, role="assistant", content="done")],
            ):
                messages = await _compose(_agent(output_style="default"), events, channels=channels)
                assert _nag_count(messages) == 0


# ── the system-prompt rules block ────────────────────────────────────────────


@pytest.mark.asyncio
class TestConciseSystemPrompt:
    async def _prelude_for(self, agent: StepSurface) -> StepPrelude:
        with mock.patch("aios.db.queries.get_open_obligations", new=AsyncMock(return_value=[])):
            return await compute_step_prelude(
                fake_pool_yielding_conn(MagicMock()),
                _SESSION,
                account_id=_ACCOUNT,
                session=mock.Mock(id=_SESSION, parent_run_id=None),
                agent=agent,
                channels=[],
                memory_store_echoes=[],
            )

    async def test_system_prompt_contains_rules_block_iff_concise(self) -> None:
        prelude = await self._prelude_for(_agent(output_style="concise"))
        assert CONCISE_STYLE_BLOCK in prelude.system_prompt
        prelude = await self._prelude_for(_agent(output_style="default"))
        assert CONCISE_STYLE_BLOCK not in prelude.system_prompt

    async def test_override_clause_carves_out_delivery(self) -> None:
        """#2262: the precedence clause must claim STYLE only.

        ``augment_with_concise_style`` appends this block directly on top of
        ``build_focal_paradigm_block``, whose contract ("bare assistant text is
        NOT delivered to any channel") reads like style guidance but is the
        rule that makes the agent audible. An unscoped "these rules win"
        outranked it and muted a live channel-attached agent, so the carve-out
        is load-bearing, not decorative — this asserts it cannot be quietly
        re-broadened by a later edit.
        """
        assert "outranks these rules absolutely" in CONCISE_STYLE_BLOCK
        assert "LENGTH and SHAPE" in CONCISE_STYLE_BLOCK
        # The old unscoped sentence must not come back verbatim.
        assert (
            "Where these rules conflict with more general style guidance "
            "elsewhere in your instructions, these rules win."
        ) not in CONCISE_STYLE_BLOCK


# ── the windowing reserve ────────────────────────────────────────────────────


class TestConciseNagReserve:
    def test_upper_bound_covers_real_builder_output(self) -> None:
        """The hardcoded reserve must cover the message the builder actually
        emits -- costed on the REAL builder output, not a re-rolled inline dict.

        BOTH variants: the reserve is computed at windowing time from the agent
        alone, before the composer knows whether the session has channels, so
        the bound has to hold for the longer channel-attached nag (#2262).
        """
        for content in (CONCISE_NAG_CONTENT, CONCISE_NAG_CONTENT_CHANNELS):
            cost = approx_tokens([{"role": "user", "content": content}])
            assert cost <= CONCISE_NAG_UPPER_BOUND_LOCAL, (
                f"nag variant {content[:40]!r}… costs {cost} local tokens, "
                f"over the {CONCISE_NAG_UPPER_BOUND_LOCAL} reserve -- raise the "
                f"constant rather than shortening the clause"
            )

    def test_reserve_summed_unconditionally(self) -> None:
        """Reserved for every agent (the omission-marker idiom) -- the prelude
        carries no per-agent reserve field to forget."""
        assert prelude_overhead_local(_prelude()).reserves >= CONCISE_NAG_UPPER_BOUND_LOCAL


# ── the wire enum ────────────────────────────────────────────────────────────


class TestOutputStyleWire:
    def test_invalid_output_style_rejected(self) -> None:
        """An unknown style is rejected by the pydantic Literal — the single
        validation point (no DB CHECK constraint); FastAPI surfaces it as 422."""
        with pytest.raises(ValidationError):
            AgentCreate(name="a", model="m", output_style="shouty")
        with pytest.raises(ValidationError):
            AgentUpdate(version=1, output_style="shouty")
