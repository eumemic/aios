"""Concise output style: the two step-time injections and the nag reserve.

Mechanism and rationale live in ``aios.harness.concise``.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from aios.harness.concise import (
    CONCISE_NAG_CONTENT,
    CONCISE_NAG_UPPER_BOUND_LOCAL,
    CONCISE_STYLE_BLOCK,
    build_concise_nag_message,
)
from aios.harness.context import EPHEMERAL_TAIL_KEY
from aios.harness.step_context import (
    StepPrelude,
    compose_step_context,
    compute_step_prelude,
    prelude_overhead_local,
)
from aios.harness.tokens import approx_tokens
from aios.models.agents import (
    AgentBinding,
    AgentCreate,
    AgentUpdate,
    OutputStyle,
    StepSurface,
)
from aios.models.events import Event
from tests.unit.conftest import fake_pool_yielding_conn

_ACCOUNT = "acc_concise"
_SESSION = "sess_concise"


def _agent(*, output_style: OutputStyle = "default") -> StepSurface:
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
        binding=AgentBinding(agent_id="agt_concise", version=1),
    )


def _evt(seq: int, *, role: str, content: str = "hi") -> Event:
    return Event(
        id=f"evt_{seq:04d}",
        session_id=_SESSION,
        seq=seq,
        kind="message",
        data={"role": role, "content": content},
        cumulative_tokens=None,
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
    )


def _prelude(*, system_prompt: str = "sys") -> StepPrelude:
    return StepPrelude(
        system_prompt=system_prompt,
        tools=[],
        skill_versions=[],
        tail_block_upper_bound_local=0,
        obligations=[],
        obligations_block_upper_bound_local=0,
    )


class _Session:
    id = _SESSION
    focal_channel = None


async def _compose(
    agent: StepSurface, events: list[Event], *, channels: list[str] | None = None
) -> list[dict[str, Any]]:
    with (
        mock.patch(
            "aios.services.sessions.load_session_workspace_path",
            new=AsyncMock(return_value=None),
        ),
        mock.patch(
            "aios.services.accounts.resolve_effective_timezone",
            new=AsyncMock(return_value="UTC"),
        ),
    ):
        step_ctx = await compose_step_context(
            pool=MagicMock(),
            session=_Session(),  # type: ignore[arg-type]
            account_id=_ACCOUNT,
            agent=agent,
            channels=channels or [],
            prelude=_prelude(),
            events=events,
            persist_image_rewrites=False,
        )
    return step_ctx.messages


def _text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def _nag_count(messages: list[dict[str, Any]]) -> int:
    return sum(_text(m).count(CONCISE_NAG_CONTENT) for m in messages)


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
        """No channels -> no tail block, but the nag still renders, standing as
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
        assert text.endswith(CONCISE_NAG_CONTENT)
        assert _nag_count(messages) == 1

    async def test_nag_carries_ephemeral_tail_marker(self) -> None:
        """The nag (and any merge containing it) must never host the
        stable-prefix cache breakpoint — its position is per-step."""
        assert build_concise_nag_message()[EPHEMERAL_TAIL_KEY] is True
        events = [_evt(1, role="user", content="please respond")]
        messages = await _compose(_agent(output_style="concise"), events)
        assert messages[-1].get(EPHEMERAL_TAIL_KEY) is True

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


# ── the windowing reserve ────────────────────────────────────────────────────


class TestConciseNagReserve:
    def test_upper_bound_covers_real_builder_output(self) -> None:
        """The hardcoded reserve must cover the message the builder actually
        emits -- costed on the REAL builder output, not a re-rolled inline dict."""
        assert approx_tokens([build_concise_nag_message()]) <= CONCISE_NAG_UPPER_BOUND_LOCAL

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
