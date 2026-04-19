"""Unit tests for the pre-inference triage gate.

Covers the two halves of the feature:

* :func:`aios.harness.triage.run_triage` — LiteLLM call + parse, with
  fail-open guarantees for broken gates.
* :func:`aios.harness.loop._has_new_user_stimulus` — the stimulus
  predicate that decides whether triage is eligible to run at all.

LiteLLM is patched at the module boundary (``aios.harness.triage.litellm``)
so these tests don't touch the network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.harness.loop import _has_new_user_stimulus
from aios.harness.triage import TriageVerdict, run_triage
from aios.models.agents import TriageConfig
from aios.models.events import Event


def _user(seq: int, content: str = "hi") -> Event:
    return Event(
        id=f"evt_{seq}",
        session_id="sess_01TEST",
        seq=seq,
        kind="message",
        data={"role": "user", "content": content},
        created_at=datetime.now(tz=UTC),
    )


def _assistant(seq: int, reacting_to: int) -> Event:
    return Event(
        id=f"evt_{seq}",
        session_id="sess_01TEST",
        seq=seq,
        kind="message",
        data={"role": "assistant", "content": "ok", "reacting_to": reacting_to},
        created_at=datetime.now(tz=UTC),
    )


def _triage(seq: int, reacting_to: int, decision: str = "ignore") -> Event:
    return Event(
        id=f"evt_{seq}",
        session_id="sess_01TEST",
        seq=seq,
        kind="lifecycle",
        data={
            "event": "triage_decision",
            "decision": decision,
            "reason": "test",
            "reacting_to": reacting_to,
        },
        created_at=datetime.now(tz=UTC),
    )


def _tool_result(seq: int, tool_call_id: str) -> Event:
    return Event(
        id=f"evt_{seq}",
        session_id="sess_01TEST",
        seq=seq,
        kind="message",
        data={"role": "tool", "tool_call_id": tool_call_id, "content": "done"},
        created_at=datetime.now(tz=UTC),
    )


def _mock_litellm_response(content: str) -> Any:
    """Shape a minimal object that matches what ``run_triage`` reads.

    LiteLLM returns a ``ModelResponse`` with ``.choices[0].message.content``;
    we build a structurally-compatible stand-in so we don't depend on
    LiteLLM internals in tests.
    """

    class _Msg:
        def __init__(self, c: str) -> None:
            self.content = c

    class _Choice:
        def __init__(self, c: str) -> None:
            self.message = _Msg(c)

    class _Resp:
        def __init__(self, c: str) -> None:
            self.choices = [_Choice(c)]

    return _Resp(content)


# ─── run_triage ─────────────────────────────────────────────────────────────


class TestRunTriage:
    async def test_valid_respond_verdict(self) -> None:
        config = TriageConfig(model="openrouter/fake", system="")
        messages = [{"role": "user", "content": "hey bot"}]
        fake = _mock_litellm_response('{"decision": "respond", "reason": "addressed"}')
        with patch("aios.harness.triage.litellm.acompletion", AsyncMock(return_value=fake)) as m:
            verdict = await run_triage(config=config, messages=messages)
        assert verdict == TriageVerdict(decision="respond", reason="addressed")
        # The agent persona shouldn't leak into the gate call — the
        # system message the gate sees is built from config.system plus
        # the default rubric, not whatever system_prompt the main flow
        # would use.
        call_kwargs = m.call_args.kwargs
        assert call_kwargs["model"] == "openrouter/fake"
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["stream"] is False

    async def test_valid_ignore_verdict(self) -> None:
        config = TriageConfig(model="openrouter/fake")
        fake = _mock_litellm_response('{"decision": "ignore", "reason": "noise"}')
        with patch("aios.harness.triage.litellm.acompletion", AsyncMock(return_value=fake)):
            verdict = await run_triage(config=config, messages=[])
        assert verdict.decision == "ignore"
        assert verdict.reason == "noise"

    async def test_invalid_json_fails_open_to_respond(self) -> None:
        """Fail-open: a broken gate returning unparseable text must not
        silence the agent. Going silent on gate bugs is worse than the
        occasional unneeded reply."""
        config = TriageConfig(model="openrouter/fake")
        fake = _mock_litellm_response("not json at all")
        with patch("aios.harness.triage.litellm.acompletion", AsyncMock(return_value=fake)):
            verdict = await run_triage(config=config, messages=[])
        assert verdict.decision == "respond"
        assert "invalid_json" in verdict.reason

    async def test_schema_mismatch_fails_open(self) -> None:
        config = TriageConfig(model="openrouter/fake")
        fake = _mock_litellm_response('{"decision": "maybe", "reason": "hmm"}')
        with patch("aios.harness.triage.litellm.acompletion", AsyncMock(return_value=fake)):
            verdict = await run_triage(config=config, messages=[])
        assert verdict.decision == "respond"
        assert "schema_mismatch" in verdict.reason

    async def test_provider_exception_fails_open(self) -> None:
        config = TriageConfig(model="openrouter/fake")
        with patch(
            "aios.harness.triage.litellm.acompletion",
            AsyncMock(side_effect=RuntimeError("503 from provider")),
        ):
            verdict = await run_triage(config=config, messages=[])
        assert verdict.decision == "respond"
        assert "gate_error" in verdict.reason

    async def test_empty_response_fails_open(self) -> None:
        config = TriageConfig(model="openrouter/fake")
        fake = _mock_litellm_response("")
        with patch("aios.harness.triage.litellm.acompletion", AsyncMock(return_value=fake)):
            verdict = await run_triage(config=config, messages=[])
        assert verdict.decision == "respond"
        assert "empty" in verdict.reason

    async def test_drops_main_system_message(self) -> None:
        """The agent's main system prompt must not reach the gate —
        otherwise the gate inherits the agent persona and its priors are
        skewed. The gate only sees its own rubric."""
        config = TriageConfig(model="openrouter/fake", system="be strict")
        messages = [
            {"role": "system", "content": "you are Bot, a helpful assistant"},
            {"role": "user", "content": "hey"},
        ]
        fake = _mock_litellm_response('{"decision": "ignore", "reason": "x"}')
        with patch("aios.harness.triage.litellm.acompletion", AsyncMock(return_value=fake)) as m:
            await run_triage(config=config, messages=messages)
        sent = m.call_args.kwargs["messages"]
        # Exactly one system message (the gate's), and it includes the
        # operator-supplied override.
        assert sum(1 for msg in sent if msg["role"] == "system") == 1
        assert "be strict" in sent[0]["content"]
        # The agent persona did NOT survive.
        assert "you are Bot" not in sent[0]["content"]


# ─── _has_new_user_stimulus ──────────────────────────────────────────────────


class TestHasNewUserStimulus:
    def test_fresh_user_returns_true(self) -> None:
        assert _has_new_user_stimulus([_user(1, "hi")]) is True

    def test_no_events_returns_false(self) -> None:
        assert _has_new_user_stimulus([]) is False

    def test_only_tool_result_returns_false(self) -> None:
        """Mid-chain tool completions must always proceed to inference —
        the gate is for new stimuli, not for continuing the agent's own
        multi-step work. Gating a tool-result would break tool chains."""
        events = [
            _user(1, "run task"),
            _assistant(2, reacting_to=1),
            _tool_result(3, "tc_a"),
        ]
        assert _has_new_user_stimulus(events) is False

    def test_user_after_triage_ignore_returns_true(self) -> None:
        events = [
            _user(1, "side chat"),
            _triage(2, reacting_to=1, decision="ignore"),
            _user(3, "hey bot"),
        ]
        assert _has_new_user_stimulus(events) is True

    def test_user_before_last_assistant_returns_false(self) -> None:
        """A user message already reacted to by a prior assistant must
        not re-trigger triage — the watermark has moved past it."""
        events = [
            _user(1, "hi"),
            _assistant(2, reacting_to=1),
        ]
        assert _has_new_user_stimulus(events) is False

    def test_user_during_tool_chain_returns_true(self) -> None:
        """If a user message interrupts a running tool chain, the next
        step should triage that message even though a tool result also
        arrived in the meantime."""
        events = [
            _user(1, "run task"),
            _assistant(2, reacting_to=1),
            _user(3, "actually wait"),
            _tool_result(4, "tc_a"),
        ]
        assert _has_new_user_stimulus(events) is True


# ─── TriageConfig model ──────────────────────────────────────────────────────


class TestTriageConfigModel:
    def test_defaults(self) -> None:
        c = TriageConfig(model="ollama_chat/llama3.2:1b")
        assert c.model == "ollama_chat/llama3.2:1b"
        assert c.system == ""

    def test_rejects_empty_model(self) -> None:
        with pytest.raises(ValueError):
            TriageConfig(model="")

    def test_rejects_unknown_fields(self) -> None:
        """``extra='forbid'`` prevents typos from silently shipping —
        e.g. ``prompt=...`` instead of ``system=...``."""
        with pytest.raises(ValueError):
            TriageConfig.model_validate({"model": "x", "prompt": "wrong field"})
