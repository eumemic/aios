"""Pre-inference triage gate.

When an agent is configured with :class:`~aios.models.agents.TriageConfig`,
:func:`run_triage` is invoked once per step before the main model call to
decide whether the agent should respond to the current stimulus. The gate
is intended for group-chat scenarios where most inbound messages are not
addressed to the agent.

The gate runs a single non-streaming LiteLLM call with a JSON-object
response format. The returned verdict has shape::

    {"decision": "respond" | "ignore", "reason": "<short free text>"}

Failures are **fail-open**: if the gate model errors, times out, or
returns unparseable output, the verdict defaults to ``"respond"``. Making
the agent go silent on a broken gate is worse than the occasional
unnecessary reply.

The gate call is ephemeral — no tool list, no streaming, no SSE
delivery, no event logging from this module. The caller
(:mod:`aios.harness.loop`) is responsible for persisting the resulting
``triage_decision`` lifecycle event with its ``reacting_to`` watermark.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import litellm
from pydantic import BaseModel, Field, ValidationError

from aios.logging import get_logger
from aios.models.agents import TriageConfig

log = get_logger("aios.harness.triage")

TriageDecision = Literal["respond", "ignore"]

# The triage call is a one-shot classification — cap output tightly so a
# chatty model can't burn tokens or latency here.
_TRIAGE_MAX_TOKENS = 200


class TriageVerdict(BaseModel):
    """Structured verdict returned by the gate model."""

    decision: TriageDecision
    reason: str = Field(default="", max_length=500)


_DEFAULT_GATE_INSTRUCTIONS = (
    "You are a gate for a chat assistant. Examine the conversation and decide "
    "whether the assistant should respond to the most recent inbound message(s). "
    'Respond with a single JSON object: {"decision": "respond" | "ignore", '
    '"reason": "<one short sentence>"}. '
    "Choose 'respond' when the assistant is addressed, named, mentioned, asked a "
    "direct question, or when the message clearly continues a prior exchange with "
    "the assistant. Choose 'ignore' for side conversation between other "
    "participants, acknowledgements not directed at the assistant, or noise."
)


def _compose_system_prompt(agent_system: str) -> str:
    """Join the agent-supplied triage system prompt with the default gate rubric.

    Having a default rubric means operators can set ``triage.system`` to
    something narrow (e.g. "only respond when user says 'hey bot'") without
    having to re-derive the JSON response contract.
    """
    if agent_system:
        return f"{agent_system}\n\n{_DEFAULT_GATE_INSTRUCTIONS}"
    return _DEFAULT_GATE_INSTRUCTIONS


async def run_triage(
    *,
    config: TriageConfig,
    messages: list[dict[str, Any]],
) -> TriageVerdict:
    """Call the gate model and return its verdict.

    ``messages`` is the chat-completions message list built by
    :func:`aios.harness.context.build_messages` — the same context the
    main inference would see, minus tool declarations. The system
    message (if any) is replaced with the triage-specific prompt so the
    gate model's priors aren't polluted by the main agent persona.

    Fail-open: any exception or parse failure yields a ``respond``
    verdict with ``reason`` describing the failure. The verdict is still
    a genuine :class:`TriageVerdict`, so callers never need to handle
    ``None``.
    """
    gate_messages = _build_gate_messages(config, messages)
    try:
        response = await litellm.acompletion(
            model=config.model,
            messages=gate_messages,
            response_format={"type": "json_object"},
            max_tokens=_TRIAGE_MAX_TOKENS,
            stream=False,
        )
    except Exception as exc:
        log.warning("triage.model_call_failed", error=str(exc))
        return TriageVerdict(decision="respond", reason=f"gate_error: {exc}")

    content = _extract_content(response)
    if not content:
        log.warning("triage.empty_response")
        return TriageVerdict(decision="respond", reason="gate_returned_empty")

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        log.warning("triage.invalid_json", content=content[:200])
        return TriageVerdict(decision="respond", reason="gate_invalid_json")

    try:
        verdict = TriageVerdict.model_validate(data)
    except ValidationError as exc:
        log.warning("triage.schema_mismatch", error=str(exc), content=content[:200])
        return TriageVerdict(decision="respond", reason="gate_schema_mismatch")

    log.info("triage.verdict", decision=verdict.decision, reason=verdict.reason)
    return verdict


def _build_gate_messages(
    config: TriageConfig, messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return a fresh message list with the gate's system prompt on top.

    Drops the main agent's system message (if present) so the gate model
    classifies based on the gate's own rubric rather than the agent
    persona. The conversation history is preserved verbatim.
    """
    body = [m for m in messages if m.get("role") != "system"]
    return [{"role": "system", "content": _compose_system_prompt(config.system)}, *body]


def _extract_content(response: Any) -> str:
    """Pull the text content out of a LiteLLM ModelResponse.

    LiteLLM normalizes providers to the OpenAI shape, so we can rely on
    ``choices[0].message.content`` being a string (or None). Keeping the
    extraction isolated means a provider quirk only needs fixing here.
    """
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError):
        return ""
    return content or ""
