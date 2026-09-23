"""Exhaustive state table for the explicit output cap (#2451 / #2453).

Four review rounds each found a NEW edge in the same area — spellings, validity,
precedence, windowing, admission — because each was checked by a hand-picked
sample. This file enumerates the WHOLE input space instead:

* route family: every route shape the cap logic branches on;
* each of ``max_output_tokens`` / ``max_tokens`` / ``max_completion_tokens``
  independently absent, a valid positive int (a DIFFERENT value per spelling, so
  a precedence swap is observable), or each invalid value
  (``None``, ``0``, negative, ``bool``, ``str``) —
  7 states per spelling, 7**3 = 343 combinations per route.

For every cell it asserts against an ORACLE written here from the documented
rules, not from the implementation:

(a) the payload aios hands LiteLLM carries at most ONE cap key (on every route —
    Anthropic-shaped routes are where two keys become two competing caps);
(b) its value equals what admission reads off that same payload AND what
    windowing reserves (``loop`` feeds ``resolve_output_cap(...).value`` to
    ``effective_window_max``);
(c) precedence: the first VALID value in
    ``max_output_tokens > max_tokens > max_completion_tokens`` wins, else the
    model-ceiling default on the routes that get one, else no cap;
(d) no invalid value reaches the payload under any spelling.

Failures are collected per route rather than raised on the first cell, so a
mutation reports how many of the 343 cells it breaks.
"""

from __future__ import annotations

import itertools
from typing import Any

import litellm
import pytest

from aios.harness import completion
from aios.harness.context_admission import AdmissionMode, admit_context
from aios.harness.context_budget import effective_window_max

_CAP_KEYS = ("max_output_tokens", "max_tokens", "max_completion_tokens")
# Documented precedence, restated independently of EXPLICIT_OUTPUT_CAP_KEYS so a
# reorder of the shared tuple is caught here rather than silently agreed with.
_PRECEDENCE = ("max_output_tokens", "max_tokens", "max_completion_tokens")
_VALID = {"max_output_tokens": 1111, "max_tokens": 2222, "max_completion_tokens": 3333}
_ABSENT = object()
_INVALID: tuple[Any, ...] = (None, 0, -5, True, "4096")


def _ceiling(model: str) -> int:
    value = litellm.get_model_info(model)["max_output_tokens"]
    assert isinstance(value, int) and value > 0
    return value


# (id, model, extra params, anthropic_shaped, default ceiling or None)
# ``anthropic_shaped``: the route reads ``max_tokens``, so the one cap must use
# that spelling. ``default``: the ceiling injected when the caller names no
# valid cap (None = route deliberately gets no default / catalog has none).
_ROUTES: list[tuple[str, str, dict[str, Any], bool, int | None]] = [
    (
        "anthropic-in-catalog",
        "anthropic/claude-opus-4-1",
        {},
        True,
        _ceiling("anthropic/claude-opus-4-1"),
    ),
    ("anthropic-unmapped", "anthropic/claude-not-a-real-model-2451", {}, True, None),
    ("openai-chat", "openai/gpt-4.1", {}, False, None),
    ("openai-responses", "openai/responses/gpt-5.6-sol", {}, False, None),
    ("openrouter-anthropic", "openrouter/anthropic/claude-opus-4-1", {}, True, None),
    (
        "anthropic-via-openrouter-override",
        "anthropic/claude-opus-4-1",
        {"custom_llm_provider": "openrouter"},
        True,
        None,
    ),
    ("vertex-claude", "vertex_ai/claude-opus-4-1", {}, True, _ceiling("vertex_ai/claude-opus-4-1")),
    # ``custom_llm_provider`` overrides the provider, so route shape follows the
    # override (``litellm.get_llm_provider(model, custom_llm_provider=...)``, the
    # credential resolver's sniff), not the bare model string.
    #
    # An alias litellm can't place from the string alone, sent to Anthropic via
    # the override: litellm 1.96.2 dispatches this to its Anthropic handler, so
    # an omitted cap is the 4096 trap and ``max_output_tokens`` must become
    # ``max_tokens``. Unmapped there, so no ceiling default.
    (
        "alias-via-anthropic-override",
        "opus-alias-2451",
        {"custom_llm_provider": "anthropic"},
        True,
        None,
    ),
    # ``gpt-4`` + ``anthropic``: classified Anthropic-shaped, unmapped (NOT
    # OpenAI's gpt-4 catalog entry). NOTE litellm 1.96.2 ``completion()`` checks
    # ``model in open_ai_chat_completion_models`` BEFORE the provider branch and
    # actually sends this to its OpenAI handler; ``max_tokens`` is valid there
    # too and no default is injected, so the outcome is safe either way.
    ("gpt4-via-anthropic-override", "gpt-4", {"custom_llm_provider": "anthropic"}, True, None),
    # ``claude-*`` dispatched to OpenAI is OpenAI-shaped: no ceiling default.
    (
        "claude-via-openai-override",
        "claude-opus-4-1",
        {"custom_llm_provider": "openai"},
        False,
        None,
    ),
]


def _cells() -> list[dict[str, Any]]:
    states: list[Any] = [_ABSENT, "VALID", *_INVALID]
    out = []
    for combo in itertools.product(states, repeat=len(_CAP_KEYS)):
        params: dict[str, Any] = {}
        for key, state in zip(_CAP_KEYS, combo, strict=True):
            if state is _ABSENT:
                continue
            params[key] = _VALID[key] if isinstance(state, str) and state == "VALID" else state
        out.append(params)
    return out


_CELLS = _cells()


def _is_valid(value: Any) -> bool:
    return type(value) is int and value > 0


def _oracle(params: dict[str, Any], anthropic_shaped: bool, default: int | None) -> dict[str, int]:
    for key in _PRECEDENCE:
        if key in params and _is_valid(params[key]):
            return {("max_tokens" if anthropic_shaped else key): params[key]}
    return {"max_tokens": default} if default is not None else {}


def test_cell_space_is_complete() -> None:
    assert len(_CELLS) == 7**3
    assert len({repr(sorted(c.items(), key=lambda kv: kv[0])) for c in _CELLS}) == 7**3


@pytest.mark.parametrize(
    ("model", "base", "anthropic_shaped", "default"),
    [pytest.param(m, b, a, d, id=i) for i, m, b, a, d in _ROUTES],
)
def test_every_cell_sends_one_cap_that_admission_and_windowing_agree_on(
    model: str, base: dict[str, Any], anthropic_shaped: bool, default: int | None
) -> None:
    failures: list[str] = []
    for cell in _CELLS:
        params = {**base, **cell}
        kwargs = completion._build_litellm_kwargs(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            auth=None,
            extra=dict(params),
            session_id="sess_table",
            stream=False,
        )
        wire = {k: kwargs[k] for k in _CAP_KEYS if k in kwargs}
        expected = _oracle(cell, anthropic_shaped, default)
        admission = admit_context(
            kwargs, mode=AdmissionMode.OBSERVE, attestation=None
        ).output_reserve
        # Windowing exactly as ``loop`` composes it; the reservation actually
        # subtracted is recovered against a known limit. (With no cap on the
        # wire, windowing still reserves any ``thinking.budget_tokens`` — none
        # here, so 0.)
        cap = completion.resolve_output_cap(model, params)
        limit = 10_000_000
        window = limit - effective_window_max(
            model="table/no-served-ceiling",
            window_max=limit,
            params=params,
            output_reserve=cap.value,
            context_limit=limit,
        )
        wire_value = next(iter(wire.values())) if wire else None

        problems = []
        if len(wire) > 1:
            problems.append("(a) >1 cap key")
        if wire != expected:
            problems.append(f"(c) wire {wire} != oracle {expected}")
        if admission != wire_value:
            problems.append(f"(b) admission {admission} != wire {wire_value}")
        if window != (wire_value or 0):
            problems.append(f"(b) windowing {window} != wire {wire_value}")
        if any(not _is_valid(v) for v in wire.values()):
            problems.append(f"(d) invalid value on wire {wire}")
        if problems:
            failures.append(f"{cell}: {'; '.join(problems)}")
    assert not failures, f"{len(failures)}/{len(_CELLS)} cells wrong, e.g.:\n" + "\n".join(
        failures[:8]
    )


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"max_output_tokens": 1111, "max_tokens": 2222}, 1111),
        ({"max_output_tokens": 1111, "max_completion_tokens": 3333}, 1111),
        ({"max_tokens": 2222, "max_completion_tokens": 3333}, 2222),
        ({"max_output_tokens": 1111, "max_tokens": 2222, "max_completion_tokens": 3333}, 1111),
        ({"max_output_tokens": 0, "max_tokens": 2222, "max_completion_tokens": 3333}, 2222),
    ],
    ids=["mot+mt", "mot+mct", "mt+mct", "all3", "invalid-mot+mt+mct"],
)
def test_real_anthropic_wire_body_carries_only_the_resolved_cap(
    monkeypatch: pytest.MonkeyPatch, params: dict[str, Any], expected: int
) -> None:
    """The #2453 P1 (competing spellings) asserted on the serialized BYTES.

    ``_build_litellm_kwargs`` is what aios hands LiteLLM; LiteLLM then maps
    ``max_completion_tokens`` onto ``max_tokens`` and passes ``max_output_tokens``
    through, so a surviving extra spelling is a second cap on the wire.
    """
    from tests.unit.test_completion_max_tokens import TestRealLiteLLMWireBody

    body = TestRealLiteLLMWireBody._wire_body(
        monkeypatch, model="anthropic/claude-opus-4-1", params=params
    )
    assert {k: body[k] for k in _CAP_KEYS if k in body} == {"max_tokens": expected}


def test_real_wire_body_honours_the_custom_llm_provider_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model string litellm can't place, sent to Anthropic via ``custom_llm_provider``.

    Classified from the bare string, the caller's ``max_output_tokens`` passed
    through verbatim and litellm added its own ``max_tokens: 4096`` — two
    competing caps, the caller's ignored. Classified from the dispatch provider,
    the one cap travels as ``max_tokens``.
    """
    from tests.unit.test_completion_max_tokens import TestRealLiteLLMWireBody

    body = TestRealLiteLLMWireBody._wire_body(
        monkeypatch,
        model="opus-alias-2451",
        params={"custom_llm_provider": "anthropic", "max_output_tokens": 1111},
    )
    assert {k: body[k] for k in _CAP_KEYS if k in body} == {"max_tokens": 1111}
