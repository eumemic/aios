"""The eval workflow templates and the helpers their scripts define.

A workflow script can't import aios, so ``eval_item`` carries its own reading of an
arm's output; the drift test holds it to the binding boundary's
(``harness/model_binding.map_run_output_to_response``)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from evals.workflows import (
    eval_analysis,
    eval_item,
    eval_judge,
    eval_r0,
    load,
    paired_eval,
    render,
)

from aios.errors import ValidationError
from aios.harness.model_binding import BindingBoundaryError, map_run_output_to_response
from aios.models.agents import ToolSpec
from aios.workflows.script_validation import validate_workflow_script

_REF = {"id": "wf_x", "version": 1}
ITEM = load(eval_item.build(r0=_REF, judge=_REF))
JUDGE = load(eval_judge.build())


def _templates() -> list[tuple[str, str, list[dict[str, str]]]]:
    return [
        ("r0", eval_r0.build(), eval_r0.TOOLS),
        ("judge", eval_judge.build(), eval_judge.TOOLS),
        ("analysis", eval_analysis.build(), eval_analysis.TOOLS),
        ("item", eval_item.build(r0=_REF, judge=_REF), eval_item.TOOLS),
        (
            "gate",
            paired_eval.build(mode="gate", bar={"x": 1}, item=_REF, analysis=_REF),
            paired_eval.TOOLS,
        ),
        (
            "monitor",
            paired_eval.build(mode="monitor", bar={"x": 1}, item=_REF, analysis=_REF),
            paired_eval.TOOLS,
        ),
    ]


@pytest.mark.parametrize(("name", "script", "tools"), _templates())
def test_every_template_validates_against_its_declared_tools(
    name: str, script: str, tools: list[dict[str, str]]
) -> None:
    validate_workflow_script(script, [ToolSpec.model_validate(t) for t in tools])


def test_a_template_missing_its_tools_is_rejected() -> None:
    with pytest.raises(ValidationError, match="sample_requests"):
        validate_workflow_script(
            paired_eval.build(mode="gate", bar={}, item=_REF, analysis=_REF), []
        )


def test_render_fills_every_token_and_only_known_ones() -> None:
    assert render("A = __CONFIG__", config={"k": [1, None]}) == "A = " + repr('{"k": [1, null]}')
    with pytest.raises(KeyError, match="__OTHER__"):
        render("A = __CONFIG__", other=1)
    with pytest.raises(KeyError, match="__CONFIG2__"):
        render("A = __CONFIG__ + __CONFIG2__", config=1)


def test_a_rendered_value_reads_back_unchanged() -> None:
    bar = {"delta": 0.1, "judge": {"model": "m", "params": {"temperature": 0}}, "flag": True}
    gate = load(paired_eval.build(mode="gate", bar=bar, item=_REF, analysis=_REF))
    assert gate["BAR"] == bar


def test_the_gate_template_has_two_modes_only() -> None:
    with pytest.raises(ValueError, match="mode"):
        paired_eval.build(mode="weekly", bar={}, item=_REF, analysis=_REF)


# ── the monitor's week ────────────────────────────────────────────────────────

MONITOR = load(paired_eval.build(mode="monitor", bar={}, item=_REF, analysis=_REF))


@pytest.mark.parametrize("day", range(0, 800, 7))
def test_the_window_is_the_utc_iso_week_before_the_fire(day: int) -> None:
    for hour in (0, 13, 23):
        fired = datetime(2025, 12, 1, tzinfo=UTC) + timedelta(days=day + day % 5, hours=hour)
        monday = (fired - timedelta(days=fired.weekday())).replace(hour=0)
        window = MONITOR["monitor_window"](fired.isoformat())
        assert window == {
            "start": (monday - timedelta(days=7)).isoformat(),
            "end": monday.isoformat(),
        }


def test_every_fire_in_a_week_picks_the_same_window() -> None:
    windows = {
        json.dumps(MONITOR["monitor_window"](f"2026-10-{d:02d}T{h:02d}:30:00.123456+00:00"))
        for d in range(5, 12)
        for h in (0, 12, 23)
    }
    assert len(windows) == 1


# ── model families ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("anthropic/claude-opus-4-8", "anthropic"),
        ("claude-3-5-sonnet-20240620", "anthropic"),
        ("bedrock/anthropic.claude-3-sonnet-20240229-v1:0", "anthropic"),
        ("openrouter/anthropic/claude-sonnet-4.5", "anthropic"),
        ("openai/gpt-4o", "openai"),
        ("openrouter/openai/gpt-5.5", "openai"),
        ("openai/responses/gpt-5.5", "openai"),
        ("openrouter/openai/gpt-6-luna", "openai"),
        ("openai/gpt-6-luna", "openai"),
        ("gpt-6-luna", "openai"),
        ("o3-mini", "openai"),
        ("azure/gpt-4o", "openai"),
        ("gemini/gemini-2.5-pro", "google"),
        ("vertex_ai/gemini-pro", "google"),
        ("openrouter/moonshotai/kimi-k2.6", "moonshot"),
        ("openrouter/z-ai/glm-5.2", "zhipu"),
        ("xai/grok-4", "xai"),
        ("openrouter/meta-llama/llama-3.1-70b-instruct", "meta"),
        ("groq/llama3-70b-8192", "meta"),
        ("deepseek/deepseek-chat", "deepseek"),
        ("openrouter/qwen/qwen-2.5-72b", "qwen"),
        ("mistral/mistral-large-latest", "mistral"),
        ("bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", "anthropic"),
        # A gateway string naming one lab's protocol and another's model can't be
        # placed: an Anthropic judge must not pass as unrelated to openai/claude-...
        ("openai/claude-sonnet-4-5", None),
        ("anthropic/gpt-4o", None),
        ("openai/some-private-model", None),
        ("openrouter/auto", None),
        ("workflow:wf_1@2", None),
        ("<unknown>", None),
        (None, None),
    ],
)
def test_model_family(model: str | None, family: str | None) -> None:
    assert ITEM["family"](model) == family


def _facts(*nodes: dict[str, Any], truncated: bool = False) -> dict[str, Any]:
    return {"nodes": list(nodes), "truncated": truncated, "litellm_version": "x"}


def _run(id: str, parent: str, label: str | None, models: list[str | None]) -> dict[str, Any]:
    return {
        "kind": "run",
        "id": id,
        "parent": {"kind": "run", "id": parent},
        "label": label,
        "workflow_id": "wf_" + id,
        "workflow_version": 1,
        "duration_ms": 10,
        "usage": [{"model": m, "cost_microusd": 5, "uncached_cost_microusd": 7} for m in models],
    }


def test_arm_facts_cover_the_arms_whole_subtree() -> None:
    facts = _facts(
        _run("a", "me", "arm:cand", ["openai/gpt-4o"]),
        _run("b", "a", "inner", ["openrouter/moonshotai/kimi-k2.6"]),
        _run("c", "me", "arm:base", ["openai/gpt-4o"]),
    )
    cand = ITEM["summarize"](facts, "arm:cand")
    assert cand["models"] == ["openai/gpt-4o", "openrouter/moonshotai/kimi-k2.6"]
    assert cand["workflows"] == ["wf_a@1", "wf_b@1"]
    assert cand["uncached_cost_microusd"] == 14 and cand["cost_microusd"] == 10


def test_a_candidates_own_sub_run_cannot_pose_as_an_arm() -> None:
    facts = _facts(_run("a", "me", "arm:cand", []), _run("b", "a", "judge", ["openai/x"]))
    assert ITEM["labelled"](facts, "judge") is None


def test_an_unpriced_usage_makes_the_arm_unpriced() -> None:
    node = _run("a", "me", "arm:cand", ["openai/gpt-4o"])
    node["usage"].append({"model": "x/y", "cost_microusd": 1, "uncached_cost_microusd": None})
    assert ITEM["summarize"](_facts(node), "arm:cand")["uncached_cost_microusd"] is None


@pytest.mark.parametrize(
    ("facts", "violates"),
    [
        (_facts(_run("a", "me", "arm:base", ["openai/gpt-4o"])), False),
        (_facts(_run("a", "me", "arm:base", ["anthropic/claude-x"])), True),
        (_facts(_run("a", "me", "arm:base", [None])), True),
        (_facts(_run("a", "me", "arm:base", ["openrouter/auto"])), True),
        (_facts(_run("a", "me", "arm:base", ["openai/gpt-4o"]), truncated=True), True),
    ],
)
def test_a_family_overlap_or_an_unplaceable_model_violates(
    facts: dict[str, Any], violates: bool
) -> None:
    summaries = [ITEM["summarize"](facts, "arm:base")]
    assert ITEM["violation"](facts, {"anthropic"}, summaries) is violates


# ── the request's windows ─────────────────────────────────────────────────────


def _msgs(*roles: str) -> list[dict[str, Any]]:
    return [{"role": r, "content": f"{r}{i}"} for i, r in enumerate(roles)]


def test_the_tail_starts_at_a_user_message() -> None:
    rest = _msgs("user", "assistant", "tool", "user", "assistant", "tool", "user")
    assert [m["role"] for m in ITEM["judge_tail"](rest, 5)] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    # Never less than the last user turn, even when the size cuts inside it.
    assert ITEM["judge_tail"](rest, 1) == rest[6:]
    assert ITEM["judge_tail"](rest, 100) == rest


def test_the_last_user_turn_keeps_its_tool_results() -> None:
    rest = _msgs("user", "assistant", "user", "assistant", "tool")
    assert rest[ITEM["last_user"](rest) :] == rest[2:]


def test_split_system_takes_only_the_leading_system_messages() -> None:
    messages = _msgs("system", "system", "user", "system")
    system, rest = ITEM["split_system"](messages)
    assert len(system) == 2 and len(rest) == 2


# ── an arm's output ───────────────────────────────────────────────────────────

_TC = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}


@pytest.mark.parametrize(
    "output",
    [
        {"content": "hi", "finish_reason": "stop"},
        {"content": "hi", "tool_calls": [_TC], "finish_reason": "tool_calls"},
        {"content": None, "tool_calls": [_TC]},
        {"content": "", "tool_calls": []},
        {"content": "cut", "finish_reason": "length"},
        {"content": "", "tool_calls": [_TC], "finish_reason": "length"},
        {"content": "no", "finish_reason": "content_filter"},
        {"content": "no", "finish_reason": "refusal"},
        {"content": "x", "message": {"role": "assistant", "content": "x"}},
        {"content": 3},
        {"content": "x", "tool_calls": "f()"},
        {"content": "x", "tool_calls": ["f"]},
        {"content": "x", "message": "m"},
        "text",
        None,
        [],
    ],
)
def test_reading_an_arm_matches_the_binding_boundary(output: Any) -> None:
    turn = ITEM["as_turn"](output)
    try:
        response = map_run_output_to_response(output)
    except BindingBoundaryError:
        assert turn is None
        return
    assert turn == {
        "content": response.content,
        "tool_calls": response.tool_calls,
        "finish_reason": response.finish_reason,
    }


def _tools(**required: list[str]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": name, "parameters": {"required": keys}}}
        for name, keys in required.items()
    ]


def _call(name: str, args: Any) -> dict[str, Any]:
    return {"function": {"name": name, "arguments": args}}


@pytest.mark.parametrize(
    ("calls", "valid"),
    [
        ([], True),
        ([_call("f", '{"a": 1}')], True),
        ([_call("f", {"a": 1})], True),
        ([_call("g", "{}")], False),
        ([_call("f", "{}")], False),
        ([_call("f", "not json")], False),
        ([_call("f", "[1]")], False),
    ],
)
def test_tool_call_validity(calls: list[dict[str, Any]], valid: bool) -> None:
    assert ITEM["calls_valid"](calls, _tools(f=["a"])) is valid


def test_arm_record_marks_errors_and_degenerate_turns() -> None:
    rec, reply = ITEM["arm_record"]({"output": {"error": "call_llm failed: boom"}}, None)
    assert reply is None and rec["error_kind"] == "error" and rec["degenerate"]
    # A budget is read from the arm's spend, never from text an arm controls.
    rec, _ = ITEM["arm_record"]({"output": {"error": "run budget exhausted: x"}}, None)
    assert rec["error_kind"] == "error"
    rec, _ = ITEM["arm_record"]({"raised": {"kind": "child_errored", "message": "m"}}, None)
    assert rec["error_kind"] == "child_errored"
    rec, reply = ITEM["arm_record"]({"output": "text"}, None)
    assert rec["invalid_output"] and reply is None
    rec, reply = ITEM["arm_record"]({"output": {"content": "", "tool_calls": []}}, None)
    assert rec["degenerate"] and reply == {"content": "", "tool_calls": []}
    rec, _ = ITEM["arm_record"]({"output": {"content": "ok"}}, None)
    assert not rec["degenerate"] and rec["chars"] == 2


@pytest.mark.parametrize(
    ("result", "refused", "overloaded"),
    [
        ({"raised": {"kind": "invoke_workflow_refused", "message": "run cap"}}, True, False),
        ({"raised": {"kind": "invoke_workflow_forbidden", "message": "no"}}, False, False),
        ({"output": {"error": "call_llm failed: RateLimitError: 429"}}, False, True),
        ({"output": {"error": "call_llm failed: InternalServerError: Overloaded"}}, False, True),
        ({"output": {"error": "call_llm timed out: deadline"}}, False, True),
        ({"raised": {"kind": "author_exception", "message": "provider 503"}}, False, True),
        ({"output": {"error": "call_llm failed: BadRequestError: bad"}}, False, False),
        ({"output": {"content": "fine"}}, False, False),
    ],
)
def test_launch_refusals_and_overloads(
    result: dict[str, Any], refused: bool, overloaded: bool
) -> None:
    """Only core sets a launch refusal's kind; overload is read from text, which for
    the candidate arm is the candidate's own (so eval_item trusts it there once)."""
    assert ITEM["launch_refused"](result) is refused
    assert ITEM["overloaded"](result) is overloaded


# ── the judge ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("reasoning\nWINNER: 1", "1"),
        ("winner: tie", "TIE"),
        ("WINNER: 2 ... no wait\nWINNER: 1", "1"),
        ("WINNER:2", "2"),
        ("I can't decide", None),
        (None, None),
    ],
)
def test_the_verdict_line_is_parsed(text: str | None, verdict: str | None) -> None:
    assert JUDGE["parse"](text) == verdict


@pytest.mark.parametrize(
    ("first", "second", "outcome"),
    [
        ("1", "2", "win"),
        ("2", "1", "loss"),
        ("1", "1", "tie"),  # position bias: the orders disagree
        ("2", "2", "tie"),
        ("TIE", "2", "tie"),
        ("TIE", "TIE", "tie"),
    ],
)
def test_the_two_orders_combine(first: str, second: str, outcome: str) -> None:
    assert JUDGE["combine"](first, second) == outcome


def test_replies_that_differ_only_in_whitespace_and_call_ids_are_identical() -> None:
    a = {
        "content": "Hello  there\n",
        "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": '{"b": 2, "a": 1}'}}],
    }
    b = {
        "content": "Hello there",
        "tool_calls": [{"id": "2", "function": {"name": "f", "arguments": {"a": 1, "b": 2}}}],
    }
    assert JUDGE["normalized"](a) == JUDGE["normalized"](b)
    c = dict(b, tool_calls=[{"function": {"name": "f", "arguments": '{"a": 2}'}}])
    assert JUDGE["normalized"](a) != JUDGE["normalized"](c)


def test_the_judge_prompt_shows_both_replies_and_their_tool_calls() -> None:
    text = JUDGE["prompt"](
        {"max_chars": 50, "system": "sys", "conversation": "[user] hi"},
        {"content": "one", "tool_calls": [_TC]},
        {"content": "x" * 200},
    )
    assert "REPLY 1:\none\n-> requests tool call f({})" in text
    assert "[... clipped ...]" in text
    assert JUDGE["RUBRIC"] == eval_judge.RUBRIC
