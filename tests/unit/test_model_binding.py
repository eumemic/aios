"""Unit tests for the ``workflow:`` model binding boundary (issue #1634).

Pins the two pure transforms that bracket the async two-step model-dispatch /
harvest flow:

* :func:`parse_workflow_model` — recognise + split ``workflow:<id>[@version]``,
  with the ``@version`` pin and the malformed-binding fail-loud.
* :func:`map_run_output_to_response` — the binding boundary: validate the inner
  run's structured return and project it into an ``LlmResponse``, mapping the
  inner ``finish_reason`` (refusal / empty / length / normal) onto the outer
  standardized value the dispatch tail branches on.
"""

from __future__ import annotations

import pytest

from aios.harness.completion import LlmResponse
from aios.harness.model_binding import (
    BindingBoundaryError,
    WorkflowModelRef,
    is_workflow_model,
    map_finish_reason,
    map_run_output_to_response,
    parse_workflow_model,
)

# ─── parse_workflow_model ────────────────────────────────────────────────────


def test_non_workflow_model_returns_none() -> None:
    assert parse_workflow_model("anthropic/claude-opus-4-6") is None
    assert is_workflow_model("anthropic/claude-opus-4-6") is False


def test_parse_unversioned_binding() -> None:
    ref = parse_workflow_model("workflow:wf_debate")
    assert ref == WorkflowModelRef(workflow_id="wf_debate", version=None)
    assert is_workflow_model("workflow:wf_debate") is True


def test_parse_version_pin() -> None:
    ref = parse_workflow_model("workflow:wf_debate@7")
    assert ref == WorkflowModelRef(workflow_id="wf_debate", version=7)


def test_parse_empty_id_fails_loud() -> None:
    with pytest.raises(BindingBoundaryError):
        parse_workflow_model("workflow:")


def test_parse_empty_version_fails_loud() -> None:
    with pytest.raises(BindingBoundaryError):
        parse_workflow_model("workflow:wf_debate@")


def test_parse_non_integer_version_fails_loud() -> None:
    with pytest.raises(BindingBoundaryError):
        parse_workflow_model("workflow:wf_debate@latest")


# ─── map_finish_reason ───────────────────────────────────────────────────────


@pytest.mark.parametrize("inner", ["content_filter", "refusal"])
def test_inner_refusal_maps_to_content_filter(inner: str) -> None:
    assert map_finish_reason(inner, has_content=True, has_tool_calls=False) == "content_filter"


def test_empty_result_maps_to_content_filter() -> None:
    # No content and no tool_calls — an empty inner deliberation is a non-answer.
    assert map_finish_reason("stop", has_content=False, has_tool_calls=False) == "content_filter"
    assert map_finish_reason(None, has_content=False, has_tool_calls=False) == "content_filter"


def test_length_passes_through() -> None:
    assert map_finish_reason("length", has_content=True, has_tool_calls=False) == "length"


def test_default_terminal_derives_from_presence() -> None:
    assert map_finish_reason("stop", has_content=True, has_tool_calls=False) == "stop"
    assert map_finish_reason(None, has_content=False, has_tool_calls=True) == "tool_calls"
    # An unknown inner reason normalizes to the presence-derived terminal.
    assert map_finish_reason("weird", has_content=True, has_tool_calls=True) == "tool_calls"


# ─── map_run_output_to_response ──────────────────────────────────────────────


def test_maps_text_answer() -> None:
    resp = map_run_output_to_response({"content": "the answer is 42"})
    assert isinstance(resp, LlmResponse)
    assert resp.content == "the answer is 42"
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"
    # A synthesized assistant message round-trips the content.
    assert resp.message == {"role": "assistant", "content": "the answer is 42"}


def test_maps_tool_calls_turn() -> None:
    tc = {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
    resp = map_run_output_to_response({"content": "", "tool_calls": [tc]})
    assert resp.tool_calls == [tc]
    assert resp.finish_reason == "tool_calls"
    assert resp.message["tool_calls"] == [tc]


def test_refusal_finish_reason_preserved() -> None:
    resp = map_run_output_to_response(
        {"content": "I can't help with that", "finish_reason": "refusal"}
    )
    assert resp.finish_reason == "content_filter"


def test_empty_inner_result_bricks() -> None:
    resp = map_run_output_to_response({"content": "", "tool_calls": []})
    assert resp.finish_reason == "content_filter"


def test_full_message_carried_through() -> None:
    msg = {
        "role": "assistant",
        "content": "hi",
        "thinking_blocks": [{"type": "thinking", "thinking": "...", "signature": "sig"}],
    }
    resp = map_run_output_to_response({"content": "hi", "message": msg})
    assert resp.message == msg
    # A defensive copy — the boundary must not alias the inner dict.
    assert resp.message is not msg


def test_usage_and_cost_carried_for_span_only() -> None:
    resp = map_run_output_to_response(
        {"content": "ok", "usage": {"input_tokens": 3}, "cost": 0.002}
    )
    assert resp.usage == {"input_tokens": 3}
    assert resp.cost == 0.002


@pytest.mark.parametrize(
    "bad",
    [
        "not a dict",
        42,
        None,
        {"content": 123},
        {"content": "ok", "tool_calls": "nope"},
        {"content": "ok", "tool_calls": [1, 2]},
        {"content": "ok", "message": "not a dict"},
    ],
)
def test_malformed_return_fails_loud(bad: object) -> None:
    with pytest.raises(BindingBoundaryError):
        map_run_output_to_response(bad)
