"""Unit tests for create-time workflow-script validation (#1285).

These exercise the **pure** validator (``aios.workflows.script_validation``) — no DB,
no pool — covering the structural and script-local-surface acceptance criteria:

* (1) syntax / compile failure
* (2) missing top-level ``async def main``
* (3) mis-signatured ``main`` (plain ``def``; wrong arity)
* (4) under-declared tool surface (literal ``tool("X")`` not in ``tools``)
* (6) a valid, fully-covered script passes
* (7) un-AST-able (computed/variable) tool names + agent ids cause no false rejection

The named-``agent(agent_id="…")`` surface union (criterion 5) is resolved against the
live agent in the service layer and is covered by the integration tests
(``test_wf_create_time_validation``); here we assert only the AST extraction it builds on.
"""

from __future__ import annotations

import pytest

from aios.errors import ValidationError
from aios.models.agents import ToolSpec
from aios.workflows.script_validation import (
    declared_tool_names,
    extract_required_agent_ids,
    validate_workflow_script,
)

_VALID_MAIN = "async def main(input):\n    return input\n"


# ─── (1) syntax / compile ────────────────────────────────────────────────────


def test_syntax_error_is_rejected_with_compile_message() -> None:
    script = "async def main(input):\n    return (\n"  # unbalanced paren
    with pytest.raises(ValidationError) as exc:
        validate_workflow_script(script, [])
    msg = str(exc.value)
    assert "compile" in msg
    # Line is surfaced where the SyntaxError carries it.
    assert "line" in msg
    assert exc.value.detail is not None and exc.value.detail.get("lineno") is not None


def test_stray_token_is_a_compile_failure() -> None:
    script = "async def main(input):\n    x = 1 2 3\n    return x\n"
    with pytest.raises(ValidationError) as exc:
        validate_workflow_script(script, [])
    assert "compile" in str(exc.value)


# ─── (2) missing main ────────────────────────────────────────────────────────


def test_missing_main_is_rejected() -> None:
    script = "x = 1\nasync def helper(input):\n    return input\n"
    with pytest.raises(ValidationError) as exc:
        validate_workflow_script(script, [])
    assert "async def main(input)" in str(exc.value)


def test_main_nested_inside_function_is_not_top_level() -> None:
    script = "def outer():\n    async def main(input):\n        return input\n"
    with pytest.raises(ValidationError) as exc:
        validate_workflow_script(script, [])
    assert "async def main(input)" in str(exc.value)


# ─── (3) mis-signatured main ─────────────────────────────────────────────────


def test_plain_def_main_is_rejected() -> None:
    script = "def main(input):\n    return input\n"
    with pytest.raises(ValidationError) as exc:
        validate_workflow_script(script, [])
    assert "async" in str(exc.value)


def test_main_with_no_params_is_rejected() -> None:
    script = "async def main():\n    return 1\n"
    with pytest.raises(ValidationError) as exc:
        validate_workflow_script(script, [])
    assert "input" in str(exc.value)


def test_main_with_two_params_is_rejected() -> None:
    script = "async def main(a, b):\n    return a\n"
    with pytest.raises(ValidationError) as exc:
        validate_workflow_script(script, [])
    assert "input" in str(exc.value)


def test_main_with_differently_named_single_param_is_accepted() -> None:
    # The host calls the entry point positionally, so arity — not the parameter name —
    # is what matters. `main(i)` / `main(payload)` bind the run's input just like
    # `main(input)`. Many existing workflows use `i`; rejecting on the name would break
    # them.
    validate_workflow_script("async def main(payload):\n    return payload\n", [])
    validate_workflow_script("async def main(i):\n    return i\n", [])


def test_main_keyword_only_input_is_rejected() -> None:
    # `main(*, input)` cannot be called positionally as `main(value)`.
    script = "async def main(*, input):\n    return input\n"
    with pytest.raises(ValidationError):
        validate_workflow_script(script, [])


def test_main_posonly_input_is_accepted() -> None:
    validate_workflow_script("async def main(input, /):\n    return input\n", [])


def test_main_star_args_is_accepted() -> None:
    # A `*args` catch-all still binds the single positional `input`.
    validate_workflow_script("async def main(*args):\n    return args\n", [])


def test_last_main_binding_wins() -> None:
    # exec semantics: a later top-level `main` shadows an earlier one. The valid (async)
    # one is last -> accepted.
    script = "def main(input):\n    return 1\nasync def main(input):\n    return 2\n"
    validate_workflow_script(script, [])
    # ...and the reverse (valid first, broken last) is rejected on the last binding.
    script2 = "async def main(input):\n    return 1\ndef main(input):\n    return 2\n"
    with pytest.raises(ValidationError):
        validate_workflow_script(script2, [])


# ─── (4) under-declared tool surface ─────────────────────────────────────────


def test_under_declared_tool_is_rejected_naming_the_tool() -> None:
    script = "async def main(input):\n    r = await tool('bash', {'command': 'ls'})\n    return r\n"
    with pytest.raises(ValidationError) as exc:
        validate_workflow_script(script, [])
    assert "bash" in str(exc.value)
    assert exc.value.detail is not None
    assert "bash" in exc.value.detail["missing_tools"]


def test_under_declared_lists_all_missing_tools() -> None:
    script = (
        "async def main(input):\n"
        "    await tool('bash', {})\n"
        "    await tool('http_request', {})\n"
        "    return 1\n"
    )
    with pytest.raises(ValidationError) as exc:
        validate_workflow_script(script, [ToolSpec(type="bash")])
    # bash is declared; http_request is not -> only http_request is named.
    assert exc.value.detail is not None
    assert exc.value.detail["missing_tools"] == ["http_request"]


def test_custom_tool_name_satisfies_the_declared_surface() -> None:
    script = "async def main(input):\n    return await tool('my_tool', {})\n"
    tools = [
        ToolSpec(type="custom", name="my_tool", description="d", input_schema={"type": "object"})
    ]
    validate_workflow_script(script, tools)


# ─── (6) valid, fully-covered ────────────────────────────────────────────────


def test_valid_fully_covered_script_passes() -> None:
    script = (
        "async def main(input):\n"
        "    out = await tool('bash', {'command': 'echo hi'})\n"
        "    res = await agent({'task': 't'}, agent_id='impl')\n"
        "    return {'out': out, 'res': res}\n"
    )
    validate_workflow_script(script, [ToolSpec(type="bash")])  # agent surface checked in service


def test_no_tools_no_agents_minimal_script_passes() -> None:
    validate_workflow_script(_VALID_MAIN, [])


# ─── (7) un-AST-able names — no false rejection ──────────────────────────────


def test_computed_tool_name_is_excluded_not_rejected() -> None:
    script = "async def main(input):\n    name = input['tool']\n    return await tool(name, {})\n"
    # Un-AST-able first arg -> excluded from the required-surface check. Accepted.
    validate_workflow_script(script, [])


def test_fstring_tool_name_is_excluded() -> None:
    script = "async def main(input):\n    return await tool(f\"t_{input['x']}\", {})\n"
    validate_workflow_script(script, [])


def test_agent_id_from_input_is_excluded() -> None:
    script = "async def main(input):\n    return await agent({'x': 1}, agent_id=input['a'])\n"
    # No literal agent_id -> nothing required of the surface; accepted.
    assert extract_required_agent_ids(script) == set()
    validate_workflow_script(script, [])


# ─── AST extraction helpers ──────────────────────────────────────────────────


def test_extract_required_agent_ids_literal_only() -> None:
    script = (
        "async def main(input):\n"
        "    await agent({}, agent_id='impl')\n"
        "    await agent({}, agent_id='review')\n"
        "    await agent({}, agent_id=input['x'])\n"  # excluded
        "    await agent({})\n"  # generic subagent, no agent_id — excluded
        "    return 1\n"
    )
    assert extract_required_agent_ids(script) == {"impl", "review"}


def test_declared_tool_names_covers_type_and_custom_name() -> None:
    names = declared_tool_names(
        [
            ToolSpec(type="bash"),
            ToolSpec(type="custom", name="foo", description="d", input_schema={}),
        ]
    )
    assert "bash" in names
    assert "foo" in names
    assert "custom" in names
