"""Unit tests for create-time workflow ``script`` validation (#1285).

Covers the acceptance criteria of #1285 at the validator level (pure, no DB):

1. Syntax error → compile/syntax validation error (line/offset surfaced).
2. Missing top-level ``main`` → rejected, message names the required ``main``.
3. Mis-signatured ``main`` (not async / wrong params) → rejected, names the signature.
4. Under-declared tool surface (literal ``tool("X")`` not in declared tools) → rejected,
   names ``X``.
5. Literal ``agent(agent_id="A")`` participation (the documented agent depth).
6. Valid, fully-covered script → accepted (no raise).
7. Un-AST-able (computed/variable) tool name / agent_id → accepted (excluded from the
   required-surface check, never a false rejection).

The service-path + tool-handler-path enforcement (criterion 8) and update enforcement
are covered in ``test_workflow_management.py``.
"""

from __future__ import annotations

import pytest

from aios.models.agents import ToolSpec
from aios.workflows.script_validation import (
    WorkflowScriptValidationError,
    validate_workflow_script,
)

_VALID_MAIN = "async def main(input):\n    return input\n"


# ─── criterion 1: syntax / compile error ─────────────────────────────────────


def test_syntax_error_rejected_as_compile_failure() -> None:
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script("async def main(input):\n    x = (\n")
    msg = str(exc.value)
    assert "compile" in msg
    # Line is surfaced where available.
    assert "line" in msg


def test_stray_token_rejected() -> None:
    with pytest.raises(WorkflowScriptValidationError, match="compile"):
        validate_workflow_script("async def main(input):\n    return 1 1\n")


# ─── criterion 2: missing main ───────────────────────────────────────────────


def test_missing_main_rejected() -> None:
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script("x = 1\n\nasync def helper(input):\n    return input\n")
    assert "main" in str(exc.value)


def test_main_not_top_level_rejected() -> None:
    # A ``main`` nested inside another function is not a top-level entry point.
    script = (
        "def outer():\n"
        "    async def main(input):\n"
        "        return input\n"
        "    return main\n"
    )
    with pytest.raises(WorkflowScriptValidationError, match="main"):
        validate_workflow_script(script)


# ─── criterion 3: mis-signatured main ────────────────────────────────────────


def test_plain_def_main_rejected() -> None:
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script("def main(input):\n    return input\n")
    assert "async" in str(exc.value)


def test_main_zero_args_rejected() -> None:
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script("async def main():\n    return 1\n")
    assert "input" in str(exc.value)


def test_main_two_args_rejected() -> None:
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script("async def main(a, b):\n    return 1\n")
    assert "input" in str(exc.value)


def test_main_wrong_param_name_rejected() -> None:
    with pytest.raises(WorkflowScriptValidationError, match="input"):
        validate_workflow_script("async def main(payload):\n    return payload\n")


def test_main_varargs_rejected() -> None:
    with pytest.raises(WorkflowScriptValidationError, match="input"):
        validate_workflow_script("async def main(*args):\n    return 1\n")


def test_main_kwonly_rejected() -> None:
    with pytest.raises(WorkflowScriptValidationError, match="input"):
        validate_workflow_script("async def main(input, *, x=1):\n    return input\n")


# ─── criterion 4: under-declared tool surface ────────────────────────────────


def test_under_declared_tool_rejected_names_the_tool() -> None:
    script = 'async def main(input):\n    return await tool("http_request", {})\n'
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script(script, tools=[ToolSpec(type="bash")])
    assert "http_request" in str(exc.value)


def test_under_declared_tool_with_no_declared_tools_rejected() -> None:
    script = 'async def main(input):\n    return await tool("bash", {})\n'
    with pytest.raises(WorkflowScriptValidationError, match="bash"):
        validate_workflow_script(script, tools=[])


def test_multiple_missing_tools_all_named() -> None:
    script = (
        "async def main(input):\n"
        '    await tool("web_search", {})\n'
        '    return await tool("web_fetch", {})\n'
    )
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script(script, tools=[])
    msg = str(exc.value)
    assert "web_search" in msg and "web_fetch" in msg


# ─── criterion 5: literal agent_id participation (documented depth) ──────────


def test_literal_agent_id_accepted_when_no_tool_drift() -> None:
    # A named-agent child runs over ``agent ∩ run`` (it can only narrow the run's
    # surface), so a bare literal agent reference adds no required tool and is
    # accepted. The literal agent_id participates in the analysis (extracted), but
    # imposes no DB-free surface requirement (see the module docstring on depth).
    script = 'async def main(input):\n    return await agent({"x": 1}, agent_id="reviewer")\n'
    validate_workflow_script(script, tools=[])


def test_agent_reference_with_under_declared_tool_still_rejected() -> None:
    # An agent reference does not excuse an under-declared literal tool call in the
    # same script — the tool union still gates.
    script = (
        "async def main(input):\n"
        '    await agent({"x": 1}, agent_id="reviewer")\n'
        '    return await tool("bash", {})\n'
    )
    with pytest.raises(WorkflowScriptValidationError, match="bash"):
        validate_workflow_script(script, tools=[])


# ─── criterion 6: valid, fully-covered script accepted ───────────────────────


def test_valid_fully_covered_script_accepted() -> None:
    script = (
        "async def main(input):\n"
        '    out = await tool("bash", {"command": "echo hi"})\n'
        '    return await agent(out, agent_id="reviewer")\n'
    )
    # No raise.
    validate_workflow_script(script, tools=[ToolSpec(type="bash")])


def test_minimal_valid_script_accepted() -> None:
    validate_workflow_script(_VALID_MAIN, tools=[])


def test_custom_tool_name_resolved_by_name() -> None:
    # A custom tool's declared name is its ``name`` (not ``type == "custom"``).
    script = 'async def main(input):\n    return await tool("my_tool", {})\n'
    validate_workflow_script(
        script,
        tools=[ToolSpec(type="custom", name="my_tool", description="d", input_schema={})],
    )


def test_posonly_input_param_accepted() -> None:
    validate_workflow_script("async def main(input, /):\n    return input\n", tools=[])


# ─── criterion 7: un-AST-able names never cause false rejection ──────────────


def test_computed_tool_name_accepted() -> None:
    script = (
        "async def main(input):\n"
        '    name = input["tool"]\n'
        "    return await tool(name, {})\n"
    )
    # The variable tool name is un-AST-able → excluded, not a violation.
    validate_workflow_script(script, tools=[])


def test_agent_id_from_input_accepted() -> None:
    script = (
        "async def main(input):\n"
        '    return await agent({"x": 1}, agent_id=input["agent_id"])\n'
    )
    validate_workflow_script(script, tools=[])


def test_mixed_literal_and_computed_only_literal_checked() -> None:
    # The literal "bash" is checked (and covered); the computed name is skipped.
    script = (
        "async def main(input):\n"
        '    await tool("bash", {})\n'
        "    n = input[\"t\"]\n"
        "    return await tool(n, {})\n"
    )
    validate_workflow_script(script, tools=[ToolSpec(type="bash")])
