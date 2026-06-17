"""Reference tests for create-time workflow-script validation (#1284).

These exercise the pure validator (:func:`aios.workflows.script_validation.
validate_workflow_script`) directly — no DB — so each acceptance criterion in
#1284 has an objective pass/fail test:

1. syntax / compile failure  → rejected, message identifies a syntax failure
2. missing top-level ``main`` → rejected, message names ``async def main(input)``
3. mis-signatured ``main``    → rejected (plain ``def``, ``main()``, ``main(a, b)``)
4. under-declared tool        → rejected, message names the missing tool ``X``
5. under-declared agent ref   → rejected, message names the missing element
6. valid, fully-covered       → accepted (happy path unchanged)
7. un-AST-able names          → accepted (excluded from the required-surface check)

The service-path wiring (create + update both enforce it, the tool-handler path
and HTTP path both flow through the service) is covered in
``test_workflow_service_validation.py``.
"""

from __future__ import annotations

import pytest

from aios.errors import ValidationError
from aios.models.agents import McpServerSpec, ToolSpec
from aios.workflows.script_validation import (
    WorkflowScriptValidationError,
    validate_workflow_script,
)

_VALID_MAIN = "async def main(input):\n    return input\n"


# ── 1. syntax / compile failure ──────────────────────────────────────────────


def test_syntax_error_rejected_with_compile_message() -> None:
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script("async def main(input):\n    return (\n")
    msg = exc.value.message
    assert "compile" in msg or "syntax" in msg
    # Line/offset surfaced where available.
    assert "line" in msg
    assert exc.value.detail.get("lineno") is not None


def test_stray_token_rejected() -> None:
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script("async def main(input):\n    return 1 @\n")
    assert "syntax" in exc.value.message or "compile" in exc.value.message


def test_validation_error_is_422_aios_error() -> None:
    err = WorkflowScriptValidationError("x")
    assert isinstance(err, ValidationError)
    assert err.status_code == 422


# ── 2. missing main ──────────────────────────────────────────────────────────


def test_missing_main_rejected() -> None:
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script("async def helper(input):\n    return 1\n")
    assert "main(input)" in exc.value.message


def test_main_not_top_level_rejected() -> None:
    # A nested ``main`` is not the entry point.
    script = "async def wrapper():\n    async def main(input):\n        return 1\n"
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script(script)
    assert "main(input)" in exc.value.message


# ── 3. mis-signatured main ───────────────────────────────────────────────────


def test_plain_def_main_rejected() -> None:
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script("def main(input):\n    return 1\n")
    assert "async" in exc.value.message


def test_main_no_params_rejected() -> None:
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script("async def main():\n    return 1\n")
    assert "input" in exc.value.message


def test_main_two_params_rejected() -> None:
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script("async def main(a, b):\n    return 1\n")
    assert "input" in exc.value.message


def test_main_required_kwonly_rejected() -> None:
    with pytest.raises(WorkflowScriptValidationError):
        validate_workflow_script("async def main(input, *, x):\n    return 1\n")


def test_main_varargs_only_rejected() -> None:
    with pytest.raises(WorkflowScriptValidationError):
        validate_workflow_script("async def main(*args):\n    return 1\n")


def test_main_input_with_default_kwonly_accepted() -> None:
    # A defaulted keyword-only extra still leaves ``main(input_value)`` callable.
    validate_workflow_script("async def main(input, *, label=None):\n    return 1\n")


# ── 4. under-declared tool surface ───────────────────────────────────────────


def test_undeclared_tool_rejected_naming_it() -> None:
    script = 'async def main(input):\n    return await tool("bash", {"command": "ls"})\n'
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script(script, tools=[])
    assert "bash" in exc.value.message
    assert "bash" in exc.value.detail.get("missing_tools", [])


def test_declared_builtin_tool_accepted() -> None:
    script = 'async def main(input):\n    return await tool("bash", {"command": "ls"})\n'
    validate_workflow_script(script, tools=[ToolSpec(type="bash")])


def test_declared_custom_tool_accepted_by_name() -> None:
    script = 'async def main(input):\n    return await tool("my_tool", {})\n'
    validate_workflow_script(
        script, tools=[ToolSpec(type="custom", name="my_tool", description="d", input_schema={})]
    )


def test_multiple_undeclared_tools_all_named() -> None:
    script = (
        'async def main(input):\n    await tool("bash", {})\n    await tool("web_search", {})\n'
    )
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script(script, tools=[])
    assert "bash" in exc.value.message
    assert "web_search" in exc.value.message


def test_partially_declared_tools_names_only_missing() -> None:
    script = (
        'async def main(input):\n    await tool("bash", {})\n    await tool("web_search", {})\n'
    )
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script(script, tools=[ToolSpec(type="bash")])
    assert "web_search" in exc.value.message
    assert "bash" not in exc.value.detail.get("missing_tools", [])


# ── 5. under-declared agent reference ────────────────────────────────────────


def test_undeclared_literal_agent_id_rejected_naming_it() -> None:
    script = 'async def main(input):\n    return await agent({}, agent_id="scout")\n'
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script(script)
    assert "scout" in exc.value.message
    assert "scout" in exc.value.detail.get("missing_agents", [])


def test_literal_agent_id_covered_by_mcp_accepted() -> None:
    script = 'async def main(input):\n    return await agent({}, agent_id="scout")\n'
    validate_workflow_script(
        script, mcp_servers=[McpServerSpec(name="scout", url="https://example.test/mcp")]
    )


def test_literal_agent_id_covered_by_tool_accepted() -> None:
    script = 'async def main(input):\n    return await agent({}, agent_id="scout")\n'
    validate_workflow_script(
        script, tools=[ToolSpec(type="custom", name="scout", description="d", input_schema={})]
    )


# ── 6. valid, fully-covered ──────────────────────────────────────────────────


def test_valid_minimal_accepted() -> None:
    validate_workflow_script(_VALID_MAIN)


def test_valid_fully_covered_accepted() -> None:
    script = (
        "async def main(input):\n"
        '    r = await tool("bash", {"command": "ls"})\n'
        '    return await agent({}, agent_id="scout")\n'
    )
    validate_workflow_script(
        script,
        tools=[ToolSpec(type="bash")],
        mcp_servers=[McpServerSpec(name="scout", url="https://example.test/mcp")],
    )


def test_posonly_input_accepted() -> None:
    validate_workflow_script("async def main(input, /):\n    return input\n")


def test_last_main_binding_wins() -> None:
    # exec semantics: a later top-level ``main`` shadows an earlier one — the valid
    # async one is the effective entry, so this is accepted.
    script = "def main(input):\n    return 1\nasync def main(input):\n    return 2\n"
    validate_workflow_script(script)


# ── 7. un-AST-able names excluded (no false rejection) ───────────────────────


def test_computed_tool_name_accepted() -> None:
    script = (
        "async def main(input):\n"
        '    name_var = input["tool"]\n'
        "    return await tool(name_var, {})\n"
    )
    validate_workflow_script(script, tools=[])


def test_agent_id_from_input_accepted() -> None:
    script = 'async def main(input):\n    return await agent({}, agent_id=input["a"])\n'
    validate_workflow_script(script)


def test_agent_id_from_variable_accepted() -> None:
    script = (
        "SCOUT = input_unused = None\n"
        "async def main(input):\n"
        '    aid = input["who"]\n'
        "    return await agent({}, agent_id=aid)\n"
    )
    validate_workflow_script(script)


def test_fstring_tool_name_accepted() -> None:
    script = "async def main(input):\n    return await tool(f\"{input['kind']}_tool\", {})\n"
    validate_workflow_script(script)


def test_mixed_literal_and_computed_only_literal_checked() -> None:
    # The computed call is excluded; the literal ``"bash"`` is the only required name.
    script = 'async def main(input):\n    await tool(input["x"], {})\n    await tool("bash", {})\n'
    with pytest.raises(WorkflowScriptValidationError) as exc:
        validate_workflow_script(script, tools=[])
    assert exc.value.detail.get("missing_tools") == ["bash"]


def test_attribute_call_not_treated_as_capability() -> None:
    # ``obj.tool("x")`` is not the injected ``tool(...)`` capability.
    script = (
        "class C:\n    def tool(self, n, i):\n        return None\n"
        "async def main(input):\n"
        '    return C().tool("bash", {})\n'
    )
    validate_workflow_script(script, tools=[])
