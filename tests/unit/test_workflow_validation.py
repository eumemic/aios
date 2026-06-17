"""Unit tests for create-time workflow-script validation (#1284, part of #777).

Pure (no DB): exercises the validator core in ``aios.workflows.validation`` —
compile-check, ``async def main(input)`` structure assertion, and the literal-only
**validate-declared** tool-surface superset check. The service-path / agent-surface
and update-time wiring is covered in ``test_workflow_validation_service.py``.

Maps to the issue's acceptance criteria:
  1. syntax error          → rejected (compile/syntax message, line/offset)
  2. missing ``main``       → rejected (names the required ``async def main(input)``)
  3. mis-signatured ``main``→ rejected (not async / wrong arity)
  4. under-declared tool    → rejected (names the missing tool)
  6. valid covered          → accepted
  7. un-AST-able names      → accepted (excluded from the required-surface check)
"""

from __future__ import annotations

import pytest

from aios.errors import ValidationError
from aios.models.agents import ToolSpec
from aios.workflows.validation import (
    extract_required_surface,
    validate_workflow_script,
)

_GOOD_MAIN = "async def main(input):\n    return input\n"


def _err(fn) -> ValidationError:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError) as ei:
        fn()
    return ei.value


# ── criterion 1: syntax / structure errors ───────────────────────────────────


class TestSyntaxError:
    def test_unbalanced_paren_rejected_as_compile_error(self) -> None:
        err = _err(lambda: validate_workflow_script("async def main(input):\n    x = (\n"))
        assert "compile" in str(err).lower()
        assert err.detail["reason"] == "compile_error"
        # The error identifies it as a compile/syntax failure and surfaces a line.
        assert err.detail["lineno"] is not None

    def test_stray_token_rejected(self) -> None:
        err = _err(lambda: validate_workflow_script("async def main(input):\n    return @@\n"))
        assert err.error_type == "validation_error"
        assert err.detail["reason"] == "compile_error"


# ── criterion 2: missing main ─────────────────────────────────────────────────


class TestMissingMain:
    def test_no_main_at_all(self) -> None:
        err = _err(lambda: validate_workflow_script("x = 1\n"))
        assert "main" in str(err)
        assert err.detail["reason"] == "missing_main"

    def test_compiles_but_only_helper(self) -> None:
        err = _err(lambda: validate_workflow_script("async def helper(input):\n    return 1\n"))
        assert err.detail["reason"] == "missing_main"

    def test_main_must_be_top_level(self) -> None:
        # A nested `async def main` does not count (not a top-level entry point).
        src = "async def outer(input):\n    async def main(input):\n        return 1\n"
        err = _err(lambda: validate_workflow_script(src))
        assert err.detail["reason"] == "missing_main"


# ── criterion 3: mis-signatured main ──────────────────────────────────────────


class TestMisSignaturedMain:
    def test_plain_def_not_async(self) -> None:
        err = _err(lambda: validate_workflow_script("def main(input):\n    return 1\n"))
        assert err.detail["reason"] == "main_not_async"
        assert "async" in str(err)

    def test_async_main_zero_params(self) -> None:
        err = _err(lambda: validate_workflow_script("async def main():\n    return 1\n"))
        assert err.detail["reason"] == "bad_main_signature"
        assert "input" in str(err)

    def test_async_main_two_params(self) -> None:
        err = _err(lambda: validate_workflow_script("async def main(a, b):\n    return 1\n"))
        assert err.detail["reason"] == "bad_main_signature"

    def test_async_main_varargs_only(self) -> None:
        err = _err(lambda: validate_workflow_script("async def main(*args):\n    return 1\n"))
        assert err.detail["reason"] == "bad_main_signature"

    def test_async_main_kwargs_only(self) -> None:
        err = _err(lambda: validate_workflow_script("async def main(**kw):\n    return 1\n"))
        assert err.detail["reason"] == "bad_main_signature"

    def test_async_main_single_param_with_default_accepted(self) -> None:
        # A trailing default on the single `input` parameter still binds main(input).
        validate_workflow_script("async def main(input=None):\n    return input\n")

    def test_single_input_param_accepted(self) -> None:
        # The happy-path shape — accepted (no raise).
        validate_workflow_script(_GOOD_MAIN)

    def test_positional_only_input_accepted(self) -> None:
        validate_workflow_script("async def main(input, /):\n    return input\n")


# ── criterion 4: under-declared tool surface ──────────────────────────────────


class TestUnderDeclaredTool:
    def test_literal_tool_not_declared_rejected(self) -> None:
        src = _GOOD_MAIN + "\nasync def _x(input):\n    await tool('http_request', {})\n"
        # tool() must be inside main for realism, but extraction is module-wide:
        src = "async def main(input):\n    return await tool('http_request', {})\n"
        err = _err(lambda: validate_workflow_script(src, tools=[]))
        assert err.detail["reason"] == "undeclared_tools"
        assert "http_request" in err.detail["missing_tools"]
        assert "http_request" in str(err)

    def test_partial_declaration_names_only_missing(self) -> None:
        src = (
            "async def main(input):\n"
            "    await tool('bash', {})\n"
            "    return await tool('http_request', {})\n"
        )
        err = _err(lambda: validate_workflow_script(src, tools=[ToolSpec(type="bash")]))
        assert err.detail["missing_tools"] == ["http_request"]

    def test_custom_tool_referenced_by_name(self) -> None:
        src = "async def main(input):\n    return await tool('my_custom', {})\n"
        # Declared as a custom tool whose *name* is 'my_custom' → covered.
        validate_workflow_script(
            src, tools=[ToolSpec(type="custom", name="my_custom", description="d", input_schema={})]
        )
        # A custom tool with a different name does NOT cover it.
        err = _err(
            lambda: validate_workflow_script(
                src, tools=[ToolSpec(type="custom", name="other", description="d", input_schema={})]
            )
        )
        assert "my_custom" in err.detail["missing_tools"]


# ── criterion 6: valid, fully-covered script accepted ─────────────────────────


class TestValidCovered:
    def test_fully_covered_script_accepted(self) -> None:
        src = (
            "async def main(input):\n"
            "    a = await tool('bash', {'command': 'echo hi'})\n"
            "    b = await tool('http_request', {})\n"
            "    return [a, b]\n"
        )
        extracted = validate_workflow_script(
            src,
            tools=[ToolSpec(type="bash"), ToolSpec(type="http_request")],
        )
        assert extracted.tool_names == {"bash", "http_request"}
        assert extracted.agent_ids == set()

    def test_no_tool_calls_accepted(self) -> None:
        validate_workflow_script(_GOOD_MAIN, tools=[])


# ── criterion 7: un-AST-able names do not cause false rejection ────────────────


class TestUnAstableExcluded:
    def test_variable_tool_name_accepted(self) -> None:
        src = "async def main(input):\n    name = input['tool']\n    return await tool(name, {})\n"
        # Declared tools empty — still accepted, the computed name is excluded.
        extracted = validate_workflow_script(src, tools=[])
        assert extracted.tool_names == set()

    def test_agent_id_from_input_accepted(self) -> None:
        src = (
            "async def main(input):\n    return await agent({'x': 1}, agent_id=input['agent_id'])\n"
        )
        extracted = validate_workflow_script(src, tools=[])
        assert extracted.agent_ids == set()

    def test_mixed_literal_and_computed(self) -> None:
        src = (
            "async def main(input):\n"
            "    n = input['t']\n"
            "    await tool(n, {})\n"
            "    return await tool('bash', {})\n"
        )
        # Only the literal 'bash' participates; the computed name is excluded.
        err = _err(lambda: validate_workflow_script(src, tools=[]))
        assert err.detail["missing_tools"] == ["bash"]


# ── extraction surface (agent_id literals participate) ────────────────────────


class TestExtraction:
    def test_literal_agent_id_extracted(self) -> None:
        src = "async def main(input):\n    return await agent({'x': 1}, agent_id='researcher')\n"
        extracted = extract_required_surface(src)
        assert extracted.agent_ids == {"researcher"}

    def test_attribute_call_not_treated_as_builtin_tool(self) -> None:
        # `obj.tool('x', ...)` is an attribute access, not the injected builtin.
        src = "async def main(input):\n    obj = input\n    return obj.tool('x', {})\n"
        extracted = extract_required_surface(src)
        assert extracted.tool_names == set()
