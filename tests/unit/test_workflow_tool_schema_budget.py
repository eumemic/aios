"""The model-facing schema budget for the workflow-authoring trio (#2294).

``create_workflow`` / ``update_workflow`` / ``call_workflow`` used to render at
~69KB combined — the fattest schemas on the tool surface — and that payload rode
along in EVERY request of every agent holding them, workflow turn or not. The
diet (:mod:`aios.tools.schema_diet`) cut it to ~6KB by moving the authoring
manual to an on-demand tool and collapsing the declared-surface config trees.

These tests pin all three halves of that trade, because each can silently rot:

* the **budget** — a future field description, or a re-inlined ``$ref``, quietly
  reinflates the always-on cost; the byte assertion is the only thing that
  notices;
* the **reachability** — the trimmed ``script`` description promises a tool that
  serves the contract, so that tool must exist, be offered alongside the
  authoring tools, and return the real contract; and
* the **validation** — the diet loosens what the dispatch-time jsonschema check
  enforces, so the pydantic models behind the handlers must still reject a
  malformed body with a field-precise error, and the HTTP/SDK schema must be
  entirely untouched.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import aios.tools  # noqa: F401 - trigger built-in registration side effects
from aios.harness.step_context import _inject_workflow_script_contract
from aios.harness.tokens import approx_tokens
from aios.models.workflows import WORKFLOW_SCRIPT_CONTRACT, WorkflowCreate
from aios.tools.input import tool_input
from aios.tools.invoke import ToolBail, prepare_builtin
from aios.tools.registry import openai_tool_entry, registry
from aios.tools.workflow_management import (
    SCRIPT_CONTRACT_TOOL_NAME,
    WORKFLOW_AUTHORING_TOOL_NAMES,
    get_workflow_script_contract_handler,
)

#: Per-tool ceiling on the rendered chat-completions entry (description +
#: schema), from #2294's acceptance criteria. Deliberately tight: crossing it is
#: a decision to re-spend always-on context, and should be made consciously —
#: trim the prose, or move it behind ``get_workflow_script_contract``.
MAX_RENDERED_BYTES = 2_500

#: The trio's combined ceiling, benchmarked against Claude Code's own
#: ``Workflow`` tool (5,473 bytes for the same conceptual surface).
MAX_TRIO_BYTES = 6_500


def _rendered_size(name: str) -> int:
    return len(json.dumps(openai_tool_entry(registry.get(name)), separators=(",", ":")))


@pytest.mark.parametrize("name", sorted(WORKFLOW_AUTHORING_TOOL_NAMES))
def test_authoring_tool_schema_within_budget(name: str) -> None:
    size = _rendered_size(name)
    assert size <= MAX_RENDERED_BYTES, (
        f"{name} renders at {size} bytes (budget {MAX_RENDERED_BYTES}). This ships in "
        "every request of every agent holding the tool. Trim the prose, or move it "
        f"behind {SCRIPT_CONTRACT_TOOL_NAME} (#2294)."
    )


def test_authoring_trio_within_combined_budget() -> None:
    total = sum(_rendered_size(name) for name in WORKFLOW_AUTHORING_TOOL_NAMES)
    assert total <= MAX_TRIO_BYTES, f"trio renders at {total} bytes (budget {MAX_TRIO_BYTES})"


@pytest.mark.parametrize("name", sorted(WORKFLOW_AUTHORING_TOOL_NAMES))
def test_authoring_manual_is_not_inlined(name: str) -> None:
    """The contract's own prose must not reappear in any always-on schema."""
    rendered = json.dumps(openai_tool_entry(registry.get(name)))
    # A distinctive line from the contract body — present iff it got re-inlined.
    assert "Injected capability API" not in rendered
    assert len(WORKFLOW_SCRIPT_CONTRACT) > len(rendered)


@pytest.mark.parametrize("name", sorted(WORKFLOW_AUTHORING_TOOL_NAMES))
def test_declared_surface_fields_are_schema_opaque(name: str) -> None:
    """No ToolSpec / McpServerSpec / HttpServerSpec tree survives in the schema."""
    schema = registry.get(name).parameters_schema
    rendered = json.dumps(schema)
    for model_name in ("ToolSpec", "McpServerSpec", "HttpServerSpec", "McpToolsetConfig"):
        assert model_name not in rendered, f"{name} still expands {model_name}"


@pytest.mark.parametrize("name", sorted(WORKFLOW_AUTHORING_TOOL_NAMES))
def test_trusted_id_injection_still_refused(name: str) -> None:
    """``additionalProperties: false`` is load-bearing and survives the diet."""
    schema = registry.get(name).parameters_schema
    assert schema["additionalProperties"] is False
    with pytest.raises(ToolBail):
        prepare_builtin(name, {"creator_session_id": "sess_evil"})


# ── every rendered schema must be countable by the real token counter ──────
#
# The byte budgets above measure ``json.dumps`` length; the loop unit tests
# patch ``prelude_overhead_local`` to 0. Neither ever runs the REAL
# ``litellm.token_counter`` over a rendered registry schema — which is exactly
# how #2294's bare ``{"type": "array"}`` (no ``items``) shipped: litellm's
# ``_format_type`` does ``props['items']`` unconditionally, so every step of
# every workflow-capable agent raised ``KeyError: 'items'`` in production and
# the fleet was rolled back. These tests close that gap: every registered
# tool's rendered entry must survive the counter, individually and all at once.


@pytest.mark.parametrize("name", registry.names())
def test_every_registered_schema_is_countable(name: str) -> None:
    assert approx_tokens([], tools=[openai_tool_entry(registry.get(name))]) > 0


def test_whole_registry_is_countable_at_once() -> None:
    tools = [openai_tool_entry(registry.get(name)) for name in registry.names()]
    assert approx_tokens([], tools=tools) > 0


# ── the on-demand half ─────────────────────────────────────────────────────


def test_script_field_points_at_a_real_tool() -> None:
    """The trimmed ``script`` description names a tool that actually exists."""
    schema = registry.get("create_workflow").parameters_schema
    description = schema["properties"]["script"]["description"]
    assert SCRIPT_CONTRACT_TOOL_NAME in description
    assert registry.has(SCRIPT_CONTRACT_TOOL_NAME)
    assert registry.get(SCRIPT_CONTRACT_TOOL_NAME).transport == "agent_tool"


async def test_contract_tool_returns_the_whole_contract() -> None:
    result = await get_workflow_script_contract_handler("sess_x", {})
    assert result == {"contract": WORKFLOW_SCRIPT_CONTRACT}


def _offered(*names: str) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": name}} for name in names]


def _names(tools: list[dict[str, Any]]) -> list[str]:
    return [entry["function"]["name"] for entry in tools]


def test_contract_tool_is_injected_alongside_the_authoring_tools() -> None:
    tools = _offered("create_workflow")
    _inject_workflow_script_contract(tools)
    assert _names(tools) == ["create_workflow", SCRIPT_CONTRACT_TOOL_NAME]


def test_contract_tool_injection_is_idempotent() -> None:
    """An agent that also declares the tool gets exactly one entry."""
    tools = _offered("create_workflow", SCRIPT_CONTRACT_TOOL_NAME)
    _inject_workflow_script_contract(tools)
    assert _names(tools).count(SCRIPT_CONTRACT_TOOL_NAME) == 1


def test_contract_tool_not_injected_without_an_authoring_tool() -> None:
    tools = _offered("list_workflows")
    _inject_workflow_script_contract(tools)
    assert _names(tools) == ["list_workflows"]


# ── validation and the HTTP/SDK contract are untouched ─────────────────────


def test_malformed_declared_surface_still_gets_a_field_precise_error() -> None:
    """The schema is opaque; pydantic is not. A bad entry names its own field."""
    with pytest.raises(ToolBail) as excinfo:
        tool_input(
            WorkflowCreate,
            {"name": "wf", "script": "async def main(input): ...", "tools": [{"type": "nope"}]},
        )
    message = str(excinfo.value)
    assert "tools" in message
    assert "nope" in message


def test_valid_declared_surface_still_admitted_through_dispatch() -> None:
    """Positive control: the diet loosens, so a well-formed surface still passes."""
    arguments = {
        "name": "wf",
        "script": "async def main(input): ...",
        "tools": [{"type": "bash"}],
    }
    assert prepare_builtin("create_workflow", arguments) == arguments
    assert tool_input(WorkflowCreate, arguments).tools[0].type == "bash"


def test_http_sdk_schema_keeps_the_full_models() -> None:
    """#2294 is a model-facing change only — openapi.json must not move."""
    schema = WorkflowCreate.model_json_schema()
    assert "ToolSpec" in schema["$defs"]
    assert "McpServerSpec" in schema["$defs"]
    assert schema["properties"]["script"]["description"] == WORKFLOW_SCRIPT_CONTRACT
