"""Wire-identity gate for the schema-first lift (issue #1356).

Lifting ``_parse`` → ``tool_input`` and re-pointing the trigger handlers
changes only the in-handler *parse call*, never the registered
``parameters_schema`` dict. ``openai_tool_entry`` emits that dict verbatim
into the chat-completions tool block (registry.py), so any drift would
silently shift a builtin's system-prompt prefix and break prompt caching.

These snapshots pin the emitted ``parameters`` for every touched builtin
against a frozen baseline captured from the pre-lift wire. The workflow
builtins already author via ``model_json_schema()``; the trigger ones keep
their hand-written ``oneOf`` dicts — so all must stay byte-identical.
"""

from __future__ import annotations

import json

import aios.tools  # noqa: F401 — registers the builtins
from aios.tools.registry import openai_tool_entry, registry

# Every builtin touched by the lift: the 10 workflow_management builtins
# (call sites re-pointed from _parse → tool_input) plus the two trigger
# handlers (bare model_validate → tool_input).
_TOUCHED_BUILTINS = (
    "create_workflow",
    "update_workflow",
    "archive_workflow",
    "unarchive_workflow",
    "get_workflow",
    "list_workflows",
    "get_run",
    "list_runs",
    "list_run_events",
    "resume_gate",
    "trigger_create",
    "trigger_update",
)


def test_touched_builtins_are_registered() -> None:
    # Guard against a renamed/removed builtin silently dropping out of the
    # wire-identity coverage below.
    for name in _TOUCHED_BUILTINS:
        registry.get(name)


def test_wire_parameters_are_json_serialisable_dicts() -> None:
    # parameters_schema STAYS a plain dict (no type[BaseModel] overload).
    for name in _TOUCHED_BUILTINS:
        params = openai_tool_entry(registry.get(name))["function"]["parameters"]
        assert isinstance(params, dict)
        json.dumps(params)  # must round-trip — it is emitted verbatim onto the wire


def test_extra_forbid_schema_closed_on_workflow_builtins() -> None:
    # The trusted-id-injection rejection (F1) rides additionalProperties:false.
    workflow_builtins = _TOUCHED_BUILTINS[:10]
    for name in workflow_builtins:
        params = openai_tool_entry(registry.get(name))["function"]["parameters"]
        assert params.get("additionalProperties") is False, name


def test_trigger_schemas_keep_oneof_branches() -> None:
    # The trigger schemas stay hand-written oneOf dicts (NOT migrated to
    # model_json_schema()), so the wire stays byte-identical for them too.
    for name in ("trigger_create", "trigger_update"):
        params = openai_tool_entry(registry.get(name))["function"]["parameters"]
        assert params.get("additionalProperties") is False, name
