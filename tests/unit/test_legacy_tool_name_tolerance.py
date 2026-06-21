"""Read-tolerance for the builtin tool renames (#1419 invoke*→call_*, #1428 cancel_run→stop_task).

Agent/workflow/run/session rows persisted before a rename carry legacy builtin names
(``invoke``/``invoke_agent``/``invoke_workflow``/``create_run``/``await_run`` from #1419,
``cancel_run`` from #1428) in their ``tools`` JSONB. The ``ToolSpec`` ``mode="before"`` validator
maps them so old rows still load; ``to_openai_tools`` dedupes the ``create_run``/``invoke_workflow``
collapse so the model never sees a duplicate ``call_workflow``. (The data migrations 0116 / 0117
rewrite the persisted rows; the shim covers the post-deploy/pre-migrate window.)
"""

from __future__ import annotations

import pytest

from aios.models.agents import ToolSpec
from aios.tools.registry import to_openai_tools


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        ("invoke", "call_session"),
        ("invoke_agent", "call_agent"),
        ("invoke_workflow", "call_workflow"),
        ("create_run", "call_workflow"),
        ("await_run", "call_workflow"),
        # #1428: the cancel_run MODEL tool superseded by stop_task. Maps to a REGISTERED
        # name so to_openai_tools' registry.get doesn't raise in the post-deploy window.
        ("cancel_run", "stop_task"),
    ],
)
def test_legacy_builtin_name_maps_to_canonical(legacy: str, canonical: str) -> None:
    """A ToolSpec persisted with a pre-rename builtin name validates to the new name."""
    spec = ToolSpec.model_validate({"type": legacy})
    assert spec.type == canonical


def test_legacy_name_preserves_overrides() -> None:
    """The rename keeps the spec's other fields (permission/transport overrides)."""
    spec = ToolSpec.model_validate(
        {"type": "invoke_workflow", "permission": "always_ask", "enabled": False}
    )
    assert spec.type == "call_workflow"
    assert spec.permission == "always_ask" and spec.enabled is False


def test_canonical_names_unaffected() -> None:
    spec = ToolSpec.model_validate({"type": "call_workflow"})
    assert spec.type == "call_workflow"


def test_collapse_is_deduped_in_model_surface() -> None:
    """invoke_workflow + create_run both map to call_workflow; the model sees it ONCE,
    with original order otherwise preserved."""
    tools = [
        ToolSpec(type="bash"),
        ToolSpec.model_validate({"type": "invoke_workflow"}),
        ToolSpec.model_validate({"type": "create_run"}),
        ToolSpec.model_validate({"type": "invoke_agent"}),
    ]
    names = [t["function"]["name"] for t in to_openai_tools(tools)]
    assert names == ["bash", "call_workflow", "call_agent"]
