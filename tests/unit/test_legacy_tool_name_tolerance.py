"""The contracted builtin-tool renames are no longer read-tolerated."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.models.agents import ToolSpec


@pytest.mark.parametrize(
    "legacy",
    ["invoke", "invoke_agent", "invoke_workflow", "create_run", "await_run", "cancel_run"],
)
def test_legacy_builtin_name_is_rejected_after_contract(legacy: str) -> None:
    with pytest.raises(ValidationError):
        ToolSpec.model_validate({"type": legacy})


def test_canonical_names_remain_valid() -> None:
    assert ToolSpec.model_validate({"type": "call_workflow"}).type == "call_workflow"
