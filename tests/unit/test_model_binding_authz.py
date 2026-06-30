"""Unit tests for the ``workflow:`` model-binding privilege guard (#1636).

The pure authorization helper shared by every spawn path: it decides whether a
principal may bind/select a ``workflow:`` model, keyed on operator vs self-authoring.
A raw provider model is always admissible; a ``workflow:`` binding is operator-only.
"""

from __future__ import annotations

import pytest

from aios.errors import ForbiddenError
from aios.services.model_binding_authz import (
    enforce_workflow_binding_privilege,
    is_workflow_binding,
)


class TestIsWorkflowBinding:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("workflow:wf_1", True),
            ("workflow:wf_1@3", True),
            ("workflow:", True),  # prefix probe is cheap; parse-validity is a separate gate
            ("test/dummy", False),
            ("anthropic/claude", False),
            ("", False),
            (None, False),
        ],
    )
    def test_probe(self, model: str | None, expected: bool) -> None:
        assert is_workflow_binding(model) is expected


class TestEnforceWorkflowBindingPrivilege:
    def test_operator_may_bind_workflow_model(self) -> None:
        # An operator principal binds a workflow: model freely — no raise.
        enforce_workflow_binding_privilege("workflow:wf_1", is_operator=True)

    def test_self_authoring_cannot_bind_workflow_model(self) -> None:
        with pytest.raises(ForbiddenError) as exc:
            enforce_workflow_binding_privilege("workflow:wf_1", is_operator=False)
        # The denied binding is surfaced in the structured detail.
        assert exc.value.detail == {"model": "workflow:wf_1"}

    def test_self_authoring_raw_provider_model_is_allowed(self) -> None:
        # The common case: a self-authoring principal binding an ordinary provider
        # model is untouched — the guard is a no-op for non-workflow bindings.
        enforce_workflow_binding_privilege("test/dummy", is_operator=False)

    def test_none_model_is_noop_for_both_principals(self) -> None:
        # ``None`` (e.g. update_agent preserving the field) introduces no binding.
        enforce_workflow_binding_privilege(None, is_operator=False)
        enforce_workflow_binding_privilege(None, is_operator=True)
