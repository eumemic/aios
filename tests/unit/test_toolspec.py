"""Unit tests for ToolSpec model — enabled, permission, backward compat."""

from __future__ import annotations

import json

import pytest

from aios.models.agents import PermissionPolicy, ToolSpec


class TestDefaults:
    def test_builtin_defaults(self) -> None:
        spec = ToolSpec(type="bash")
        assert spec.enabled is True
        assert spec.permission is None

    def test_custom_defaults(self) -> None:
        spec = ToolSpec(
            type="custom",
            name="foo",
            description="bar",
            input_schema={"type": "object"},
        )
        assert spec.enabled is True
        assert spec.permission is None


class TestPermissionField:
    def test_always_allow(self) -> None:
        spec = ToolSpec(type="bash", permission="always_allow")
        assert spec.permission == "always_allow"

    def test_always_ask(self) -> None:
        spec = ToolSpec(type="bash", permission="always_ask")
        assert spec.permission == "always_ask"

    def test_invalid_policy_rejected(self) -> None:
        with pytest.raises(ValueError):
            ToolSpec(type="bash", permission="always_deny")  # type: ignore[arg-type]


class TestEnabledField:
    def test_disabled(self) -> None:
        spec = ToolSpec(type="bash", enabled=False)
        assert spec.enabled is False

    def test_disabled_custom(self) -> None:
        spec = ToolSpec(
            type="custom",
            name="foo",
            description="bar",
            input_schema={"type": "object"},
            enabled=False,
        )
        assert spec.enabled is False


class TestSerialization:
    def test_round_trip(self) -> None:
        spec = ToolSpec(type="bash", enabled=False, permission="always_ask")
        d = spec.model_dump()
        restored = ToolSpec.model_validate(d)
        assert restored.enabled is False
        assert restored.permission == "always_ask"

    def test_json_round_trip(self) -> None:
        spec = ToolSpec(type="bash", permission="always_ask")
        j = spec.model_dump_json()
        restored = ToolSpec.model_validate_json(j)
        assert restored.permission == "always_ask"


class TestBackwardCompat:
    """Old JSONB rows from before the enabled/permission fields were added."""

    def test_old_builtin_json(self) -> None:
        old_json = '{"type": "bash"}'
        spec = ToolSpec.model_validate_json(old_json)
        assert spec.enabled is True
        assert spec.permission is None

    def test_old_custom_json(self) -> None:
        old_json = json.dumps(
            {
                "type": "custom",
                "name": "foo",
                "description": "bar",
                "input_schema": {"type": "object"},
            }
        )
        spec = ToolSpec.model_validate_json(old_json)
        assert spec.enabled is True
        assert spec.permission is None

    def test_old_tools_list(self) -> None:
        """Simulate deserializing an agent's tools list from Postgres JSONB."""
        raw = [{"type": "bash"}, {"type": "read"}]
        specs = [ToolSpec.model_validate(t) for t in raw]
        assert all(s.enabled is True for s in specs)
        assert all(s.permission is None for s in specs)


class TestPermissionPolicyType:
    def test_type_alias_values(self) -> None:
        # Just verify the type alias is accessible and correct.
        assert PermissionPolicy.__args__ == ("always_allow", "always_ask")  # type: ignore[attr-defined]
