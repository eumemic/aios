"""Unit tests for tool_dispatch helpers.

``_validate_arguments`` is the interesting one — it converts JSON Schema
failures into a model-readable error string that lists every problem at
once so a single retry can fix them all.
"""

from __future__ import annotations

from typing import Any

from aios.harness.tool_dispatch import _validate_arguments

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "channel_id": {"type": ["string", "null"]},
    },
    "required": ["channel_id"],
    "additionalProperties": False,
}


class TestValidateArguments:
    def test_valid_arguments_return_none(self) -> None:
        assert _validate_arguments({"channel_id": "signal/a/b"}, _SCHEMA) is None
        assert _validate_arguments({"channel_id": None}, _SCHEMA) is None

    def test_missing_required_key_is_reported(self) -> None:
        err = _validate_arguments({}, _SCHEMA)
        assert err is not None
        assert "'channel_id' is a required property" in err

    def test_unexpected_property_is_reported(self) -> None:
        """The common weak-model failure mode: wrong param name.  The
        extra-property error must surface so the model can correct it.
        """
        err = _validate_arguments({"target": "signal/a/b"}, _SCHEMA)
        assert err is not None
        assert "'target' was unexpected" in err

    def test_wrong_type_is_reported(self) -> None:
        err = _validate_arguments({"channel_id": 42}, _SCHEMA)
        assert err is not None
        assert "42" in err  # what was sent
        # jsonschema's message varies; just assert we got SOMETHING useful
        assert "channel_id" in err or "type" in err.lower() or "42" in err

    def test_multi_error_accumulation(self) -> None:
        """Missing required + extra property → BOTH surface in a single
        error.  Model sees every issue at once; one retry fixes all.
        """
        err = _validate_arguments({"target": "x"}, _SCHEMA)
        assert err is not None
        assert "'channel_id' is a required property" in err
        assert "'target' was unexpected" in err

    def test_passed_arguments_echoed_for_context(self) -> None:
        """The error includes the arguments the model sent so the model
        can see exactly what shape it produced vs. what was expected.
        """
        err = _validate_arguments({"target": "oops"}, _SCHEMA)
        assert err is not None
        assert '"target": "oops"' in err

    def test_error_ends_with_hint_to_consult_schema(self) -> None:
        """Closing guidance points the model back at the tool's
        declared ``parameters`` — its authoritative reference."""
        err = _validate_arguments({"target": "x"}, _SCHEMA)
        assert err is not None
        assert "parameters" in err.lower()

    def test_permissive_schema_allows_anything(self) -> None:
        """A schema with no constraints (no required, no additionalProps
        false) accepts arbitrary payloads — validation is opt-in per
        tool."""
        loose: dict[str, Any] = {"type": "object"}
        assert _validate_arguments({"anything": "goes"}, loose) is None
        assert _validate_arguments({}, loose) is None
