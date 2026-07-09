"""Unit tests for the shared schema-violation formatter (#1769).

Covers the ratified spec-v2 rules directly against
:func:`aios.tools.schema_errors.format_schema_violation`:

* no instance echo — long/container instances never appear verbatim;
* short scalar instances (<=100 chars) may appear where they ARE the
  diagnosis (enum/const);
* every ``type`` mismatch line reads ``expected <X>, got <json-type>``;
* the schema is included (full when small, else the failing subschema);
* the stringified-JSON quirk is detected (hint fires) without ever coercing —
  the return value from :func:`format_schema_violation` on a match is the
  human-facing string, the CALLER decides whether to accept (it never does);
* a conforming instance returns ``None``.
"""

from __future__ import annotations

import json
from typing import Any

from aios.tools.schema_errors import format_schema_violation, json_type_name

_OBJ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class TestJsonTypeName:
    def test_bool_before_int(self) -> None:
        # The load-bearing ordering: bool is an int subclass in Python.
        assert json_type_name(True) == "boolean"
        assert json_type_name(False) == "boolean"

    def test_ladder(self) -> None:
        assert json_type_name(None) == "null"
        assert json_type_name(3) == "integer"
        assert json_type_name(3.5) == "number"
        assert json_type_name("x") == "string"
        assert json_type_name([1, 2]) == "array"
        assert json_type_name({"a": 1}) == "object"


class TestFormatSchemaViolation:
    def test_conforming_value_returns_none(self) -> None:
        assert (
            format_schema_violation(
                {"answer": "hi"},
                _OBJ_SCHEMA,
                root="value",
                intro="bad",
                retry_hint="retry",
                site="test",
            )
            is None
        )

    def test_no_instance_echo_for_large_payload(self) -> None:
        big_blob = json.dumps({"branch": "x" * 3000, "pr_title": "y" * 2000})
        err = format_schema_violation(
            big_blob,
            _OBJ_SCHEMA,
            root="value",
            intro="output_schema_violation: `value` does not conform to the request's output_schema.",
            retry_hint="Provide `value` as a conforming object and call `return` again.",
            site="test",
        )
        assert err is not None
        # The multi-KB payload must NOT appear verbatim anywhere in the message.
        assert big_blob not in err
        assert "x" * 3000 not in err
        # The type mismatch line is present, no-echo style.
        assert "expected object, got string" in err
        assert "at value:" in err

    def test_type_mismatch_reports_expected_and_got(self) -> None:
        err = format_schema_violation(
            {"answer": 1},
            _OBJ_SCHEMA,
            root="value",
            intro="bad",
            retry_hint="retry",
            site="test",
        )
        assert err is not None
        assert "at value.answer: expected string, got integer" in err

    def test_missing_required_and_additional_property_both_surface(self) -> None:
        err = format_schema_violation(
            {"extra": 1},
            _OBJ_SCHEMA,
            root="value",
            intro="bad",
            retry_hint="retry",
            site="test",
        )
        assert err is not None
        assert "'answer' is a required property" in err
        assert "'extra' was unexpected" in err

    def test_short_scalar_enum_instance_shown_verbatim(self) -> None:
        schema = {"enum": ["x", "y"]}
        err = format_schema_violation(
            "z", schema, root="value", intro="bad", retry_hint="retry", site="test"
        )
        assert err is not None
        assert "'z' is not one of ['x', 'y']" in err  # short scalar: verbatim is fine

    def test_long_string_pattern_mismatch_not_echoed(self) -> None:
        schema = {"type": "string", "pattern": "^a"}
        long_str = "b" * 500
        err = format_schema_violation(
            long_str, schema, root="value", intro="bad", retry_hint="retry", site="test"
        )
        assert err is not None
        assert long_str not in err
        assert "<string>" in err  # collapsed to type name, not the 500-char value

    def test_schema_included_when_small(self) -> None:
        err = format_schema_violation(
            {"answer": 1}, _OBJ_SCHEMA, root="value", intro="bad", retry_hint="retry", site="test"
        )
        assert err is not None
        assert '"answer"' in err  # the schema itself rendered

    def test_large_schema_falls_back_to_failing_subschema(self) -> None:
        big_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                **{f"filler_{i}": {"type": "string", "description": "x" * 50} for i in range(60)},
            },
            "required": ["answer"],
        }
        assert len(json.dumps(big_schema)) > 2000
        err = format_schema_violation(
            {"answer": 1}, big_schema, root="value", intro="bad", retry_hint="retry", site="test"
        )
        assert err is not None
        # Only the failing subschema `{"type": "string"}` should appear, not the
        # full multi-KB schema (no `filler_` keys leaking through).
        assert "filler_" not in err

    def test_stringified_json_quirk_detected_with_hint(self) -> None:
        payload = json.dumps({"answer": "hi"})
        err = format_schema_violation(
            payload, _OBJ_SCHEMA, root="value", intro="bad", retry_hint="retry", site="test"
        )
        assert err is not None
        assert "pass that value directly, not wrapped in a string" in err

    def test_non_json_string_no_hint(self) -> None:
        err = format_schema_violation(
            "not json at all",
            _OBJ_SCHEMA,
            root="value",
            intro="bad",
            retry_hint="retry",
            site="test",
        )
        assert err is not None
        assert "pass that value directly" not in err

    def test_json_string_that_does_not_match_schema_no_hint(self) -> None:
        payload = json.dumps({"answer": 1})  # parses, but still fails the schema
        err = format_schema_violation(
            payload, _OBJ_SCHEMA, root="value", intro="bad", retry_hint="retry", site="test"
        )
        assert err is not None
        assert "pass that value directly" not in err

    def test_retry_hint_appended_when_provided(self) -> None:
        err = format_schema_violation(
            {"answer": 1},
            _OBJ_SCHEMA,
            root="value",
            intro="bad",
            retry_hint="call return again",
            site="test",
        )
        assert err is not None
        assert err.rstrip().endswith("call return again")

    def test_retry_hint_omitted_when_none(self) -> None:
        """A fail-loud caller (workflow run output) passes no retry hint — the
        message must not end with a stray retry instruction."""
        err = format_schema_violation(
            {"answer": 1}, _OBJ_SCHEMA, root="output", intro="bad", retry_hint=None, site="test"
        )
        assert err is not None
        assert "call return again" not in err
        assert "retry" not in err.lower()

    def test_bare_root_uses_root_token_when_path_empty(self) -> None:
        err = format_schema_violation(
            "not-an-object",
            {"type": "object"},
            root="output",
            intro="bad",
            retry_hint=None,
            site="test",
        )
        assert err is not None
        assert "at output: expected object, got string" in err

    def test_empty_root_uses_angle_root_token(self) -> None:
        err = format_schema_violation(
            "nope",
            {"type": "object"},
            root="",
            intro="bad",
            retry_hint=None,
            site="test",
        )
        assert err is not None
        assert "at <root>: expected object, got string" in err
