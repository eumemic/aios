"""The ``return``-tool stringified-JSON coercion gate (#1769 / PR #2096).

``_enforce_output_schema`` may accept a double-encoded ``value`` (a JSON string
that parses to a conforming value) — but acceptance and transformation must not
split: whatever the schema gate validated is what ``return_handler`` must hand
to ``_finish`` (and therefore to the awaiting caller).

Three properties, one per failure direction:

* **Propagation** — a JSON-encoded object that parses to a conforming object is
  accepted AND the value the consumer receives is the parsed ``dict``, never the
  original ``str``.
* **Positive control** — a value that ALREADY conforms passes through byte-for-byte
  UNCHANGED. Coercion must be a *repair* of an otherwise-failing value, not an
  unconditional ``json.loads`` of every string: for a ``{"type": "string"}`` schema
  the conforming value ``'"hello"'`` must stay ``'"hello"'``, not become ``'hello'``.
* **Regression guard** — a genuinely non-conforming value is still REJECTED.
  "Accept coercions" must not degrade into "accept everything".

The pool/queries calls are mocked out (the pattern in
test_closed_request_liveness.py / test_workflow_output_schema.py); the DB-backed
read itself is covered in tests/integration.
"""

from __future__ import annotations

from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

from aios.db import queries
from aios.harness import runtime
from aios.models.sessions import Ok
from aios.tools import workflow_completion
from aios.tools.registry import ToolResult
from aios.tools.workflow_completion import _enforce_output_schema, return_handler

_OBJ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
}
_STR_SCHEMA: dict[str, Any] = {"type": "string"}
_ANY_SCHEMA: dict[str, Any] = {}


def _mock_schema(monkeypatch: Any, schema: dict[str, Any] | None) -> None:
    """Point ``get_request_output_schema`` at ``schema`` with a stubbed pool."""
    monkeypatch.setattr(queries, "get_request_output_schema", AsyncMock(return_value=schema))
    pool = mock.MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock.MagicMock())
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(runtime, "require_pool", lambda: pool)


class TestCoercionPropagates:
    """Whatever the gate validated is what the consumer must receive."""

    async def test_json_encoded_object_is_accepted_as_a_dict(self, monkeypatch: Any) -> None:
        _mock_schema(monkeypatch, _OBJ_SCHEMA)

        value, error = await _enforce_output_schema("ses_1", "req_1", '{"n": 1}')

        assert error is None, "a JSON-encoded conforming object must be accepted"
        assert value == {"n": 1}
        assert isinstance(value, dict), "the consumer must receive the PARSED value, not the string"

    async def test_return_handler_persists_the_coerced_value(self, monkeypatch: Any) -> None:
        """End of the propagation chain: the coerced dict reaches ``_finish``."""
        _mock_schema(monkeypatch, _OBJ_SCHEMA)
        monkeypatch.setattr(
            workflow_completion, "_closed_request_error", AsyncMock(return_value=None)
        )
        finish = AsyncMock(return_value={"status": "returned"})
        monkeypatch.setattr(workflow_completion, "_finish", finish)

        result = await return_handler("ses_1", {"request_id": "req_1", "value": '{"n": 1}'})

        assert result == {"status": "returned"}
        finish.assert_awaited_once()
        assert finish.await_args is not None
        outcome = finish.await_args.kwargs["outcome"]
        assert outcome == Ok(result={"n": 1})
        assert isinstance(outcome.result, dict), "a str here is the split-acceptance bug"


class TestPositiveControlUnchanged:
    """An ALREADY-conforming value must survive the gate byte-for-byte."""

    async def test_conforming_dict_passes_through_unchanged(self, monkeypatch: Any) -> None:
        _mock_schema(monkeypatch, _OBJ_SCHEMA)
        original = {"n": 1}

        value, error = await _enforce_output_schema("ses_1", "req_1", original)

        assert error is None
        assert value == {"n": 1}

    async def test_conforming_string_that_is_also_json_is_not_parsed(
        self, monkeypatch: Any
    ) -> None:
        """A ``{"type": "string"}`` request whose value is the literal ``'"hello"'``.

        That string ALREADY conforms. Parsing it yields ``'hello'`` — which also
        conforms — so an unconditional ``json.loads`` silently rewrites the
        caller's answer. Coercion must only fire to RESCUE a value the schema
        would otherwise reject.
        """
        _mock_schema(monkeypatch, _STR_SCHEMA)

        value, error = await _enforce_output_schema("ses_1", "req_1", '"hello"')

        assert error is None
        assert value == '"hello"', "an already-conforming value must not be rewritten"

    async def test_permissive_schema_does_not_rewrite_a_json_looking_string(
        self, monkeypatch: Any
    ) -> None:
        """A schema of ``{}`` accepts anything, so EVERY string already conforms."""
        _mock_schema(monkeypatch, _ANY_SCHEMA)

        for original in ('{"n": 1}', "42", "true", "null", '"hi"'):
            value, error = await _enforce_output_schema("ses_1", "req_1", original)
            assert error is None
            assert value == original, f"{original!r} already conformed; it must not be parsed"


class TestRegressionGuardStillRejects:
    """Accepting coercions must not degrade into accepting everything."""

    async def test_non_conforming_json_string_is_rejected(self, monkeypatch: Any) -> None:
        _mock_schema(monkeypatch, _OBJ_SCHEMA)

        value, error = await _enforce_output_schema("ses_1", "req_1", '{"n": "not-an-int"}')

        assert error is not None, "a value that conforms neither raw nor parsed must be REJECTED"
        assert "output_schema_violation" in error
        assert value == '{"n": "not-an-int"}', "a rejected value must not be silently rewritten"

    async def test_non_json_string_is_rejected(self, monkeypatch: Any) -> None:
        _mock_schema(monkeypatch, _OBJ_SCHEMA)

        value, error = await _enforce_output_schema("ses_1", "req_1", "not json at all")

        assert error is not None
        assert value == "not json at all"

    async def test_wrong_typed_scalar_is_rejected(self, monkeypatch: Any) -> None:
        _mock_schema(monkeypatch, _OBJ_SCHEMA)

        _value, error = await _enforce_output_schema("ses_1", "req_1", 123)

        assert error is not None

    async def test_return_handler_bounces_a_rejected_value(self, monkeypatch: Any) -> None:
        """A rejection must reach the model as a tool error, with nothing persisted."""
        _mock_schema(monkeypatch, _OBJ_SCHEMA)
        monkeypatch.setattr(
            workflow_completion, "_closed_request_error", AsyncMock(return_value=None)
        )
        finish = AsyncMock()
        monkeypatch.setattr(workflow_completion, "_finish", finish)

        result = await return_handler("ses_1", {"request_id": "req_1", "value": "not json"})

        assert isinstance(result, ToolResult)
        assert result.is_error is True
        finish.assert_not_called()


class TestNoSchemaPassesThrough:
    """No ``output_schema`` on the request → no validation and no coercion."""

    async def test_schemaless_request_never_coerces(self, monkeypatch: Any) -> None:
        _mock_schema(monkeypatch, None)

        value, error = await _enforce_output_schema("ses_1", "req_1", '{"n": 1}')

        assert error is None
        assert value == '{"n": 1}', "with no schema there is nothing to coerce toward"
