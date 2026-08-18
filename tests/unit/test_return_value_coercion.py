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

from datetime import UTC, datetime
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

from aios.db import queries
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.models.sessions import Ok
from aios.models.tasks import AwaitResponse
from aios.models.workflows import WfRun
from aios.tools import workflow_completion
from aios.tools.registry import ToolResult
from aios.tools.workflow_completion import _enforce_output_schema, return_handler
from aios.workflows import step as workflow_step

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


def _run_with_schema() -> WfRun:
    now = datetime.now(UTC)
    return WfRun(
        id="wfr_1",
        workflow_id="wf_1",
        account_id="acc_1",
        environment_id="env_1",
        request_id="req_1",
        caller={"kind": "run", "id": "wfr_parent"},
        request_output_schema=_OBJ_SCHEMA,
        script="async def main(input): return None",
        script_sha="sha",
        host_semantics_epoch=1,
        status="running",
        last_event_seq=0,
        created_at=now,
        updated_at=now,
    )


async def _complete_workflow_output(monkeypatch: Any, value: Any) -> tuple[str, Any]:
    monkeypatch.setattr(
        wf_queries,
        "run_children_usage",
        AsyncMock(return_value=wf_queries.RunChildrenUsage(0, 0, 0, 0, 0)),
    )
    commit = AsyncMock()
    monkeypatch.setattr(workflow_step, "_commit_terminal_and_dispatch", commit)
    await workflow_step._complete_run(
        mock.MagicMock(), _run_with_schema(), output=value, is_error=False
    )
    assert commit.await_args is not None
    return commit.await_args.kwargs["status"], commit.await_args.kwargs["output"]


async def _resolve_call_output(monkeypatch: Any, value: Any) -> Any:
    monkeypatch.setattr(
        "aios.tools.invoke_session._park_on_task",
        AsyncMock(return_value=AwaitResponse(outcome="ok", result=value)),
    )
    from aios.tools.invoke_session import _park_and_resolve

    return await _park_and_resolve(
        object(),
        servicer_kind="session",
        servicer_id="ses_1",
        request_id="req_1",
        account_id="acc_1",
        output_schema=_OBJ_SCHEMA,
    )


class TestCrossPathConsistency:
    """The same terminal payload receives the same verdict at all four boundaries."""

    async def test_encoded_object_is_accepted_and_normalized_everywhere(
        self, monkeypatch: Any
    ) -> None:
        encoded = '{"n": 1}'
        _mock_schema(monkeypatch, _OBJ_SCHEMA)

        session_value, session_error = await _enforce_output_schema("ses_1", "req_1", encoded)
        workflow_status, workflow_value = await _complete_workflow_output(monkeypatch, encoded)
        call_session = await _resolve_call_output(monkeypatch, encoded)
        call_workflow = await _resolve_call_output(monkeypatch, encoded)

        assert session_error is None
        assert session_value == {"n": 1}  # return / workflow_completion
        assert (workflow_status, workflow_value) == ("completed", {"n": 1})
        assert call_session == {"ok": {"n": 1}}
        assert call_workflow == {"ok": {"n": 1}}

    async def test_parseable_nonconforming_object_is_rejected_everywhere(
        self, monkeypatch: Any
    ) -> None:
        encoded_invalid = '{"n": "not-an-int"}'
        _mock_schema(monkeypatch, _OBJ_SCHEMA)

        session_value, session_error = await _enforce_output_schema(
            "ses_1", "req_1", encoded_invalid
        )
        workflow_status, workflow_value = await _complete_workflow_output(
            monkeypatch, encoded_invalid
        )
        call_session = await _resolve_call_output(monkeypatch, encoded_invalid)
        call_workflow = await _resolve_call_output(monkeypatch, encoded_invalid)

        assert session_error is not None
        assert session_value == encoded_invalid
        assert workflow_status == "errored"
        assert "does not conform" in workflow_value
        for result in (call_session, call_workflow):
            assert isinstance(result, ToolResult)
            assert result.is_error
            assert "output_schema_violation" in result.content


class TestNoSchemaPassesThrough:
    """No ``output_schema`` on the request → no validation and no coercion."""

    async def test_schemaless_request_never_coerces(self, monkeypatch: Any) -> None:
        _mock_schema(monkeypatch, None)

        value, error = await _enforce_output_schema("ses_1", "req_1", '{"n": 1}')

        assert error is None
        assert value == '{"n": 1}', "with no schema there is nothing to coerce toward"
