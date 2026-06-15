"""Unit tests for the invocation request/response models (#1128).

The API caller's request-writer surface ships two plain ``BaseModel``s — no
opaque encoding. These cover the wire-shape contract: discriminated
``target_kind``, optional ``output_schema`` / ``environment_id``, and the
structured (un-encoded) handle.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.models.invocations import InvocationHandle, InvocationRequest


class TestInvocationRequest:
    def test_minimal_agent_request(self) -> None:
        req = InvocationRequest(target_kind="agent", target="agent_x", environment_id="env_y")
        assert req.target_kind == "agent"
        assert req.target == "agent_x"
        assert req.input is None
        assert req.output_schema is None
        assert req.environment_id == "env_y"

    def test_session_target_needs_no_environment(self) -> None:
        req = InvocationRequest(target_kind="session", target="sess_z", input={"q": 1})
        assert req.environment_id is None
        assert req.input == {"q": 1}

    def test_output_schema_rides_the_request(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
        req = InvocationRequest(
            target_kind="workflow",
            target="wf_a",
            environment_id="env_b",
            output_schema=schema,
        )
        assert req.output_schema == schema

    def test_bad_target_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InvocationRequest(target_kind="robot", target="x")

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            InvocationRequest(target_kind="agent", target="x", bogus="nope")  # type: ignore[call-arg]

    def test_target_required(self) -> None:
        with pytest.raises(ValidationError):
            InvocationRequest(target_kind="agent")  # type: ignore[call-arg]


class TestInvocationHandle:
    def test_structured_fields_plain(self) -> None:
        handle = InvocationHandle(servicer_kind="session", servicer_id="sess_1", request_id="req_1")
        # No opaque encoding — the fields serialize verbatim.
        assert handle.model_dump() == {
            "servicer_kind": "session",
            "servicer_id": "sess_1",
            "request_id": "req_1",
        }

    def test_run_servicer_kind(self) -> None:
        handle = InvocationHandle(servicer_kind="run", servicer_id="wfr_1", request_id="req_2")
        assert handle.servicer_kind == "run"

    def test_bad_servicer_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InvocationHandle(servicer_kind="agent", servicer_id="x", request_id="r")
