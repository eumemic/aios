"""Unit tests for the model-path tool dispatch: the ``validate_arguments`` helper and
the ``_classify_tool_error`` eviction policy.

``validate_arguments`` converts JSON Schema failures into a model-readable error string
that lists every problem at once. ``_classify_tool_error`` decides whether a handler
exception is an expected model-visible refusal (clean ``is_error`` result, no eviction)
or a genuine failure (also evicts the sandbox).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.errors import (
    AiosError,
    ConflictError,
    CryptoDecryptError,
    ForbiddenError,
    NotFoundError,
    RateLimitedError,
    ValidationError,
)
from aios.harness import tool_dispatch
from aios.harness.tool_dispatch import _classify_tool_error, _tool_lifecycle
from aios.services import sessions as sessions_service
from aios.tools.bash import BashArgumentError
from aios.tools.invoke import ToolBail, validate_arguments

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
        assert validate_arguments({"channel_id": "signal/a/b"}, _SCHEMA) is None
        assert validate_arguments({"channel_id": None}, _SCHEMA) is None

    def test_missing_required_key_is_reported(self) -> None:
        err = validate_arguments({}, _SCHEMA)
        assert err is not None
        assert "'channel_id' is a required property" in err

    def test_unexpected_property_is_reported(self) -> None:
        """The common weak-model failure mode: wrong param name.  The
        extra-property error must surface so the model can correct it.
        """
        err = validate_arguments({"target": "signal/a/b"}, _SCHEMA)
        assert err is not None
        assert "'target' was unexpected" in err

    def test_wrong_type_is_reported(self) -> None:
        err = validate_arguments({"channel_id": 42}, _SCHEMA)
        assert err is not None
        assert "42" in err  # what was sent
        # jsonschema's message varies; just assert we got SOMETHING useful
        assert "channel_id" in err or "type" in err.lower() or "42" in err

    def test_multi_error_accumulation(self) -> None:
        """Missing required + extra property → BOTH surface in a single
        error.  Model sees every issue at once; one retry fixes all.
        """
        err = validate_arguments({"target": "x"}, _SCHEMA)
        assert err is not None
        assert "'channel_id' is a required property" in err
        assert "'target' was unexpected" in err

    def test_passed_arguments_echoed_for_context(self) -> None:
        """The error includes the arguments the model sent so the model
        can see exactly what shape it produced vs. what was expected.
        """
        err = validate_arguments({"target": "oops"}, _SCHEMA)
        assert err is not None
        assert '"target": "oops"' in err

    def test_error_ends_with_hint_to_consult_schema(self) -> None:
        """Closing guidance points the model back at the tool's
        declared ``parameters`` — its authoritative reference."""
        err = validate_arguments({"target": "x"}, _SCHEMA)
        assert err is not None
        assert "parameters" in err.lower()

    def test_permissive_schema_allows_anything(self) -> None:
        """A schema with no constraints (no required, no additionalProps
        false) accepts arbitrary payloads — validation is opt-in per
        tool."""
        loose: dict[str, Any] = {"type": "object"}
        assert validate_arguments({"anything": "goes"}, loose) is None
        assert validate_arguments({}, loose) is None


class _InternalAiosError(AiosError):
    error_type = "internal_thing"
    status_code = 500


class TestClassifyToolError:
    @pytest.mark.parametrize(
        ("err", "evict", "message"),
        [
            # Expected, model-visible refusals — never evict the sandbox. The headline
            # fix: a 400 *ArgumentError (BashArgumentError) is a clean refusal now,
            # where it used to evict the container along with every other AiosError.
            (ToolBail("bad args"), False, "bad args"),
            (BashArgumentError("command must be non-empty"), False, "command must be non-empty"),
            (ForbiddenError("denied", detail={"x": 1}), False, 'denied ({"x": 1})'),
            (NotFoundError("no such workflow"), False, "no such workflow"),
            (ConflictError("stale version"), False, "stale version"),
            (RateLimitedError("slow down"), False, "slow down"),
            (ValidationError("bad field"), False, "bad field"),
            # Genuine failures — also evict the sandbox.
            (_InternalAiosError("internal boom"), True, "internal boom"),
            # A 5xx WITH detail still gets the json-suffixed message (the detail-format
            # branch is independent of the evict decision).
            (_InternalAiosError("boom", detail={"k": 2}), True, 'boom ({"k": 2})'),
            (CryptoDecryptError("decrypt failed"), True, "decrypt failed"),
            (ValueError("oops"), True, "ValueError: oops"),
            (RuntimeError("kaboom"), True, "RuntimeError: kaboom"),
        ],
    )
    def test_classification(self, err: BaseException, evict: bool, message: str) -> None:
        assert _classify_tool_error(err) == (evict, message)


class TestToolLifecycleEviction:
    """End-to-end: the classifier wired into ``_tool_lifecycle`` — a refusal appends a
    clean is_error result without evicting; a genuine failure also evicts."""

    @staticmethod
    async def _drive(monkeypatch: Any, err: BaseException) -> tuple[list[str], Any]:
        """Run a handler that raises ``err`` through the lifecycle; the DB writes are
        stubbed. Returns ``(evict_calls, append_result_mock)``."""
        monkeypatch.setattr(
            sessions_service,
            "append_event",
            AsyncMock(return_value=MagicMock(id="span_1")),
        )
        append_result = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "_append_tool_result", append_result)
        monkeypatch.setattr(tool_dispatch, "_trigger_sweep", AsyncMock())

        evicted: list[str] = []
        call = {"id": "tc_1", "function": {"name": "demo", "arguments": "{}"}}
        async with _tool_lifecycle(
            MagicMock(),
            "ses_1",
            call,
            account_id="acc_1",
            log_prefix="tool",
            on_exception=evicted.append,
        ):
            raise err
        return evicted, append_result  # type: ignore[unreachable]

    async def test_client_error_refusal_does_not_evict(self, monkeypatch: Any) -> None:
        evicted, append_result = await self._drive(
            monkeypatch, ForbiddenError("denied", detail={"x": 1})
        )
        assert evicted == []  # a 4xx refusal must NOT evict the sandbox
        assert append_result.await_count == 1  # but still appends a model-visible result
        assert append_result.await_args.kwargs["error"] == 'denied ({"x": 1})'

    async def test_toolbail_does_not_evict(self, monkeypatch: Any) -> None:
        evicted, append_result = await self._drive(monkeypatch, ToolBail("bad json"))
        assert evicted == []
        assert append_result.await_args.kwargs["error"] == "bad json"

    async def test_server_error_evicts(self, monkeypatch: Any) -> None:
        evicted, append_result = await self._drive(monkeypatch, CryptoDecryptError("boom"))
        assert evicted == ["ses_1"]  # a 5xx is a genuine failure → evict
        assert append_result.await_count == 1

    async def test_unexpected_exception_evicts(self, monkeypatch: Any) -> None:
        evicted, append_result = await self._drive(monkeypatch, ValueError("oops"))
        assert evicted == ["ses_1"]
        assert append_result.await_args.kwargs["error"] == "ValueError: oops"
