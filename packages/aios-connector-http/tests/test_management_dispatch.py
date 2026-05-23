"""Unit tests for :meth:`HttpConnector.dispatch_management_call` (#348).

Mirrors :mod:`test_runner`'s ``_ProbeConnector`` pattern: we override
the result POST hook and capture calls in-memory.  The SSE plumbing is
not exercised here — that's the e2e test's job.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from aios_connector_http import (
    HttpConnector,
    ManagementHandlerError,
    management_handler,
)


class _RecordedResult:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _ManagementProbeConnector(HttpConnector):
    """Three management handlers — happy / structured-error / generic-error."""

    connector = "probe"

    def __init__(self) -> None:
        super().__init__(base_url="http://x", token="aios_runtime_x")
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.results: list[_RecordedResult] = []
        self._client = MagicMock()

    @management_handler()
    async def register(self, *, external_account_id: str) -> dict[str, str]:
        self.calls.append(("register", {"external_account_id": external_account_id}))
        return {"status": "sms_sent", "external_account_id": external_account_id}

    @management_handler()
    async def captcha_path(self, *, external_account_id: str) -> dict[str, str]:
        self.calls.append(("captcha_path", {"external_account_id": external_account_id}))
        raise ManagementHandlerError(
            {
                "status": "captcha_required",
                "captcha_url": "https://x",
                "external_account_id": external_account_id,
            }
        )

    @management_handler(method="updateProfile")
    async def update_profile(self, *, external_account_id: str, given_name: str) -> dict[str, str]:
        self.calls.append(
            (
                "update_profile",
                {"external_account_id": external_account_id, "given_name": given_name},
            )
        )
        raise RuntimeError("daemon down")

    async def _post_management_call_result(  # type: ignore[override]
        self,
        client: Any,
        *,
        call_id: str,
        result: Any,
        is_error: bool,
    ) -> None:
        del client
        self.results.append(_RecordedResult(call_id=call_id, result=result, is_error=is_error))


@pytest.fixture
def probe() -> _ManagementProbeConnector:
    return _ManagementProbeConnector()


class TestDispatchManagementCall:
    @pytest.mark.asyncio
    async def test_routes_call_to_decorated_handler(self, probe: _ManagementProbeConnector) -> None:
        await probe.dispatch_management_call(
            {
                "call_id": "mgmt_1",
                "method": "register",
                "params": {"external_account_id": "+15551234567"},
            }
        )
        assert probe.calls == [("register", {"external_account_id": "+15551234567"})]
        r = probe.results[0]
        assert r.kwargs == {
            "call_id": "mgmt_1",
            "result": {"status": "sms_sent", "external_account_id": "+15551234567"},
            "is_error": False,
        }

    @pytest.mark.asyncio
    async def test_management_handler_error_passes_payload_through(
        self, probe: _ManagementProbeConnector
    ) -> None:
        await probe.dispatch_management_call(
            {
                "call_id": "mgmt_2",
                "method": "captcha_path",
                "params": {"external_account_id": "+1"},
            }
        )
        r = probe.results[0]
        assert r.kwargs["is_error"] is True
        assert r.kwargs["result"] == {
            "status": "captcha_required",
            "captcha_url": "https://x",
            "external_account_id": "+1",
        }

    @pytest.mark.asyncio
    async def test_generic_exception_becomes_string_error(
        self, probe: _ManagementProbeConnector
    ) -> None:
        await probe.dispatch_management_call(
            {
                "call_id": "mgmt_3",
                "method": "updateProfile",
                "params": {"external_account_id": "+1", "given_name": "X"},
            }
        )
        r = probe.results[0]
        assert r.kwargs["is_error"] is True
        assert r.kwargs["result"] == {"error": "daemon down"}

    @pytest.mark.asyncio
    async def test_unknown_method_posts_error_result(
        self, probe: _ManagementProbeConnector
    ) -> None:
        await probe.dispatch_management_call(
            {"call_id": "mgmt_4", "method": "no_such_method", "params": {}}
        )
        r = probe.results[0]
        assert r.kwargs["is_error"] is True
        assert "no_such_method" in r.kwargs["result"]["error"]
        assert probe.calls == []

    @pytest.mark.asyncio
    async def test_decorator_method_override_routes_to_aliased_name(
        self, probe: _ManagementProbeConnector
    ) -> None:
        # update_profile is decorated as @management_handler(method="updateProfile")
        assert "updateProfile" in probe._management
        assert "update_profile" not in probe._management


class TestCollection:
    def test_tool_and_management_handler_cannot_coexist(self) -> None:
        from aios_connector_http import tool

        with pytest.raises(RuntimeError, match="@tool and @management_handler"):

            class _Bad(HttpConnector):
                connector = "bad"

                @tool()
                @management_handler()
                async def both(self) -> str:
                    return "x"

            _Bad(base_url="http://x", token="aios_runtime_x")
