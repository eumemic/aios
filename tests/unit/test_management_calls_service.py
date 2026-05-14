"""Unit coverage for :mod:`aios.services.management_calls`."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.errors import ManagementCallTimeoutError
from aios.services.management_calls import submit_call
from tests.unit.conftest import fake_pool_yielding_conn


@contextlib.asynccontextmanager
async def _listener_yielding(payload: str | None) -> AsyncIterator[asyncio.Queue[str]]:
    """Stand-in for ``listen_for_connector_result`` that pre-populates the queue.

    Passing ``None`` yields an empty queue so the caller hits its
    ``asyncio.wait_for`` timeout — covers the timeout-path assertion.
    """
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
    if payload is not None:
        await queue.put(payload)
    yield queue


def _row(result: Any, *, is_error: bool, status: str = "succeeded") -> dict[str, Any]:
    return {
        "id": "mgmt_x",
        "connector": "signal",
        "method": "register",
        "params": {},
        "status": status,
        "result": result,
        "is_error": is_error,
    }


class TestSubmitCall:
    @pytest.mark.asyncio
    async def test_happy_path_returns_result(self) -> None:
        conn = MagicMock()
        pool = fake_pool_yielding_conn(conn)
        expected: dict[str, Any] = {"status": "sms_sent", "account": "+15551234567"}

        with (
            patch(
                "aios.services.management_calls.listen.listen_for_connector_result",
                return_value=_listener_yielding(""),
            ),
            patch(
                "aios.services.management_calls.queries.insert_management_call", AsyncMock()
            ) as insert,
            patch(
                "aios.services.management_calls.queries.notify_management_call_dispatch",
                AsyncMock(),
            ) as notify,
            patch(
                "aios.services.management_calls.queries.get_management_call",
                AsyncMock(return_value=_row(expected, is_error=False)),
            ),
        ):
            result, is_error = await submit_call(
                "postgresql://test/test",
                pool,
                connector="signal",
                method="register",
                params={"account": "+15551234567"},
                timeout_s=1.0,
            )

        assert (result, is_error) == (expected, False)
        insert.assert_awaited_once()
        notify.assert_awaited_once()
        notified_call_id = notify.await_args.kwargs["call_id"]
        assert notified_call_id.startswith("mgmt_")
        assert notify.await_args.kwargs["connector"] == "signal"
        assert insert.await_args.kwargs["call_id"] == notified_call_id

    @pytest.mark.asyncio
    async def test_is_error_propagates(self) -> None:
        conn = MagicMock()
        pool = fake_pool_yielding_conn(conn)
        captcha: dict[str, Any] = {"status": "captcha_required", "captcha_url": "https://..."}

        with (
            patch(
                "aios.services.management_calls.listen.listen_for_connector_result",
                return_value=_listener_yielding(""),
            ),
            patch("aios.services.management_calls.queries.insert_management_call", AsyncMock()),
            patch(
                "aios.services.management_calls.queries.notify_management_call_dispatch",
                AsyncMock(),
            ),
            patch(
                "aios.services.management_calls.queries.get_management_call",
                AsyncMock(return_value=_row(captcha, is_error=True, status="failed")),
            ),
        ):
            result, is_error = await submit_call(
                "postgresql://test/test",
                pool,
                connector="signal",
                method="register",
                params={"account": "+15551234567"},
                timeout_s=1.0,
            )

        assert (result, is_error) == (captcha, True)

    @pytest.mark.asyncio
    async def test_timeout_raises(self) -> None:
        conn = MagicMock()
        pool = fake_pool_yielding_conn(conn)

        with (
            patch(
                "aios.services.management_calls.listen.listen_for_connector_result",
                return_value=_listener_yielding(None),
            ),
            patch("aios.services.management_calls.queries.insert_management_call", AsyncMock()),
            patch(
                "aios.services.management_calls.queries.notify_management_call_dispatch",
                AsyncMock(),
            ),
            pytest.raises(ManagementCallTimeoutError) as exc_info,
        ):
            await submit_call(
                "postgresql://test/test",
                pool,
                connector="signal",
                method="register",
                params={"account": "+15551234567"},
                timeout_s=0.05,
            )

        assert exc_info.value.detail["connector"] == "signal"
        assert exc_info.value.detail["method"] == "register"
        assert exc_info.value.detail["call_id"].startswith("mgmt_")
