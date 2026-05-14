"""Unit coverage for :mod:`aios.services.management_calls`."""

from __future__ import annotations

import asyncio
import contextlib
import json
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


class TestSubmitCall:
    @pytest.mark.asyncio
    async def test_happy_path_returns_result_and_clears_flag(self) -> None:
        conn = MagicMock()
        conn.execute = AsyncMock()
        pool = fake_pool_yielding_conn(conn)
        expected: dict[str, Any] = {"status": "sms_sent", "account": "+15551234567"}

        with (
            patch(
                "aios.services.management_calls.listen.listen_for_connector_result",
                return_value=_listener_yielding(
                    json.dumps({"result": expected, "is_error": False})
                ),
            ),
            patch(
                "aios.services.management_calls.queries.insert_management_call", AsyncMock()
            ) as insert,
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
        # NOTIFY fired on the correct channel with the same call_id we INSERTed.
        notify_call = conn.execute.await_args
        assert notify_call.args[0] == "SELECT pg_notify($1, $2)"
        assert notify_call.args[1] == "connector_management_calls_signal"
        notified_call_id = notify_call.args[2]
        assert notified_call_id.startswith("mgmt_")
        # ...and the INSERT used the same call_id.
        insert.assert_awaited_once()
        assert insert.await_args.kwargs["call_id"] == notified_call_id
        assert insert.await_args.kwargs["connector"] == "signal"
        assert insert.await_args.kwargs["method"] == "register"

    @pytest.mark.asyncio
    async def test_is_error_propagates(self) -> None:
        conn = MagicMock()
        conn.execute = AsyncMock()
        pool = fake_pool_yielding_conn(conn)
        captcha: dict[str, Any] = {"status": "captcha_required", "captcha_url": "https://..."}

        with (
            patch(
                "aios.services.management_calls.listen.listen_for_connector_result",
                return_value=_listener_yielding(json.dumps({"result": captcha, "is_error": True})),
            ),
            patch("aios.services.management_calls.queries.insert_management_call", AsyncMock()),
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
        conn.execute = AsyncMock()
        pool = fake_pool_yielding_conn(conn)

        with (
            patch(
                "aios.services.management_calls.listen.listen_for_connector_result",
                return_value=_listener_yielding(None),
            ),
            patch("aios.services.management_calls.queries.insert_management_call", AsyncMock()),
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
