"""Unit coverage for the single-sourced SSE preflight-or-503 helper (#1079).

The five SSE route handlers shared a verbatim ``try: sub = await open_*()
except SSE_PREFLIGHT_EXCEPTIONS: log.warning(...); raise
SSEPreflightFailedError`` block. It was extracted to
:func:`aios.api.sse.preflight_subscription`; these tests pin both the
success pass-through and the 503-with-diagnostic failure path.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import asyncpg
import pytest

from aios.api.sse import preflight_subscription
from aios.db.listen import SSESubscriberCapacityError
from aios.errors import SSEPreflightFailedError


async def test_preflight_returns_subscription_on_success() -> None:
    sentinel = MagicMock(name="subscription")

    async def _open() -> MagicMock:
        return sentinel

    log = MagicMock()
    sub = await preflight_subscription(
        _open(),
        stream_name="runtime_calls",
        log_key="sse.runtime_calls.preflight_failed",
        log_fields={"connector": "telegram"},
        log=log,
    )
    assert sub is sentinel
    log.warning.assert_not_called()


async def test_preflight_raises_503_and_logs_on_transient_failure() -> None:
    async def _open() -> MagicMock:
        raise asyncpg.CannotConnectNowError("startup blip")

    log = MagicMock()
    with pytest.raises(SSEPreflightFailedError) as excinfo:
        await preflight_subscription(
            _open(),
            stream_name="runtime_calls",
            log_key="sse.runtime_calls.preflight_failed",
            log_fields={"connector": "telegram"},
            log=log,
        )

    assert excinfo.value.detail == {"stream": "runtime_calls"}
    log.warning.assert_called_once()
    call = log.warning.call_args
    assert call.args == ("sse.runtime_calls.preflight_failed",)
    assert call.kwargs["connector"] == "telegram"
    assert call.kwargs["error"] == "startup blip"
    assert call.kwargs["error_type"] == "CannotConnectNowError"


async def test_preflight_does_not_swallow_unexpected_errors() -> None:
    """Non-preflight exceptions bubble as-is (not converted to 503)."""

    async def _open() -> MagicMock:
        raise ValueError("programmer error")

    log = MagicMock()
    with pytest.raises(ValueError, match="programmer error"):
        await preflight_subscription(
            _open(),
            stream_name="runtime_calls",
            log_key="sse.runtime_calls.preflight_failed",
            log_fields={"connector": "telegram"},
            log=log,
        )
    log.warning.assert_not_called()


async def test_preflight_translates_sse_capacity_exhaustion_to_503() -> None:
    async def _open() -> MagicMock:
        raise SSESubscriberCapacityError("subscriber cap reached")

    with pytest.raises(SSEPreflightFailedError) as excinfo:
        await preflight_subscription(
            _open(),
            stream_name="session_events",
            log_key="sse.session.preflight_failed",
            log_fields={},
            log=MagicMock(),
        )
    assert excinfo.value.status_code == 503
