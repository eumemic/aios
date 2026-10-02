"""``POST /v1/connectors/signal/unregister`` (#2322).

Detaching a connection only changes the aios binding; Signal keeps the
number registered to the connector's signal-cli device until it is
unregistered.  The route dispatches ``unregister`` over the
management-call plane so the connector performs it against its own
signal-cli — no container exec required.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.api.routers.connectors import SignalUnregisterRequest, post_signal_unregister
from aios.errors import ConflictError, ConnectorCallFailedError, ValidationError


def _pool() -> MagicMock:
    pool = MagicMock()

    @asynccontextmanager
    async def _acquire() -> Any:
        yield MagicMock()

    pool.acquire = _acquire
    return pool


_NO_ATTACHED = patch(
    "aios.api.routers.connectors.queries.list_attached_connections_for_phone",
    AsyncMock(return_value=[]),
)


@pytest.mark.asyncio
async def test_dispatches_unregister_management_call() -> None:
    pool = _pool()
    submit = AsyncMock(return_value=({"external_account_id": "+16575274288"}, False))
    with _NO_ATTACHED, patch("aios.api.routers.connectors.management_calls.submit_call", submit):
        await post_signal_unregister(
            SignalUnregisterRequest(external_account_id="+16575274288"),
            "postgresql://x",
            pool,
            "acc_1",
        )
    submit.assert_awaited_once()
    assert submit.await_args is not None
    kwargs = submit.await_args.kwargs
    assert kwargs["connector"] == "signal"
    assert kwargs["method"] == "unregister"
    assert kwargs["params"] == {"external_account_id": "+16575274288"}
    assert kwargs["account_id"] == "acc_1"


@pytest.mark.asyncio
async def test_connector_error_surfaces_as_502() -> None:
    submit = AsyncMock(return_value=({"error": "Specified account does not exist"}, True))
    with (
        _NO_ATTACHED,
        patch("aios.api.routers.connectors.management_calls.submit_call", submit),
        pytest.raises(ConnectorCallFailedError) as exc_info,
    ):
        await post_signal_unregister(
            SignalUnregisterRequest(external_account_id="+1"),
            "postgresql://x",
            _pool(),
            "acc_1",
        )
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail["method"] == "unregister"


@pytest.mark.asyncio
async def test_refuses_with_409_and_no_dispatch_when_still_attached() -> None:
    """Unregistering is irreversible: a number whose connection is still
    attached must be refused BEFORE any management call is dispatched."""
    attached = MagicMock()
    attached.id = "conn_live"
    lookup = AsyncMock(return_value=[attached])
    submit = AsyncMock(return_value=({}, False))
    with (
        patch("aios.api.routers.connectors.queries.list_attached_connections_for_phone", lookup),
        patch("aios.api.routers.connectors.management_calls.submit_call", submit),
        pytest.raises(ConflictError) as exc_info,
    ):
        await post_signal_unregister(
            SignalUnregisterRequest(external_account_id="+1 (657) 527-4288"),
            "postgresql://x",
            _pool(),
            "acc_1",
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["connection_ids"] == ["conn_live"]
    assert submit.await_count == 0
    lookup.assert_awaited_once()
    assert lookup.await_args is not None
    assert lookup.await_args.args[1:] == ("signal", "16575274288")
    assert lookup.await_args.kwargs == {"account_id": "acc_1"}


@pytest.mark.asyncio
async def test_rejects_digitless_number_without_dispatch() -> None:
    submit = AsyncMock(return_value=({}, False))
    with (
        patch("aios.api.routers.connectors.management_calls.submit_call", submit),
        pytest.raises(ValidationError),
    ):
        await post_signal_unregister(
            SignalUnregisterRequest(external_account_id="not-a-phone"),
            "postgresql://x",
            _pool(),
            "acc_1",
        )
    assert submit.await_count == 0


def test_phone_digits_keeps_ascii_digits_only() -> None:
    """#2322 N2: the Python normal form must agree with the SQL side's
    ``regexp_replace(..., '[^0-9]', '', 'g')``.  ``str.isdigit`` would keep
    fullwidth / Arabic-Indic / superscript digits that the SQL strips."""
    from aios.db.queries import phone_digits

    assert phone_digits("+1 (657) 527-4288") == "16575274288"
    assert phone_digits("+1657527428\uff18") == "1657527428"  # fullwidth 8 dropped
    assert phone_digits("\u0663\u0663\u0663") == ""
    assert phone_digits("+1\u00b2") == "1"


@pytest.mark.asyncio
async def test_unicode_only_digits_are_rejected_as_422_without_dispatch() -> None:
    pool = _pool()
    submit = AsyncMock()
    with (
        _NO_ATTACHED,
        patch("aios.api.routers.connectors.management_calls.submit_call", submit),
        pytest.raises(ValidationError),
    ):
        await post_signal_unregister(
            SignalUnregisterRequest(external_account_id="+\u0663\u0663\u0663"),
            "postgresql://x",
            pool,
            "acc_1",
        )
    assert submit.await_count == 0
