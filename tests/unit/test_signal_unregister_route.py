"""``POST /v1/connectors/signal/unregister`` (#2322).

Detaching a connection only changes the aios binding; Signal keeps the
number registered to the connector's signal-cli device until it is
unregistered.  The route dispatches ``unregister`` over the
management-call plane so the connector performs it against its own
signal-cli — no container exec required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.api.routers.connectors import SignalUnregisterRequest, post_signal_unregister
from aios.errors import ConnectorCallFailedError


@pytest.mark.asyncio
async def test_dispatches_unregister_management_call() -> None:
    pool = MagicMock()
    submit = AsyncMock(return_value=({"external_account_id": "+16575274288"}, False))
    with patch("aios.api.routers.connectors.management_calls.submit_call", submit):
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
        patch("aios.api.routers.connectors.management_calls.submit_call", submit),
        pytest.raises(ConnectorCallFailedError) as exc_info,
    ):
        await post_signal_unregister(
            SignalUnregisterRequest(external_account_id="+1"),
            "postgresql://x",
            MagicMock(),
            "acc_1",
        )
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail["method"] == "unregister"
