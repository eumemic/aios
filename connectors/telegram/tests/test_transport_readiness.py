from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest
from aios_connector_http.runner import _ConnectionState

from aios_telegram.connector import TelegramConnector, _TelegramConnectionState

CONNECTION_ID = "conn_ready"


def _connector_and_state() -> tuple[TelegramConnector, _TelegramConnectionState]:
    connector = TelegramConnector()
    connector._connections[CONNECTION_ID] = _ConnectionState(CONNECTION_ID, "bot")
    updater = MagicMock()
    updater.start_polling = AsyncMock()
    application = MagicMock()
    application.start = AsyncMock()
    application.updater = updater
    state = _TelegramConnectionState(
        application=application,
        bot_id=1,
        first_name="Bot",
        username="bot",
        inbound_queue=asyncio.Queue(),
    )
    return connector, state


async def test_polling_marks_ready_only_after_transport_starts() -> None:
    connector, state = _connector_and_state()
    release = asyncio.Event()

    async def start_polling(**_kwargs: object) -> None:
        await release.wait()

    state.application.updater.start_polling.side_effect = start_polling
    task = asyncio.create_task(connector._run_polling(CONNECTION_ID, state))
    await asyncio.sleep(0)
    assert connector._connections[CONNECTION_ID].serve_status == "starting"

    release.set()
    await asyncio.sleep(0)
    assert connector._connections[CONNECTION_ID].serve_status == "serving"

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_polling_start_failure_remains_unhealthy() -> None:
    connector, state = _connector_and_state()
    state.application.updater.start_polling.side_effect = RuntimeError("polling failed")

    with pytest.raises(RuntimeError, match="polling failed"):
        await connector._run_polling(CONNECTION_ID, state)

    assert connector._connections[CONNECTION_ID].serve_status == "starting"
