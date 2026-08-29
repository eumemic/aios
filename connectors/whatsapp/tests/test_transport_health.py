from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aios_connector_http.runner import _ConnectionState

from aios_whatsapp.config import Settings
from aios_whatsapp.connector import WhatsappConnector


class _Listener:
    def __init__(self, notifications: list[tuple[str, dict[str, Any]]]) -> None:
        self._notifications = notifications

    async def notifications(self) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        for notification in self._notifications:
            yield notification


class _Daemon:
    def __init__(self, notifications: list[tuple[str, dict[str, Any]]]) -> None:
        self.listener = _Listener(notifications)


def _connector(tmp_path: Path) -> WhatsappConnector:
    connector = WhatsappConnector(Settings(data_dir=tmp_path / "data"))
    connector._connections["conn"] = _ConnectionState("conn", "account")
    connector.emit_lifecycle = AsyncMock()  # type: ignore[method-assign]
    return connector


@pytest.mark.asyncio
async def test_transport_ready_tracks_whatsapp_connection_state(tmp_path: Path) -> None:
    connector = _connector(tmp_path)

    # A reachable daemon listener is not evidence that whatsmeow completed its
    # upstream handshake, so startup remains unhealthy.
    assert connector._connections["conn"].serve_status == "starting"

    await connector._dispatch_notifications(
        "conn",
        _Daemon(
            [
                ("connectionState", {"state": "connected"}),
                ("connectionState", {"state": "disconnected"}),
            ]
        ),  # type: ignore[arg-type]
    )

    assert connector._connections["conn"].serve_status == "starting"


@pytest.mark.asyncio
async def test_logged_out_revokes_transport_readiness(tmp_path: Path) -> None:
    connector = _connector(tmp_path)
    connector.mark_transport_ready("conn")
    assert connector._connections["conn"].serve_status == "serving"

    await connector._dispatch_notifications(
        "conn",
        _Daemon([("loggedOut", {"reason": "device_removed"})]),  # type: ignore[arg-type]
    )

    assert connector._connections["conn"].serve_status == "starting"
    connector.emit_lifecycle.assert_awaited_once()  # type: ignore[attr-defined]
