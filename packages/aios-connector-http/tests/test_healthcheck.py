from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from aios_connector_http.healthcheck import (
    DEFAULT_HEARTBEAT_PATH,
    heartbeat_is_fresh,
    resolve_heartbeat_path,
)
from aios_connector_http.runner import HttpConnector, _ConnectionState


class _Connector(HttpConnector):
    connector = "probe"


def test_configured_heartbeat_path_takes_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configured = tmp_path / "configured-alive"
    monkeypatch.setenv("AIOS_CONNECTOR_HEARTBEAT_PATH", str(configured))
    assert resolve_heartbeat_path() == configured


def test_heartbeat_path_is_writable_outside_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("AIOS_CONNECTOR_HEARTBEAT_PATH", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    with patch("pathlib.Path.exists", return_value=False):
        path = resolve_heartbeat_path()
    assert path == tmp_path / DEFAULT_HEARTBEAT_PATH.name
    path.touch()


@pytest.mark.asyncio
async def test_heartbeat_stops_while_a_connection_is_restarting(tmp_path: Path) -> None:
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    connector.HEARTBEAT_INTERVAL = 0.01
    connector._connections["conn_1"] = _ConnectionState("conn_1", "account")

    task = asyncio.create_task(connector._heartbeat_loop(heartbeat))
    try:
        await asyncio.sleep(0.03)
        first_mtime = heartbeat.stat().st_mtime_ns

        connector._connections["conn_1"].serve_status = "restarting"
        await asyncio.sleep(0.03)
        assert heartbeat.stat().st_mtime_ns == first_mtime

        connector._connections["conn_1"].serve_status = "serving"
        await asyncio.sleep(0.03)
        assert heartbeat.stat().st_mtime_ns > first_mtime
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_every_connector_image_defines_a_healthcheck() -> None:
    repository = Path(__file__).parents[3]
    dockerfiles = sorted((repository / "connectors").glob("*/Dockerfile"))

    assert dockerfiles
    for dockerfile in dockerfiles:
        assert "HEALTHCHECK" in dockerfile.read_text(), dockerfile


def test_healthcheck_rejects_stale_or_missing_heartbeat(tmp_path: Path) -> None:
    heartbeat = tmp_path / "alive"
    assert not heartbeat_is_fresh(heartbeat, max_age_seconds=30)

    heartbeat.touch()
    assert heartbeat_is_fresh(heartbeat, max_age_seconds=30)

    old = time.time() - 31
    os.utime(heartbeat, (old, old))
    assert not heartbeat_is_fresh(heartbeat, max_age_seconds=30)
