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
async def test_heartbeat_withheld_until_discovery_is_authoritative(tmp_path: Path) -> None:
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    connector.HEARTBEAT_INTERVAL = 0.01

    task = asyncio.create_task(connector._heartbeat_loop(heartbeat))
    try:
        await asyncio.sleep(0.03)
        assert not heartbeat.exists()
        connector._discovery_cursor = 0
        await asyncio.sleep(0.03)
        assert heartbeat.exists()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_heartbeat_recovers_stale_file_left_by_crashed_process(tmp_path: Path) -> None:
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "crash-left-alive"
    heartbeat.touch()
    old = time.time() - 3600
    os.utime(heartbeat, (old, old))
    original_identity = (heartbeat.stat().st_dev, heartbeat.stat().st_ino)
    connector._discovery_cursor = 0
    connector.HEARTBEAT_INTERVAL = 0.01

    task = asyncio.create_task(connector._heartbeat_loop(heartbeat))
    try:
        await asyncio.sleep(0.03)
        assert heartbeat_is_fresh(heartbeat, max_age_seconds=30)
        assert connector._heartbeat_owned
        assert connector._heartbeat_identity == original_identity
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_heartbeat_does_not_claim_or_remove_preexisting_file(tmp_path: Path) -> None:
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "operator-owned"
    heartbeat.write_text("keep")
    connector._discovery_cursor = 0
    connector.HEARTBEAT_INTERVAL = 0.01

    task = asyncio.create_task(connector._heartbeat_loop(heartbeat))
    try:
        await asyncio.sleep(0.03)
        assert heartbeat.read_text() == "keep"
        assert not connector._heartbeat_owned
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_cleanup_refuses_to_unlink_replacement_inode(tmp_path: Path) -> None:
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    heartbeat.touch()
    owned_stat = heartbeat.stat()
    connector._heartbeat_owned = True
    connector._heartbeat_identity = (owned_stat.st_dev, owned_stat.st_ino)

    # Keep the old inode referenced so the filesystem cannot immediately reuse
    # its number for the replacement and make this identity test ambiguous.
    with heartbeat.open() as old_inode:
        heartbeat.unlink()
        heartbeat.write_text("replacement")
        assert (heartbeat.stat().st_dev, heartbeat.stat().st_ino) != connector._heartbeat_identity

        await connector._remove_owned_heartbeat(heartbeat)
        assert not old_inode.closed

    assert heartbeat.read_text() == "replacement"
    assert not connector._heartbeat_owned
    assert connector._heartbeat_identity is None



@pytest.mark.asyncio
async def test_heartbeat_does_not_claim_path_replaced_after_create(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    connector._discovery_cursor = 0
    connector.HEARTBEAT_INTERVAL = 0.01
    real_fstat = os.fstat
    replaced = False

    def replace_after_fstat(fd: int) -> os.stat_result:
        nonlocal replaced
        created = real_fstat(fd)
        if not replaced:
            heartbeat.unlink()
            heartbeat.write_text("operator")
            replaced = True
        return created

    monkeypatch.setattr(os, "fstat", replace_after_fstat)
    task = asyncio.create_task(connector._heartbeat_loop(heartbeat))
    try:
        await asyncio.sleep(0.03)
        assert replaced
        assert heartbeat.read_text() == "operator"
        assert not connector._heartbeat_owned
        assert connector._heartbeat_identity is None
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_heartbeat_stops_while_a_connection_is_restarting(tmp_path: Path) -> None:
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    connector.HEARTBEAT_INTERVAL = 0.01
    connector._discovery_cursor = 0  # authoritative empty snapshot completed
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
