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
    main,
    read_connection_health,
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


def test_malformed_fresh_heartbeat_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    heartbeat = tmp_path / "alive"
    heartbeat.write_text("{bad json")
    monkeypatch.setenv("AIOS_CONNECTOR_HEARTBEAT_PATH", str(heartbeat))

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        "{}",
        '{"healthy_connection_ids": [], "unhealthy_connection_ids": "conn_1"}',
        '{"healthy_connection_ids": [null], "unhealthy_connection_ids": []}',
    ],
)
def test_structurally_invalid_fresh_heartbeat_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str
) -> None:
    heartbeat = tmp_path / "alive"
    heartbeat.write_text(content)
    monkeypatch.setenv("AIOS_CONNECTOR_HEARTBEAT_PATH", str(heartbeat))

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1


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

    # Keep the old inode referenced until the replacement exists so the
    # filesystem cannot immediately reuse its number, then close it before
    # cleanup.  Verify the pathname itself survives (an open file descriptor
    # can still be read after its pathname has been unlinked).
    old_inode = heartbeat.open()
    heartbeat.unlink()
    heartbeat.write_text("replacement")
    assert (heartbeat.stat().st_dev, heartbeat.stat().st_ino) != connector._heartbeat_identity
    old_inode.close()

    await connector._remove_owned_heartbeat(heartbeat)

    assert heartbeat.exists()
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
    connector.HEARTBEAT_INTERVAL = 0.0
    connector._discovery_cursor = 0  # authoritative empty snapshot completed
    connector._connections["conn_1"] = _ConnectionState("conn_1", "account", serve_status="serving")

    # Drive the loop one iteration at a time instead of racing wall-clock
    # sleeps: the previous version sampled the mtime BEFORE the state
    # transition, so a healthy iteration scheduled between the sample and the
    # transition refreshed the mtime and the assertion compared against a stale
    # sample. Here every sample is taken only after the iteration that observed
    # the intended state has completed, so no unobserved iteration can move the
    # mtime out from under us.
    resume = asyncio.Event()
    iterated = asyncio.Event()

    async def hook() -> None:
        iterated.set()
        await resume.wait()
        resume.clear()

    connector._heartbeat_iteration_hook = hook

    async def step() -> None:
        """Advance exactly one heartbeat-loop iteration and wait for it to finish."""
        iterated.clear()
        resume.set()
        await iterated.wait()

    task = asyncio.create_task(connector._heartbeat_loop(heartbeat))
    try:
        # First iteration: serving -> heartbeat established fresh.
        await iterated.wait()
        first_mtime = heartbeat.stat().st_mtime_ns

        # An unhealthy transition must FREEZE freshness: the mtime does not
        # advance across a fail-closed iteration.
        connector._connections["conn_1"].serve_status = "restarting"
        await step()
        await step()  # a second fail-closed iteration must still not advance it
        assert heartbeat.stat().st_mtime_ns == first_mtime

        # Recovery must resume freshness: the mtime advances again.
        connector._connections["conn_1"].serve_status = "serving"
        await step()
        assert heartbeat.stat().st_mtime_ns > first_mtime
    finally:
        resume.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
@pytest.mark.parametrize("serve_returns", [False, True])
async def test_heartbeat_requires_established_receiving_transport(
    tmp_path: Path, serve_returns: bool
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class _TransportConnector(_Connector):
        async def serve_connection(self, connection_id: str, secrets: dict[str, str]) -> None:
            del connection_id, secrets
            entered.set()
            if not serve_returns:
                await release.wait()

    connector = _TransportConnector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    connector.HEARTBEAT_INTERVAL = 0.01
    connector._discovery_cursor = 0
    connector._connections["conn_1"] = _ConnectionState("conn_1", "account")

    serve_task = asyncio.create_task(connector._isolated_serve_connection("conn_1", {}))
    heartbeat_task = asyncio.create_task(connector._heartbeat_loop(heartbeat))
    try:
        await entered.wait()
        if serve_returns:
            await serve_task
        await asyncio.sleep(0.03)
        # A transport that never became ready must NOT signal the container alive:
        # Docker's freshness probe must fail. Before finding #1 this was enforced
        # by refusing to create the file at all — but that also suppressed the
        # external liveness detector's ability to correlate WHICH connection is
        # down (empty ID lists), so a multi-connection connector produced no
        # alarm. The invariant is really about FRESHNESS, not existence: the file
        # may exist to publish all-unhealthy CONTENT, but it must be born stale
        # (freshness withheld) and must report no healthy connection.
        assert not heartbeat_is_fresh(heartbeat, max_age_seconds=30)
        if heartbeat.exists():
            healthy_ids, unhealthy_ids = read_connection_health(heartbeat)
            assert healthy_ids == []
            assert unhealthy_ids == ["conn_1"]
        assert connector._connections["conn_1"].serve_status in {"starting", "stopped"}
    finally:
        release.set()
        if not serve_task.done():
            serve_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await serve_task
        heartbeat_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await heartbeat_task


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


@pytest.mark.asyncio
async def test_fail_closed_claim_is_stale_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fail-closed inode is invisible until its timestamp is stale."""
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    entered_link = asyncio.Event()
    release_link = asyncio.Event()
    loop = asyncio.get_running_loop()
    real_link = os.link

    def paused_link(source: str, destination: Path) -> None:
        loop.call_soon_threadsafe(entered_link.set)
        asyncio.run_coroutine_threadsafe(release_link.wait(), loop).result()
        real_link(source, destination)

    monkeypatch.setattr(os, "link", paused_link)
    claim = asyncio.create_task(
        asyncio.to_thread(connector._claim_heartbeat, heartbeat, b"payload", False)
    )
    await entered_link.wait()
    assert not heartbeat.exists()
    assert not heartbeat_is_fresh(heartbeat, max_age_seconds=30)
    release_link.set()
    identity = await claim

    assert identity is not None
    assert heartbeat.read_bytes() == b"payload"
    assert not heartbeat_is_fresh(heartbeat, max_age_seconds=30)
