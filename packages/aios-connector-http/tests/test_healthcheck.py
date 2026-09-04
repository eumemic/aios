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
        published = asyncio.Event()

        async def after_iteration() -> None:
            published.set()

        connector._heartbeat_iteration_hook = after_iteration
        connector._discovery_cursor = 0
        async with asyncio.timeout(1):
            await published.wait()
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
        # Recovery publishes a new inode so a paused former owner cannot resume.
        assert connector._heartbeat_identity != original_identity
        assert read_connection_health(heartbeat) == ([], [])
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.parametrize(
    "stale_payload", [None, b'{"healthy_connection_ids": ["old"], "unhealthy_connection_ids": []}']
)
@pytest.mark.asyncio
async def test_first_fresh_heartbeat_contains_current_authoritative_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stale_payload: bytes | None
) -> None:
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    if stale_payload is not None:
        heartbeat.write_bytes(stale_payload)
        old = time.time() - 3600
        os.utime(heartbeat, (old, old))

    connector._discovery_cursor = 0
    connector._connections["current"] = _ConnectionState(
        "current", "account", serve_status="serving"
    )
    published = asyncio.Event()
    release = asyncio.Event()

    async def after_iteration() -> None:
        published.set()
        await release.wait()

    connector._heartbeat_iteration_hook = after_iteration
    task = asyncio.create_task(connector._heartbeat_loop(heartbeat))
    try:
        await published.wait()
        assert heartbeat_is_fresh(heartbeat, max_age_seconds=30)
        assert read_connection_health(heartbeat) == (["current"], [])
        monkeypatch.setenv("AIOS_CONNECTOR_HEARTBEAT_PATH", str(heartbeat))
        main()
    finally:
        release.set()
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
async def test_cleanup_refuses_to_unlink_replacement_inode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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

    def refuse_unlink(_path: Path, *args: object, **kwargs: object) -> None:
        raise AssertionError("cleanup must not unlink through a replaceable pathname")

    monkeypatch.setattr(Path, "unlink", refuse_unlink)
    await connector._remove_owned_heartbeat(heartbeat)

    assert heartbeat.exists()
    assert heartbeat.read_text() == "replacement"
    assert not connector._heartbeat_owned
    assert connector._heartbeat_identity is None


@pytest.mark.asyncio
async def test_heartbeat_loop_survives_symlink_replacement(tmp_path: Path) -> None:
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    operator_file = tmp_path / "operator"
    operator_file.write_text("operator")
    connector._discovery_cursor = 0
    connector.HEARTBEAT_INTERVAL = 0.01
    first_published = asyncio.Event()
    replacement_installed = asyncio.Event()
    retried = asyncio.Event()
    iterations = 0

    async def after_iteration() -> None:
        nonlocal iterations
        iterations += 1
        if iterations == 1:
            first_published.set()
            await replacement_installed.wait()
        elif iterations == 2:
            retried.set()

    connector._heartbeat_iteration_hook = after_iteration
    task = asyncio.create_task(connector._heartbeat_loop(heartbeat))
    try:
        async with asyncio.timeout(1):
            await first_published.wait()
        heartbeat.unlink()
        heartbeat.symlink_to(operator_file)
        replacement_installed.set()
        async with asyncio.timeout(1):
            await retried.wait()

        assert not task.done()
        assert heartbeat.is_symlink()
        assert operator_file.read_text() == "operator"
        assert not connector._heartbeat_owned
        assert connector._heartbeat_identity is None
    finally:
        replacement_installed.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


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


def test_fail_closed_claim_is_stale_on_publication(tmp_path: Path) -> None:
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"

    identity = connector._claim_heartbeat(heartbeat, b"payload", False)

    assert identity is not None
    assert heartbeat.read_bytes() == b"payload"
    assert not heartbeat_is_fresh(heartbeat, max_age_seconds=30)


def test_claim_does_not_leak_internal_staging_path(tmp_path: Path) -> None:
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"

    identity = connector._claim_heartbeat(heartbeat, b"payload", True)

    assert identity is not None
    assert heartbeat.read_bytes() == b"payload"
    assert list(tmp_path.iterdir()) == [heartbeat]
    assert heartbeat.stat().st_nlink == 1


def test_fresh_owner_refusals_do_not_leak_descriptors(tmp_path: Path) -> None:
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    heartbeat.write_bytes(b"peer")
    proc_fds = Path("/proc/self/fd")
    if not proc_fds.is_dir():
        pytest.skip("descriptor accounting requires procfs")
    before = len(list(proc_fds.iterdir()))

    for _ in range(200):
        assert connector._claim_heartbeat(heartbeat, b"payload", True) is None

    after = len(list(proc_fds.iterdir()))
    assert heartbeat.read_bytes() == b"peer"
    assert after - before < 5


def test_stale_reclaim_revokes_superseded_owner(tmp_path: Path) -> None:
    old_owner = _Connector(base_url="http://example.test", token="token")
    new_owner = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    old_payload = b"old owner"
    new_payload = b"new owner"

    old_identity = old_owner._claim_heartbeat(heartbeat, old_payload, True)
    assert old_identity is not None
    stale = time.time() - 3600
    os.utime(heartbeat, (stale, stale))

    new_identity = new_owner._claim_heartbeat(heartbeat, new_payload, True)

    assert new_identity is not None
    assert new_identity != old_identity
    assert not old_owner._refresh_heartbeat(heartbeat, old_identity, old_payload, True)
    assert heartbeat.read_bytes() == new_payload


def test_repeated_stale_reclaims_do_not_accumulate_staging_paths(tmp_path: Path) -> None:
    """Restart churn retains only the single public heartbeat inode."""
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"

    for restart in range(5):
        identity = connector._claim_heartbeat(heartbeat, f"payload-{restart}".encode(), True)
        assert identity is not None
        assert heartbeat.read_bytes() == f"payload-{restart}".encode()
        assert list(tmp_path.iterdir()) == [heartbeat]
        assert heartbeat.stat().st_nlink == 1
        stale = time.time() - 3600
        os.utime(heartbeat, (stale, stale))


@pytest.mark.asyncio
async def test_fail_closed_claim_reclaims_stale_crash_debris(tmp_path: Path) -> None:
    """Finding #1: a restart whose transports are all still starting must
    reclaim a STALE heartbeat left by the crashed process and republish the
    CURRENT connection attribution, while keeping the file stale (freshness
    withheld) so Docker still ages the runtime out.
    """
    import json

    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    # Stale debris from the previous process: names an obsolete healthy id.
    heartbeat.write_text(
        json.dumps(
            {"healthy_connection_ids": ["old"], "unhealthy_connection_ids": []},
            sort_keys=True,
        )
    )
    old = time.time() - 3600
    os.utime(heartbeat, (old, old))

    connector._discovery_cursor = 0
    connector.HEARTBEAT_INTERVAL = 0.0
    # Two current connections, both still starting -> every transport unhealthy.
    connector._connections["conn_a"] = _ConnectionState("conn_a", "account")
    connector._connections["conn_b"] = _ConnectionState("conn_b", "account")

    iterated = asyncio.Event()

    async def hook() -> None:
        iterated.set()

    connector._heartbeat_iteration_hook = hook
    task = asyncio.create_task(connector._heartbeat_loop(heartbeat))
    try:
        await iterated.wait()
        # Freshness stays withheld: Docker must still age the runtime out.
        assert not heartbeat_is_fresh(heartbeat, max_age_seconds=30)
        # Attribution has converged to the current authoritative set.
        healthy_ids, unhealthy_ids = read_connection_health(heartbeat)
        assert healthy_ids == []
        assert unhealthy_ids == ["conn_a", "conn_b"]
        assert connector._heartbeat_owned
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_fail_closed_claim_refuses_fresh_preexisting_file(tmp_path: Path) -> None:
    """Over-correction guard: reclaiming stale debris must NOT extend to a
    FRESH pre-existing heartbeat (an operator replacement / a live peer). The
    fail-closed claim path must still refuse it.
    """
    import json

    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    heartbeat.write_text(
        json.dumps(
            {"healthy_connection_ids": ["peer"], "unhealthy_connection_ids": []},
            sort_keys=True,
        )
    )
    # Fresh: touched now, not aged out.
    connector._discovery_cursor = 0
    connector.HEARTBEAT_INTERVAL = 0.0
    connector._connections["conn_a"] = _ConnectionState("conn_a", "account")

    iterated = asyncio.Event()

    async def hook() -> None:
        iterated.set()

    connector._heartbeat_iteration_hook = hook
    task = asyncio.create_task(connector._heartbeat_loop(heartbeat))
    try:
        await iterated.wait()
        # The fresh peer/operator file is untouched and unclaimed.
        healthy_ids, unhealthy_ids = read_connection_health(heartbeat)
        assert healthy_ids == ["peer"]
        assert unhealthy_ids == []
        assert not connector._heartbeat_owned
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_stale_reclaim_refuses_replacement_after_inspection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A replacement installed after stale inspection is never overwritten."""
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    heartbeat.write_bytes(b"stale owner")
    old = time.time() - 3600
    os.utime(heartbeat, (old, old))

    real_ftruncate = os.ftruncate
    replaced = False

    def replace_after_guard(fd: int, length: int) -> None:
        nonlocal replaced
        if not replaced:
            # The stale inode has been inspected and captured by the guard. A
            # replacement at the public pathname must survive the stale update.
            heartbeat.unlink()
            heartbeat.write_bytes(b"operator replacement")
            replaced = True
        real_ftruncate(fd, length)

    monkeypatch.setattr(os, "ftruncate", replace_after_guard)

    identity = connector._claim_heartbeat(heartbeat, b"new owner", False)

    assert identity is None
    assert heartbeat.read_bytes() == b"operator replacement"


def test_stale_reclaim_refuses_same_inode_refresh_before_exchange(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A peer refreshing stale inode content/freshness wins reclamation."""
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    heartbeat.write_bytes(b"stale owner")
    old = time.time() - 3600
    os.utime(heartbeat, (old, old))
    original_identity = (heartbeat.stat().st_dev, heartbeat.stat().st_ino)

    real_ftruncate = os.ftruncate
    refreshed = False

    def refresh_incumbent(fd: int, length: int) -> None:
        nonlocal refreshed
        if not refreshed:
            # Refresh through the public path without replacing its inode, at the
            # claimant's final pre-exchange checkpoint.
            with heartbeat.open("r+b") as peer:
                peer.seek(0)
                peer.truncate()
                peer.write(b"live peer")
            refreshed = True
        real_ftruncate(fd, length)

    monkeypatch.setattr(os, "ftruncate", refresh_incumbent)

    identity = connector._claim_heartbeat(heartbeat, b"claimant", False)

    assert identity is None
    assert (heartbeat.stat().st_dev, heartbeat.stat().st_ino) == original_identity
    assert heartbeat.read_bytes() == b"live peer"
    assert heartbeat_is_fresh(heartbeat, max_age_seconds=30)
