"""F1: a failing filesystem step during heartbeat refresh must not raise.

Property: if any filesystem step of a heartbeat refresh fails, the refresh
reports "not published" (``None``) instead of raising, and ``_heartbeat_loop``
keeps running and retries on the next interval.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aios_connector_http.runner as runner_mod
import pytest
from aios_connector_http.runner import HttpConnector


class _Connector(HttpConnector):
    connector = "probe"


def _eio(*_args: Any, **_kwargs: Any) -> None:
    raise OSError(5, "EIO")


def test_refresh_survives_link_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "hb"
    identity = HttpConnector._claim_heartbeat(path, b"{}", True)
    assert identity is not None
    before = path.read_bytes()
    monkeypatch.setattr(runner_mod, "_link_unnamed_file", _eio)
    assert HttpConnector._refresh_heartbeat(path, identity, b'{"x": 1}', True) is None
    # Nothing was published and the incumbent is intact.
    assert path.read_bytes() == before
    assert HttpConnector._still_owns_heartbeat(path, identity)
    # No staging leftovers in the heartbeat directory.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["hb"]


@pytest.mark.parametrize("step", ["fsync", "utime", "write"])
def test_refresh_survives_other_fs_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    path = tmp_path / "hb"
    identity = HttpConnector._claim_heartbeat(path, b"{}", True)
    assert identity is not None
    monkeypatch.setattr(runner_mod.os, step, _eio)
    assert HttpConnector._refresh_heartbeat(path, identity, b'{"x": 1}', True) is None


@pytest.mark.asyncio
async def test_heartbeat_loop_survives_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connector = _Connector(base_url="http://example.test", token="token")
    path = tmp_path / "alive"
    connector.HEARTBEAT_INTERVAL = 0.0
    connector._discovery_cursor = 0

    real_link = runner_mod._link_unnamed_file
    calls = {"n": 0}

    def _flaky_link(fd: int, destination: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 2:  # first refresh's staging link (1st call is the claim)
            raise OSError(5, "EIO")
        real_link(fd, destination)

    monkeypatch.setattr(runner_mod, "_link_unnamed_file", _flaky_link)

    iterations = 0
    done = asyncio.Event()

    async def hook() -> None:
        nonlocal iterations
        iterations += 1
        if iterations >= 4:
            done.set()

    connector._heartbeat_iteration_hook = hook
    task = asyncio.create_task(connector._heartbeat_loop(path))
    try:
        async with asyncio.timeout(5):
            await done.wait()
        assert not task.done(), task.exception() if task.done() else None
        assert calls["n"] >= 3
        # Ownership was kept across the transient failure and refresh resumed.
        assert connector._heartbeat_owned
        assert connector._heartbeat_identity is not None
        assert HttpConnector._still_owns_heartbeat(path, connector._heartbeat_identity)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
