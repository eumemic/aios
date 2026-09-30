"""Heartbeat publication must work where ``O_TMPFILE`` does not (overlayfs).

Property: when the heartbeat directory cannot create an unnamed ``O_TMPFILE``
inode (overlayfs, the storage driver for every prod connector's ``/var/run``,
returns ``EOPNOTSUPP``) or the platform lacks the ``os.O_TMPFILE`` constant, the
runner still publishes a complete heartbeat at the public path. That heartbeat is
fresh, and ``python -m aios_connector_http.healthcheck`` exits 0.

Failure mode (before the fix): ``_claim_heartbeat`` returned ``None`` for every
non-empty payload, so no heartbeat was ever written and the image HEALTHCHECK
reported the connector unhealthy forever.

The fallback stages the inode under a unique name in a private directory inside
the heartbeat directory. The tests also pin down what the fallback must not
do: leave staging names behind, publish a torn file, treat crash debris as a
heartbeat, or destroy a foreign file.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import aios_connector_http.runner as runner_mod
import pytest
from aios_connector_http.healthcheck import heartbeat_is_fresh, main, read_connection_health
from aios_connector_http.runner import HttpConnector, _ConnectionState

_REAL_OPEN = os.open
_O_TMPFILE = getattr(os, "O_TMPFILE", 0)


class _Connector(HttpConnector):
    connector = "probe"


def _payload(healthy: list[str], unhealthy: list[str]) -> bytes:
    return json.dumps(
        {"healthy_connection_ids": healthy, "unhealthy_connection_ids": unhealthy},
        sort_keys=True,
    ).encode()


def _open_without_tmpfile(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
    if _O_TMPFILE and (flags & _O_TMPFILE) == _O_TMPFILE:
        raise OSError(errno.EOPNOTSUPP, os.strerror(errno.EOPNOTSUPP), str(path))
    return _REAL_OPEN(path, flags, *args, **kwargs)


@pytest.fixture(params=["eopnotsupp", "constant_absent"])
def no_tmpfile(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Make ``O_TMPFILE`` unusable in the two ways prod and other platforms do."""
    if request.param == "eopnotsupp":
        if not _O_TMPFILE:
            pytest.skip("platform has no O_TMPFILE to reject")
        monkeypatch.setattr(os, "open", _open_without_tmpfile)
    else:
        monkeypatch.delattr(os, "O_TMPFILE", raising=False)
    yield request.param


def _assert_probe_passes(monkeypatch: pytest.MonkeyPatch, heartbeat: Path) -> None:
    monkeypatch.setenv("AIOS_CONNECTOR_HEARTBEAT_PATH", str(heartbeat))
    try:
        main()
    except SystemExit as exc:  # pragma: no cover - the failure being guarded
        assert exc.code in (0, None), f"healthcheck exited {exc.code}"


def _only_public(tmp_path: Path, heartbeat: Path) -> None:
    assert sorted(tmp_path.iterdir()) == [heartbeat], "staging name leaked"
    assert heartbeat.stat().st_nlink == 1


@pytest.mark.asyncio
async def test_heartbeat_loop_publishes_without_o_tmpfile(
    no_tmpfile: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    connector = _Connector(base_url="http://example.test", token="token")
    heartbeat = tmp_path / "alive"
    connector._discovery_cursor = 0
    connector._connections["conn_1"] = _ConnectionState("conn_1", "acct", serve_status="serving")
    connector.HEARTBEAT_INTERVAL = 0.0
    iterations = 0
    done = asyncio.Event()

    async def hook() -> None:
        nonlocal iterations
        iterations += 1
        if iterations >= 3:  # one claim and at least two refreshes
            done.set()
            await asyncio.Event().wait()

    connector._heartbeat_iteration_hook = hook
    task = asyncio.create_task(connector._heartbeat_loop(heartbeat))
    try:
        async with asyncio.timeout(10):
            await done.wait()
        assert connector._heartbeat_owned
        assert heartbeat_is_fresh(heartbeat, max_age_seconds=30)
        assert read_connection_health(heartbeat) == (["conn_1"], [])
        _assert_probe_passes(monkeypatch, heartbeat)
        _only_public(tmp_path, heartbeat)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_claim_and_refresh_without_o_tmpfile(
    no_tmpfile: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    heartbeat = tmp_path / "alive"
    first = _payload(["a"], [])
    identity = HttpConnector._claim_heartbeat(heartbeat, first, True)
    assert identity is not None
    assert heartbeat.read_bytes() == first
    _only_public(tmp_path, heartbeat)

    second = _payload(["a", "b"], [])
    refreshed = HttpConnector._refresh_heartbeat(heartbeat, identity, second, True)
    assert refreshed is not None
    assert heartbeat.read_bytes() == second
    assert heartbeat_is_fresh(heartbeat, max_age_seconds=30)
    _assert_probe_passes(monkeypatch, heartbeat)
    _only_public(tmp_path, heartbeat)
    # The replacement is a new ownership generation: the old identity is revoked.
    assert HttpConnector._refresh_heartbeat(heartbeat, identity, first, True) is None


def test_stale_reclaim_without_o_tmpfile_revokes_predecessor(
    no_tmpfile: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """F1 still holds: a successor never reads or inherits the predecessor's file."""
    heartbeat = tmp_path / "alive"
    old = HttpConnector._claim_heartbeat(heartbeat, _payload(["old"], []), True)
    assert old is not None
    stale = time.time() - 3600
    os.utime(heartbeat, (stale, stale))

    new = HttpConnector._claim_heartbeat(heartbeat, _payload(["new"], []), True)
    assert new is not None
    assert read_connection_health(heartbeat) == (["new"], [])
    assert HttpConnector._refresh_heartbeat(heartbeat, old, _payload(["old"], []), True) is None
    _assert_probe_passes(monkeypatch, heartbeat)
    _only_public(tmp_path, heartbeat)


def test_fresh_foreign_file_is_untouched_without_o_tmpfile(no_tmpfile: str, tmp_path: Path) -> None:
    heartbeat = tmp_path / "alive"
    heartbeat.write_bytes(b"operator data")
    assert HttpConnector._claim_heartbeat(heartbeat, _payload(["a"], []), True) is None
    assert heartbeat.read_bytes() == b"operator data"
    _only_public(tmp_path, heartbeat)


def test_stale_foreign_file_is_not_reclaimed_without_o_tmpfile(
    no_tmpfile: str, tmp_path: Path
) -> None:
    """B1 still holds: age alone never licenses destroying a non-AIOS file."""
    heartbeat = tmp_path / "alive"
    heartbeat.write_bytes(b"operator data")
    stale = time.time() - 3600
    os.utime(heartbeat, (stale, stale))
    assert HttpConnector._claim_heartbeat(heartbeat, _payload(["a"], []), True) is None
    assert heartbeat.read_bytes() == b"operator data"
    _only_public(tmp_path, heartbeat)


@pytest.mark.parametrize("step", ["write", "fsync"])
def test_failed_staging_leaves_no_name_behind(
    no_tmpfile: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, step: str
) -> None:
    """Every failure path removes the staging name and publishes nothing torn."""
    heartbeat = tmp_path / "alive"

    def _eio(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(errno.EIO, "EIO")

    with monkeypatch.context() as patch:
        patch.setattr(os, step, _eio)
        with pytest.raises(OSError):
            HttpConnector._claim_heartbeat(heartbeat, _payload(["a"], []), True)
    assert list(tmp_path.iterdir()) == []

    identity = HttpConnector._claim_heartbeat(heartbeat, _payload(["a"], []), True)
    assert identity is not None
    before = heartbeat.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(os, step, _eio)
        assert HttpConnector._refresh_heartbeat(heartbeat, identity, _payload([], []), True) is None
    assert heartbeat.read_bytes() == before
    _only_public(tmp_path, heartbeat)


def test_link_failure_leaves_no_name_behind(
    no_tmpfile: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    heartbeat = tmp_path / "alive"

    def _eio(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(errno.EIO, "EIO")

    with monkeypatch.context() as patch:
        patch.setattr(runner_mod, "_link_unnamed_file", _eio)
        assert HttpConnector._claim_heartbeat(heartbeat, _payload(["a"], []), True) is None
    assert list(tmp_path.iterdir()) == []

    identity = HttpConnector._claim_heartbeat(heartbeat, _payload(["a"], []), True)
    assert identity is not None
    before = heartbeat.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(runner_mod, "_link_unnamed_file", _eio)
        assert HttpConnector._refresh_heartbeat(heartbeat, identity, _payload([], []), True) is None
    assert heartbeat.read_bytes() == before
    _only_public(tmp_path, heartbeat)

    with monkeypatch.context() as patch:
        patch.setattr(runner_mod, "_rename_exchange", lambda *_a: False)
        assert HttpConnector._refresh_heartbeat(heartbeat, identity, _payload([], []), True) is None
    assert heartbeat.read_bytes() == before
    _only_public(tmp_path, heartbeat)


def test_crash_debris_is_neither_a_heartbeat_nor_reclaimed(
    no_tmpfile: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A staging name left by a crash is never read as health and never destroyed.

    Debris is a fresh, well-formed, all-healthy payload. That is the worst case,
    because reading it as health would be exactly the F1 defect.
    """
    heartbeat = tmp_path / "alive"
    debris_bytes = _payload(["ghost"], [])
    debris = [
        tmp_path / ".aios-hb-staging.crashed" / "staging",  # named-staging fallback
        tmp_path / ".alive.crashed" / "claimant",  # exchange staging
    ]
    for leftover in debris:
        leftover.parent.mkdir()
        leftover.write_bytes(debris_bytes)

    monkeypatch.setenv("AIOS_CONNECTOR_HEARTBEAT_PATH", str(heartbeat))
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    assert not heartbeat_is_fresh(heartbeat, max_age_seconds=30)

    identity = HttpConnector._claim_heartbeat(heartbeat, _payload(["current"], []), True)
    assert identity is not None
    assert read_connection_health(heartbeat) == (["current"], [])
    _assert_probe_passes(monkeypatch, heartbeat)
    # The debris is left intact. It may belong to a peer, so it is not reclaimed.
    for leftover in debris:
        assert leftover.read_bytes() == debris_bytes
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        ".aios-hb-staging.crashed",
        ".alive.crashed",
        "alive",
    ]
