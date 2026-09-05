"""Reproduction + guards for heartbeat refresh atomicity (PR #2355 finding).

Property under test: readers must always observe a complete, structurally
valid heartbeat snapshot. Refreshing an owned heartbeat must never expose an
empty or partially written file as fresh.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from aios_connector_http.healthcheck import read_connection_health
from aios_connector_http.runner import HttpConnector


def _payload(healthy: list[str], unhealthy: list[str]) -> bytes:
    return json.dumps(
        {
            "healthy_connection_ids": healthy,
            "unhealthy_connection_ids": unhealthy,
        },
        sort_keys=True,
    ).encode()


def _claim(path: Path, payload: bytes) -> tuple[int, int]:
    identity = HttpConnector._claim_heartbeat(path, payload, True)
    assert identity is not None
    return identity


def test_reader_never_observes_partial_snapshot_during_refresh(tmp_path: Any) -> None:
    """Interpose at the write boundary and read concurrently.

    On the in-place ``ftruncate`` + ``os.write`` head, a reader that runs after
    truncation but before the new bytes land observes an empty file and
    ``read_connection_health`` returns ``([], [])`` even though the connection
    is healthy. With atomic publication the reader can only see the prior
    complete snapshot or the new complete snapshot.
    """
    path = tmp_path / "hb"
    first = _payload(["conn_1"], [])
    identity = _claim(path, first)
    # Sanity: the claimed heartbeat is a complete, valid snapshot.
    assert read_connection_health(path) == (["conn_1"], [])

    at_boundary = threading.Event()
    release = threading.Event()
    observed: list[tuple[list[str], list[str]]] = []

    real_write = os.write

    def _paused_write(fd: int, data: bytes) -> int:
        # Fires while the heartbeat is being replaced. On the buggy head the
        # public inode has already been truncated to empty at this point.
        at_boundary.set()
        release.wait(timeout=5)
        return real_write(fd, data)

    second = _payload(["conn_1", "conn_2"], [])

    def _refresh() -> None:
        import aios_connector_http.runner as runner_mod

        orig = runner_mod.os.write
        runner_mod.os.write = _paused_write  # type: ignore[assignment]
        try:
            HttpConnector._refresh_heartbeat(path, identity, second, True)
        finally:
            runner_mod.os.write = orig  # type: ignore[assignment]

    writer = threading.Thread(target=_refresh)
    writer.start()
    try:
        assert at_boundary.wait(timeout=5), "writer never reached the write boundary"
        # Read while the writer is paused mid-publish.
        observed.append(read_connection_health(path))
    finally:
        release.set()
        writer.join(timeout=5)

    healthy, unhealthy = observed[0]
    # The reader must have seen a complete snapshot: either the prior one or
    # the new one -- never empty/partial. On the buggy head this is ([], []).
    assert (healthy, unhealthy) in (
        (["conn_1"], []),
        (["conn_1", "conn_2"], []),
    ), f"reader observed a partial/empty snapshot mid-refresh: {observed[0]}"

    # And the final published state is the new complete snapshot.
    assert read_connection_health(path) == (["conn_1", "conn_2"], [])


def test_refresh_still_replaces_content_positive_control(tmp_path: Any) -> None:
    """Positive control / over-correction guard: an uncontended refresh must
    actually replace the published content and keep it a single valid snapshot.

    A degenerate 'never truncate / never write' fix would also make the
    partial-read test pass, so assert the new bytes really land.
    """
    path = tmp_path / "hb"
    identity = _claim(path, _payload(["a"], []))
    assert read_connection_health(path) == (["a"], [])

    new = _payload(["a", "b"], ["c"])
    result = HttpConnector._refresh_heartbeat(path, identity, new, True)
    # Refresh now returns the identity of the inode published at the path
    # (a new inode under atomic publication), never None on success.
    assert result is not None
    assert read_connection_health(path) == (["a", "b"], ["c"])
    # The published file is exactly the new snapshot -- no leftover bytes.
    assert path.read_bytes() == new


def test_refresh_preserves_identity_replacement_safety(tmp_path: Any) -> None:
    """Standing property: refresh must still refuse when the public pathname no
    longer resolves to the owned inode (replacement safety), and must not
    clobber an independent replacement.
    """
    path = tmp_path / "hb"
    identity = _claim(path, _payload(["a"], []))

    # Operator replaces the pathname with an independent inode. Create the
    # replacement as a distinct file and rename it over the path so its inode
    # number cannot coincide with the just-freed one.
    replacement = _payload(["independent"], [])
    other = tmp_path / "other"
    other.write_bytes(replacement)
    assert (other.stat().st_dev, other.stat().st_ino) != identity
    os.replace(other, path)

    result = HttpConnector._refresh_heartbeat(path, identity, _payload(["a"], []), True)
    assert result is None, "refresh must relinquish a replaced pathname"
    # The independent replacement's content is untouched.
    assert read_connection_health(path) == (["independent"], [])
