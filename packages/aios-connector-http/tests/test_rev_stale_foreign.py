import asyncio
import contextlib
import os
import time
from pathlib import Path

from aios_connector_http.runner import HttpConnector, _ConnectionState


class _C(HttpConnector):
    connector = "probe"


def test_stale_preexisting_non_aios_file_is_not_destroyed(tmp_path: Path) -> None:
    path = tmp_path / "alive"
    path.write_text("operator data - not an AIOS heartbeat")
    old = time.time() - 3600
    os.utime(path, (old, old))
    HttpConnector._claim_heartbeat(
        path, b'{"healthy_connection_ids":["conn_1"],"unhealthy_connection_ids":[]}', True
    )
    survivors = [p.read_text() for p in tmp_path.rglob("*") if p.is_file()]
    assert any("operator data" in s for s in survivors), "pre-existing non-AIOS file destroyed"


def test_loop_does_not_destroy_stale_non_aios_file(tmp_path: Path) -> None:
    path = tmp_path / "alive"
    path.write_text("operator data - not an AIOS heartbeat")
    old = time.time() - 3600
    os.utime(path, (old, old))
    c = _C(base_url="http://x", token="aios_runtime_x")
    c._connections = {"conn_1": _ConnectionState("conn_1", "a", serve_status="serving")}
    c._discovery_cursor = 0
    c.HEARTBEAT_INTERVAL = 0.0

    async def go() -> None:
        done = asyncio.Event()

        async def hook() -> None:
            done.set()
            await asyncio.Event().wait()

        c._heartbeat_iteration_hook = hook
        t = asyncio.create_task(c._heartbeat_loop(path))
        await asyncio.wait_for(done.wait(), 5)
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t

    asyncio.run(go())
    survivors = [p.read_text() for p in tmp_path.rglob("*") if p.is_file()]
    assert any("operator data" in s for s in survivors)
