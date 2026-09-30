import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aios_connector_http.runner import HttpConnector


class _C(HttpConnector):
    connector = "probe"


def test_heartbeat_setup_failure_does_not_skip_teardown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("AIOS_CONNECTOR_HEARTBEAT_PATH", str(blocker / "alive"))
    c = _C(base_url="http://x", token="aios_runtime_x")
    torn = []

    async def teardown() -> None:
        torn.append(True)

    c.teardown = teardown
    c.load_answered = AsyncMock(return_value=set())
    c._publish_tools_schema = AsyncMock()

    async def stop_soon() -> None:
        await asyncio.sleep(0.2)
        raise RuntimeError("loop exit")

    c._discovery_loop = lambda tg: stop_soon()
    c._tool_loop = AsyncMock()
    c._management_call_loop = AsyncMock()
    with contextlib.suppress(BaseException):
        asyncio.run(c.run())
    assert torn, "teardown skipped because heartbeat task exception escaped run()'s finally"


def test_heartbeat_loop_survives_and_retries_after_setup_failure(tmp_path: Path) -> None:
    """A failing iteration is retried, not fatal: once the obstruction is gone
    the loop publishes a heartbeat without being restarted."""
    from aios_connector_http.healthcheck import heartbeat_is_fresh
    from aios_connector_http.runner import _ConnectionState

    blocker = tmp_path / "dir"
    blocker.write_text("x")  # a FILE where the heartbeat directory must be
    path = blocker / "alive"
    c = _C(base_url="http://x", token="aios_runtime_x")
    c._connections = {"conn_1": _ConnectionState("conn_1", "a", serve_status="serving")}
    c._discovery_cursor = 0
    c.HEARTBEAT_INTERVAL = 0.0
    iterations = 0

    async def go() -> None:
        nonlocal iterations
        published = asyncio.Event()

        async def hook() -> None:
            nonlocal iterations
            iterations += 1
            if iterations == 1:
                blocker.unlink()  # obstruction removed after a failed iteration
            elif c._heartbeat_owned:
                published.set()

        c._heartbeat_iteration_hook = hook
        t = asyncio.create_task(c._heartbeat_loop(path))
        try:
            await asyncio.wait_for(published.wait(), 5)
            assert not t.done(), "heartbeat task died on a failed iteration"
        finally:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t

    asyncio.run(go())
    assert heartbeat_is_fresh(path, max_age_seconds=30)
