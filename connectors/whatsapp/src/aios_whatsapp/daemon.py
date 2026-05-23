"""whatsapp-daemon subprocess lifecycle.

Spawns the Go daemon, drains its stdio, waits for TCP readiness via
``version``, and exposes :class:`RpcClient` + :class:`RpcListener` for
the connector's serve loop.  Crash mid-startup raises
:class:`DaemonCrashError`; crash mid-session is observed via the
listener stream raising :class:`ListenerClosedError`.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from pathlib import Path
from typing import Self

import structlog

from .errors import DaemonCrashError
from .rpc import RpcClient, RpcListener

log = structlog.get_logger(__name__)

READY_POLL_ATTEMPTS = 150  # 150 attempts @ 200ms = 30s total
READY_POLL_INTERVAL_S = 0.2
READY_POLL_TIMEOUT_S = 2.0

SHUTDOWN_GRACE_S = 5.0


class WhatsappDaemon:
    """Manages one whatsapp-daemon subprocess serving one paired phone."""

    def __init__(
        self,
        *,
        daemon_bin: str,
        host: str,
        port: int,
        store_dir: Path,
    ) -> None:
        self.daemon_bin = daemon_bin
        self.host = host
        self.port = port
        self.store_dir = store_dir

        self._proc: asyncio.subprocess.Process | None = None
        self._drain_tasks: list[asyncio.Task[None]] = []

        self.rpc = RpcClient(host, port)
        self.listener = RpcListener(host, port)

    async def __aenter__(self) -> Self:
        await self._spawn()
        try:
            await self._wait_for_tcp()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.listener.aclose()

        if self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=SHUTDOWN_GRACE_S)
            except TimeoutError:
                log.warning("whatsapp.daemon.sigkill", port=self.port)
                with contextlib.suppress(ProcessLookupError):
                    self._proc.kill()
                await self._proc.wait()

        for t in self._drain_tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        self._drain_tasks.clear()

    async def _spawn(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        args = [
            self.daemon_bin,
            "-listen",
            f"{self.host}:{self.port}",
            "-store-dir",
            str(self.store_dir),
        ]
        log.info(
            "whatsapp.daemon.spawn",
            daemon_bin=self.daemon_bin,
            host=self.host,
            port=self.port,
            store_dir=str(self.store_dir),
        )
        self._proc = await _spawn_subprocess(args)
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._drain_tasks.append(
            asyncio.create_task(
                _drain(self._proc.stdout, "stdout"),
                name="whatsapp-daemon-stdout",
            )
        )
        self._drain_tasks.append(
            asyncio.create_task(
                _drain(self._proc.stderr, "stderr"),
                name="whatsapp-daemon-stderr",
            )
        )

    async def _wait_for_tcp(self) -> None:
        """Poll ``version`` until the daemon answers, exits, or attempts exhaust."""
        last_error: Exception | None = None
        for _ in range(READY_POLL_ATTEMPTS):
            if self._proc is not None and self._proc.returncode is not None:
                raise DaemonCrashError(
                    f"whatsapp-daemon exited with code {self._proc.returncode} during startup"
                )
            try:
                probe = RpcClient(self.host, self.port, timeout=READY_POLL_TIMEOUT_S)
                await probe.call("version")
                await self.listener.connect()
                log.info("whatsapp.daemon.ready", host=self.host, port=self.port)
                return
            except Exception as e:
                last_error = e
                await asyncio.sleep(READY_POLL_INTERVAL_S)
        raise DaemonCrashError(
            f"whatsapp-daemon never became ready on {self.host}:{self.port}: {last_error!r}"
        )


async def _spawn_subprocess(args: list[str]) -> asyncio.subprocess.Process:
    # Detach into own session so terminal SIGINT doesn't cascade — the
    # connector's own SIGTERM handler runs __aexit__, which sends a
    # clean SIGTERM here.
    return await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )


async def _drain(reader: asyncio.StreamReader, stream_name: str) -> None:
    # Background stdio forwarder.  Broad-except keeps a stream read
    # failure (broken pipe after daemon dies) from silently dropping
    # the task with an unobserved exception that asyncio would lose.
    try:
        while True:
            line = await reader.readline()
            if not line:
                return
            log.info(
                "whatsapp.daemon.stream",
                stream=stream_name,
                line=line.rstrip(b"\n").decode("utf-8", errors="replace"),
            )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.warning("whatsapp.daemon.stream.read_error", stream=stream_name, error=str(e))
