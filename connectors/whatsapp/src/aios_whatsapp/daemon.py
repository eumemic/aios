"""whatsapp-daemon subprocess lifecycle.

Spawns the Go daemon, drains its stdio, waits for TCP readiness via
``version``, and exposes :class:`RpcClient` + :class:`RpcListener` for
the per-connection :class:`aios_whatsapp.connector.WhatsappConnector`
serve loop.

Crash-is-fatal: an unexpected subprocess exit sets
:class:`DaemonCrashError` on :meth:`crashed`.  The connector's listener
drain (spawned under the runner's TaskGroup) observes this indirectly
via :meth:`RpcListener.notifications` failing once the subprocess dies,
and the resulting exception propagates through the TaskGroup — tearing
the connection's ``serve_connection`` task down (but leaving sibling
connections running, via :meth:`HttpConnector._isolated_serve_connection`).
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
    """Manages one whatsapp-daemon subprocess serving a single phone.

    Per-connection scope (unlike Signal's shared multi-account daemon):
    whatsmeow's ``Client`` is per-device, so we spawn one daemon per
    phone and let crash isolation fall out from the runner's
    :meth:`HttpConnector._isolated_serve_connection` per-connection
    task scope.
    """

    def __init__(
        self,
        *,
        daemon_bin: str,
        host: str,
        port: int,
        store_dir: Path,
        log_level: str = "info",
    ) -> None:
        self.daemon_bin = daemon_bin
        self.host = host
        self.port = port
        self.store_dir = store_dir
        self.log_level = log_level

        self._proc: asyncio.subprocess.Process | None = None
        self._drain_tasks: list[asyncio.Task[None]] = []
        self._watch_task: asyncio.Task[None] | None = None
        self._crash_future: asyncio.Future[None] | None = None

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
        if self._watch_task is not None:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task
            self._watch_task = None

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
            "-log-level",
            self.log_level,
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
                _drain(self._proc.stdout, "whatsapp.daemon.stdout"),
                name="whatsapp-daemon-stdout",
            )
        )
        self._drain_tasks.append(
            asyncio.create_task(
                _drain(self._proc.stderr, "whatsapp.daemon.stderr"),
                name="whatsapp-daemon-stderr",
            )
        )
        self._crash_future = asyncio.get_running_loop().create_future()
        self._watch_task = asyncio.create_task(self._watch_exit(), name="whatsapp-daemon-watch")

    async def _watch_exit(self) -> None:
        assert self._proc is not None
        assert self._crash_future is not None
        rc = await self._proc.wait()
        if not self._crash_future.done():
            self._crash_future.set_exception(
                DaemonCrashError(f"whatsapp-daemon exited with code {rc}")
            )

    def crashed(self) -> asyncio.Future[None]:
        assert self._crash_future is not None, "daemon not started"
        return self._crash_future

    async def _wait_for_tcp(self) -> None:
        """Poll ``version`` until the daemon answers or attempts exhaust.

        Side-effect-free probe; works whether the daemon is mid-pairing
        or fully connected to WhatsApp.  ``READY_POLL_ATTEMPTS *
        READY_POLL_INTERVAL_S`` (30 s default) gives generous headroom
        for Go's startup + sqlstore initialization.
        """
        last_error: Exception | None = None
        for _ in range(READY_POLL_ATTEMPTS):
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

    async def version(self) -> dict[str, str]:
        """Return ``{name, version}`` from the daemon.

        Cheap, side-effect-free.  Used both at startup (via
        :meth:`_wait_for_tcp`) and by tests that want to round-trip
        the RPC layer without touching whatsmeow.
        """
        result = await self.rpc.call("version")
        return result if isinstance(result, dict) else {}


async def _spawn_subprocess(args: list[str]) -> asyncio.subprocess.Process:
    """Spawn whatsapp-daemon detached into its own session + process group.

    ``start_new_session=True`` (POSIX ``setsid()``) makes the child a
    session and process group leader so foreground-terminal SIGINTs
    don't reach the daemon — the connector's own SIGINT handler runs
    ``__aexit__``, which sends a clean SIGTERM.  Same rationale as
    :func:`aios_signal.daemon._spawn_subprocess`.
    """
    spawn = asyncio.create_subprocess_exec
    return await spawn(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )


async def _drain(reader: asyncio.StreamReader, log_event: str) -> None:
    try:
        while True:
            line = await reader.readline()
            if not line:
                return
            log.info(log_event, line=line.rstrip(b"\n").decode("utf-8", errors="replace"))
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.warning(f"{log_event}.read_error", error=str(e))
