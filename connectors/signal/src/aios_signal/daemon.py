"""signal-cli subprocess lifecycle.

Owns spawning signal-cli in daemon mode, pumping its stdout/stderr into
structlog, and waiting for TCP readiness via ``listAccounts``. Crash-is-fatal:
an unexpected subprocess exit raises :class:`DaemonCrashError` through the
``crashed()`` awaitable, which app.py propagates into the TaskGroup to tear
the whole process down.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from pathlib import Path
from typing import Any, Self

import structlog

from .errors import BotAccountNotFoundError, DaemonCrashError
from .rpc import RpcClient, RpcListener

log = structlog.get_logger(__name__)

# 150 attempts @ 200ms = 30s total.
READY_POLL_ATTEMPTS = 150
READY_POLL_INTERVAL_S = 0.2
READY_POLL_TIMEOUT_S = 2.0

BOT_UUID_ATTEMPTS = 3
BOT_UUID_RETRY_INTERVAL_S = 2.0

SHUTDOWN_GRACE_S = 5.0


class SignalDaemon:
    """Async context manager owning a ``signal-cli --daemon`` subprocess.

    Usage::

        async with SignalDaemon(
            phone="+15551234567",
            config_dir=Path("~/.config/signal-cli"),
            cli_bin="signal-cli",
            host="127.0.0.1",
            port=7583,
        ) as daemon:
            bot_uuid = await daemon.discover_bot_uuid()
            async for envelope in daemon.listener.messages():
                ...
    """

    def __init__(
        self,
        *,
        phone: str,
        config_dir: Path,
        cli_bin: str,
        host: str,
        port: int,
    ) -> None:
        self.phone = phone
        self.config_dir = config_dir
        self.cli_bin = cli_bin
        self.host = host
        self.port = port

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
                log.warning("signal.daemon.sigkill", phone=self.phone)
                with contextlib.suppress(ProcessLookupError):
                    self._proc.kill()
                await self._proc.wait()

        for t in self._drain_tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        self._drain_tasks.clear()

    async def _spawn(self) -> None:
        args = [
            self.cli_bin,
            "--config",
            str(self.config_dir),
            "-a",
            self.phone,
            "-o",
            "json",
            "--trust-new-identities",
            "always",
            "daemon",
            "--tcp",
            f"{self.host}:{self.port}",
            "--receive-mode=on-connection",
        ]
        log.info("signal.daemon.spawn", phone=self.phone, host=self.host, port=self.port)
        self._proc = await _spawn_subprocess(args)
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._drain_tasks.append(
            asyncio.create_task(
                _drain(self._proc.stdout, "signal.daemon.stdout"),
                name="signal-daemon-stdout",
            )
        )
        self._drain_tasks.append(
            asyncio.create_task(
                _drain(self._proc.stderr, "signal.daemon.stderr"),
                name="signal-daemon-stderr",
            )
        )
        self._crash_future = asyncio.get_running_loop().create_future()
        self._watch_task = asyncio.create_task(self._watch_exit(), name="signal-daemon-watch")

    async def _watch_exit(self) -> None:
        assert self._proc is not None
        assert self._crash_future is not None
        rc = await self._proc.wait()
        if not self._crash_future.done():
            self._crash_future.set_exception(DaemonCrashError(f"signal-cli exited with code {rc}"))

    def crashed(self) -> asyncio.Future[None]:
        """Return a future that completes with :class:`DaemonCrashError` on crash.

        app.py awaits this as a supervision primitive inside its TaskGroup so
        that a daemon crash causes the whole process to exit non-zero.
        """
        assert self._crash_future is not None, "daemon not started"
        return self._crash_future

    async def _wait_for_tcp(self) -> None:
        """Poll ``listAccounts`` until it returns, or fail after 30s."""
        last_error: Exception | None = None
        for _ in range(READY_POLL_ATTEMPTS):
            try:
                probe = RpcClient(self.host, self.port, timeout=READY_POLL_TIMEOUT_S)
                await probe.call("listAccounts")
                await self.listener.connect()
                log.info("signal.daemon.ready", host=self.host, port=self.port)
                return
            except Exception as e:
                last_error = e
                await asyncio.sleep(READY_POLL_INTERVAL_S)
        raise DaemonCrashError(
            f"signal-cli daemon never became ready on {self.host}:{self.port}: {last_error!r}"
        )

    async def discover_bot_uuid(self) -> str:
        """Return the ACI UUID of the account whose number matches ``self.phone``.

        Retries transient RPC failures a few times; missing-account is fatal.
        """
        last_error: Exception | None = None
        for attempt in range(BOT_UUID_ATTEMPTS):
            if attempt > 0:
                await asyncio.sleep(BOT_UUID_RETRY_INTERVAL_S)
            try:
                accounts = await self.rpc.call("listAccounts")
            except Exception as e:
                last_error = e
                log.warning("signal.bot_uuid.rpc_error", attempt=attempt, error=str(e))
                continue
            uuid = _find_account_uuid(accounts, self.phone)
            if uuid is not None:
                return uuid
            raise BotAccountNotFoundError(
                f"signal-cli has no account for {self.phone}. "
                f"Run `signal-cli -a {self.phone} register` first."
            )
        raise BotAccountNotFoundError(
            f"listAccounts failed after {BOT_UUID_ATTEMPTS} attempts: {last_error!r}"
        )


async def _spawn_subprocess(args: list[str]) -> asyncio.subprocess.Process:
    """Thin wrapper over ``asyncio.create_subprocess_exec`` for mocking in tests."""
    spawn = asyncio.create_subprocess_exec
    return await spawn(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


def _normalize_phone(phone: str) -> str:
    return phone.strip()


def _find_account_uuid(accounts: Any, phone: str) -> str | None:
    """Scan the ``listAccounts`` result for the entry matching ``phone``."""
    if not isinstance(accounts, list):
        return None
    target = _normalize_phone(phone)
    for entry in accounts:
        if not isinstance(entry, dict):
            continue
        number = entry.get("number")
        if isinstance(number, str) and _normalize_phone(number) == target:
            uuid = entry.get("uuid")
            if isinstance(uuid, str) and uuid:
                return uuid
    return None


async def _drain(reader: asyncio.StreamReader, log_event: str) -> None:
    """Read lines from a subprocess pipe and log them through structlog."""
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
