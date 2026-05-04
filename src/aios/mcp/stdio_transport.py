"""Stdio MCP transport for connector subprocesses.

Thin wrapper over :func:`mcp.client.stdio.stdio_client`.  The SDK already
handles subprocess lifecycle (``asyncio.create_subprocess_exec`` under the
hood, env merging via ``get_default_environment()``, MCP-spec shutdown
sequence: close stdin → SIGTERM → SIGKILL).  This module adds three
pieces on top:

* A :class:`ConnectorSpec` dataclass so the supervisor can describe a
  connector to launch without leaking ``StdioServerParameters`` shape
  through its surface.
* :func:`open_connector_session` — async context manager that spawns
  the subprocess, builds a :class:`ClientSession`, runs ``initialize()``,
  and yields the live session plus the ``InitializeResult``.  The
  caller owns the lifetime; on exit the SDK tears the subprocess down.
* A read-stream splitter that diverts aios-namespaced notifications
  (``notifications/aios/*``) to a caller-supplied async callback before
  they reach ``ClientSession``.  Without this hop the SDK's pydantic
  validation rejects unknown notification methods (``ServerNotification``
  is a closed union in the SDK), and the connector's account snapshots
  + inbound deliveries would get silently dropped.

Restart-with-backoff and circuit-breaker live in the connector
supervisor, not here.  Each call to :func:`open_connector_session`
spawns *one* subprocess.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast

import anyio
from mcp.client.session import ClientSession, MessageHandlerFnT
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.message import SessionMessage
from mcp.types import InitializeResult, JSONRPCNotification

from aios.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream


log = get_logger("aios.mcp.stdio_transport")

AIOS_NOTIFICATION_PREFIX = "notifications/aios/"

# Initialize handshake ceiling.  A connector that opens stdio but
# stalls on ``notifications/initialized`` would otherwise park the
# supervisor in ``starting`` forever; this bound makes the timeout
# look like any other crash and trip restart-with-backoff.  Generous
# because real connectors may load credentials / open daemons during
# their initialize().
_INIT_TIMEOUT_S = 30.0


def _safe_close_fd(fd: int) -> None:
    """Close an OS fd, ignoring EBADF if it's already closed.

    Used as a defense-in-depth cleanup callback on the spawn path: the
    pump task closes the read-end on EOF, but if spawn fails before
    the pump starts, this callback closes it instead.
    """
    with contextlib.suppress(OSError):
        os.close(fd)


@dataclass(frozen=True)
class ConnectorSpec:
    """How to launch one connector subprocess.

    The ``aios.connectors`` entry-point group resolves to a callable
    that returns a :class:`ConnectorSpec` — that's the integration
    point for connector packages (signal-cli, telegram, the reference
    SDK).  Exposing the spec as data lets the supervisor log, diff,
    and re-spawn without owning entry-point loading.
    """

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: Path | None = None


# Patterns matching credentials that connector libraries embed in URL
# paths and routinely log via their HTTP clients (e.g. python-telegram-bot
# delegates to httpx, whose default logger emits the full request URL
# including the bot token).  We redact at emission time rather than
# upstream-filtering each library because stderr is a multi-source pipe
# and the supervisor has no per-line knowledge of which library produced
# each line.
#
# Telegram bot tokens have the shape ``<bot_id>:<32+ alphanum-_-_>``, used
# as a path segment after ``/bot``.  The redaction preserves the URL
# shape so the operator can still see *which* request was made (and
# spot, e.g., a getUpdates polling loop) without exposing the token.
_BOT_TOKEN_RE = re.compile(r"/bot\d+:[A-Za-z0-9_\-]+")


def _redact_secrets(line: str) -> str:
    """Strip embedded secrets from a stderr line before logging.

    The redaction is deliberately conservative — it doesn't try to
    parse log formats; it just matches well-known credential shapes.
    Add new patterns here as connectors that leak their own
    credentials are surfaced.

    The substring fast-path skips the regex on the ~all-of-stderr
    that doesn't mention ``/bot``; matters for high-frequency emitters
    like PTB's ``getUpdates`` polling loop.
    """
    if "/bot" not in line:
        return line
    return _BOT_TOKEN_RE.sub("/bot<redacted>", line)


async def _pump_stderr(connector_name: str, read_fd: int) -> None:
    """Read connector stderr via ``loop.add_reader`` and emit lines as structlog events.

    The SDK's ``stdio_client`` accepts ``errlog`` and forwards it to
    ``subprocess.Popen(stderr=...)``, which calls ``.fileno()`` to
    ``dup2`` onto the child's fd 2.  We hand the SDK the write-end of
    an OS pipe and consume the read-end here.

    A blocking read on a thread (``anyio.wrap_file``) leaks: cancelling
    the awaiting task doesn't preempt the kernel-level read, so the
    task group can't exit cleanly until the pipe EOFs.  Using
    ``loop.add_reader`` keeps the pump on the asyncio event loop,
    so cancellation removes the reader and closes the fd in O(1).
    """
    bound = log.bind(connector=connector_name)
    loop = asyncio.get_event_loop()
    os.set_blocking(read_fd, False)
    pending = b""  # immutable so split returns fresh slices in one O(n) pass
    eof = asyncio.Event()

    def on_readable() -> None:
        nonlocal pending
        try:
            chunk = os.read(read_fd, 4096)
        except BlockingIOError:
            return
        except OSError:
            eof.set()
            return
        if not chunk:
            eof.set()
            return
        parts = (pending + chunk).split(b"\n")
        pending = parts.pop()
        for line in parts:
            if line:
                bound.info(
                    "connector.stderr",
                    line=_redact_secrets(line.decode("utf-8", errors="replace")),
                )

    loop.add_reader(read_fd, on_readable)
    try:
        await eof.wait()
    finally:
        loop.remove_reader(read_fd)
        if pending:
            bound.info(
                "connector.stderr",
                line=_redact_secrets(pending.decode("utf-8", errors="replace")),
            )
        with contextlib.suppress(OSError):
            os.close(read_fd)


AiosNotificationHandler = "Callable[[str, dict[str, Any] | None], Awaitable[None]]"


async def _splitter_task(
    upstream: MemoryObjectReceiveStream[SessionMessage | Exception],
    downstream: MemoryObjectSendStream[SessionMessage | Exception],
    on_aios_notification: Callable[[str, dict[str, Any] | None], Awaitable[None]],
    connector_name: str,
    closed_event: asyncio.Event,
) -> None:
    """Forward upstream messages to ``downstream``, intercepting aios notifications.

    Runs as a sibling of ``stdio_client``'s I/O tasks inside a shared
    task group.  Anything matching ``notifications/aios/<...>`` is
    handed to ``on_aios_notification`` and *not* forwarded — it would
    otherwise fail :class:`mcp.types.ServerNotification` validation
    inside :class:`ClientSession`'s receive loop and be dropped with a
    warning.  Everything else (responses, standard notifications,
    parse errors) flows through unchanged.

    Sets ``closed_event`` when ``upstream`` ends, so the supervisor
    body can wake from its idle ``await`` and respawn — without this
    signal, a crashed subprocess leaves the supervisor parked forever.
    """
    async with downstream:
        try:
            async for msg in upstream:
                if isinstance(msg, Exception):
                    await downstream.send(msg)
                    continue
                root = msg.message.root
                if isinstance(root, JSONRPCNotification) and root.method.startswith(
                    AIOS_NOTIFICATION_PREFIX
                ):
                    try:
                        await on_aios_notification(root.method, root.params)
                    except Exception:
                        log.exception(
                            "connector.aios_notification_handler_failed",
                            connector=connector_name,
                            method=root.method,
                        )
                    continue
                await downstream.send(msg)
        finally:
            closed_event.set()


@asynccontextmanager
async def open_connector_session(
    spec: ConnectorSpec,
    *,
    on_aios_notification: Callable[[str, dict[str, Any] | None], Awaitable[None]],
    message_handler: MessageHandlerFnT | None = None,
) -> AsyncIterator[tuple[ClientSession, InitializeResult, asyncio.Event]]:
    """Spawn the connector subprocess and yield an initialized MCP session.

    Yields ``(session, init_result, closed_event)``.  ``closed_event``
    fires when the subprocess closes its stdout (a clean exit *or* a
    crash both read as EOF on the splitter's upstream).  Long-lived
    callers should park on ``asyncio.wait`` between this event and
    their own shutdown signal so a dead subprocess wakes them.

    ``on_aios_notification`` is called once for each
    ``notifications/aios/<...>`` notification the subprocess emits, in
    receive order.  The handler must be quick (anything heavy should
    spawn its own task); blocking this coroutine pauses standard
    message delivery as well, since both consumers share the splitter
    task.

    On exit (normal or exception), the SDK runs the spec'd shutdown:
    close stdin, wait, SIGTERM, SIGKILL.
    """
    if spec.cwd is not None:
        spec.cwd.mkdir(parents=True, exist_ok=True)
    server_params = StdioServerParameters(
        command=spec.command,
        args=list(spec.args),
        env=dict(spec.env) if spec.env is not None else None,
        cwd=spec.cwd,
    )
    # OS pipe so the SDK can hand a real fd to ``subprocess.Popen``.
    # The write-end is the child's stderr; the read-end feeds the
    # ``_pump_stderr`` task that turns lines into structlog events.
    stderr_r, stderr_w = os.pipe()
    errfile = os.fdopen(stderr_w, "w", buffering=1, encoding="utf-8", errors="replace")
    async with AsyncExitStack() as stack:
        # Register both pipe ends with the stack BEFORE the spawn so a
        # ``stdio_client`` failure (missing binary, bad cwd, ENOMEM)
        # doesn't leak the read-end.  The pump task takes ownership of
        # ``stderr_r`` once it starts and closes it in its own finally;
        # the duplicate ``os.close`` here is suppressed by ``EBADF``.
        stack.callback(errfile.close)
        stack.callback(_safe_close_fd, stderr_r)

        upstream_read, write_stream = await stack.enter_async_context(
            stdio_client(server_params, errlog=cast(TextIO, errfile))
        )
        # Splitter pair: upstream → splitter → downstream → ClientSession.
        # buffer=0 mirrors stdio_client's own pair so backpressure is
        # transparent end-to-end.
        downstream_send: MemoryObjectSendStream[SessionMessage | Exception]
        downstream_recv: MemoryObjectReceiveStream[SessionMessage | Exception]
        downstream_send, downstream_recv = anyio.create_memory_object_stream(0)
        stack.push_async_callback(downstream_recv.aclose)

        tg = await stack.enter_async_context(anyio.create_task_group())
        closed_event = asyncio.Event()
        tg.start_soon(_pump_stderr, spec.name, stderr_r)
        tg.start_soon(
            _splitter_task,
            upstream_read,
            downstream_send,
            on_aios_notification,
            spec.name,
            closed_event,
        )

        session = await stack.enter_async_context(
            ClientSession(downstream_recv, write_stream, message_handler=message_handler)
        )
        # Bounded handshake: a connector that opens stdio but never
        # answers ``initialize`` would otherwise pin the supervisor in
        # ``starting`` forever — no exception means no respawn and no
        # circuit-breaker progress.  The bound mirrors the per-call
        # ceiling in :class:`~aios.harness.connector_supervisor`.
        init_result = await asyncio.wait_for(session.initialize(), timeout=_INIT_TIMEOUT_S)
        yield session, init_result, closed_event
        tg.cancel_scope.cancel()
