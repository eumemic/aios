"""Connector subprocess supervisor.

Worker-scoped registry that owns the persistent stdio MCP sessions for
every entry in ``settings.connectors_enabled``.  Holds three things:

* A long-lived asyncio task per connector that opens the subprocess,
  runs ``initialize()``, validates the ``experimental.aios/connector``
  capability, and stays parked on a shutdown event so the
  :class:`~mcp.client.session.ClientSession` (and the underlying
  subprocess) stay alive until the worker exits.
* In-memory state per connector: the live session handle, the most
  recent account snapshot received via ``notifications/aios/accounts``,
  status (running / restarting / circuit_open), and the running backoff.
* Two outbound entry points: :meth:`get_session` for code that needs to
  build its own request (used by the harness's outbound MCP dispatch)
  and :meth:`dispatch_call` for the procrastinate ``connector_call``
  task that backs the API's ``POST /v1/connectors/:name/call``.

Restart semantics: on subprocess crash, sleep ``backoff`` seconds (5s
initial, doubles each consecutive failure, capped at 5 min) then
re-spawn.  After ``_CIRCUIT_THRESHOLD`` failures within
``_CIRCUIT_WINDOW``, the loop stops respawning and the connector
reports ``circuit_open`` until the worker is restarted — operator can
fix the connector or its config and restart the worker.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Literal

from mcp.client.session import ClientSession
from mcp.types import InitializeResult

from aios.config import Settings
from aios.logging import get_logger
from aios.mcp.stdio_transport import ConnectorSpec, open_connector_session

log = get_logger("aios.harness.connector_supervisor")

_AIOS_EXPERIMENTAL_KEY = "aios/connector"
_BACKOFF_INITIAL_S = 5.0
_BACKOFF_CAP_S = 300.0
_CIRCUIT_THRESHOLD = 10
_CIRCUIT_WINDOW_S = 3600.0
_DISPATCH_TIMEOUT_S = 120.0


class ConnectorNotEnabled(Exception):
    """Raised when a request targets a connector not in ``connectors_enabled``."""


class ConnectorNotReady(Exception):
    """Raised when the connector subprocess is down or hasn't finished init."""


class CircuitOpen(Exception):
    """Raised when too many consecutive failures suspended the supervisor loop."""


ConnectorStatus = Literal["starting", "running", "restarting", "circuit_open"]


@dataclass
class ConnectorState:
    """Live state for one connector subprocess.

    ``status`` is what GET ``/v1/connectors`` reports back.
    ``accounts`` is a list of account dicts (shape opaque to aios; the
    connector defines them) replaced wholesale on every
    ``notifications/aios/accounts``.
    """

    name: str
    spec: ConnectorSpec
    status: ConnectorStatus = "starting"
    instructions: str | None = None
    accounts: list[dict[str, Any]] = field(default_factory=list)
    last_error: str | None = None
    failures: deque[float] = field(default_factory=deque)
    backoff: float = _BACKOFF_INITIAL_S
    session: ClientSession | None = None
    init_result: InitializeResult | None = None
    # ``session is not None`` is the data form of "ready"; ``ready``
    # is the awaitable handle for first-init wait.
    ready: asyncio.Event = field(default_factory=asyncio.Event)

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "instructions": self.instructions,
            "accounts": list(self.accounts),
            "last_error": self.last_error,
        }


def resolve_connector_specs(settings: Settings) -> list[ConnectorSpec]:
    """Resolve ``connectors_enabled`` against the ``aios.connectors`` entry-point group.

    Each entry point's ``load()`` must return a callable that accepts
    ``(connector_name, settings)`` and returns a :class:`ConnectorSpec`.
    Unknown names raise — the operator listed something that isn't
    installed, which should fail loudly at boot rather than silently
    skip.
    """
    available = {ep.name: ep for ep in entry_points(group="aios.connectors")}
    specs: list[ConnectorSpec] = []
    for name in settings.connectors_enabled:
        ep = available.get(name)
        if ep is None:
            raise RuntimeError(
                f"connector {name!r} listed in connectors_enabled but no "
                f"aios.connectors entry point with that name is installed"
            )
        factory = ep.load()
        spec = factory(name, settings)
        if not isinstance(spec, ConnectorSpec):
            raise RuntimeError(
                f"entry point aios.connectors:{name} returned {type(spec).__name__!r}, "
                f"expected ConnectorSpec"
            )
        specs.append(spec)
    return specs


class ConnectorSubprocessRegistry:
    """Per-worker registry of long-lived connector subprocess sessions.

    Constructed once in :func:`aios.harness.worker.worker_main` and
    stashed on :mod:`aios.harness.runtime` so procrastinate tasks
    (registered at import time) can reach it.

    Thread/loop model: single asyncio event loop, one task per
    connector, no shared mutable state across loops.
    """

    def __init__(self, specs: list[ConnectorSpec]) -> None:
        self._states: dict[str, ConnectorState] = {
            s.name: ConnectorState(name=s.name, spec=s) for s in specs
        }
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._shutdown = asyncio.Event()

    @property
    def names(self) -> list[str]:
        return list(self._states.keys())

    async def start(self) -> None:
        """Spawn one supervisor task per connector. Returns immediately."""
        for name in self._states:
            self._tasks[name] = asyncio.create_task(
                self._supervisor_loop(name),
                name=f"connector_supervisor:{name}",
            )

    async def shutdown(self) -> None:
        """Stop supervisor loops and tear down subprocesses.

        Sets the shutdown flag (each loop returns at its next park)
        then cancels in case a loop is mid-restart-sleep.  Waits for
        every task to finish so the worker's ``finally`` block can
        rely on a clean exit.
        """
        self._shutdown.set()
        for task in self._tasks.values():
            task.cancel()
        results = await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        for name, result in zip(self._tasks.keys(), results, strict=True):
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                log.warning(
                    "connector_supervisor.task_failed",
                    connector=name,
                    error=f"{type(result).__name__}: {result}",
                )
        self._tasks.clear()

    def state(self, name: str) -> ConnectorState | None:
        return self._states.get(name)

    def snapshot_all(self) -> list[dict[str, Any]]:
        return [s.snapshot() for s in self._states.values()]

    async def get_session(self, name: str) -> ClientSession:
        """Return the live session for ``name``, blocking until first init.

        Raises :class:`ConnectorNotEnabled` if the name isn't in the
        configured set or :class:`CircuitOpen` if the supervisor loop
        gave up.  Otherwise blocks indefinitely on first init — wrap
        the call in :func:`asyncio.wait_for` if you need a deadline
        (the dispatch path here does so via :data:`_DISPATCH_TIMEOUT_S`).
        """
        state = self._states.get(name)
        if state is None:
            raise ConnectorNotEnabled(name)
        if state.status == "circuit_open":
            raise CircuitOpen(name)
        await state.ready.wait()
        if state.session is None:
            raise ConnectorNotReady(name)
        return state.session

    async def dispatch_call(
        self,
        name: str,
        tool: str,
        arguments: dict[str, Any],
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke a tool on the named connector.

        Returns the raw MCP-call response shape: ``{"content": str}`` on
        success, ``{"error": str}`` on tool error or transport failure.
        Mirrors :func:`aios.mcp.client.call_mcp_tool` so call sites can
        union the two without conditional shaping.

        Bounds first-init wait by :data:`_DISPATCH_TIMEOUT_S` so a
        connector stuck in ``starting`` doesn't pin the worker
        indefinitely; downstream the call itself shares the same bound.
        """
        try:
            session = await asyncio.wait_for(self.get_session(name), timeout=_DISPATCH_TIMEOUT_S)
        except ConnectorNotEnabled:
            return {"error": f"connector {name!r} not enabled", "code": "not_enabled"}
        except CircuitOpen:
            return {
                "error": f"connector {name!r} circuit open after repeated failures",
                "code": "circuit_open",
            }
        except (ConnectorNotReady, TimeoutError):
            return {"error": f"connector {name!r} not ready", "code": "not_ready"}

        try:
            result = await asyncio.wait_for(
                session.call_tool(tool, arguments, meta=meta),
                timeout=_DISPATCH_TIMEOUT_S,
            )
        except Exception as err:
            log.warning(
                "connector.call_failed",
                connector=name,
                tool=tool,
                exc_info=True,
            )
            return {
                "error": f"connector transport error: {type(err).__name__}: {err}",
                "code": "transport_error",
            }

        from aios.mcp.client import shape_call_result

        return shape_call_result(result)

    # ── notifications ─────────────────────────────────────────────────

    async def _on_aios_notification(
        self, name: str, method: str, params: dict[str, Any] | None
    ) -> None:
        """Route a ``notifications/aios/<...>`` payload from connector ``name``."""
        state = self._states[name]
        if method == "notifications/aios/accounts":
            accounts = (params or {}).get("accounts") or []
            if isinstance(accounts, list):
                state.accounts = list(accounts)
                log.info(
                    "connector.accounts_updated",
                    connector=name,
                    count=len(state.accounts),
                )
            else:
                # Surface contract violations on ``aios connector list`` so
                # the operator notices without having to grep logs.
                state.last_error = "malformed accounts payload"
                log.warning(
                    "connector.accounts_malformed",
                    connector=name,
                    payload_type=type(accounts).__name__,
                )
        elif method == "notifications/aios/inbound":
            # No-op until inbound dispatch lands (lookup connection,
            # append event, ack via aios_inbound_ack tool call).
            log.info(
                "connector.inbound_stub",
                connector=name,
                params=params,
            )
        else:
            log.info("connector.unknown_aios_notification", connector=name, method=method)

    # ── supervisor loop ───────────────────────────────────────────────

    async def _supervisor_loop(self, name: str) -> None:
        """Open subprocess, park on shutdown, restart on crash with backoff.

        Returns when :meth:`shutdown` flips the shutdown flag *or* the
        circuit opens.  Either way the corresponding :class:`ConnectorState`
        reflects terminal status before this coroutine exits, so a GET
        call right after sees the right thing.
        """
        state = self._states[name]
        spec = state.spec

        async def handler(method: str, params: dict[str, Any] | None) -> None:
            await self._on_aios_notification(name, method, params)

        while not self._shutdown.is_set():
            state.status = "starting"
            state.last_error = None
            try:
                async with open_connector_session(spec, on_aios_notification=handler) as (
                    session,
                    init_result,
                    closed_event,
                ):
                    self._validate_capability(name, init_result)
                    state.session = session
                    state.init_result = init_result
                    state.instructions = init_result.instructions
                    state.status = "running"
                    state.backoff = _BACKOFF_INITIAL_S
                    state.ready.set()
                    log.info(
                        "connector.running",
                        connector=name,
                        server_name=init_result.serverInfo.name,
                    )
                    # Park until shutdown OR the subprocess closes its
                    # stdout (closed_event fires on EOF).  Racing the
                    # two is what lets a dead subprocess wake the loop
                    # for respawn — anyio's cancel-propagation alone
                    # leaves this body parked.
                    shutdown_task = asyncio.create_task(self._shutdown.wait())
                    closed_task = asyncio.create_task(closed_event.wait())
                    try:
                        done, _ = await asyncio.wait(
                            {shutdown_task, closed_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        for t in (shutdown_task, closed_task):
                            if not t.done():
                                t.cancel()
                    if shutdown_task in done:
                        return
                    # Subprocess closed.  Fall through to respawn.
                    state.last_error = "subprocess closed"
                    log.warning("connector.crashed", connector=name, error=state.last_error)
            except asyncio.CancelledError:
                if self._shutdown.is_set():
                    log.info("connector.shutdown", connector=name)
                    return
                # Cancellation that wasn't initiated by us — almost always
                # the subprocess closed its end and anyio's task group
                # propagated cancellation through ``open_connector_session``.
                # Fall through to the crash branch and respawn.
                state.last_error = "subprocess closed"
                log.warning("connector.crashed", connector=name, error=state.last_error)
            except Exception as exc:
                state.last_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "connector.crashed",
                    connector=name,
                    error=state.last_error,
                    exc_info=True,
                )
            finally:
                state.session = None
                state.init_result = None
                state.ready.clear()

            # Crash bookkeeping + circuit breaker.
            now = time.monotonic()
            state.failures.append(now)
            cutoff = now - _CIRCUIT_WINDOW_S
            while state.failures and state.failures[0] < cutoff:
                state.failures.popleft()
            if len(state.failures) >= _CIRCUIT_THRESHOLD:
                state.status = "circuit_open"
                log.error(
                    "connector.circuit_open",
                    connector=name,
                    failures=len(state.failures),
                    window_s=_CIRCUIT_WINDOW_S,
                )
                return

            state.status = "restarting"
            log.info(
                "connector.restart_scheduled",
                connector=name,
                delay_s=state.backoff,
            )
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=state.backoff)
                # Shutdown won the race.
                return
            except TimeoutError:
                pass
            state.backoff = min(state.backoff * 2, _BACKOFF_CAP_S)

    def _validate_capability(self, name: str, init_result: InitializeResult) -> None:
        """Hard-fail if the connector didn't declare ``experimental.aios/connector``."""
        experimental = init_result.capabilities.experimental or {}
        if _AIOS_EXPERIMENTAL_KEY not in experimental:
            raise RuntimeError(
                f"connector {name!r} did not declare "
                f"experimental.{_AIOS_EXPERIMENTAL_KEY!r} capability"
            )
