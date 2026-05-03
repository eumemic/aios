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
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from importlib.metadata import entry_points
from typing import Any, Literal

from mcp.client.session import ClientSession
from mcp.types import InitializeResult

from aios.config import Settings
from aios.db import queries
from aios.errors import ConflictError, NotFoundError
from aios.harness import runtime
from aios.harness.wake import defer_wake
from aios.logging import get_logger
from aios.mcp.client import shape_call_result
from aios.mcp.stdio_transport import ConnectorSpec, open_connector_session
from aios.services import sessions as sessions_service

log = get_logger("aios.harness.connector_supervisor")

_AIOS_EXPERIMENTAL_KEY = "aios/connector"
_BACKOFF_INITIAL_S = 5.0
_BACKOFF_CAP_S = 300.0
_CIRCUIT_THRESHOLD = 10
_CIRCUIT_WINDOW_S = 3600.0
# Per-call ceiling for ``dispatch_call`` (and the wait for first init).
# Matches the API router's ``_RESULT_TIMEOUT_S`` so a slow tool surfaces
# as a 408 to the operator at the same moment the worker would give up,
# rather than the worker continuing to execute past the API's deadline
# and dropping its NOTIFY into a closed channel.  Per plan decision #4.
_DISPATCH_TIMEOUT_S = 60.0


class ConnectorNotEnabled(Exception):
    """Raised when a request targets a connector not in ``connectors_enabled``."""


class ConnectorNotReady(Exception):
    """Raised when the connector subprocess is down or hasn't finished init."""


class CircuitOpen(Exception):
    """Raised when too many consecutive failures suspended the supervisor loop."""


class _DedupRollback(Exception):
    """Internal-only: raised inside the inbound transaction to undo a duplicate.

    Caught by the same method that raised it.  Existing as a typed
    exception keeps the rollback intent legible — ``raise Exception``
    in the same code path would invite a maintainer to add error
    handling that masks the deliberate rollback.
    """


ConnectorStatus = Literal["starting", "running", "restarting", "circuit_open"]


@dataclass
class ConnectorState:
    """Live state for one connector subprocess.

    ``status`` is what GET ``/v1/connectors`` reports back.
    ``accounts`` is a list of account dicts (shape opaque to aios; the
    connector defines them) replaced wholesale on every
    ``notifications/aios/accounts``.  ``drops`` is a per-reason
    counter surfaced as ``recent_drops`` in :meth:`snapshot` so
    operators see at-a-glance from ``aios connector list`` whether
    inbound traffic is landing — plan §15.
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
    drops: Counter[str] = field(default_factory=Counter)
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
            "recent_drops": dict(self.drops),
        }


def resolve_connector_specs(settings: Settings) -> list[ConnectorSpec]:
    """Resolve ``connectors_enabled`` against the ``aios.connectors`` entry-point group.

    Each entry point's ``load()`` must return a callable that accepts
    ``(connector_name, settings)`` and returns a :class:`ConnectorSpec`.
    Unknown names raise — the operator listed something that isn't
    installed, which should fail loudly at boot rather than silently
    skip.

    Defaults the spec's cwd to ``settings.connectors_dir / name`` when
    the factory leaves it unset, so connectors land their state files
    (PR3's spool, signal-cli's data dir) under the operator-controlled
    root by default per plan decision #11.  Factories that need a
    different layout can override by setting ``cwd`` themselves.
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
        if spec.cwd is None:
            spec = replace(spec, cwd=settings.connectors_dir / name)
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

    def __init__(self, specs: list[ConnectorSpec], *, settings: Settings) -> None:
        self._settings = settings
        self._states: dict[str, ConnectorState] = {
            s.name: ConnectorState(name=s.name, spec=s) for s in specs
        }
        self._tasks: dict[str, asyncio.Task[None]] = {}
        # Inbound handler tasks: spawned per-notification so the splitter
        # task in :func:`open_connector_session` doesn't pause receive
        # while a single inbound walks the DB + sends the ack.  Tracked
        # so :meth:`shutdown` can wait them out cleanly.
        self._inbound_tasks: set[asyncio.Task[None]] = set()
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

        In-flight inbound handler tasks are awaited (not cancelled) so
        an inbound that's mid-transaction commits cleanly — losing it
        would force the connector to replay the same message on next
        boot, which is correct but noisy.
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
        if self._inbound_tasks:
            await asyncio.gather(*self._inbound_tasks, return_exceptions=True)
            self._inbound_tasks.clear()

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

        return shape_call_result(result)

    # ── notifications ─────────────────────────────────────────────────

    async def _on_aios_notification(
        self, name: str, method: str, params: dict[str, Any] | None
    ) -> None:
        """Route a ``notifications/aios/<...>`` payload from connector ``name``."""
        state = self._states[name]
        if method == "notifications/aios/accounts":
            raw_accounts = (params or {}).get("accounts")
            if isinstance(raw_accounts, list):
                state.accounts = list(raw_accounts)
                log.info(
                    "connector.accounts_updated",
                    connector=name,
                    count=len(state.accounts),
                )
            else:
                # Surface contract violations on ``aios connector list`` so
                # the operator notices without having to grep logs.  An
                # empty list is fine ("zero accounts available"); any
                # other shape (null, missing key, wrong type) is a bug
                # in the connector, not a recovery signal.
                state.last_error = "malformed accounts payload"
                log.warning(
                    "connector.accounts_malformed",
                    connector=name,
                    payload_type=type(raw_accounts).__name__,
                )
        elif method == "notifications/aios/inbound":
            # Spawn a separate task: the splitter holds the SDK's receive
            # stream while this handler runs, so doing DB work inline
            # would pause every other message (init responses, account
            # snapshots) for the duration.
            task = asyncio.create_task(
                self._handle_inbound(name, params or {}),
                name=f"connector_inbound:{name}",
            )
            self._inbound_tasks.add(task)
            task.add_done_callback(self._inbound_tasks.discard)
        else:
            log.info("connector.unknown_aios_notification", connector=name, method=method)

    # ── inbound dispatch ──────────────────────────────────────────────

    async def _handle_inbound(self, name: str, params: dict[str, Any]) -> None:
        """Append an inbound event, dedupe via the ledger, ack the spool.

        Implements the crash walkthrough from plan decision #16: ledger
        INSERT runs in the same transaction as ``append_event`` so a
        replayed inbound (same ULID after worker SIGKILL) hits
        ``ON CONFLICT DO NOTHING`` and the txn rolls back without a
        second event row.  Ack runs after commit, fire-and-forget — a
        failed ack just means the connector replays again on its next
        reconnect, where the ledger conflict path takes over.

        Drops are counted but never fail the splitter — a malformed
        inbound surfaces in ``recent_drops`` and operator logs without
        stalling the connector pipeline.
        """

        state = self._states[name]
        event_id = params.get("event_id")
        account = params.get("account")
        chat_id = params.get("chat_id")
        sender = params.get("sender") or {}
        content = params.get("content")
        if not (
            isinstance(event_id, str)
            and isinstance(account, str)
            and isinstance(chat_id, str)
            and isinstance(content, str)
        ):
            log.warning(
                "connector.inbound_malformed",
                connector=name,
                missing=[
                    field
                    for field, value in (
                        ("event_id", event_id),
                        ("account", account),
                        ("chat_id", chat_id),
                        ("content", content),
                    )
                    if not isinstance(value, str)
                ],
            )
            self._record_drop(state, "malformed")
            return

        pool = runtime.require_pool()

        # 1. Look up connection (fresh read — no caching, since the
        #    operator may have just attached/configured the connection).
        async with pool.acquire() as conn:
            connection = await queries.get_connection_for_account(
                conn, connector=name, account=account
            )
        if connection is None:
            await self._auto_create_or_drop(state, name=name, account=account)
            # Even if auto-create succeeds, this inbound is dropped:
            # the freshly-detached connection has no session to deliver
            # to.  Operator must explicitly attach / configure-per-chat.
            await self._send_ack(name, event_id)
            return

        # 2. Branch on routing mode.
        target_session_id: str | None = None
        if connection.session_id is not None:
            target_session_id = connection.session_id
        elif connection.session_template_id is not None:
            target_session_id = await self._resolve_per_chat_session(
                state,
                connection_id=connection.id,
                template_id=connection.session_template_id,
                connector_name=name,
                account=account,
                chat_id=chat_id,
            )
        else:
            self._record_drop(state, "detached")
            await self._send_ack(name, event_id)
            return

        if target_session_id is None:
            # Per-chat resolution already recorded the drop reason.
            await self._send_ack(name, event_id)
            return

        # 3. Append event + ledger row in the same transaction.
        appended = await self._append_with_dedup(
            connector_name=name,
            account=account,
            event_id=event_id,
            session_id=target_session_id,
            chat_id=chat_id,
            sender=sender if isinstance(sender, dict) else {},
            content=content,
            attachments=params.get("attachments"),
        )

        # 4. Outside the transaction: defer wake (idempotent) and ack.
        if appended:
            try:
                await defer_wake(pool, target_session_id, cause="inbound")
            except Exception:
                # Defer-wake failures are non-fatal — the next sweep
                # picks up the unhandled message.  Logged so operator
                # diagnoses if a connector goes silent.
                log.warning(
                    "connector.inbound_defer_wake_failed",
                    connector=name,
                    session_id=target_session_id,
                    exc_info=True,
                )
        await self._send_ack(name, event_id)

    async def _auto_create_or_drop(self, state: ConnectorState, *, name: str, account: str) -> None:
        """Insert a detached connection if ``auto_create_connections`` is on.

        Mirrors plan decision #13: missing pair drops the inbound
        regardless, but auto-create gives operators a row they can
        attach to without re-issuing the original message.  Both paths
        increment the ``no_connection`` counter so the operator sees
        the surface (auto-creating a row doesn't deliver the message
        that prompted it).
        """

        self._record_drop(state, "no_connection")
        if self._settings.connectors_auto_create.get(name, True):
            pool = runtime.require_pool()
            try:
                async with pool.acquire() as conn:
                    await queries.insert_connection(
                        conn, connector=name, account=account, metadata={}
                    )
                    log.info(
                        "connector.auto_create_connection",
                        connector=name,
                        account=account,
                    )
            except ConflictError:
                # Race: another handler created the row first; the
                # outcome is what we wanted, no error.
                pass
            except Exception:
                log.warning(
                    "connector.auto_create_failed",
                    connector=name,
                    account=account,
                    exc_info=True,
                )

    async def _resolve_per_chat_session(
        self,
        state: ConnectorState,
        *,
        connection_id: str,
        template_id: str,
        connector_name: str,
        account: str,
        chat_id: str,
    ) -> str | None:
        """Return the session id for ``(connection_id, chat_id)``, spawning if needed.

        Cheap path: existing row in ``connection_chat_sessions`` →
        return its session id.  Spawn path: read the template (drop
        on archived), create a session via :func:`services.sessions.create_session`,
        race-safely register it in ``connection_chat_sessions``.
        """

        pool = runtime.require_pool()
        async with pool.acquire() as conn:
            existing = await queries.lookup_chat_session(conn, connection_id, chat_id)
            if existing is not None:
                return existing
            template = await queries.get_session_template(conn, template_id)
        if template.archived_at is not None:
            self._record_drop(state, "archived_template")
            return None

        focal_channel = f"{connector_name}/{account}/{chat_id}"
        session = await sessions_service.create_session(
            pool,
            agent_id=template.agent_id,
            environment_id=template.environment_id,
            agent_version=template.agent_version,
            title=None,
            metadata={},
            vault_ids=template.vault_ids or None,
            focal_channel=focal_channel,
            spawned_from_connection_id=connection_id,
        )

        # Race-safe register: on conflict the existing session_id wins
        # and the just-spawned session is intentionally orphaned (plan
        # decision #8 — sessions table tracks it; operator can archive
        # later).
        async with pool.acquire() as conn:
            registered = await queries.insert_chat_session(
                conn,
                connection_id=connection_id,
                chat_id=chat_id,
                session_id=session.id,
            )
        if registered != session.id:
            log.info(
                "connector.per_chat_race_orphaned_session",
                connector=connector_name,
                lost_session_id=session.id,
                kept_session_id=registered,
            )
        return registered

    async def _append_with_dedup(
        self,
        *,
        connector_name: str,
        account: str,
        event_id: str,
        session_id: str,
        chat_id: str,
        sender: dict[str, Any],
        content: str,
        attachments: Any,
    ) -> bool:
        """Append a user-message event AND record the dedup-ledger row.

        Both writes run in one transaction.  Returns ``True`` if the
        event was appended, ``False`` if the ledger detected a
        duplicate (already-processed inbound replayed by the
        connector).  In the duplicate case the txn rolls back — no
        second event row, no second seq increment, but the caller
        still sends the ack so the connector clears its spool.
        """

        pool = runtime.require_pool()
        channel = f"{connector_name}/{account}/{chat_id}"
        sender_name = sender.get("display_name")
        metadata: dict[str, Any] = {"channel": channel}
        if isinstance(sender_name, str):
            metadata["sender"] = sender_name
        if isinstance(attachments, list) and attachments:
            metadata["attachments"] = attachments
        data: dict[str, Any] = {
            "role": "user",
            "content": content,
            "metadata": metadata,
        }

        try:
            async with pool.acquire() as conn, conn.transaction():
                event = await queries.append_event(
                    conn,
                    session_id=session_id,
                    kind="message",
                    data=data,
                    orig_channel=channel,
                )
                inserted = await queries.try_record_inbound_ack(
                    conn,
                    connector=connector_name,
                    account=account,
                    event_id=event_id,
                    appended_seq=event.seq,
                )
                if not inserted:
                    # Duplicate inbound — undo this txn entirely.  The
                    # ack still fires from the caller so the connector's
                    # spool clears.
                    raise _DedupRollback()
                # Same-txn pending flip (mirrors append_user_message)
                # so polling orchestrators see the session leave idle.
                await conn.execute(
                    "UPDATE sessions SET status = 'pending', updated_at = now() "
                    "WHERE id = $1 AND status = 'idle'",
                    session_id,
                )
        except _DedupRollback:
            log.info(
                "connector.inbound_duplicate",
                connector=connector_name,
                event_id=event_id,
            )
            return False
        except NotFoundError:
            # The session vanished between resolution and append.  Rare
            # but possible if an operator just archived a per_chat
            # session.  Drop with counter so it surfaces.
            state = self._states[connector_name]
            self._record_drop(state, "session_missing")
            return False
        return True

    async def _send_ack(self, name: str, event_id: str) -> None:
        """Fire-and-forget ``aios_inbound_ack`` to clear the connector's spool.

        Failures here aren't fatal: the connector replays on its next
        reconnect, where the ledger conflict path catches the
        duplicate.  We log but don't retry — retries would risk an
        infinite loop if the connector is wedged.
        """
        result = await self.dispatch_call(name, "aios_inbound_ack", {"event_id": event_id})
        if "error" in result:
            log.info(
                "connector.inbound_ack_failed",
                connector=name,
                event_id=event_id,
                error=result["error"],
            )

    def _record_drop(self, state: ConnectorState, reason: str) -> None:
        """Bump the per-reason drop counter, log, and let snapshots reflect it."""
        state.drops[reason] += 1
        log.info(
            "connector.inbound_dropped",
            connector=state.name,
            reason=reason,
            count=state.drops[reason],
        )

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
