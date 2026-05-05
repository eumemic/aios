"""Connector subprocess supervisor.

Worker-scoped registry that owns the persistent stdio MCP sessions for
every connector instance in ``settings.connectors_enabled``.  Holds
three things:

* A long-lived asyncio task per ``(connector, instance)`` pair that
  opens the subprocess, runs ``initialize()``, validates the
  ``experimental.aios/connector`` capability, and stays parked on a
  shutdown event so the :class:`~mcp.client.session.ClientSession`
  (and the underlying subprocess) stay alive until the worker exits.
* In-memory state per ``(connector, instance)``: the live session
  handle, the most recent account snapshot received via
  ``notifications/aios/accounts``, status (running / restarting /
  circuit_open), and the running backoff.
* An ``account → instance`` routing map maintained from the per-instance
  account snapshots.  Outbound calls referencing a ``(connector,
  account)`` pair dispatch to the owning instance via this map.
* Two outbound entry points: :meth:`get_session` for code that needs to
  build its own request (used by the harness's outbound MCP dispatch)
  and :meth:`dispatch_call` for the procrastinate ``connector_call``
  task that backs the API's
  ``POST /v1/connectors/:connector/:instance/call``.

Multi-instance: ``connectors_enabled`` accepts ``<connector>[:<instance>]``
entries.  Default instance = connector name when ``:`` is omitted, so
single-instance setups (``connectors_enabled=signal``) continue to work
without per-instance scoping.  Per-instance:

* cwd is ``<connectors_dir>/<connector>/`` for default instances,
  ``<connectors_dir>/<connector>/<instance>/`` for non-default — keeps
  PR3's spool location stable for single-instance deployments.
* env is the parent's ``os.environ`` for default instances; for
  non-default, ``AIOS_<CONN>_<INST>_*`` vars are re-exported as
  ``AIOS_<CONN>_*`` so connector subprocess code stays instance-naive.

Restart semantics: on subprocess crash, sleep ``backoff`` seconds (5s
initial, doubles each consecutive failure, capped at 5 min) then
re-spawn.  After ``_CIRCUIT_THRESHOLD`` failures within
``_CIRCUIT_WINDOW``, the loop stops respawning and the connector
reports ``circuit_open`` until the worker is restarted — operator can
fix the connector or its config and restart the worker.

Architectural constraint
------------------------

This module currently lives in the ``aios worker`` process and bakes in
the single-worker invariant: the worker's ``pg_try_advisory_lock``
refuses a second concurrent ``aios worker`` so the connector supervisor
has exclusive ownership of stdio handles, restart bookkeeping, and the
in-memory account map.  That's fine today (one worker comfortably
multiplexes hundreds of in-flight session-step coroutines on the
asyncio loop), but it's a real architectural ceiling for horizontal
scaling — any future need for multiple workers would force connector
supervision into a separate ``aios connectors`` process.

To preserve that option without committing to it, this module's public
surface is intentionally narrow:

* :class:`ConnectorSubprocessRegistry` — entry-point lookup, lifecycle
  bookkeeping, and the ``account → instance`` routing map.  External
  consumers should reach for ``state``, ``states_for_connector``,
  ``snapshot_all``, ``get_session``, ``dispatch_call``, and
  ``dispatch_call_for_account`` only.

* The procrastinate ``connector_call`` task in
  :mod:`aios.harness.connector_tasks` is the canonical IPC primitive
  for cross-process calls into the supervisor.  When the API process
  needs supervisor data it goes through that task, not by importing
  the registry.

* The one in-process call site that bypasses the procrastinate IPC is
  the outbound MCP dispatcher in :mod:`aios.harness.tool_dispatch`,
  justified there as the per-tool-call hot path where ~50-200ms of
  procrastinate round-trip would dominate the latency budget.

If new functionality on the supervisor is tempted to add another
in-process consumer that imports the registry directly, prefer
defining a procrastinate task in ``connector_tasks.py`` and routing
new callers through it.  Same end behaviour today; one less call site
to migrate the day the supervisor splits out.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from importlib.metadata import entry_points
from typing import Any, Literal

from aios_connector import ConnectorSpec
from mcp.client.session import ClientSession
from mcp.types import InitializeResult

from aios.config import ConnectorInstance, Settings, parse_connector_entry
from aios.db import queries
from aios.errors import NotFoundError
from aios.harness import runtime
from aios.harness.attachment_staging import (
    AttachmentStagingError,
    stage_inbound_attachments,
)
from aios.harness.wake import defer_wake
from aios.logging import get_logger
from aios.mcp.client import MAX_TOOLS_PER_SERVER, shape_call_result
from aios.mcp.stdio_transport import open_connector_session
from aios.models.sessions import MAX_USER_MESSAGE_CHARS
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
    """Raised inside the inbound transaction to trigger rollback on duplicate."""


ConnectorStatus = Literal["starting", "running", "restarting", "circuit_open"]
DropReason = Literal[
    "no_connection",
    "detached",
    "archived_template",
    "session_missing",
    "malformed",
    "payload_too_large",
    "account_drift",
    "attachment_staging_failed",
]


def instance_label(connector: str, instance: str) -> str:
    """Human-readable identifier for ``(connector, instance)``.

    Collapses ``signal:signal`` to ``signal`` for default-instance
    setups so single-instance deployments don't get a confusing
    redundant suffix in logs, snapshots, and error envelopes.
    """
    return connector if connector == instance else f"{connector}:{instance}"


@dataclass
class ConnectorState:
    """Live state for one connector subprocess instance.

    ``status`` is what GET ``/v1/connectors`` reports back.
    ``accounts`` is a list of account dicts (shape opaque to aios; the
    connector defines them) replaced wholesale on every
    ``notifications/aios/accounts``.  ``drops`` is a per-reason
    counter surfaced as ``recent_drops`` in :meth:`snapshot` so
    operators see at-a-glance from ``aios connector list`` whether
    inbound traffic is landing — plan §15.

    ``connector`` and ``instance`` together identify the subprocess.
    For single-instance deployments they're equal (default-instance
    convention); for multi-instance setups the operator picks the
    instance name explicitly.
    """

    connector: str
    instance: str
    spec: ConnectorSpec
    status: ConnectorStatus = "starting"
    instructions: str | None = None
    accounts: list[dict[str, Any]] = field(default_factory=list)
    last_error: str | None = None
    failures: deque[float] = field(default_factory=deque)
    backoff: float = _BACKOFF_INITIAL_S
    session: ClientSession | None = None
    init_result: InitializeResult | None = None
    drops: Counter[DropReason] = field(default_factory=Counter)
    # ``session is not None`` is the data form of "ready"; ``ready``
    # is the awaitable handle for first-init wait.
    ready: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def key(self) -> tuple[str, str]:
        """Compound registry key — used as the dict key in ``_states``."""
        return (self.connector, self.instance)

    @property
    def label(self) -> str:
        return instance_label(self.connector, self.instance)

    def snapshot(self) -> dict[str, Any]:
        """Build the admin-facing dict for ``connector_status`` / GET ``/v1/connectors``.

        ``instructions`` is intentionally absent.  The MCP server's
        ``InitializeResult.instructions`` is consumed for model context
        via :func:`aios.harness.loop.discover_session_mcp_tools` (which
        re-fetches it directly), not via this snapshot.  Including it
        here previously broke ``connector_status``: the task pg_notifies
        the JSON-encoded snapshot, and Postgres NOTIFY caps payloads at
        8000 bytes — a signal connector serving N phones with their
        contacts and groups easily exceeded that.
        """
        return {
            "connector": self.connector,
            "instance": self.instance,
            "status": self.status,
            "accounts": list(self.accounts),
            "last_error": self.last_error,
            "recent_drops": dict(self.drops),
        }


def resolve_connector_specs(settings: Settings) -> list[tuple[ConnectorInstance, ConnectorSpec]]:
    """Resolve ``connectors_enabled`` into ``(instance, spec)`` pairs.

    Each enabled connector launches as
    ``python -m aios_connector <connector_name>``; the SDK's runner
    loads the ``aios.connectors`` entry point and instantiates the
    connector.  Unknown names raise so a typo in ``connectors_enabled``
    fails loudly at boot.

    :func:`_apply_instance_overlay` stamps a per-instance cwd
    (``settings.connectors_dir / connector`` for default instances,
    ``.../instance`` for non-default) and ``mkdir -p``s it so the first
    spawn doesn't ``FileNotFoundError``.
    """
    available = {ep.name for ep in entry_points(group="aios.connectors")}
    out: list[tuple[ConnectorInstance, ConnectorSpec]] = []
    for raw in settings.connectors_enabled:
        ci = parse_connector_entry(raw)
        if ci.connector not in available:
            raise RuntimeError(
                f"connector {ci.connector!r} listed in connectors_enabled (entry "
                f"{raw!r}) but no aios.connectors entry point with that name is "
                f"installed"
            )
        spec = ConnectorSpec(
            name=ci.connector,
            command=sys.executable,
            args=["-m", "aios_connector", ci.connector],
        )
        spec = _apply_instance_overlay(spec, ci, settings)
        out.append((ci, spec))
    return out


def _apply_instance_overlay(
    spec: ConnectorSpec, ci: ConnectorInstance, settings: Settings
) -> ConnectorSpec:
    """Layer per-instance cwd + env on top of a factory-returned spec.

    cwd: default-instance keeps the PR3 single-segment path
    (``<connectors_dir>/<connector>/``); non-default gets a per-instance
    subdir.  Either way an explicit ``spec.cwd`` from the factory wins
    (factories that hard-code their own state-file layout opt out).
    The resolved cwd is ``mkdir -p``'d so the first
    :func:`asyncio.create_subprocess_exec` doesn't ``FileNotFoundError``
    on a fresh install.

    env: always pass the worker's ``os.environ`` to the subprocess so
    pydantic-settings reads the same vars the operator set, with
    path-shaped settings (``AIOS_WORKSPACE_ROOT``) absolutized so the
    subprocess's different cwd doesn't shift the resolved location.
    For non-default instances, also re-export
    ``AIOS_<CONN_UPPER>_<INST_UPPER>_*`` as ``AIOS_<CONN_UPPER>_*`` so
    connector subprocess code stays instance-naive — it just reads
    config under the standard prefix.  Single-instance deployments use
    the unscoped vars directly with no transform.
    """
    cwd = spec.cwd
    if cwd is None:
        cwd = settings.connectors_dir / ci.connector
        if ci.instance != ci.connector:
            cwd = cwd / ci.instance
    cwd.mkdir(parents=True, exist_ok=True)
    env = _build_instance_env(ci, settings=settings, base=spec.env)
    return replace(spec, cwd=cwd, env=env)


def _build_instance_env(
    ci: ConnectorInstance, *, settings: Settings, base: dict[str, str] | None
) -> dict[str, str]:
    """Construct the env dict for a connector subprocess.

    Starts from ``os.environ`` (so the subprocess inherits operator-set
    config), layers the factory-supplied ``base`` on top (factory wins
    over inherited if both set the same key), then absolutizes the
    path-shaped settings the SDK and connector code rely on
    (``AIOS_WORKSPACE_ROOT`` today — see below), and finally — for
    non-default instances — re-exports
    ``AIOS_<CONN_UPPER>_<INST_UPPER>_<FIELD>`` as
    ``AIOS_<CONN_UPPER>_<FIELD>`` so the connector reads its config
    under the standard prefix.
    from the resulting env so each subprocess sees only its own
    credentials (a wedged or chatty connector that iterates
    ``os.environ`` won't surface sibling tokens).

    Path absolutization: the worker resolves ``settings.workspace_root``
    against its own cwd, but spawns connector subprocesses under
    ``<connectors_dir>/<connector>/[<instance>/]``.  Inheriting the
    operator's relative ``AIOS_WORKSPACE_ROOT`` (e.g. ``./workspaces``
    in ``.env``) would let the SDK's ``SandboxPath`` resolution resolve
    ``/workspace/foo.png`` against the wrong directory.  We stamp the
    worker-resolved absolute path so harness and SDK agree on the
    bind-mount root regardless of subprocess cwd.  Factory-supplied
    overrides lose to this absolutized value: a connector that sets
    ``AIOS_WORKSPACE_ROOT`` in ``spec.env`` would silently desynchronize
    from the harness's bind-mount, which is almost certainly a bug.

    The re-export overrides any unscoped value: an operator running
    ``connectors_enabled=telegram:bot1,telegram:bot2`` with both
    ``AIOS_TELEGRAM_BOT_TOKEN`` (legacy unscoped) and
    ``AIOS_TELEGRAM_BOT1_BOT_TOKEN`` should see ``bot1`` use the
    scoped value, never the unscoped fallback (which would map two
    instances to the same bot).
    """
    env: dict[str, str] = dict(os.environ)
    if base:
        env.update(base)
    # Absolutize path-shaped settings the SDK/harness share via env so
    # subprocesses don't resolve them against their own cwd.  Add new
    # path-shaped settings here when they're introduced rather than
    # relying on per-setting fixes scattered across call sites.
    env["AIOS_WORKSPACE_ROOT"] = str(settings.workspace_root.resolve())
    if ci.instance == ci.connector:
        return env
    scope_prefix = f"AIOS_{ci.connector.upper()}_{ci.instance.upper()}_"
    target_prefix = f"AIOS_{ci.connector.upper()}_"
    for key, value in list(env.items()):
        if key.startswith(scope_prefix):
            env[target_prefix + key[len(scope_prefix) :]] = value
    return env


class ConnectorSubprocessRegistry:
    """Per-worker registry of long-lived connector subprocess sessions.

    Constructed once in :func:`aios.harness.worker.worker_main` and
    stashed on :mod:`aios.harness.runtime` so procrastinate tasks
    (registered at import time) can reach it.

    Thread/loop model: single asyncio event loop, one task per
    ``(connector, instance)``, no shared mutable state across loops.
    """

    def __init__(
        self,
        specs: list[tuple[ConnectorInstance, ConnectorSpec]],
        *,
        settings: Settings,
    ) -> None:
        self._settings = settings
        self._states: dict[tuple[str, str], ConnectorState] = {
            (ci.connector, ci.instance): ConnectorState(
                connector=ci.connector, instance=ci.instance, spec=spec
            )
            for ci, spec in specs
        }
        self._tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        # Inbound handler tasks: spawned per-notification so the splitter
        # task in :func:`open_connector_session` doesn't pause receive
        # while a single inbound walks the DB + sends the ack.  Tracked
        # so :meth:`shutdown` can wait them out cleanly.
        self._inbound_tasks: set[asyncio.Task[None]] = set()
        # ``(connector, account)`` → instance.  Rebuilt per-instance on
        # every ``notifications/aios/accounts``: clear all entries
        # pointing at the reporting instance, then insert from the new
        # snapshot.  This keeps removed accounts from leaving stale
        # entries pointing at the instance that used to serve them.
        self._account_to_instance: dict[tuple[str, str], str] = {}
        self._shutdown = asyncio.Event()

    @property
    def keys(self) -> list[tuple[str, str]]:
        return list(self._states.keys())

    async def start(self) -> None:
        """Spawn one supervisor task per ``(connector, instance)`` pair."""
        for key, state in self._states.items():
            self._tasks[key] = asyncio.create_task(
                self._supervisor_loop(key),
                name=f"connector_supervisor:{state.label}",
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
        for key, result in zip(self._tasks.keys(), results, strict=True):
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                connector, instance = key
                log.warning(
                    "connector_supervisor.task_failed",
                    connector=connector,
                    instance=instance,
                    error=f"{type(result).__name__}: {result}",
                )
        self._tasks.clear()
        if self._inbound_tasks:
            await asyncio.gather(*self._inbound_tasks, return_exceptions=True)
            self._inbound_tasks.clear()

    def state(self, connector: str, instance: str) -> ConnectorState | None:
        return self._states.get((connector, instance))

    def states_for_connector(self, connector: str) -> list[ConnectorState]:
        """All ``ConnectorState``s for instances of one connector type."""
        return [s for (c, _i), s in self._states.items() if c == connector]

    def resolve_default_instance(self, connector: str) -> str | None:
        """Return the sole instance name for ``connector``, or ``None``.

        Used by the procrastinate tasks to auto-resolve the ``_`` sentinel
        that the CLI passes when the operator omits ``<instance>``.
        Returns ``None`` for both "no instances enabled" and "multiple
        instances enabled" — callers distinguish via
        :meth:`states_for_connector`.
        """
        states = self.states_for_connector(connector)
        return states[0].instance if len(states) == 1 else None

    def snapshot_all(self) -> list[dict[str, Any]]:
        return [s.snapshot() for s in self._states.values()]

    def lookup_instance_for_account(self, connector: str, account: str) -> str | None:
        """Return which instance serves ``(connector, account)`` per the routing map."""
        return self._account_to_instance.get((connector, account))

    async def list_tools(self) -> list[dict[str, Any]]:
        """Enumerate connector-subprocess tools as OpenAI-format tool dicts.

        Each ``mcp__<connector>__<tool>`` namespace matches the HTTP-MCP
        path in :func:`aios.mcp.client.discover_mcp_tools` so the
        dispatcher resolves either kind without a branch.  Multi-instance
        connectors collapse duplicates by name with first-wins ordering.
        Instances not in ``running`` are skipped; truncation cap mirrors
        the HTTP-MCP path.  ``aios_inbound_ack`` is internal harness
        machinery and never reaches the model.
        """
        running = [
            (connector, state.session)
            for (connector, _instance), state in self._states.items()
            if state.status == "running" and state.session is not None
        ]
        if not running:
            return []
        results = await asyncio.gather(*(session.list_tools() for _, session in running))
        seen: set[str] = set()
        tools: list[dict[str, Any]] = []
        for (connector, _session), result in zip(running, results, strict=True):
            if len(result.tools) > MAX_TOOLS_PER_SERVER:
                log.warning(
                    "connector.tools_truncated",
                    connector=connector,
                    total=len(result.tools),
                    limit=MAX_TOOLS_PER_SERVER,
                )
            for tool in result.tools[:MAX_TOOLS_PER_SERVER]:
                if tool.name == "aios_inbound_ack":
                    continue
                qualified = f"mcp__{connector}__{tool.name}"
                if qualified in seen:
                    continue
                seen.add(qualified)
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": qualified,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema,
                        },
                    }
                )
        return tools

    async def get_session(self, connector: str, instance: str) -> ClientSession:
        """Return the live session for ``(connector, instance)``.

        Raises :class:`ConnectorNotEnabled` if the pair isn't in the
        configured set or :class:`CircuitOpen` if the supervisor loop
        gave up.  Otherwise blocks indefinitely on first init — wrap
        the call in :func:`asyncio.wait_for` if you need a deadline
        (the dispatch path here does so via :data:`_DISPATCH_TIMEOUT_S`).
        """
        state = self._states.get((connector, instance))
        if state is None:
            raise ConnectorNotEnabled(f"{connector}:{instance}")
        if state.status == "circuit_open":
            raise CircuitOpen(state.label)
        await state.ready.wait()
        if state.session is None:
            raise ConnectorNotReady(state.label)
        return state.session

    async def dispatch_call(
        self,
        connector: str,
        instance: str,
        tool: str,
        arguments: dict[str, Any],
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke a tool on the named ``(connector, instance)``.

        Returns the raw MCP-call response shape: ``{"content": str}`` on
        success, ``{"error": str}`` on tool error or transport failure.
        Mirrors :func:`aios.mcp.client.call_mcp_tool` so call sites can
        union the two without conditional shaping.

        Bounds first-init wait by :data:`_DISPATCH_TIMEOUT_S` so an
        instance stuck in ``starting`` doesn't pin the worker
        indefinitely; downstream the call itself shares the same bound.
        """
        label = instance_label(connector, instance)
        try:
            session = await asyncio.wait_for(
                self.get_session(connector, instance), timeout=_DISPATCH_TIMEOUT_S
            )
        except ConnectorNotEnabled:
            return {"error": f"connector instance {label!r} not enabled", "code": "not_enabled"}
        except CircuitOpen:
            return {
                "error": f"connector instance {label!r} circuit open after repeated failures",
                "code": "circuit_open",
            }
        except (ConnectorNotReady, TimeoutError):
            return {"error": f"connector instance {label!r} not ready", "code": "not_ready"}

        try:
            result = await asyncio.wait_for(
                session.call_tool(tool, arguments, meta=meta),
                timeout=_DISPATCH_TIMEOUT_S,
            )
        except Exception as err:
            log.warning(
                "connector.call_failed",
                connector=connector,
                instance=instance,
                tool=tool,
                exc_info=True,
            )
            return {
                "error": f"connector transport error: {type(err).__name__}: {err}",
                "code": "transport_error",
            }

        return shape_call_result(result)

    async def dispatch_call_for_account(
        self,
        connector: str,
        account: str,
        tool: str,
        arguments: dict[str, Any],
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dispatch when caller knows the account but not the instance.

        Resolves the owning instance via the account → instance map
        built from per-instance ``notifications/aios/accounts`` snapshots.
        Returns a ``not_enabled`` envelope if no instance currently
        serves that ``(connector, account)`` pair — covers both "operator
        listed the account but its instance hasn't started yet" and
        "account was reported by the connector subprocess but later
        retracted".
        """
        instance = self._account_to_instance.get((connector, account))
        if instance is None:
            return {
                "error": f"no instance serves account {account!r} on connector {connector!r}",
                "code": "not_enabled",
            }
        return await self.dispatch_call(connector, instance, tool, arguments, meta=meta)

    # ── notifications ─────────────────────────────────────────────────

    async def _on_aios_notification(
        self,
        key: tuple[str, str],
        method: str,
        params: dict[str, Any] | None,
    ) -> None:
        """Route a ``notifications/aios/<...>`` payload from one instance."""
        connector, instance = key
        state = self._states[key]
        if method == "notifications/aios/accounts":
            # Each fresh accounts payload is the connector's authoritative
            # statement of its account set; clear any prior accounts-payload-
            # derived ``last_error`` (malformed payload, account conflict)
            # before re-evaluating.  Subprocess-state errors (crashes) are
            # set in ``_supervisor_loop`` and not affected here.
            state.last_error = None
            raw_accounts = (params or {}).get("accounts")
            if isinstance(raw_accounts, list):
                state.accounts = list(raw_accounts)
                self._rebuild_account_map(state, raw_accounts)
                log.info(
                    "connector.accounts_updated",
                    connector=connector,
                    instance=instance,
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
                    connector=connector,
                    instance=instance,
                    payload_type=type(raw_accounts).__name__,
                )
        elif method == "notifications/aios/inbound":
            # Spawn a separate task: the splitter holds the SDK's receive
            # stream while this handler runs, so doing DB work inline
            # would pause every other message (init responses, account
            # snapshots) for the duration.
            task = asyncio.create_task(
                self._handle_inbound(key, params or {}),
                name=f"connector_inbound:{state.label}",
            )
            self._inbound_tasks.add(task)
            task.add_done_callback(self._inbound_tasks.discard)
        else:
            log.info(
                "connector.unknown_aios_notification",
                connector=connector,
                instance=instance,
                method=method,
            )

    def _rebuild_account_map(self, state: ConnectorState, raw_accounts: list[Any]) -> None:
        """Refresh ``_account_to_instance`` for the reporting instance.

        Atomic per-instance: clear all entries currently pointing at
        ``state.instance`` for this connector, then insert from the new
        snapshot.  Tolerates concurrent snapshots from sibling instances
        (each only touches its own entries).

        Conflict policy: if another instance of the same connector type
        has already claimed this account, log a warning, set
        ``state.last_error``, and DO NOT overwrite the existing claim.
        Schema's ``UNIQUE (connector, account) WHERE archived_at IS
        NULL`` makes operator-side double-attach impossible; this surfaces
        the misconfig at the connector layer so it's visible in
        ``aios connector list``.
        """
        connector = state.connector
        # Clear stale claims for this instance.
        for key in list(self._account_to_instance.keys()):
            if key[0] == connector and self._account_to_instance[key] == state.instance:
                del self._account_to_instance[key]
        conflicts: list[str] = []
        for entry in raw_accounts:
            if not isinstance(entry, dict):
                continue
            account_id = entry.get("id")
            if not isinstance(account_id, str) or not account_id:
                continue
            map_key = (connector, account_id)
            existing = self._account_to_instance.get(map_key)
            if existing is not None and existing != state.instance:
                conflicts.append(f"{account_id!r} (already claimed by {existing!r})")
                log.warning(
                    "connector.account_conflict",
                    connector=connector,
                    instance=state.instance,
                    other_instance=existing,
                    account=account_id,
                )
                continue
            self._account_to_instance[map_key] = state.instance
        if conflicts:
            state.last_error = f"account conflict with sibling instance(s): {', '.join(conflicts)}"

    # ── inbound dispatch ──────────────────────────────────────────────

    async def _handle_inbound(self, key: tuple[str, str], params: dict[str, Any]) -> None:
        """Append an inbound event, dedupe via the ledger, ack the spool.

        Implements the crash walkthrough from plan decision #16: ledger
        INSERT runs in the same transaction as ``append_event`` so a
        replayed inbound (same ULID after worker SIGKILL) hits
        ``ON CONFLICT DO NOTHING`` and the txn rolls back without a
        second event row.  Ack runs after commit and is awaited inline
        (see :meth:`_send_ack` — bounded by ``_DISPATCH_TIMEOUT_S`` so
        a wedged connector pins the inbound task for at most 60 s); a
        failed ack just means the connector replays again on its next
        reconnect, where the ledger conflict path takes over.

        Drops are counted but never fail the splitter — a malformed
        inbound surfaces in ``recent_drops`` and operator logs without
        stalling the connector pipeline.
        """

        connector, instance = key
        state = self._states[key]
        event_id = params.get("event_id")
        account = params.get("account")
        chat_id = params.get("chat_id")
        sender = params.get("sender") or {}
        content = params.get("content")
        raw_metadata = params.get("metadata")
        connector_metadata = raw_metadata if isinstance(raw_metadata, dict) else None
        # Optional ISO-8601 platform timestamp (when the source platform
        # supplies one).  Stamped onto the appended event as
        # ``metadata.platform_timestamp`` so operators can compare against
        # the server-side ``created_at`` when debugging delivery latency.
        raw_timestamp = params.get("timestamp")
        platform_timestamp = raw_timestamp if isinstance(raw_timestamp, str) else None

        # event_id is checked first because it's the only thing that lets
        # us ack the spool entry — without it we can't dedup or clear,
        # so the connector author must fix their emit shape.
        if not isinstance(event_id, str):
            log.warning(
                "connector.inbound_missing_event_id",
                connector=connector,
                instance=instance,
            )
            self._record_drop(state, "malformed")
            return

        if not (isinstance(account, str) and isinstance(chat_id, str) and isinstance(content, str)):
            log.warning(
                "connector.inbound_malformed",
                connector=connector,
                instance=instance,
                missing=[
                    field_name
                    for field_name, value in (
                        ("account", account),
                        ("chat_id", chat_id),
                        ("content", content),
                    )
                    if not isinstance(value, str)
                ],
            )
            self._record_drop(state, "malformed")
            await self._send_ack(state, event_id)
            return

        if len(content) > MAX_USER_MESSAGE_CHARS:
            log.warning(
                "connector.inbound_too_large",
                connector=connector,
                instance=instance,
                account=account,
                length=len(content),
                limit=MAX_USER_MESSAGE_CHARS,
            )
            self._record_drop(state, "payload_too_large")
            await self._send_ack(state, event_id)
            return

        pool = runtime.require_pool()

        # Drift guard: an account that's persisted in ``connections``
        # but absent from the connector's live snapshot is a no-op
        # delivery target — the connector can't reach it on outbound,
        # so silently appending an event would orphan the conversation.
        # Surface as a counter so ``aios connector list`` shows the
        # stuck account; the operator can detach + re-attach against a
        # known-good account.  Plan §15 lists this as a drop reason.
        known_account_ids = {entry.get("id") for entry in state.accounts if isinstance(entry, dict)}
        if state.accounts and account not in known_account_ids:
            self._record_drop(state, "account_drift")
            await self._send_ack(state, event_id)
            return

        # 1. Look up connection (fresh read — no caching, since the
        #    operator may have just attached/configured the connection).
        async with pool.acquire() as conn:
            connection = await queries.get_connection_for_account(
                conn, connector=connector, account=account
            )
        if connection is None:
            await self._auto_create_or_drop(state, account=account)
            # Even if auto-create succeeds, this inbound is dropped:
            # the freshly-detached connection has no session to deliver
            # to.  Operator must explicitly attach / configure-per-chat.
            await self._send_ack(state, event_id)
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
                connector_name=connector,
                account=account,
                chat_id=chat_id,
            )
        else:
            self._record_drop(state, "detached")
            await self._send_ack(state, event_id)
            return

        if target_session_id is None:
            # Per-chat resolution already recorded the drop reason.
            await self._send_ack(state, event_id)
            return

        # 3. Stage attachments before the dedup transaction.  Renames
        # are idempotent on the replayed-event_id path; a failure here
        # acks the spool because the connector's temp file is gone and
        # endless replay would only re-fail.
        try:
            staged_attachments, newly_staged_paths = stage_inbound_attachments(
                session_id=target_session_id,
                connector_name=connector,
                event_id=event_id,
                raw_attachments=params.get("attachments"),
            )
        except AttachmentStagingError as err:
            log.warning(
                "connector.attachment_staging_failed",
                connector=connector,
                instance=instance,
                event_id=event_id,
                session_id=target_session_id,
                error=str(err),
            )
            self._record_drop(state, "attachment_staging_failed")
            await self._send_ack(state, event_id)
            return

        # 4. Append event + ledger row in the same transaction.
        try:
            await self._append_with_dedup(
                connector_name=connector,
                account=account,
                event_id=event_id,
                session_id=target_session_id,
                chat_id=chat_id,
                sender=sender if isinstance(sender, dict) else {},
                content=content,
                attachments=staged_attachments,
                connector_metadata=connector_metadata,
                platform_timestamp=platform_timestamp,
            )
        except NotFoundError:
            # The session vanished between resolution and append.  Per
            # design #216 §5, an append failure triggers the synchronous
            # compensating action (the orphan GC sweep at startup is
            # the catch-all for crashes, not an excuse to leak files
            # we already know are unreferenced).  Replay-skipped paths
            # are excluded by ``stage_inbound_attachments`` so this
            # only unlinks bytes this call materialized.
            for path in newly_staged_paths:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
            self._record_drop(state, "session_missing")
            await self._send_ack(state, event_id)
            return

        # 5. Outside the transaction: defer wake unconditionally.
        # ``defer_wake`` is idempotent (procrastinate's queueing_lock
        # coalesces duplicates), so we call it on both first-append and
        # dedup paths — the dedup path heals the case where the prior
        # attempt committed the event but failed to defer the wake.
        # Failures propagate so the inbound task fails and the connector
        # replays on reconnect, where this same path can re-run.
        await defer_wake(pool, target_session_id, cause="inbound")
        await self._send_ack(state, event_id)

    async def _auto_create_or_drop(self, state: ConnectorState, *, account: str) -> None:
        """Insert a detached connection if ``auto_create_connections`` is on.

        Mirrors plan decision #13: missing pair drops the inbound
        regardless, but auto-create gives operators a row they can
        attach to without re-issuing the original message.  Both paths
        increment the ``no_connection`` counter so the operator sees
        the surface (auto-creating a row doesn't deliver the message
        that prompted it).

        Race-safe via :func:`queries.insert_connection`'s
        ``ON CONFLICT DO NOTHING RETURNING`` shape (plan decision #5);
        a concurrent explicit POST or sibling inbound handler that
        beat us simply hands back the existing row, no exception.

        No dedup-ledger row is written here on purpose: if the operator
        attaches the auto-created connection later, the connector's
        replay should deliver the message that prompted creation —
        which a ledger row would block.
        """

        self._record_drop(state, "no_connection")
        # ``connectors_auto_create`` is keyed by connector type, not
        # instance — auto-create policy is per-platform.
        if not self._settings.connectors_auto_create.get(state.connector, True):
            return
        pool = runtime.require_pool()
        async with pool.acquire() as conn:
            await queries.insert_connection(
                conn, connector=state.connector, account=account, metadata={}
            )
        log.info(
            "connector.auto_create_connection",
            connector=state.connector,
            instance=state.instance,
            account=account,
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
        connector_metadata: dict[str, Any] | None,
        platform_timestamp: str | None = None,
    ) -> None:
        """Append a user-message event AND record the dedup-ledger row.

        Both writes run in one transaction.  Silently returns on dedup
        (already-processed inbound replayed by the connector) — the
        ``_DedupRollback`` path logs at INFO and the caller proceeds to
        defer a wake regardless (idempotent via procrastinate's
        queueing_lock).  Raises :class:`NotFoundError` if the target
        session vanished between resolution and append.

        Connector-supplied metadata (signal's ``sender_uuid``,
        ``timestamp_ms``, ``reply_to``, ``reaction``; telegram's
        ``message_id``, ``reply_to``; anything else the connector
        emits) is merged in BEFORE the supervisor's stamps so
        supervisor-canonical fields (``channel``, ``sender``,
        ``attachments``, ``platform_timestamp``) win on key conflicts.
        The model relies on these fields — e.g. ``signal_react`` is
        documented to copy ``sender_uuid`` and ``timestamp_ms`` from
        the inbound header.

        ``platform_timestamp`` (when supplied by the connector via the
        SDK's ``emit_inbound(timestamp=...)`` kwarg) is the source
        platform's actual delivery time as ISO-8601.  Server-side
        ``created_at`` always wins for ordering invariants; this field
        is purely a debugging aid.
        """

        pool = runtime.require_pool()
        channel = f"{connector_name}/{account}/{chat_id}"
        sender_name = sender.get("display_name")
        metadata: dict[str, Any] = {}
        if connector_metadata is not None:
            metadata.update(connector_metadata)
        metadata["channel"] = channel
        if isinstance(sender_name, str):
            metadata["sender"] = sender_name
        if isinstance(attachments, list) and attachments:
            metadata["attachments"] = attachments
        if platform_timestamp is not None:
            metadata["platform_timestamp"] = platform_timestamp
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
                    raise _DedupRollback()
                await queries.flip_idle_to_pending(conn, session_id)
        except _DedupRollback:
            log.info(
                "connector.inbound_duplicate",
                connector=connector_name,
                event_id=event_id,
            )

    async def _send_ack(self, state: ConnectorState, event_id: str) -> None:
        """Send ``aios_inbound_ack`` to clear the spool of the originating instance.

        Awaited inline so :meth:`shutdown` waits for in-flight acks
        (or hits the 60s ``_DISPATCH_TIMEOUT_S``) before returning.
        Failures here aren't fatal — the connector replays on its next
        reconnect where the ledger conflict path catches the duplicate;
        we log but don't retry to avoid pinning a wedged connector.
        """
        result = await self.dispatch_call(
            state.connector, state.instance, "aios_inbound_ack", {"event_id": event_id}
        )
        if "error" in result:
            log.info(
                "connector.inbound_ack_failed",
                connector=state.connector,
                instance=state.instance,
                event_id=event_id,
                error=result["error"],
            )

    def _record_drop(self, state: ConnectorState, reason: DropReason) -> None:
        """Bump the per-reason drop counter, log, and let snapshots reflect it."""
        state.drops[reason] += 1
        log.info(
            "connector.inbound_dropped",
            connector=state.connector,
            instance=state.instance,
            reason=reason,
            count=state.drops[reason],
        )

    # ── supervisor loop ───────────────────────────────────────────────

    async def _supervisor_loop(self, key: tuple[str, str]) -> None:
        """Open subprocess, park on shutdown, restart on crash with backoff.

        Returns when :meth:`shutdown` flips the shutdown flag *or* the
        circuit opens.  Either way the corresponding :class:`ConnectorState`
        reflects terminal status before this coroutine exits, so a GET
        call right after sees the right thing.
        """
        connector, instance = key
        state = self._states[key]
        spec = state.spec

        async def handler(method: str, params: dict[str, Any] | None) -> None:
            await self._on_aios_notification(key, method, params)

        while not self._shutdown.is_set():
            state.status = "starting"
            state.last_error = None
            try:
                async with open_connector_session(spec, on_aios_notification=handler) as (
                    session,
                    init_result,
                    closed_event,
                ):
                    self._validate_capability(state, init_result)
                    state.session = session
                    state.init_result = init_result
                    state.instructions = init_result.instructions
                    state.status = "running"
                    state.backoff = _BACKOFF_INITIAL_S
                    state.ready.set()
                    log.info(
                        "connector.running",
                        connector=connector,
                        instance=instance,
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
                    log.warning(
                        "connector.crashed",
                        connector=connector,
                        instance=instance,
                        error=state.last_error,
                    )
            except asyncio.CancelledError:
                if self._shutdown.is_set():
                    log.info(
                        "connector.shutdown",
                        connector=connector,
                        instance=instance,
                    )
                    return
                # Cancellation that wasn't initiated by us — almost always
                # the subprocess closed its end and anyio's task group
                # propagated cancellation through ``open_connector_session``.
                # Fall through to the crash branch and respawn.
                state.last_error = "subprocess closed"
                log.warning(
                    "connector.crashed",
                    connector=connector,
                    instance=instance,
                    error=state.last_error,
                )
            except Exception as exc:
                state.last_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "connector.crashed",
                    connector=connector,
                    instance=instance,
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
                    connector=connector,
                    instance=instance,
                    failures=len(state.failures),
                    window_s=_CIRCUIT_WINDOW_S,
                )
                return

            state.status = "restarting"
            log.info(
                "connector.restart_scheduled",
                connector=connector,
                instance=instance,
                delay_s=state.backoff,
            )
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=state.backoff)
                # Shutdown won the race.
                return
            except TimeoutError:
                pass
            state.backoff = min(state.backoff * 2, _BACKOFF_CAP_S)

    def _validate_capability(self, state: ConnectorState, init_result: InitializeResult) -> None:
        """Hard-fail if the connector didn't declare ``experimental.aios/connector``."""
        experimental = init_result.capabilities.experimental or {}
        if _AIOS_EXPERIMENTAL_KEY not in experimental:
            raise RuntimeError(
                f"connector instance {state.label!r} did not declare "
                f"experimental.{_AIOS_EXPERIMENTAL_KEY!r} capability"
            )
