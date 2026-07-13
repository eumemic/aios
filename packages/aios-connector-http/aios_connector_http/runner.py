"""Base class for runtime-container connector authors.

One container, one ``connector`` *type* (e.g. ``"telegram"``), N
connections discovered at runtime via SSE.  Subclass
:class:`HttpConnector`, set ``connector``, decorate methods with
:func:`tool`, implement :meth:`serve_connection` if your platform pushes
inbound from each connection (e.g. signal/telegram polling)::

    class MyConnector(HttpConnector):
        connector = "myplatform"

        async def serve_connection(self, connection_id, secrets):
            async with platform.client(secrets["bot_token"]) as plat:
                async for inbound in plat.poll():
                    await self.emit_inbound(
                        connection_id=connection_id,
                        chat_id=inbound.chat_id,
                        sender={"display_name": inbound.sender},
                        content=inbound.text,
                    )

    if __name__ == "__main__":
        import asyncio
        asyncio.run(MyConnector().run())

The runner reads ``AIOS_URL`` and ``AIOS_RUNTIME_TOKEN`` from env, opens
an :class:`aios_sdk.Client`, publishes the tool catalog for the
connector type once, then races three jobs concurrently:

1. ``GET /v1/connectors/connections`` — discovery SSE.  Each ``added``
   triggers ``serve_connection(connection_id, secrets)`` in its own task.
   Each ``removed`` cancels the matching task.
2. ``GET /v1/connectors/runtime/calls`` — calls SSE.  Each event is
   dispatched to its ``@tool`` method; ``connection_id`` is one of the
   focal-injectable kwargs.
3. The per-connection ``serve_connection`` tasks themselves, supervised
   in a :class:`asyncio.TaskGroup` for clean shutdown.

The runner tracks answered ``tool_call_id``s in memory so SSE
reconnects (which replay the backfill) don't double-execute.  Each
answered id maps to its serialized tool-result payload (or ``None``):
the persist point sits *between* the side-effecting tool body and the
tool-result POST, so a POST that fails after a successful platform send
won't re-run the body on replay — the persisted result is simply
re-POSTed (the AIOS append is idempotent on ``(session_id,
tool_call_id)``).  For durability across container restart, override
:meth:`load_answered` / :meth:`save_answered` (a small SQLite spool
ships at ``spool.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import signal as _signal
import types
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import httpx
import structlog
from aios_sdk import (
    Client,
    stream_connection_discovery,
    stream_connector_calls,
    stream_management_calls,
)
from aios_sdk._generated.api.connectors.get_connector_runtime_secrets import (
    asyncio_detailed as _get_runtime_secrets,
)
from aios_sdk._generated.api.connectors.post_connector_runtime_management_call_result import (
    asyncio_detailed as _post_runtime_management_call_result,
)
from aios_sdk._generated.api.connectors.post_connector_runtime_tool_result import (
    asyncio_detailed as _post_runtime_tool_result,
)
from aios_sdk._generated.api.connectors.put_connector_tools_schema import (
    asyncio_detailed as _put_tools_schema,
)
from aios_sdk._generated.models.http_validation_error import HTTPValidationError
from aios_sdk._generated.models.runtime_management_call_result_request import (
    RuntimeManagementCallResultRequest,
)
from aios_sdk._generated.models.runtime_tool_result_request import (
    RuntimeToolResultRequest,
)
from aios_sdk._generated.models.runtime_tool_result_request_content_type_1_item import (
    RuntimeToolResultRequestContentType1Item,
)
from aios_sdk._generated.models.tools_schema_update import ToolsSchemaUpdate
from aios_sdk._generated.models.tools_schema_update_tools_item import (
    ToolsSchemaUpdateToolsItem,
)
from aios_sdk._generated.types import Unset
from ulid import ULID

from .sandbox import _SandboxPathMarker, resolve_sandbox_path
from .schema import derive_tool_spec

ToolFn = Callable[..., Awaitable[Any]]

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_TOOL_ATTR = "__aios_http_tool__"
_TOOL_FAF_ATTR = "__aios_http_tool_fire_and_forget__"
_MGMT_ATTR = "__aios_http_management__"


class ManagementHandlerError(Exception):
    """Raised by a ``@management_handler`` to POST a structured ``is_error=true`` payload.

    Plain exceptions become ``{"error": str(exc)}``; raise this when the
    operator side needs to discriminate on a structured shape (e.g.
    captcha-required carrying a URL).
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(json.dumps(payload))
        self.payload = payload


def _is_fatal_inbound_status(status_code: int) -> bool:
    """``emit_inbound`` raises for these; everything else (routine 4xx)
    drops the envelope and returns ``None``.

    ``401`` / ``403`` mean the runtime token is busted (revoked or
    misconfigured) — fatal for every connection the container serves.
    ``5xx`` is a server-side outage / bug — also fatal so the
    container crash-and-restarts.  Routine 4xx (``400``, ``422``)
    indicates one specific envelope is malformed; dropping it lets
    sibling connections keep serving.
    """
    return status_code in (401, 403) or status_code >= 500


def tool(
    *,
    name: str | None = None,
    fire_and_forget: bool = False,
) -> Callable[[ToolFn], ToolFn]:
    """Decorate a method as a connector tool.

    The method's name becomes the published tool name (or the explicit
    ``name=`` override).  Arguments come from the model's tool_call
    arguments, parsed as JSON and passed as kwargs.  The method's
    return value is the tool result string the session log stores.

    The runner publishes a derived JSON Schema for every ``@tool`` once
    at startup via ``PUT /v1/connectors/{connector}/tools_schema``;
    individual connections inherit it from the type catalog.

    ``fire_and_forget`` declares the tool a *delivery* action — a message
    send or reaction whose only output is a ``{"sent_at_ms": N}`` ack.  It
    types a mid-dispatch failure as ``delivery_failed`` (#1722) so the model
    can tell "my send attempt itself errored" from an ordinary tool
    exception.  It no longer affects the wake decision: every tool result —
    including a delivery ack — is a stimulus the session wakes to react to,
    so the standard agentic loop (tool call → result → continue) holds for
    connector sends too (#1919).  Leave it ``False`` for ordinary
    list/create/get/edit/delete tools.
    """

    def _wrap(f: ToolFn) -> ToolFn:
        setattr(f, _TOOL_ATTR, name or f.__name__)
        setattr(f, _TOOL_FAF_ATTR, fire_and_forget)
        return f

    return _wrap


@dataclass(frozen=True, slots=True)
class _ToolMeta:
    """Per-tool reflection cache populated once at construction."""

    fn: ToolFn
    focal_params: frozenset[str]
    # Subset of ``focal_params`` the signature requires (no default) — the
    # method literally cannot run without them. Used at dispatch time
    # (#1722) to detect an unresolvable focal channel BEFORE calling the
    # tool body: a required ``chat_id``/``external_account_id`` that
    # ``_inject_focal_kwargs`` couldn't fill (empty/malformed
    # ``focal_channel``, e.g. cleared mid-turn) fails loud with a typed
    # ``channel_unresolved`` error instead of the tool method raising an
    # opaque ``TypeError: missing required keyword-only argument``.
    required_focal_params: frozenset[str]
    sandbox_params: tuple[tuple[str, str], ...]  # (param_name, "scalar" | "list")
    fire_and_forget: bool = False


def management_handler(
    *,
    method: str | None = None,
) -> Callable[[ToolFn], ToolFn]:
    """Decorate a method as a management-call handler.

    Sibling of :func:`tool` for operator-initiated calls (not model-callable).
    The decorated method's name (or the explicit ``method=`` override) is
    the wire-level method the operator-facing route dispatches to.
    """

    def _wrap(f: ToolFn) -> ToolFn:
        setattr(f, _MGMT_ATTR, method or f.__name__)
        return f

    return _wrap


@dataclass(frozen=True, slots=True)
class _ManagementHandlerMeta:
    """Per-handler reflection cache."""

    fn: ToolFn


@dataclass
class _ConnectionState:
    """Per-connection runtime state.  One row per active connection."""

    connection_id: str
    external_account_id: str
    secrets: dict[str, str] = field(default_factory=dict)
    worker: asyncio.Task[None] | None = None


class HttpConnector:
    """Base class for runtime-container connectors.

    Set ``connector`` on the subclass to the platform type (e.g.
    ``"telegram"``).  Decorate methods with :func:`tool`.  Override
    :meth:`serve_connection` to do per-connection platform work.
    """

    # Subclasses MUST set this — the connector type the container serves.
    connector: str = ""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
    ) -> None:
        if not self.connector:
            raise RuntimeError(
                f"{type(self).__name__} must set ``connector`` class attribute "
                "(e.g. ``connector = 'telegram'``)"
            )
        self._base_url = base_url or os.environ["AIOS_URL"]
        self._token = token or os.environ["AIOS_RUNTIME_TOKEN"]
        self._client: Client | None = None
        self._tools: dict[str, _ToolMeta] = self._collect_tools()
        self._management: dict[str, _ManagementHandlerMeta] = self._collect_management_handlers()
        # tool_call_id → serialized tool-result payload (or None when no
        # result is persisted, e.g. management calls).  Persisted between
        # the side-effecting tool body and the result POST so a failed POST
        # replays the persisted result instead of re-running the body.
        self._answered: dict[str, str | None] = {}
        self._connections: dict[str, _ConnectionState] = {}
        # Subclasses narrow the value type via a class-body annotation::
        #
        #     class SignalConnector(HttpConnector):
        #         state: dict[str, _SignalConnectionState]
        self.state: dict[str, Any] = {}
        self._ready_event: asyncio.Event = asyncio.Event()
        self._connection_served: dict[str, asyncio.Event] = {}
        # Counts how many of the three background loops (discovery, tool,
        # management) have completed their first successful SSE backfill and
        # entered their live-tail phase.  wait_ready() waits until this
        # reaches 3 — guaranteeing all loops are actively listening before
        # the caller proceeds.  Reset to 0 on teardown so re-runs work.
        self._loops_backfilled: int = 0
        self._all_loops_live: asyncio.Event = asyncio.Event()
        # Per-connection reconnect backoff, in seconds.  Keyed by
        # connection_id.  ``_isolated_serve_connection`` pops
        # ``_connections[connection_id]`` on a terminal (non-cancel)
        # ``serve_connection`` failure so a later discovery ``added`` (or
        # manual re-add) re-spawns the worker; this dict bounds how fast
        # that re-spawn actually starts platform work, so a hard-failing
        # connection (revoked token, unregistered phone) backs off
        # exponentially instead of hot-looping on every backfill replay.
        # An entry is cleared on a clean serve (one that ran past the
        # backoff sleep without immediately failing) and on ``removed``.
        self._reconnect_backoff: dict[str, float] = {}

    # ── reconnect-backoff tunables (overridable on subclasses) ────────
    #
    # Bounds the re-spawn rate of a connection whose ``serve_connection``
    # keeps failing terminally.  The first (re)spawn after a failure
    # sleeps ``RECONNECT_BACKOFF_INITIAL``; each subsequent consecutive
    # failure doubles it up to ``RECONNECT_BACKOFF_MAX``.
    RECONNECT_BACKOFF_INITIAL: float = 1.0
    RECONNECT_BACKOFF_MAX: float = 60.0

    # ─── helpers (subclasses use these) ──────────────────────────────

    def focal_channel(self, external_account_id: str, chat_id: str) -> str:
        """Return the canonical ``<connector>/<external_account_id>/<chat_id>`` string.

        Outbound tool methods write this to messages they edit/react to;
        the runtime parses it back into ``external_account_id`` /
        ``chat_id`` kwargs on the next call (see
        :func:`_inject_focal_kwargs`).
        """
        return f"{self.connector}/{external_account_id}/{chat_id}"

    # ─── lifecycle hooks (override as needed) ────────────────────────

    async def setup(self, tg: asyncio.TaskGroup) -> None:
        """Override: container-wide init before discovery + tool loops.

        Called once inside :meth:`run` after the SDK client opens and
        after :meth:`_publish_tools_schema`, immediately upon entering
        the TaskGroup that owns the discovery + tool loops.  Use this
        for per-container resources whose lifetime spans every
        connection (e.g. a shared signal-cli daemon serving multiple
        accounts) — spawn any long-running tasks via ``tg.create_task``
        so an unhandled crash propagates and tears the container down,
        rather than silently stalling inbound delivery.
        Default is a no-op.
        """
        del tg

    async def serve_connection(self, connection_id: str, secrets: dict[str, str]) -> None:
        """Override: per-connection long-running platform task.

        Spawned when the discovery SSE emits ``added`` for this
        ``connection_id``; cancelled when ``removed``.  The default
        is a no-op block — connectors that only respond to outbound
        tool calls (no inbound platform feed) leave it as-is.

        ``secrets`` is the per-connection decrypted credentials dict.
        Cached at spawn time — if operators rotate secrets, the
        operator restarts the runtime container to pick them up.
        """
        del connection_id, secrets
        await asyncio.Event().wait()  # block forever

    async def teardown(self) -> None:
        """Override: cleanup before the runner exits."""

    # ── test-coordination primitives ──────────────────────────────────

    async def wait_ready(self, deadline: float = 30.0) -> None:
        """Block until all three background loops have opened their SSE streams.

        Specifically: each of the discovery, tool, and management-call loops has
        received the synthetic ``"_open"`` event the SDK's
        :func:`aios_sdk.streaming._stream_sse` yields right after
        ``response.raise_for_status()`` succeeds — i.e. the HTTP handshake
        returned 2xx headers.  Since the server-side ``listen_for_*`` setup
        runs in the route handler BEFORE ``EventSourceResponse`` is constructed
        (issue #376), 2xx headers imply the LISTEN connection is already
        established and any subsequent NOTIFY is queued.

        We deliberately do NOT depend on a server-emitted ``"connected"``
        event yielded before backfill, because yielding from an sse-starlette
        generator before any natural data chunk has surfaced an "ASGI callable
        returned without completing response" failure mode in CI (see aios#366
        commit-message archaeology).

        Caveat for test authors: in tests that use an in-process uvicorn
        fixture, watch for the cross-test ``sse_starlette.AppStatus.should_exit``
        contamination noted in :mod:`tests.conftest`
        (``_reset_sse_starlette_shutdown_state``); without that reset, every
        SSE handshake after the first fixture teardown can be cancelled
        immediately by the library's shutdown watcher.

        The default deadline is 30 s — comfortable headroom over worst-case
        healthy CI scheduling (LISTEN preflight is now a single asyncpg
        connect + add_listener inside the route handler, on the order of
        tens of milliseconds in steady state).

        Raises ``TimeoutError`` if the loops have not all reached the live
        phase within ``deadline`` seconds.  Intended for test coordination.
        """
        await asyncio.wait_for(self._all_loops_live.wait(), timeout=deadline)

    async def wait_connection_served(self, connection_id: str, deadline: float = 30.0) -> None:
        """Block until serve_connection has been spawned for connection_id.

        The discovery loop must receive and process the ``added`` event for
        ``connection_id`` before this returns.  Default deadline 30 s
        matches :meth:`wait_ready`'s so callers don't need to think about
        which is tighter; the SSE handshake itself is preflighted in the
        route handler now (#376) so the discovery loop sees backfill
        immediately after 2xx.

        Raises ``TimeoutError`` if the connection is not added within
        ``deadline`` seconds.  Intended for test coordination.
        """
        event = self._connection_served.setdefault(connection_id, asyncio.Event())
        await asyncio.wait_for(event.wait(), timeout=deadline)

    def _mark_loop_backfilled(self) -> None:
        """Called by each loop once it receives the SDK's ``"_open"`` marker.

        The marker is the SDK's synthetic SSE event yielded as soon as the
        underlying HTTP response returns 2xx headers (see
        :func:`aios_sdk.streaming._stream_sse`).  At that point the server-
        side SSE generator has begun executing and its ``listen_for_*``
        context manager is in the process of registering the LISTEN —
        within a single event-loop iteration any subsequent NOTIFY will be
        delivered.  When all three loops have called this,
        ``_all_loops_live`` is set so :meth:`wait_ready` can unblock.

        Using a plain int (not a Lock) is safe: asyncio is single-threaded
        so the increment is atomic within a single event-loop iteration.
        """
        self._loops_backfilled += 1
        if self._loops_backfilled >= 3:
            self._all_loops_live.set()

    async def load_answered(self) -> dict[str, str | None]:
        """Override: persisted tool_call_ids from a previous lifetime.

        Returns a map of ``tool_call_id`` → serialized tool-result payload
        (or ``None`` when no result was persisted).  Default ``{}``.
        """
        return {}

    async def save_answered(self, tool_call_id: str, result: str | None = None) -> None:
        """Override: persist a freshly-answered tool_call_id.

        ``result`` is the serialized platform-send result to persist
        alongside the id (``None`` when there is nothing to replay, e.g.
        management calls).  Default no-op.
        """

    async def _mark_answered(self, tool_call_id: str, result: str | None) -> None:
        """Record ``tool_call_id`` as answered and persist its result.

        Co-located with the side-effecting tool body so the persist lands
        *before* the result POST: a POST that fails after a successful
        platform send won't re-run the body on replay.
        """
        self._answered[tool_call_id] = result
        await self.save_answered(tool_call_id, result)

    async def _replay_tool_result(self, client: Client, call: dict[str, Any]) -> None:
        """Re-POST the persisted result for an already-answered call.

        Heals the send-succeeded/result-POST-failed window: the tool body
        already ran (and its side effect happened), so we only re-deliver
        the persisted result.  When no result was persisted (value is
        ``None``) there is nothing to replay and we skip.
        """
        tool_call_id = call.get("tool_call_id", "")
        result = self._answered.get(tool_call_id)
        if result is None:
            return
        await self._post_tool_result(
            client,
            connection_id=call.get("connection_id", ""),
            session_id=call.get("session_id", ""),
            tool_call_id=tool_call_id,
            content=result,
        )

    # ─── connector → aios glue ───────────────────────────────────────

    async def emit_inbound(
        self,
        *,
        connection_id: str,
        chat_id: str,
        sender: dict[str, Any],
        content: str,
        attachments: list[tuple[str, bytes, str]] | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: str | None = None,
        event_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Forward a platform inbound to aios for ``connection_id``.

        Idempotent on ``event_id``.  ``attachments`` is a list of
        ``(filename, bytes, content_type)`` tuples that ride as
        multipart parts.

        Routine 4xx responses (a 422 from one malformed envelope, a 400
        from a stale connection_id mid-removal) drop the envelope and
        return ``None`` so one bad inbound can't tear down the whole
        container via the parent TaskGroup.  ``401`` / ``403`` (auth
        broken — runtime token revoked or misconfigured) and ``5xx``
        (server outage / bug) still raise: those are container-level
        problems that warrant the crash-and-restart loop.  Callers that
        need the bad-envelope to surface (e.g. integration-test tools
        that want loud validation failures) check for ``None`` and
        raise themselves.
        """
        client = self._require_client()
        eid = event_id or str(ULID())
        log.info(
            "connector.inbound",
            connector=self.connector,
            connection_id=connection_id,
            chat_id=chat_id,
            sender=sender,
            content_len=len(content),
            attachment_count=len(attachments or []),
            event_id=eid,
        )
        # Hand-rolled multipart: the generated body's ``attachments``
        # field shape ``list[str]`` from the openapi spec (FastAPI's
        # ``UploadFile`` parts don't survive the schema round-trip),
        # so we POST directly via the underlying httpx client.
        files: list[tuple[str, tuple[str | None, bytes, str]]] = [
            ("connection_id", (None, connection_id.encode(), "text/plain")),
            ("event_id", (None, eid.encode(), "text/plain")),
            ("chat_id", (None, chat_id.encode(), "text/plain")),
            ("content", (None, content.encode(), "text/plain")),
            ("sender", (None, json.dumps(sender).encode(), "text/plain")),
        ]
        if metadata is not None:
            files.append(("metadata", (None, json.dumps(metadata).encode(), "text/plain")))
        if timestamp is not None:
            files.append(("timestamp", (None, timestamp.encode(), "text/plain")))
        for filename, blob, content_type in attachments or []:
            files.append(("attachments", (filename, blob, content_type)))

        response = await client.get_async_httpx_client().post(
            "/v1/connectors/runtime/inbound",
            files=files,
        )
        if response.is_error:
            # ``response.raise_for_status()`` discards the body, which is
            # exactly where FastAPI's validation diagnostics live for a 422
            # (and where our error envelope lives for everything else).  Log
            # the body verbatim so the operator has the field path /
            # message in the container log instead of just the bare status
            # + URL from ``httpx.HTTPStatusError``.
            sc = response.status_code
            log.warning(
                "connector.inbound.failed",
                status_code=sc,
                event_id=eid,
                connection_id=connection_id,
                body=response.text[:2000],
            )
            if _is_fatal_inbound_status(sc):
                response.raise_for_status()
            return None
        return dict(response.json())

    async def emit_lifecycle(
        self,
        *,
        connection_id: str,
        event: str,
        reason: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Append a ``kind=lifecycle`` event onto each session bound to
        ``connection_id``.

        Use for connection-broken signals the model needs to see in its
        context (e.g. "whatsapp.connection.lost" with reason
        "daemon_crashed" or "peer_logout") so it doesn't silently send
        into the void.  Returns the appended-session-ids payload, or
        ``None`` when the API returned a non-fatal 4xx (matching
        :meth:`emit_inbound`'s drop-don't-raise stance).
        """
        client = self._require_client()
        log.info(
            "connector.lifecycle",
            connector=self.connector,
            connection_id=connection_id,
            lifecycle_event=event,
            reason=reason,
        )
        body: dict[str, Any] = {
            "connection_id": connection_id,
            "event": event,
        }
        if reason is not None:
            body["reason"] = reason
        if data is not None:
            body["data"] = data
        response = await client.get_async_httpx_client().post(
            "/v1/connectors/runtime/lifecycle",
            json=body,
        )
        if response.is_error:
            sc = response.status_code
            log.warning(
                "connector.lifecycle.failed",
                status_code=sc,
                connection_id=connection_id,
                lifecycle_event=event,
                body=response.text[:2000],
            )
            if _is_fatal_inbound_status(sc):
                response.raise_for_status()
            return None
        return dict(response.json())

    async def emit_session_lifecycle(
        self,
        *,
        connection_id: str,
        session_id: str,
        event: str,
        reason: str | None = None,
        data: dict[str, Any] | None = None,
        wake: bool = False,
    ) -> dict[str, Any] | None:
        """Append a ``kind=lifecycle`` event onto **one** session bound to
        ``connection_id`` (#1261), optionally waking it.

        The per-session-targeted sibling of :meth:`emit_lifecycle`: where that
        broadcasts a transport-down notice across every bound session, this
        targets a single ``session_id``. Used when a fact concerns one
        conversation, not the whole connection — e.g. a carrier-block /
        delivery failure that must reach the *originating* session
        (``event="connector_delivery_failed"``, ``wake=True``) rather than be
        broadcast.

        Set ``wake=True`` to make the failure wake the session (stimulus-
        bearing) instead of merely being visible on its next turn. Returns the
        appended-session payload, or ``None`` when the API returned a non-fatal
        4xx (matching :meth:`emit_inbound`/:meth:`emit_lifecycle`'s
        drop-don't-raise stance).
        """
        client = self._require_client()
        log.info(
            "connector.session_lifecycle",
            connector=self.connector,
            connection_id=connection_id,
            session_id=session_id,
            lifecycle_event=event,
            reason=reason,
            wake=wake,
        )
        body: dict[str, Any] = {
            "connection_id": connection_id,
            "session_id": session_id,
            "event": event,
            "wake": wake,
        }
        if reason is not None:
            body["reason"] = reason
        if data is not None:
            body["data"] = data
        response = await client.get_async_httpx_client().post(
            "/v1/connectors/runtime/session-lifecycle",
            json=body,
        )
        if response.is_error:
            sc = response.status_code
            log.warning(
                "connector.session_lifecycle.failed",
                status_code=sc,
                connection_id=connection_id,
                session_id=session_id,
                lifecycle_event=event,
                body=response.text[:2000],
            )
            if _is_fatal_inbound_status(sc):
                response.raise_for_status()
            return None
        return dict(response.json())

    async def emit_chat_lifecycle(
        self,
        *,
        connection_id: str,
        chat_id: str,
        event: str,
        reason: str | None = None,
        data: dict[str, Any] | None = None,
        wake: bool = False,
    ) -> dict[str, Any] | None:
        """Append a ``kind=lifecycle`` event onto the single session that
        ``chat_id`` resolves to on ``connection_id`` (#1260), optionally
        waking it.

        The routing-key sibling of :meth:`emit_session_lifecycle`: where that
        needs the resolved ``session_id``, this carries the connector's
        per-peer routing key (``chat_id``) and lets AIOS resolve it through
        the connection's per-chat binding. Use when a fact concerns one peer
        and the connector knows the routing key but not the session — e.g. a
        Twilio status callback that has the peer number (→ ``chat_id``) but
        not the AIOS ``session_id``.

        Set ``wake=True`` to make the failure wake the originating session
        (stimulus-bearing) instead of merely being visible on its next turn.
        Returns the appended-session payload (including the resolved
        ``session_id``), or ``None`` when the API returned a non-fatal 4xx —
        notably a ``404`` when the ``chat_id`` has no bound session on the
        connection (a correlation gap is dropped, not raised), matching
        :meth:`emit_inbound`/:meth:`emit_lifecycle`'s drop-don't-raise stance.
        """
        client = self._require_client()
        log.info(
            "connector.chat_lifecycle",
            connector=self.connector,
            connection_id=connection_id,
            chat_id=chat_id,
            lifecycle_event=event,
            reason=reason,
            wake=wake,
        )
        body: dict[str, Any] = {
            "connection_id": connection_id,
            "chat_id": chat_id,
            "event": event,
            "wake": wake,
        }
        if reason is not None:
            body["reason"] = reason
        if data is not None:
            body["data"] = data
        response = await client.get_async_httpx_client().post(
            "/v1/connectors/runtime/chat-lifecycle",
            json=body,
        )
        if response.is_error:
            sc = response.status_code
            log.warning(
                "connector.chat_lifecycle.failed",
                status_code=sc,
                connection_id=connection_id,
                chat_id=chat_id,
                lifecycle_event=event,
                body=response.text[:2000],
            )
            if _is_fatal_inbound_status(sc):
                response.raise_for_status()
            return None
        return dict(response.json())

    # ─── delivery / edit acks (#1341) ────────────────────────────────

    async def ack_delivered(
        self,
        *,
        connection_id: str,
        session_id: str,
        platform_message_id: str,
        tool_call_id: str | None = None,
        wake: bool = False,
    ) -> dict[str, Any] | None:
        """Tell the originating ``session_id`` that an outbound the model
        consciously sent was confirmed delivered by the platform (#1341).

        The success-path complement to a ``connector_delivery_failed`` emit: a
        thin wrapper over :meth:`emit_session_lifecycle` that supplies the
        reserved ``event="connector_message_delivered"`` and carries the
        platform correlation ids in ``data``. Informational by default
        (``wake=False``) — the model reads the ack on its next turn rather than
        being woken. Inherits the drop-don't-raise posture: a non-fatal 4xx →
        ``None``; a fatal status re-raises.
        """
        return await self.emit_session_lifecycle(
            connection_id=connection_id,
            session_id=session_id,
            event="connector_message_delivered",
            data={"platform_message_id": platform_message_id, "tool_call_id": tool_call_id},
            wake=wake,
        )

    async def ack_edited(
        self,
        *,
        connection_id: str,
        session_id: str,
        platform_message_id: str,
        tool_call_id: str | None = None,
        wake: bool = False,
    ) -> dict[str, Any] | None:
        """Tell the originating ``session_id`` that an edit the model made to a
        prior outbound landed on the platform (#1341).

        Identical to :meth:`ack_delivered` but supplies the reserved
        ``event="connector_message_edited"``. Informational by default
        (``wake=False``); the edit-FAILED case is expressed as
        ``connector_delivery_failed`` + ``wake=True`` (#1308), not here.
        """
        return await self.emit_session_lifecycle(
            connection_id=connection_id,
            session_id=session_id,
            event="connector_message_edited",
            data={"platform_message_id": platform_message_id, "tool_call_id": tool_call_id},
            wake=wake,
        )

    async def ack_delivered_chat(
        self,
        *,
        connection_id: str,
        chat_id: str,
        platform_message_id: str,
        tool_call_id: str | None = None,
        wake: bool = False,
    ) -> dict[str, Any] | None:
        """Routing-key (``chat_id``) variant of :meth:`ack_delivered` (#1341).

        A thin wrapper over :meth:`emit_chat_lifecycle` for connectors that
        know the per-peer routing key but not the resolved ``session_id`` (e.g.
        a Twilio status callback). A ``chat_id`` with no bound session 404s,
        dropped to ``None`` (drop-don't-raise).
        """
        return await self.emit_chat_lifecycle(
            connection_id=connection_id,
            chat_id=chat_id,
            event="connector_message_delivered",
            data={"platform_message_id": platform_message_id, "tool_call_id": tool_call_id},
            wake=wake,
        )

    async def ack_edited_chat(
        self,
        *,
        connection_id: str,
        chat_id: str,
        platform_message_id: str,
        tool_call_id: str | None = None,
        wake: bool = False,
    ) -> dict[str, Any] | None:
        """Routing-key (``chat_id``) variant of :meth:`ack_edited` (#1341)."""
        return await self.emit_chat_lifecycle(
            connection_id=connection_id,
            chat_id=chat_id,
            event="connector_message_edited",
            data={"platform_message_id": platform_message_id, "tool_call_id": tool_call_id},
            wake=wake,
        )

    # ─── runner ──────────────────────────────────────────────────────

    async def run_until_stopped(self, *, install_signal_handlers: bool = True) -> None:
        """Run the connector until SIGINT/SIGTERM (or cancelled).

        Wraps :meth:`run` with a process-level stop signal: a SIGTERM
        (the Docker stop signal) or SIGINT (Ctrl-C) cancels the run
        task cleanly so :meth:`teardown` fires.  ``asyncio.run`` traps
        SIGINT by default but **not** SIGTERM — without this handler,
        ``docker stop`` would kill Python before subclasses get a
        chance to clean up subprocesses, sockets, etc.  Pass
        ``install_signal_handlers=False`` from tests that drive
        cancellation directly.
        """
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        signals = (_signal.SIGINT, _signal.SIGTERM) if install_signal_handlers else ()
        for sig in signals:
            loop.add_signal_handler(sig, stop.set)
        run_task = asyncio.create_task(self.run(), name=f"aios-{self.connector}-run")
        stop_task = asyncio.create_task(stop.wait(), name=f"aios-{self.connector}-stop-wait")
        try:
            await asyncio.wait({run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            # Removing handlers in the finally lets embedded callers
            # invoke ``run_until_stopped`` multiple times without the
            # previous ``stop.set`` target outliving its Event.
            for sig in signals:
                with contextlib.suppress(NotImplementedError, ValueError):
                    loop.remove_signal_handler(sig)
            stop_task.cancel()
            if not run_task.done():
                run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await run_task

    async def run(self) -> None:
        """Open the client, publish tools, race discovery + tool loops."""
        async with Client(base_url=self._base_url, token=self._token) as client:
            self._client = client
            self._answered = await self.load_answered()
            await self._publish_tools_schema()
            try:
                async with asyncio.TaskGroup() as tg:
                    # ``setup()`` registers any container-wide long-running
                    # tasks under this TG so their crashes propagate and
                    # tear the container down, rather than silently
                    # stalling inbound delivery.
                    await self.setup(tg)
                    tg.create_task(self._discovery_loop(tg), name="aios-discovery")
                    tg.create_task(self._tool_loop(), name="aios-tool-loop")
                    tg.create_task(self._management_call_loop(), name="aios-management-loop")
                    self._ready_event.set()  # all background loops scheduled
            finally:
                self._ready_event.clear()
                self._all_loops_live.clear()
                self._loops_backfilled = 0
                # Reset each event rather than replacing the dict, so callers
                # holding existing Event references are not orphaned on re-run.
                for event in self._connection_served.values():
                    event.clear()
                await self.teardown()

    async def _publish_tools_schema(self) -> None:
        """Derive a ToolSpec from each ``@tool`` method and PUT the catalog.

        Replaces ``connectors.tools_schema`` for this type wholesale on
        every startup — the runtime container is the source of truth.
        """
        client = self._require_client()
        specs = [derive_tool_spec(name, meta.fn) for name, meta in self._tools.items()]
        body = ToolsSchemaUpdate(
            tools=[ToolsSchemaUpdateToolsItem.from_dict(spec) for spec in specs]
        )
        response = await _put_tools_schema(client=client, connector=self.connector, body=body)
        if response.status_code >= 400:
            raise RuntimeError(
                f"failed to publish tools schema: {response.status_code} {response.content!r}"
            )
        log.info(
            "connector.tools.published",
            connector=self.connector,
            tool_count=len(specs),
            tool_names=sorted(self._tools.keys()),
        )

    async def _discovery_loop(self, tg: asyncio.TaskGroup) -> None:
        """Tail the connection-discovery SSE; spawn / cancel workers."""
        client = self._require_client()
        backoff = 1.0
        _signalled_ready = False
        while True:
            try:
                async for msg in stream_connection_discovery(
                    client.get_async_httpx_client(), self.connector
                ):
                    if msg.event == "_open":
                        # SDK-synthetic marker: the SSE handshake succeeded
                        # (2xx headers received).  The server-side generator
                        # has begun executing; its ``listen_for_*`` context
                        # manager will register LISTEN within microseconds
                        # of the response body's first iteration.  Signal
                        # readiness once on the first successful connection.
                        #
                        # Note: we deliberately do NOT reset ``backoff`` on
                        # ``_open`` — the synthetic event fires before any
                        # server-side state is established, so a 2xx-then-
                        # immediate-disconnect loop should still back off
                        # exponentially.  Resetting only on real messages
                        # ensures CI-side flapping doesn't pin us at 1 s.
                        if not _signalled_ready:
                            _signalled_ready = True
                            self._mark_loop_backfilled()
                        continue
                    backoff = 1.0
                    if msg.event != "connection":
                        continue
                    payload = json.loads(msg.data)
                    event = payload.get("event")
                    connection_id = payload.get("connection_id", "")
                    external_account_id = payload.get("external_account_id", "")
                    if event == "added":
                        await self._on_connection_added(tg, connection_id, external_account_id)
                    elif event == "removed":
                        await self._on_connection_removed(connection_id)
            # Only retry on transport-level errors (timeouts, dropped
            # connections, server restarts). Application-level exceptions
            # — including secrets-fetch failures from
            # :meth:`_on_connection_added` and programmer bugs — propagate
            # to the TaskGroup so the operator sees the failure.
            except httpx.HTTPError as exc:
                log.warning(
                    "connector.discovery.stream_error",
                    error=str(exc),
                    backoff=backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    async def _on_connection_added(
        self, tg: asyncio.TaskGroup, connection_id: str, external_account_id: str
    ) -> None:
        """Fetch secrets, then spawn ``serve_connection`` in its own task.

        Transient 5xx raises :class:`httpx.HTTPStatusError` so the
        discovery loop's ``except httpx.HTTPError`` retries with
        backoff; a permanent 4xx or malformed body raises
        :class:`RuntimeError` so the container crashes and the operator
        sees the bug.
        """
        if connection_id in self._connections:
            # Replay from backfill after reconnect — already running.
            return
        client = self._require_client()
        response = await _get_runtime_secrets(client=client, connection_id=connection_id)
        if response.status_code >= 500:
            raise httpx.HTTPStatusError(
                f"secrets fetch 5xx for connection {connection_id!r}: "
                f"{response.status_code} {response.content!r}",
                request=httpx.Request("GET", "/v1/connectors/runtime/secrets"),
                response=httpx.Response(response.status_code, content=response.content),
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"secrets fetch 4xx for connection {connection_id!r}: "
                f"{response.status_code} {response.content!r}"
            )
        body = response.parsed
        if body is None or isinstance(body, HTTPValidationError):
            raise RuntimeError(f"unparseable secrets response for connection {connection_id!r}")
        raw_secrets = body.secrets
        if isinstance(raw_secrets, Unset):
            secrets_map: dict[str, str] = {}
        else:
            secrets_map = {str(k): str(v) for k, v in raw_secrets.additional_properties.items()}
        state = _ConnectionState(
            connection_id=connection_id,
            external_account_id=external_account_id,
            secrets=secrets_map,
        )
        state.worker = tg.create_task(
            self._isolated_serve_connection(connection_id, secrets_map),
            name=f"aios-conn-{connection_id}",
        )
        self._connection_served.setdefault(connection_id, asyncio.Event()).set()
        self._connections[connection_id] = state
        log.info(
            "connector.connection.added",
            connector=self.connector,
            connection_id=connection_id,
            external_account_id=external_account_id,
        )

    async def _isolated_serve_connection(self, connection_id: str, secrets: dict[str, str]) -> None:
        """Run :meth:`serve_connection` with failure-isolated cleanup.

        Catches any non-``CancelledError`` exception so a single bad
        connection (typo'd secret, unregistered phone, revoked bot
        token) can't tear down sibling connections via the parent
        TaskGroup.  Always pops the user-state slot on exit so
        connections cycling in/out don't leak stale state.

        On a terminal (non-cancel) failure this ALSO pops
        ``self._connections[connection_id]`` and clears the
        ``_connection_served`` event, so the connection is no longer
        treated as "already running": a subsequent discovery-backfill
        ``added`` (or a manual re-add) re-spawns the worker with freshly
        re-fetched secrets, rather than the connection becoming a
        permanent zombie that ignores every ``added`` while process-level
        health shows green (issue #1233).  A bounded exponential backoff
        (:meth:`_arm_reconnect_backoff`) gates how fast that re-spawn
        starts platform work, so a hard-failing connection backs off
        instead of hot-looping on every backfill replay.

        Cancellation (a ``removed`` event, or container shutdown) is NOT
        a failure: it re-raises so the TaskGroup sees a clean cancel,
        leaves ``_connections`` untouched here (``_on_connection_removed``
        already owns that pop), and does not arm the backoff.
        """
        # Re-spawn backoff: if this connection failed terminally on a
        # previous spawn, sleep before doing platform work so a
        # hard-failing connection doesn't hot-loop on every backfill
        # ``added``.  Sleeping inside the worker task (not the discovery
        # loop) keeps sibling connections' bring-up unblocked.  A
        # CancelledError mid-sleep (connection ``removed`` while backing
        # off) aborts the sleep and skips serve_connection cleanly.
        delay = self._reconnect_backoff.get(connection_id, 0.0)
        if delay:
            log.info(
                "connector.connection.reconnect_backoff",
                connector=self.connector,
                connection_id=connection_id,
                delay=delay,
            )
            await asyncio.sleep(delay)
        terminal_failure = False
        try:
            await self.serve_connection(connection_id, secrets)
        except asyncio.CancelledError:
            # Cooperative cancellation (removed / shutdown): re-raise for
            # a clean TaskGroup cancel and do NOT arm backoff or pop
            # ``_connections`` — that's _on_connection_removed's job.
            raise
        except Exception as exc:
            terminal_failure = True
            log.exception(
                "connector.connection.serve_failed",
                connector=self.connector,
                connection_id=connection_id,
                error=type(exc).__name__,
            )
        finally:
            self.state.pop(connection_id, None)
            if terminal_failure:
                self._arm_reconnect_backoff(connection_id)
                # Drop the live-connection slot so a later ``added``
                # re-spawns the worker instead of short-circuiting on
                # "already running".
                self._connections.pop(connection_id, None)
                if event := self._connection_served.get(connection_id):
                    event.clear()
                log.warning(
                    "connector.connection.worker_terminated",
                    connector=self.connector,
                    connection_id=connection_id,
                    reconnect_backoff=self._reconnect_backoff.get(connection_id),
                )
            else:
                # Clean exit (serve_connection returned, or was cancelled):
                # reset the failure backoff so the next bring-up is prompt.
                self._reconnect_backoff.pop(connection_id, None)

    def _arm_reconnect_backoff(self, connection_id: str) -> None:
        """Bump the per-connection re-spawn backoff after a terminal failure.

        Starts at :attr:`RECONNECT_BACKOFF_INITIAL` and doubles on each
        consecutive terminal failure up to :attr:`RECONNECT_BACKOFF_MAX`.
        Reset to absent by a clean serve / ``removed`` so an intermittent
        connection doesn't accumulate stale backoff.
        """
        current = self._reconnect_backoff.get(connection_id)
        if current is None:
            self._reconnect_backoff[connection_id] = self.RECONNECT_BACKOFF_INITIAL
        else:
            self._reconnect_backoff[connection_id] = min(current * 2, self.RECONNECT_BACKOFF_MAX)

    async def _on_connection_removed(self, connection_id: str) -> None:
        """Cancel the worker task for a vanished connection."""
        state = self._connections.pop(connection_id, None)
        if event := self._connection_served.get(connection_id):
            event.clear()
        # A genuine ``removed`` (operator deleted the connection) clears
        # any armed re-spawn backoff so a later re-add of the same id
        # starts fresh rather than inheriting a stale failure backoff.
        self._reconnect_backoff.pop(connection_id, None)
        if state is None or state.worker is None:
            return
        state.worker.cancel()
        log.info(
            "connector.connection.removed",
            connector=self.connector,
            connection_id=connection_id,
        )

    async def _tool_loop(self) -> None:
        """Tail the per-type calls SSE and dispatch each call."""
        client = self._require_client()
        backoff = 1.0
        _signalled_ready = False
        while True:
            try:
                async for msg in stream_connector_calls(
                    client.get_async_httpx_client(), self.connector
                ):
                    if msg.event == "_open":
                        # See ``_discovery_loop`` for the readiness +
                        # don't-reset-backoff rationale.
                        if not _signalled_ready:
                            _signalled_ready = True
                            self._mark_loop_backfilled()
                        continue
                    backoff = 1.0
                    if msg.event != "call":
                        continue
                    call = json.loads(msg.data)
                    tool_call_id = call.get("tool_call_id", "")
                    if not tool_call_id:
                        continue
                    if tool_call_id in self._answered:
                        # Already executed in a prior lifetime/connection.
                        # The side effect already happened, so re-POST the
                        # persisted result (healing a result-POST that failed
                        # after a successful platform send) instead of
                        # re-running the body.  No-op when nothing persisted.
                        await self._replay_tool_result(client, call)
                        continue
                    # ``dispatch_call`` marks the call answered (via
                    # ``_mark_answered``) *between* the side-effecting tool
                    # body and the result POST, so a POST failure can't
                    # re-run the body on replay.
                    await self.dispatch_call(call)
            # Same retry posture as ``_discovery_loop`` — only catch transport
            # errors; let ``_post_tool_result`` RuntimeErrors and other
            # application bugs propagate. A persistent 4xx/5xx on tool-result
            # POST would otherwise loop forever; but because the call is
            # marked answered *before* the POST, a reconnect replays via
            # ``_replay_tool_result`` (re-POST only) rather than re-running
            # the side-effecting body.
            except httpx.HTTPError as exc:
                log.warning(
                    "connector.tool_loop.stream_error",
                    error=str(exc),
                    backoff=backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    async def dispatch_call(self, call: dict[str, Any]) -> None:
        """Run the tool method for ``call`` and POST the result.

        Tool method signatures may accept any subset of the
        focal-injectable kwargs: ``connection_id``,
        ``external_account_id``, ``chat_id``.  ``connection_id`` comes
        from the call payload; ``external_account_id`` and ``chat_id``
        are parsed out of ``focal_channel``
        (``<connector>/<external_account_id>/<chat_id>``).

        A REQUIRED focal kwarg (no default) that stays unfilled after
        injection — e.g. ``chat_id`` on a send tool when the session's
        focal channel is empty/malformed/cleared mid-turn — fails loud
        with a typed ``{"code": "channel_unresolved"}`` error result
        instead of running the tool body against garbage state or
        letting it crash with an opaque exception (#1722).  A
        fire-and-forget tool (send/react) whose body then raises for
        any other reason is reported as ``{"code": "delivery_failed"}``
        so the model can distinguish a failed send from an ordinary
        tool bug and retry/re-target — never silently dropped.

        ``SandboxPath``-annotated params get their in-sandbox path
        strings resolved to host :class:`Path`s before the body runs.

        Public surface so tests can drive dispatch without going
        through SSE — the production path is :meth:`_tool_loop`.
        """
        client = self._require_client()
        name = call.get("name", "")
        tool_call_id = call.get("tool_call_id", "")
        session_id = call.get("session_id", "")
        connection_id = call.get("connection_id", "")
        # ``workspace_path`` must be an absolute string — anything else
        # (empty, whitespace, relative, wrong type) is treated as absent
        # and routed through the fail-closed branch below.  An absolute
        # path is the only shape the worker writes via ``insert_session``.
        raw_workspace_path = call.get("workspace_path")
        workspace_path: Path | None
        if isinstance(raw_workspace_path, str) and raw_workspace_path.startswith("/"):
            workspace_path = Path(raw_workspace_path)
        else:
            workspace_path = None
        log.info(
            "connector.tool_call.dispatched",
            name=name,
            tool_call_id=tool_call_id,
            session_id=session_id,
            connection_id=connection_id,
            workspace_path=str(workspace_path) if workspace_path else None,
        )
        meta = self._tools.get(name)
        if meta is None:
            log.warning(
                "connector.tool_call.failed",
                name=name,
                tool_call_id=tool_call_id,
                session_id=session_id,
                reason="unknown_tool",
            )
            error_content = json.dumps({"error": f"unknown tool {name!r}"})
            await self._mark_answered(tool_call_id, error_content)
            await self._post_tool_result(
                client,
                connection_id=connection_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                content=error_content,
                is_error=True,
            )
            return
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        focal_channel = call.get("focal_channel") or ""
        args = _inject_focal_kwargs(meta, args, focal_channel, connection_id)
        # #1722: a send (or any focal-required tool) whose channel can't be
        # resolved must fail loud, not silently no-op. ``_inject_focal_kwargs``
        # fills a required param only when it has a source (``connection_id``
        # from the call payload; ``chat_id``/``external_account_id`` parsed out
        # of ``focal_channel``) — a still-missing required focal param here
        # means the session's focal channel was empty/malformed/cleared
        # mid-turn (or ``connection_id`` itself was blank), and the model
        # never explicitly supplied it either. Without this check the tool
        # body would either KeyError on an empty ``self.state[""]`` lookup or
        # (worse) run with a garbage chat_id, both surfacing as opaque errors
        # at best and a vanished send at worst. Detect it BEFORE the body
        # runs and return a typed, retriable error instead.
        missing_focal = sorted(
            p for p in meta.required_focal_params if p not in args or not args[p]
        )
        if missing_focal:
            log.warning(
                "connector.tool_call.failed",
                name=name,
                tool_call_id=tool_call_id,
                session_id=session_id,
                reason="channel_unresolved",
                missing=missing_focal,
            )
            error_content = json.dumps(
                {
                    "error": (
                        "focal channel could not be resolved "
                        f"(missing: {', '.join(missing_focal)}) — the session's "
                        "focal channel may have been cleared or changed mid-turn; "
                        "switch_channel to a valid channel and retry"
                    ),
                    "code": "channel_unresolved",
                }
            )
            await self._mark_answered(tool_call_id, error_content)
            await self._post_tool_result(
                client,
                connection_id=connection_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                content=error_content,
                is_error=True,
            )
            return
        try:
            args = _resolve_sandbox_paths(
                meta, args, session_id=session_id, workspace_path=workspace_path
            )
        except SandboxPathError as exc:
            log.warning(
                "connector.tool_call.failed",
                name=name,
                tool_call_id=tool_call_id,
                session_id=session_id,
                reason="sandbox_path",
                error=str(exc),
            )
            error_content = json.dumps({"error": str(exc)})
            await self._mark_answered(tool_call_id, error_content)
            await self._post_tool_result(
                client,
                connection_id=connection_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                content=error_content,
                is_error=True,
            )
            return
        try:
            result = await meta.fn(**args)
        except Exception as exc:
            # Tool-call SSE backfill can deliver a call for ``connection_id``
            # before the connection-discovery SSE has finished registering
            # the connection's state in the connector's ``state`` dict
            # (the two streams are independent).  The tool method's
            # ``self.state[connection_id]`` lookup KeyErrors and the
            # raw ``str(KeyError("'conn_01...'"))`` reaches the model as an
            # opaque quoted ID with no indication that the connection just
            # wasn't ready yet.  Detect that shape and produce an
            # interpretable error so the model retries sensibly instead of
            # guessing at "is `'conn_01...'` an arg I should pass?".
            error_payload: dict[str, Any]
            if (
                isinstance(exc, KeyError)
                and len(exc.args) == 1
                and isinstance(exc.args[0], str)
                and exc.args[0] == connection_id
            ):
                error_payload = {
                    "error": "connection not yet active; retry shortly",
                    "connection_id": connection_id,
                }
                reason = "connection_state_race"
            elif meta.fire_and_forget:
                # #1722: a fire-and-forget tool (a send/react — the turn is
                # "over" once it runs) that raises mid-dispatch is a connector-side
                # delivery failure, not a generic tool bug: the model believed
                # (or was about to believe) the message was on its way. Typed
                # ``delivery_failed`` so the model can distinguish "my send
                # attempt itself errored" from an ordinary tool exception and
                # decide whether to retry or re-target — this exception branch
                # always sets ``is_error=True`` below, so a failed send is a
                # clear, typed result the session wakes to react to.
                error_payload = {"error": str(exc), "code": "delivery_failed"}
                reason = "delivery_failed"
            else:
                error_payload = {"error": str(exc)}
                reason = "tool_exception"
            log.warning(
                "connector.tool_call.failed",
                name=name,
                tool_call_id=tool_call_id,
                session_id=session_id,
                reason=reason,
                error=str(exc),
            )
            error_content = json.dumps(error_payload)
            await self._mark_answered(tool_call_id, error_content)
            await self._post_tool_result(
                client,
                connection_id=connection_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                content=error_content,
                is_error=True,
            )
            return
        if not isinstance(result, str | list):
            result = json.dumps(result)
        # Persist the (side-effecting) tool result BEFORE the result POST.
        # A list result is serialized to JSON for persistence; it is still
        # POSTed in its native list shape below.
        serialized_result = result if isinstance(result, str) else json.dumps(result)
        await self._mark_answered(tool_call_id, serialized_result)
        # Every tool result is a stimulus (#1919): POST the successful result and
        # let aios wake the session to react to the completion — including a bare
        # delivery ack, which the model may simply acknowledge and end the turn.
        await self._post_tool_result(
            client,
            connection_id=connection_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            content=result,
        )
        log.info(
            "connector.tool_call.completed",
            name=name,
            tool_call_id=tool_call_id,
            session_id=session_id,
            is_error=False,
        )

    async def _management_call_loop(self) -> None:
        """Tail the per-type management-calls SSE and dispatch each call.

        Shares ``self._answered`` with :meth:`_tool_loop`; ``mgmt_*`` and
        ``call_*`` ULIDs are namespace-disjoint.

        Unlike :meth:`_tool_loop`, a post-result POST failure does NOT
        propagate.  Management handlers have real side effects (signal-cli's
        ``register`` dispatches an SMS); a wedge-and-restart loop would
        amplify those side effects on every SSE backfill replay.  We mark
        the call answered regardless of POST outcome and continue; the
        audit row stays ``pending`` and the operator times out — visible
        failure beats silent SMS-storm.
        """
        client = self._require_client()
        backoff = 1.0
        _signalled_ready = False
        while True:
            try:
                async for msg in stream_management_calls(
                    client.get_async_httpx_client(), self.connector
                ):
                    if msg.event == "_open":
                        # See ``_discovery_loop`` for the readiness +
                        # don't-reset-backoff rationale.
                        if not _signalled_ready:
                            _signalled_ready = True
                            self._mark_loop_backfilled()
                        continue
                    backoff = 1.0
                    if msg.event != "call":
                        continue
                    call = json.loads(msg.data)
                    call_id = call.get("call_id", "")
                    if not call_id or call_id in self._answered:
                        continue
                    try:
                        await self.dispatch_management_call(call)
                    except Exception:
                        log.exception(
                            "connector.management_call.dispatch_failed",
                            call_id=call_id,
                        )
                    # Management calls persist no replayable result; ``None``
                    # keeps the existing skip-on-replay semantics.
                    await self._mark_answered(call_id, None)
            except httpx.HTTPError as exc:
                log.warning(
                    "connector.management_loop.stream_error",
                    error=str(exc),
                    backoff=backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    async def dispatch_management_call(self, call: dict[str, Any]) -> None:
        """Run the management handler for ``call`` and POST the result.

        Public so tests can drive dispatch without SSE.
        """
        client = self._require_client()
        call_id = call.get("call_id", "")
        method = call.get("method", "")
        log.info(
            "connector.management_call.dispatched",
            connector=self.connector,
            method=method,
            call_id=call_id,
        )
        meta = self._management.get(method)
        if meta is None:
            log.warning(
                "connector.management_call.failed",
                connector=self.connector,
                method=method,
                call_id=call_id,
                reason="unknown_method",
            )
            await self._post_management_call_result(
                client,
                call_id=call_id,
                result={"error": f"unknown management method {method!r}"},
                is_error=True,
            )
            return
        params = call.get("params") or {}
        try:
            result = await meta.fn(**params)
        except ManagementHandlerError as exc:
            log.info(
                "connector.management_call.structured_error",
                connector=self.connector,
                method=method,
                call_id=call_id,
                payload=exc.payload,
            )
            await self._post_management_call_result(
                client, call_id=call_id, result=exc.payload, is_error=True
            )
            return
        except Exception as exc:
            log.warning(
                "connector.management_call.failed",
                connector=self.connector,
                method=method,
                call_id=call_id,
                reason="handler_exception",
                error=str(exc),
            )
            await self._post_management_call_result(
                client,
                call_id=call_id,
                result={"error": str(exc)},
                is_error=True,
            )
            return
        await self._post_management_call_result(
            client, call_id=call_id, result=result, is_error=False
        )
        log.info(
            "connector.management_call.completed",
            connector=self.connector,
            method=method,
            call_id=call_id,
        )

    @staticmethod
    async def _post_management_call_result(
        client: Client,
        *,
        call_id: str,
        result: Any,
        is_error: bool,
    ) -> None:
        body = RuntimeManagementCallResultRequest(
            call_id=call_id,
            result=result,
            is_error=is_error,
        )
        response = await _post_runtime_management_call_result(client=client, body=body)
        if response.status_code >= 400:
            raise RuntimeError(
                f"management-call result POST failed: {response.status_code} {response.content!r}"
            )

    @staticmethod
    async def _post_tool_result(
        client: Client,
        *,
        connection_id: str,
        session_id: str,
        tool_call_id: str,
        content: str | list[dict[str, Any]],
        is_error: bool = False,
    ) -> None:
        """POST one tool result via the generated runtime op."""
        # The generated model's list branch is typed as
        # ``list[RuntimeToolResultRequestContentType1Item]``; that item type is
        # a thin ``additional_properties`` bag whose ``from_dict``/``to_dict``
        # round-trips a plain dict unchanged, so adapting our ``list[dict]``
        # here keeps the wire payload byte-identical while satisfying the type.
        body_content: list[RuntimeToolResultRequestContentType1Item] | str = (
            [RuntimeToolResultRequestContentType1Item.from_dict(item) for item in content]
            if isinstance(content, list)
            else content
        )
        body = RuntimeToolResultRequest(
            connection_id=connection_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            content=body_content,
            is_error=is_error,
        )
        response = await _post_runtime_tool_result(client=client, body=body)
        if response.status_code >= 400:
            raise RuntimeError(
                f"tool result POST failed: {response.status_code} {response.content!r}"
            )

    def _require_client(self) -> Client:
        if self._client is None:
            raise RuntimeError("connector client not opened — call run() first")
        return self._client

    def _collect_tools(self) -> dict[str, _ToolMeta]:
        """Walk ``self``, pick up every method tagged with @tool."""
        out: dict[str, _ToolMeta] = {}
        for _, member in inspect.getmembers(self):
            if not (callable(member) and hasattr(member, _TOOL_ATTR)):
                continue
            if hasattr(member, _MGMT_ATTR):
                raise RuntimeError(
                    f"method {member.__qualname__!r} carries both @tool and "
                    "@management_handler; pick one — tools are model-callable, "
                    "management handlers are operator-callable"
                )
            tool_name: str = getattr(member, _TOOL_ATTR)
            out[tool_name] = _build_tool_meta(member)
        return out

    def _collect_management_handlers(self) -> dict[str, _ManagementHandlerMeta]:
        """Walk ``self``, pick up every method tagged with @management_handler."""
        out: dict[str, _ManagementHandlerMeta] = {}
        for _, member in inspect.getmembers(self):
            if not (callable(member) and hasattr(member, _MGMT_ATTR)):
                continue
            method_name: str = getattr(member, _MGMT_ATTR)
            out[method_name] = _ManagementHandlerMeta(fn=member)
        return out


_FOCAL_INJECTABLE: frozenset[str] = frozenset({"connection_id", "external_account_id", "chat_id"})


def _build_tool_meta(fn: ToolFn) -> _ToolMeta:
    """Inspect ``fn`` once and freeze the dispatch-time metadata."""
    sig = inspect.signature(fn)
    focal = frozenset(set(sig.parameters) & _FOCAL_INJECTABLE)
    required_focal = frozenset(
        name for name in focal if sig.parameters[name].default is inspect.Parameter.empty
    )
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}
    sandbox: list[tuple[str, str]] = []
    for param_name, hint in hints.items():
        kind = _sandbox_path_kind(hint)
        if kind is not None:
            sandbox.append((param_name, kind))
    fire_and_forget = bool(getattr(fn, _TOOL_FAF_ATTR, False))
    return _ToolMeta(
        fn=fn,
        focal_params=focal,
        required_focal_params=required_focal,
        sandbox_params=tuple(sandbox),
        fire_and_forget=fire_and_forget,
    )


class SandboxPathError(ValueError):
    """Raised by the dispatcher when a SandboxPath argument escapes
    ``/workspace/`` or ``/mnt/attachments/``.

    Surfaces to the model as a tool error result so it can self-correct
    on the next turn (e.g. retry with a path inside ``/workspace/``).
    """


def _resolve_sandbox_paths(
    meta: _ToolMeta,
    args: dict[str, Any],
    *,
    session_id: str,
    workspace_path: Path | None,
) -> dict[str, Any]:
    """Translate ``SandboxPath`` argument strings into host :class:`Path`s."""
    if not meta.sandbox_params or not session_id:
        return args
    out = dict(args)
    for param_name, kind in meta.sandbox_params:
        if param_name not in out:
            continue
        value = out[param_name]
        if kind == "scalar":
            out[param_name] = _resolve_one(
                value, session_id=session_id, workspace_path=workspace_path, name=param_name
            )
        else:
            if not isinstance(value, list):
                raise SandboxPathError(f"{param_name!r} must be a list of in-sandbox paths")
            out[param_name] = [
                _resolve_one(
                    v, session_id=session_id, workspace_path=workspace_path, name=param_name
                )
                for v in value
            ]
    return out


def _resolve_one(value: Any, *, session_id: str, workspace_path: Path | None, name: str) -> Path:
    if not isinstance(value, str):
        raise SandboxPathError(f"{name!r} must be an in-sandbox path string; got {value!r}")
    resolved = resolve_sandbox_path(
        session_id=session_id, sandbox_path=value, workspace_path=workspace_path
    )
    if resolved is None:
        # The resolver returns None for two distinct reasons: the path
        # really is outside the bind-mounted prefixes (model picked a
        # wrong prefix — model-actionable), or the path is under
        # ``/workspace`` but the runtime didn't supply a
        # ``workspace_path`` (operator/transport bug — not
        # model-actionable).  Disambiguate so the model isn't told its
        # prefix is wrong when the real issue is upstream.
        if workspace_path is None and (value == "/workspace" or value.startswith("/workspace/")):
            log.error(
                "connector.tool_call.missing_workspace_path",
                name=name,
                value=value,
                session_id=session_id,
            )
            raise SandboxPathError(f"{name!r}: workspace bind-mount unavailable for this call")
        raise SandboxPathError(
            f"{name!r}: path {value!r} is outside the connector-readable mounts. "
            f"The connector can only read files under /workspace/ or /mnt/attachments/; "
            f"files written elsewhere in the sandbox (e.g. /tmp) are invisible to it. "
            f"Copy the file into /workspace/ and resend with that /workspace/ path."
        )
    return resolved


def _sandbox_path_kind(hint: Any) -> str | None:
    """Return ``'scalar'`` / ``'list'`` / ``None`` for a type hint."""
    inner = _strip_optional(hint)
    if _is_sandbox_path(inner):
        return "scalar"
    origin = get_origin(inner)
    if origin in (list, tuple):
        elems = get_args(inner)
        if elems and _is_sandbox_path(elems[0]):
            return "list"
    return None


def _strip_optional(hint: Any) -> Any:
    """``T | None`` → ``T``; everything else passes through unchanged."""
    origin = get_origin(hint)
    if origin not in (Union, types.UnionType):
        return hint
    args = [a for a in get_args(hint) if a is not type(None)]
    return args[0] if len(args) == 1 else hint


def _is_sandbox_path(hint: Any) -> bool:
    """Detect ``Annotated[Path, SANDBOX_PATH_MARKER]`` regardless of nesting."""
    return any(isinstance(meta, _SandboxPathMarker) for meta in getattr(hint, "__metadata__", ()))


def _inject_focal_kwargs(
    meta: _ToolMeta,
    args: dict[str, Any],
    focal_channel: str,
    connection_id: str,
) -> dict[str, Any]:
    """Inject ``connection_id`` / ``external_account_id`` / ``chat_id`` kwargs.

    The runtime SSE payload includes ``connection_id`` directly;
    ``external_account_id`` and ``chat_id`` are parsed out of
    ``focal_channel`` (``<connector>/<external_account_id>/<chat_id>``).
    Each is injected only when the tool's signature accepts it AND the
    caller didn't already pass it explicitly.
    """
    out = dict(args)
    if connection_id and "connection_id" in meta.focal_params and "connection_id" not in out:
        out["connection_id"] = connection_id
    if focal_channel and meta.focal_params & {"external_account_id", "chat_id"}:
        parts = focal_channel.split("/", 2)
        if len(parts) >= 3:
            _connector, external_account_id, chat_id = parts
            if "external_account_id" in meta.focal_params and "external_account_id" not in out:
                out["external_account_id"] = external_account_id
            if "chat_id" in meta.focal_params and "chat_id" not in out:
                out["chat_id"] = chat_id
    return out
