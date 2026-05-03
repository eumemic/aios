"""Reference :class:`Connector` base class for aios connectors.

Pairs with :mod:`aios.harness.connector_supervisor` over stdio.  Subclass,
declare tools with :func:`tool` (or :func:`focal_required` to pull the
chat-id from MCP ``_meta``), call :meth:`emit_inbound` to deliver
inbound messages.  See :doc:`/packages/aios-connector/README` and the
echo example for the full contract.

The class wires three bits the lowlevel MCP SDK leaves to integrators:

* **Spool durability.**  :meth:`emit_inbound` writes to the SQLite
  spool BEFORE pushing the notification, then replays unacked entries
  on every subprocess startup.  The ``aios_inbound_ack`` tool, registered
  automatically by this class, deletes spool rows by ``event_id``.
* **Readiness deferral.**  ``setup()`` and ``discover_accounts()`` run
  BEFORE :meth:`mcp.server.lowlevel.Server.run` is started.  The
  supervisor's bounded ``initialize()`` handshake (30s ceiling per
  ``aios.mcp.stdio_transport._INIT_TIMEOUT_S``) catches connectors that
  hang in setup.  signal-cli's 5+ second startup fits comfortably.
* **Aios notifications.**  ``notifications/aios/accounts`` and
  ``notifications/aios/inbound`` go directly onto the write stream as
  :class:`mcp.types.JSONRPCNotification` — the SDK's ``ServerNotification``
  union is closed and rejects custom methods.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, get_type_hints

import anyio
import anyio.abc
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage
from mcp.types import (
    JSONRPCMessage,
    JSONRPCNotification,
    RequestParams,
    TextContent,
    Tool,
)
from ulid import ULID

from aios_connector.spool import SOFT_WARN_BYTES, Spool

# Method names the connector emits over stdio.  Mirrors the constant in
# ``aios.mcp.stdio_transport`` — duplicated rather than imported so the
# reference SDK has zero dependency on aios server code.  If the prefix
# ever changes both ends must update.
_AIOS_NOTIFICATION_PREFIX = "notifications/aios/"
_ACCOUNTS_METHOD = f"{_AIOS_NOTIFICATION_PREFIX}accounts"
_INBOUND_METHOD = f"{_AIOS_NOTIFICATION_PREFIX}inbound"

# MCP ``_meta`` key the harness stamps for outbound tool calls when
# the calling session has a focal channel set.  Value is the channel
# address with its ``<connector>/<account>`` prefix stripped.
_FOCAL_CHANNEL_META_KEY = "aios.focal_channel_path"

ToolFn = Callable[..., Awaitable[Any]]


@dataclass
class ToolDescriptor:
    """Metadata for a tool registered via :func:`tool`.

    Constructed by the decorator at class-body time, attached to the
    method so :meth:`Connector._collect_tools` can find it during
    ``__init__``.  ``input_schema`` is the JSON Schema published in the
    ``tools/list`` response; the SDK builds it from the function's
    type hints (Python types → JSON Schema primitives).
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    fn: ToolFn
    focal_required: bool = False


_TOOL_ATTR = "__aios_connector_tool__"


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[ToolFn], ToolFn]:
    """Decorator marking a coroutine method as an MCP tool.

    The method's name (or ``name=`` override) becomes the published
    tool name; its docstring (or ``description=`` override) becomes
    the description; its parameter type hints become the JSON Schema
    properties.  Self isn't included in the schema.

    Stack with :func:`focal_required` to inject the chat-id parsed
    from ``_meta.aios.focal_channel_path``::

        @tool
        @focal_required
        async def signal_send(self, text: str, *, focal: str) -> dict[str, Any]: ...
    """

    def decorate(fn: ToolFn) -> ToolFn:
        tool_name = name or fn.__name__
        doc = (fn.__doc__ or "").strip()
        if not doc and description is None:
            raise ValueError(
                f"@tool {tool_name!r} needs a docstring or explicit description= "
                "(MCP exposes the description to the model)"
            )
        schema = _build_input_schema(fn)
        descriptor = ToolDescriptor(
            name=tool_name,
            description=description or doc,
            input_schema=schema,
            fn=fn,
            focal_required=getattr(fn, "__aios_focal_required__", False),
        )
        setattr(fn, _TOOL_ATTR, descriptor)
        return fn

    return decorate


def focal_required(fn: ToolFn) -> ToolFn:
    """Decorator marking a tool as requiring a focal channel.

    The MCP request's ``_meta.aios.focal_channel_path`` is parsed and
    passed as the ``focal`` keyword argument (a string — the chat-id /
    suffix the harness stamps).  Calls without focal raise so the
    model sees a tool error rather than a silent fallback.

    Apply BELOW :func:`tool` so the descriptor's ``focal_required``
    flag is set before the tool registration reads it.
    """
    fn.__aios_focal_required__ = True  # type: ignore[attr-defined]
    return fn


def _build_input_schema(fn: ToolFn) -> dict[str, Any]:
    """Generate a JSON Schema for ``fn``'s non-self parameters.

    Walks the resolved type hints (so ``from __future__ import
    annotations`` callers still get usable schemas) and maps Python
    primitives to the JSON Schema equivalents.  Optional parameters
    (defaulting to anything) are non-required; the rest are required.

    ``focal`` is a SDK-injected kwarg — not a model-visible argument —
    so it's excluded from the schema regardless of typing.
    """
    import inspect

    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        # Forward references that can't resolve in a partially-built
        # module — degrade to "any" for those params rather than fail
        # registration.  Real connectors should make hints resolvable.
        hints = {}

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        if param_name == "self" or param_name == "focal":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        properties[param_name] = _hint_to_schema(hints.get(param_name))
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _hint_to_schema(hint: Any) -> dict[str, Any]:
    """Map a Python type hint to a JSON Schema fragment."""
    if hint is None or hint is type(None):
        return {"type": "null"}
    if hint is str:
        return {"type": "string"}
    if hint is int:
        return {"type": "integer"}
    if hint is float:
        return {"type": "number"}
    if hint is bool:
        return {"type": "boolean"}
    if hint is list:
        return {"type": "array"}
    if hint is dict:
        return {"type": "object"}
    # Best effort for parameterized generics: ``list[str]`` etc.  The
    # SDK keeps this minimal; connector authors who need richer schemas
    # should pass description= on @tool or supply schemas via a
    # follow-up enhancement.
    origin = getattr(hint, "__origin__", None)
    if origin is list:
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    return {}


@dataclass
class _Account:
    """A connector account snapshot entry.

    Shape is opaque to aios — the harness just forwards the dict — so
    the dataclass is internal; serialization is via :meth:`as_dict`.
    """

    id: str
    display_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "display_name": self.display_name}
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out


def make_account(
    id: str, display_name: str, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Convenience builder so connector code doesn't reach into ``_Account``."""
    return _Account(id=id, display_name=display_name, metadata=metadata or {}).as_dict()


class Connector:
    """Base class connector authors subclass.

    Lifecycle (driven by :meth:`run`):

    1. ``setup()`` — open daemon connections, sign in, etc.
    2. ``discover_accounts()`` — initial account snapshot.
    3. Start MCP server over stdio.
    4. Send ``notifications/aios/accounts`` once the client is initialized.
    5. Replay unacked spool entries, then enter live mode.
    6. ``teardown()`` on exit (normal or exception).

    Subclass contract:

    * Override ``name`` (e.g. ``"signal"``); the spool path and outbound
      channel-address prefix derive from it.
    * Override :meth:`setup`, :meth:`discover_accounts`, :meth:`teardown`.
    * Decorate methods with :func:`tool` (and optionally
      :func:`focal_required`) to publish tools.
    * Call :meth:`emit_inbound` from your inbound pump.
    * Optionally set ``instructions`` for a class-level
      :class:`mcp.types.InitializeResult.instructions` block.
    """

    name: ClassVar[str] = ""
    instructions: ClassVar[str | None] = None
    version: ClassVar[str] = "0.0.0"

    def __init__(self, *, spool_dir: Path | None = None) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__}.name must be set")
        # The supervisor cd's into ``~/.aios/connectors/<name>/`` before
        # spawning the subprocess (see ``resolve_connector_specs``); the
        # default spool path is therefore ``./spool.sqlite`` from the
        # connector's perspective.  Override via ``spool_dir=`` for tests.
        self._spool_dir = (spool_dir or Path.cwd()).expanduser()
        self._spool: Spool = Spool(self._spool_dir / "spool.sqlite")
        self._tools: list[ToolDescriptor] = self._collect_tools()
        self._accounts_payload: dict[str, Any] = {"accounts": []}
        self._write_stream: anyio.abc.ObjectSendStream[SessionMessage] | None = None
        self._client_initialized: anyio.Event = anyio.Event()

    # ── Subclass hooks ────────────────────────────────────────────────

    async def setup(self) -> None:
        """Open daemon connections, register listeners, etc.

        Runs before the MCP server is started.  Blocking here keeps the
        supervisor in ``starting`` until either this returns or its
        bounded init handshake (30s) trips and respawns.
        """

    async def discover_accounts(self) -> list[dict[str, Any]]:
        """Return the initial account snapshot.

        Called once between :meth:`setup` and the MCP server start.
        Most connectors maintain a single account (signal phone,
        telegram bot) and return a one-element list.  Use
        :func:`make_account` to build entries.
        """
        return []

    async def teardown(self) -> None:
        """Tear down resources opened in :meth:`setup`.

        Runs in the ``finally`` of :meth:`run`, so always fires —
        normal exit, exception, cancellation.
        """

    async def serve(self) -> None:
        """Long-running task siblinged with the MCP server.

        Override to drive an inbound pump or other background work
        (e.g. signal-cli's listener loop).  Cancelled when the MCP
        server exits.  Default implementation parks forever — most
        connectors don't need this hook because their inbound source
        is event-driven from a daemon callback.
        """
        forever: anyio.Event = anyio.Event()
        await forever.wait()

    # ── Public API ────────────────────────────────────────────────────

    async def emit_inbound(
        self,
        *,
        account: str,
        chat_id: str,
        sender: dict[str, Any],
        content: str,
        attachments: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist an inbound message to the spool, then push the notification.

        Returns the ULID assigned as ``event_id``.  Order matters: the
        spool write happens first so a crash between ``add()`` and the
        stdio write replays the entry on reconnect.

        ``account`` is one of the ids from :meth:`discover_accounts`'
        return value.  ``chat_id`` is the connector-specific identifier
        the model addresses via focal channel; the harness builds the
        full channel address as ``<connector>/<account>/<chat_id>``.
        """
        if self._write_stream is None:
            raise RuntimeError(
                "emit_inbound called before the MCP server started; "
                "call from setup() / live operation, not __init__"
            )

        event_id = str(ULID())
        params: dict[str, Any] = {
            "event_id": event_id,
            "account": account,
            "chat_id": chat_id,
            "sender": dict(sender),
            "content": content,
        }
        if attachments:
            params["attachments"] = list(attachments)
        if metadata:
            params["metadata"] = dict(metadata)

        payload = json.dumps(params).encode("utf-8")
        self._spool.add(event_id, payload, created_at=time.time())
        self._maybe_warn_spool_growth()

        await self._send_aios_notification(_INBOUND_METHOD, params)
        return event_id

    async def update_accounts(self, accounts: list[dict[str, Any]]) -> None:
        """Replace the snapshot and push the new payload to aios.

        Pre-init updates are absorbed: the snapshot is stamped now and
        sent once :meth:`_emit_initial_state` runs after the client
        sends ``notifications/initialized``.  Post-init updates fire
        ``notifications/aios/accounts`` immediately so the supervisor's
        in-memory snapshot stays current.
        """
        self._accounts_payload = {"accounts": list(accounts)}
        if self._write_stream is not None and self._client_initialized.is_set():
            await self._send_aios_notification(_ACCOUNTS_METHOD, self._accounts_payload)

    # ── Run loop ──────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main entry point.  Subclass entry-point factories call this.

        Runs ``setup()`` → ``discover_accounts()`` BEFORE the MCP server
        starts so a stuck setup trips the supervisor's bounded init
        handshake (30s) rather than parking the supervisor forever.
        """
        await self.setup()
        try:
            self._accounts_payload = {"accounts": list(await self.discover_accounts())}
            server = self._build_server()
            init_opts = server.create_initialization_options(
                notification_options=NotificationOptions(),
                experimental_capabilities={"aios/connector": {}},
            )
            async with (
                stdio_server() as (read_stream, write_stream),
                anyio.create_task_group() as tg,
            ):
                self._write_stream = write_stream
                tg.start_soon(self._emit_initial_state)
                tg.start_soon(self.serve)
                await server.run(read_stream, write_stream, init_opts)
                tg.cancel_scope.cancel()
        finally:
            self._write_stream = None
            await self.teardown()

    # ── Internal: tool collection ─────────────────────────────────────

    def _collect_tools(self) -> list[ToolDescriptor]:
        """Walk the class hierarchy for methods marked by :func:`tool`."""
        seen: dict[str, ToolDescriptor] = {}
        for klass in type(self).__mro__:
            for attr in vars(klass).values():
                descriptor = getattr(attr, _TOOL_ATTR, None)
                if descriptor is None:
                    continue
                if descriptor.name in seen:
                    continue
                seen[descriptor.name] = descriptor
        return list(seen.values())

    # ── Internal: server build ────────────────────────────────────────

    def _build_server(self) -> Server[Any, Any]:
        """Construct the MCP lowlevel server with our tools.

        Two built-ins coexist with the connector-author tools:

        * Subclass tools (decorated with :func:`tool`).
        * ``aios_inbound_ack`` — the worker calls this after a
          successful event append to clear the spool.  Registered here
          so connector authors can't accidentally shadow it.
        """
        server: Server[Any, Any] = Server(
            self.name,
            version=self.version,
            instructions=self.instructions,
        )

        @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
        async def list_tools() -> list[Tool]:
            tools: list[Tool] = [
                Tool(
                    name=descriptor.name,
                    description=descriptor.description,
                    inputSchema=descriptor.input_schema,
                )
                for descriptor in self._tools
            ]
            tools.append(_aios_inbound_ack_descriptor())
            return tools

        @server.call_tool()  # type: ignore[untyped-decorator]
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            if name == "aios_inbound_ack":
                return await self._handle_aios_inbound_ack(arguments)
            descriptor = next((d for d in self._tools if d.name == name), None)
            if descriptor is None:
                raise ValueError(f"unknown tool: {name}")
            return await self._invoke_tool(descriptor, arguments)

        # Hook the InitializedNotification: once the client confirms
        # initialization, it's safe to push the accounts snapshot and
        # replay the spool.  ``server.notification_handlers`` is a
        # plain dict keyed by notification class.
        from mcp.types import InitializedNotification

        async def on_initialized(_notification: InitializedNotification) -> None:
            self._client_initialized.set()

        server.notification_handlers[InitializedNotification] = on_initialized
        return server

    async def _invoke_tool(
        self, descriptor: ToolDescriptor, arguments: dict[str, Any]
    ) -> list[TextContent]:
        """Dispatch a tool, parsing focal meta if the descriptor opted in.

        :func:`focal_required` tools fail loudly when the meta key is
        missing — the agent shouldn't reach a focal-required tool
        without focal set, so any absence is a bug to surface, not a
        condition to fall back on.
        """
        kwargs = dict(arguments)
        if descriptor.focal_required:
            focal = self._focal_from_request_meta()
            if focal is None:
                raise ValueError(
                    f"tool {descriptor.name!r} requires a focal channel — "
                    "aios should inject _meta.aios.focal_channel_path"
                )
            kwargs["focal"] = focal

        result = await descriptor.fn(self, **kwargs)
        if isinstance(result, list) and all(isinstance(x, TextContent) for x in result):
            return result
        if isinstance(result, str):
            return [TextContent(type="text", text=result)]
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

    def _focal_from_request_meta(self) -> str | None:
        """Read ``_meta.aios.focal_channel_path`` from the active request.

        The lowlevel server stashes the raw request on a context var
        the call_tool handler can reach via :attr:`Server.request_context`.
        Missing or malformed meta returns ``None`` so the caller raises
        a tool-level error.
        """
        from mcp.server.lowlevel.server import request_ctx

        try:
            ctx = request_ctx.get()
        except LookupError:
            return None
        meta: RequestParams.Meta | None = getattr(ctx.request, "meta", None) if ctx else None
        if meta is None:
            return None
        extra = getattr(meta, "model_extra", None) or {}
        path: Any = extra.get(_FOCAL_CHANNEL_META_KEY)
        if not isinstance(path, str) or not path:
            return None
        return path

    # ── Internal: aios_inbound_ack tool ───────────────────────────────

    async def _handle_aios_inbound_ack(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Delete a spool entry by ``event_id``. Idempotent."""
        event_id = arguments.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("aios_inbound_ack requires a string event_id argument")
        removed = self._spool.ack(event_id)
        return [TextContent(type="text", text=json.dumps({"removed": removed}))]

    # ── Internal: aios notifications ──────────────────────────────────

    async def _send_aios_notification(self, method: str, params: dict[str, Any]) -> None:
        """Push an aios-namespaced notification onto the write stream.

        Bypasses :meth:`mcp.server.session.BaseSession.send_notification`
        because its ``ServerNotification`` union is closed — unknown
        methods raise pydantic validation errors.  The supervisor's
        splitter task picks these off the stream before the SDK's
        receive validation sees them.

        Callers MUST ensure ``self._write_stream`` is non-None
        (``emit_inbound`` raises pre-init; ``update_accounts`` and
        ``_emit_initial_state`` gate on the readiness event) — no
        defensive guard here, an AttributeError is the right failure
        mode for a code path that escaped the gates.
        """
        message = JSONRPCMessage(
            JSONRPCNotification(
                jsonrpc="2.0",
                method=method,
                params=dict(params),
            )
        )
        assert self._write_stream is not None
        await self._write_stream.send(SessionMessage(message=message))

    async def _emit_initial_state(self) -> None:
        """Wait for ``notifications/initialized``, then send accounts + replay spool.

        Runs as a sibling task of :meth:`mcp.server.lowlevel.Server.run`
        inside :meth:`run`.  The wait avoids a race where the
        supervisor's splitter forwards an early aios notification before
        the client's session has registered its message handlers.
        """
        await self._client_initialized.wait()
        await self._send_aios_notification(_ACCOUNTS_METHOD, self._accounts_payload)
        for event_id, payload in self._spool.unacked():
            params = json.loads(payload.decode("utf-8"))
            # Authoritative source of truth for the event_id is the
            # spool key — the JSON payload should already match, but
            # if a future schema migration adds keys we want the key
            # to win.
            params["event_id"] = event_id
            await self._send_aios_notification(_INBOUND_METHOD, params)

    def _maybe_warn_spool_growth(self) -> None:
        """Log once per emit when the spool exceeds the soft size threshold."""
        size = self._spool.size_bytes()
        if size > SOFT_WARN_BYTES:
            # Keep this dependency-light: structlog is optional in the
            # SDK, so write a stderr line the supervisor's pump picks up.
            import sys

            print(
                f"aios-connector: spool {self._spool.path} is {size} bytes "
                f"(>{SOFT_WARN_BYTES}); aios is likely down — operator action recommended",
                file=sys.stderr,
                flush=True,
            )


def _aios_inbound_ack_descriptor() -> Tool:
    """Built-in tool descriptor — kept module-level so it's stable for tests."""
    return Tool(
        name="aios_inbound_ack",
        description=(
            "aios calls this after a successful event append to clear "
            "the corresponding spool entry. Connector authors should "
            "not call it directly."
        ),
        inputSchema={
            "type": "object",
            "properties": {"event_id": {"type": "string"}},
            "required": ["event_id"],
            "additionalProperties": False,
        },
    )
