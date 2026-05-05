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

import inspect
import json
import os
import stat
import time
import types
import typing
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, get_type_hints, overload

import anyio
import anyio.abc
import structlog
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
from pydantic import TypeAdapter
from ulid import ULID

from aios_connector.media import _resolve_sandbox_path, _SandboxPathMarker
from aios_connector.spool import SOFT_WARN_BYTES, Spool

log = structlog.get_logger("aios_connector")

# Method names the connector emits over stdio.  Mirrors the constant in
# ``aios.mcp.stdio_transport`` — duplicated rather than imported so the
# reference SDK has zero dependency on aios server code.  If the prefix
# ever changes both ends must update.
_AIOS_NOTIFICATION_PREFIX = "notifications/aios/"
_ACCOUNTS_METHOD = f"{_AIOS_NOTIFICATION_PREFIX}accounts"
_INBOUND_METHOD = f"{_AIOS_NOTIFICATION_PREFIX}inbound"

# MCP ``_meta`` key the harness stamps for outbound tool calls when
# the calling session has a focal channel set.  Value is the channel
# address with only its leading ``<connector>/`` segment stripped, so
# the connector receives ``<account>/<chat_id>`` (or
# ``<account>/<chat_id>/<thread>`` for nested forms).  The base class
# splits on the first ``/`` to expose ``account`` and ``chat_id`` to
# focal-required tool methods.
_FOCAL_CHANNEL_META_KEY = "aios.focal_channel_path"

# MCP ``_meta`` key carrying the calling session's id.  Mirrors
# ``aios.harness.channels.SESSION_ID_META_KEY`` server-side.
_SESSION_ID_META_KEY = "aios.session_id"

# Default sandbox bind-mount root.  Connector subprocesses inherit
# ``AIOS_WORKSPACE_ROOT`` from the worker; this constant is only the
# fallback for tests + dev environments without it set.  Must match
# :attr:`aios.config.Settings.workspace_root`'s default.
_DEFAULT_WORKSPACE_ROOT = "/var/lib/aios/workspaces"

ToolFn = Callable[..., Awaitable[Any]]


@dataclass
class ToolDescriptor:
    """Metadata for a tool registered via :func:`tool`.

    Constructed by the decorator at class-body time, attached to the
    method so :meth:`Connector._collect_tools` can find it during
    ``__init__``.  ``input_schema`` is the JSON Schema published in the
    ``tools/list`` response; the SDK builds it from the function's
    type hints (Python types → JSON Schema primitives).

    ``focal_kwargs`` records which subset of ``{"account", "chat_id"}``
    the tool method declared in its signature.  At dispatch time the SDK
    splits the focal-channel path on the first ``/`` and injects only
    those kwargs the method asked for — connectors that serve a single
    account omit ``account`` from their signature and pay no
    multi-account routing cost.

    ``sandbox_params`` maps each parameter annotated with
    :data:`SandboxPath` (or ``list[SandboxPath]``) to its kind so the
    dispatch wrapper resolves model-supplied path strings to host
    :class:`Path` objects before the tool body runs.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    fn: ToolFn
    focal_required: bool = False
    focal_kwargs: frozenset[str] = frozenset()
    sandbox_params: dict[str, Literal["scalar", "list"]] = field(default_factory=dict)


_TOOL_ATTR = "__aios_connector_tool__"

# Kwargs the SDK injects from the focal-channel path on
# ``@focal_required`` tools.  Tool methods opt in by listing them in
# their signature; the SDK passes only the ones present.
_FOCAL_INJECTED_KWARGS = frozenset({"account", "chat_id"})


@overload
def tool(fn: ToolFn, /) -> ToolFn: ...
@overload
def tool(
    *, name: str | None = None, description: str | None = None
) -> Callable[[ToolFn], ToolFn]: ...
def tool(
    fn: ToolFn | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> ToolFn | Callable[[ToolFn], ToolFn]:
    """Decorator marking a coroutine method as an MCP tool.

    The method's name (or ``name=`` override) becomes the published
    tool name; its docstring (or ``description=`` override) becomes
    the description; its parameter type hints become the JSON Schema
    properties.  Self isn't included in the schema.

    Both ``@tool`` and ``@tool()`` are supported; pass ``name=`` /
    ``description=`` to override the defaults.

    Stack with :func:`focal_required` to receive the focal-channel
    parts as kwargs.  Declare whichever subset of ``account`` and
    ``chat_id`` your tool needs::

        # Multi-account connector — route by account
        @tool
        @focal_required
        async def signal_send(self, text: str, *, account: str, chat_id: str): ...

        # Single-account connector — chat is enough
        @tool
        @focal_required
        async def telegram_send(self, text: str, *, chat_id: str): ...
    """
    if fn is not None:
        # Bare ``@tool`` form — fn is the decorated function.
        return _decorate_tool(fn, name=None, description=None)
    return lambda f: _decorate_tool(f, name=name, description=description)


def _decorate_tool(fn: ToolFn, *, name: str | None, description: str | None) -> ToolFn:
    tool_name = name or fn.__name__
    doc = (fn.__doc__ or "").strip()
    if not doc and description is None:
        raise ValueError(
            f"@tool {tool_name!r} needs a docstring or explicit description= "
            "(MCP exposes the description to the model)"
        )
    is_focal = bool(getattr(fn, "__aios_focal_required__", False))
    focal_kwargs = _detect_focal_kwargs(fn) if is_focal else frozenset()
    if is_focal and not focal_kwargs:
        raise ValueError(
            f"@focal_required tool {tool_name!r} must declare at least one of "
            f"{{account, chat_id}} as a kwarg — the SDK injects what's "
            "present in the signature, and a tool that wants neither "
            "shouldn't be focal-required"
        )
    hints = get_type_hints(fn, include_extras=True)
    schema = _build_input_schema(fn, hints=hints, focal_required=is_focal)
    sandbox_params: dict[str, Literal["scalar", "list"]] = {}
    for param_name, hint in hints.items():
        found = _extract_sandbox_kind(hint)
        if found is not None:
            sandbox_params[param_name] = found[1]
    descriptor = ToolDescriptor(
        name=tool_name,
        description=description or doc,
        input_schema=schema,
        fn=fn,
        focal_required=is_focal,
        focal_kwargs=focal_kwargs,
        sandbox_params=sandbox_params,
    )
    setattr(fn, _TOOL_ATTR, descriptor)
    return fn


def focal_required(fn: ToolFn) -> ToolFn:
    """Decorator marking a tool as requiring a focal channel.

    The harness's ``_meta.aios.focal_channel_path`` carries the
    connector-relative focal address (``<account>/<chat_id>`` or
    ``<account>/<chat_id>/<thread>`` for nested forms).  The SDK splits
    on the first ``/`` and injects ``account`` and/or ``chat_id`` based
    on which the tool method declares in its signature.  Calls without
    focal raise so the model sees a tool error rather than a silent
    fallback.

    Apply BELOW :func:`tool` so the descriptor's ``focal_required``
    flag is set before the tool registration reads it.
    """
    fn.__aios_focal_required__ = True  # type: ignore[attr-defined]
    return fn


def _detect_focal_kwargs(fn: ToolFn) -> frozenset[str]:
    """Return the subset of :data:`_FOCAL_INJECTED_KWARGS` ``fn`` declares.

    `account` and `chat_id` can't appear as ``**kwargs`` catches — the
    intersection naturally only matches named parameters.
    """
    return frozenset(inspect.signature(fn).parameters) & _FOCAL_INJECTED_KWARGS


def _build_input_schema(
    fn: ToolFn, *, hints: dict[str, Any] | None = None, focal_required: bool
) -> dict[str, Any]:
    """Generate a JSON Schema for ``fn``'s non-self parameters.

    When ``focal_required=True``, ``account`` and ``chat_id`` are
    SDK-injected and excluded from the model-facing schema.

    ``hints`` may be passed by callers that already resolved them with
    ``get_type_hints(fn, include_extras=True)``; otherwise we resolve
    here.  ``include_extras`` is required so ``Annotated`` metadata
    (e.g. the :data:`SandboxPath` marker) survives.
    """
    sig = inspect.signature(fn)
    if hints is None:
        hints = get_type_hints(fn, include_extras=True)

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if focal_required and param_name in _FOCAL_INJECTED_KWARGS:
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
    """Map a Python type hint to a JSON Schema fragment via pydantic, with
    ``T | None`` flattened to ``T`` and :data:`SandboxPath` rendered as a
    string (the model passes in-sandbox path strings; the SDK auto-resolves)."""
    if hint is None or hint is inspect.Parameter.empty:
        return {}
    sandbox_schema = _sandbox_path_schema(hint)
    if sandbox_schema is not None:
        return sandbox_schema
    schema: dict[str, Any] = TypeAdapter(hint).json_schema()
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        non_null: list[dict[str, Any]] = [s for s in any_of if s != {"type": "null"}]
        if len(non_null) == 1:
            return non_null[0]
    return schema


def _sandbox_path_schema(hint: Any) -> dict[str, Any] | None:
    """JSON Schema for a hint involving :data:`SandboxPath`, or ``None``."""
    found = _extract_sandbox_kind(hint)
    if found is None:
        return None
    marker, kind = found
    if kind == "list":
        return {
            "type": "array",
            "items": {"type": "string", "description": marker.description},
        }
    return {"type": "string", "description": marker.description}


def _strip_optional(hint: Any) -> Any:
    """Strip a ``| None`` arm; return ``hint`` unchanged for any other union."""
    origin = typing.get_origin(hint)
    if origin is None:
        return hint
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


def _extract_sandbox_kind(
    hint: Any,
) -> tuple[_SandboxPathMarker, Literal["scalar", "list"]] | None:
    """Return ``(marker, kind)`` if ``hint`` is :data:`SandboxPath` or
    ``list[SandboxPath]`` (with an optional ``| None``), else ``None``."""
    bare = _strip_optional(hint)
    marker = _marker_in_metadata(bare)
    if marker is not None:
        return (marker, "scalar")
    if typing.get_origin(bare) is list:
        args = typing.get_args(bare)
        if len(args) == 1:
            inner = _marker_in_metadata(_strip_optional(args[0]))
            if inner is not None:
                return (inner, "list")
    return None


def _marker_in_metadata(hint: Any) -> _SandboxPathMarker | None:
    metadata = getattr(hint, "__metadata__", None)
    if not metadata:
        return None
    for meta in metadata:
        if isinstance(meta, _SandboxPathMarker):
            return meta
    return None


def _resolve_one_sandbox_path(
    *,
    sandbox_path: str,
    session_id: str,
    workspace_root: Path,
    tool_name: str,
) -> Path:
    """Resolve one in-sandbox path string to its host :class:`Path`, with
    uniform error wording across connectors."""
    host = _resolve_sandbox_path(
        session_id=session_id,
        sandbox_path=sandbox_path,
        workspace_root=workspace_root,
    )
    if host is None:
        raise ValueError(
            f"{tool_name}: sandbox path {sandbox_path!r} could not be resolved "
            f"to a host path (must be under /workspace/ or "
            f"/mnt/attachments/, no .. escapes)"
        )
    if not host.is_file():
        raise ValueError(
            f"{tool_name}: sandbox path {sandbox_path!r} does not exist (resolved to {host})"
        )
    return host


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


_MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024


class AttachmentError(ValueError):
    """Raised by :meth:`Attachment.as_params` when the host path is unreadable
    or exceeds the 5 MiB SDK boundary cap.

    Connector code that catches this can decide whether to skip the
    attachment, send a placeholder text message, or fail loudly.
    """


@dataclass(frozen=True, slots=True)
class Attachment:
    """An inbound binary blob (photo, voice note, document) carried on a
    chat message.

    The connector hands aios a *host path* — bytes never traverse stdio.
    The supervisor stages the file into the session's read-only
    ``/mnt/attachments`` bind mount before appending the inbound event.
    """

    host_path: str
    filename: str
    content_type: str

    def as_params(self) -> dict[str, Any]:
        """Validate the host path and return the JSON-RPC wire dict.

        Raises :class:`AttachmentError` when the file is missing or
        exceeds the 5 MiB cap.  Stat'ing here rather than at
        construction lets callers build :class:`Attachment` instances
        before their backing files are fully written.
        """
        try:
            st = os.stat(self.host_path)
        except FileNotFoundError as err:
            raise AttachmentError(
                f"attachment host_path does not exist: {self.host_path!r}"
            ) from err
        if not stat.S_ISREG(st.st_mode):
            raise AttachmentError(f"attachment host_path is not a regular file: {self.host_path!r}")
        if st.st_size > _MAX_ATTACHMENT_BYTES:
            raise AttachmentError(
                f"attachment {self.filename!r} is {st.st_size} bytes; "
                f"SDK cap is {_MAX_ATTACHMENT_BYTES} bytes (5 MiB)."
            )
        return {
            "host_path": self.host_path,
            "filename": self.filename,
            "content_type": self.content_type,
            "size": st.st_size,
        }


class Connector:
    """Base class connector authors subclass.

    Lifecycle (driven by :meth:`run`):

    1. ``setup()`` — open daemon connections, sign in, etc.  Runs
       before the MCP server starts; blocking here keeps the supervisor
       in ``starting`` until either this returns or the bounded init
       handshake (30s) trips and respawns.
    2. ``discover_accounts()`` — return the initial account snapshot.
       Returns ``list[dict[str, Any]]`` (NOT a list of account-id
       strings) — use :func:`make_account` to build entries.  The ``id``
       field is what aios uses as the opaque ``account`` segment in
       channel addresses.
    3. Start MCP server over stdio.  ``notifications/initialized`` is
       only sent after steps 1-2 complete: outbound dispatch into a
       not-yet-ready connector therefore waits on first init rather
       than racing past unset state.
    4. Send ``notifications/aios/accounts`` once the client is initialized.
    5. Replay unacked spool entries, then enter live mode.
    6. ``serve()`` — sibling task to the MCP server.  Default parks
       forever; override to drive an inbound pump (signal-cli's
       listener loop, etc.).  Calls to :meth:`emit_inbound` from here
       block until step 5 completes so live inbounds can't race past
       spool replay.
    7. ``teardown()`` on exit (normal or exception, in ``run()``'s
       ``finally``).

    Subclass contract:

    * Override ``name`` (e.g. ``"signal"``); the spool path and outbound
      channel-address prefix derive from it.
    * Override :meth:`setup`, :meth:`discover_accounts`, :meth:`teardown`,
      and (if event-driven inbound is needed) :meth:`serve`.
    * Decorate methods with :func:`tool` (and optionally
      :func:`focal_required`) to publish tools.
    * Call :meth:`emit_inbound` from your inbound pump.
    * Optionally set ``instructions`` for an
      :class:`mcp.types.InitializeResult.instructions` block surfaced
      to the model on every turn.

    Register the connector via the ``aios.connectors`` entry-point group;
    the entry point points at a zero-argument callable that returns a
    :class:`Connector` instance (e.g. ``my_pkg:make_connector``).  The
    supervisor synthesizes the launch command itself
    (``python -m aios_connector run <name>``) and the SDK's runner loads
    the entry point.  See ``packages/aios-echo`` for a minimal example.
    """

    name: ClassVar[str] = ""
    version: ClassVar[str] = "0.0.0"
    # ``instructions`` is intentionally NOT a ClassVar so that
    # connectors can set it on the instance from :meth:`setup` (e.g.,
    # signal embedding the bot UUID + contacts in the prompt) without
    # mutating a class-level attribute that would leak across instances.
    instructions: str | None = None

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
        # Flipped at the end of :meth:`_emit_initial_state` so live
        # :meth:`emit_inbound` calls block until the accounts snapshot
        # and any spool replay have been flushed.  Otherwise a
        # connector whose :meth:`serve` fires inbounds eagerly during
        # startup could race past the initial-state ordering invariant
        # documented on :meth:`_emit_initial_state`.
        self._initial_state_done: anyio.Event = anyio.Event()
        self._emit_count: int = 0
        self._warned: bool = False

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

        Each entry is a dict with at minimum ``id`` and ``display_name``
        — NOT a bare account-id string.  The ``id`` field is the opaque
        ``account`` segment aios uses in channel addresses
        (``<connector>/<account>/<chat_id>``) and the value
        :meth:`emit_inbound`'s ``account=`` argument must match.
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
        attachments: list[Attachment] | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> str:
        """Persist an inbound message to the spool, then push the notification.

        Returns the ULID assigned as ``event_id``.  Order matters: the
        spool write happens first so a crash between ``add()`` and the
        stdio write replays the entry on reconnect.

        ``account`` is one of the ids from :meth:`discover_accounts`'
        return value.  ``chat_id`` is the connector-specific identifier
        the model addresses via focal channel; the harness builds the
        full channel address as ``<connector>/<account>/<chat_id>``.

        ``attachments`` is an optional list of :class:`Attachment`
        records carrying binary blobs (photos, voice notes, documents).
        Each attachment's ``host_path`` is validated (readable, ≤5 MiB)
        before the notification is pushed — invalid attachments raise
        :class:`AttachmentError` and the connector decides whether to
        skip, fall back to a text placeholder, or refuse the inbound.
        Bytes never traverse stdio; the supervisor stages the host path
        into the session's ``/mnt/attachments`` bind mount.

        ``timestamp`` is an optional ISO-8601 string carrying the
        platform's actual delivery time (Signal envelope timestamp,
        Telegram message ``date``, etc.).  When supplied, the supervisor
        stamps it as ``metadata.platform_timestamp`` on the appended
        event so operators debugging "why did this arrive 30s late?"
        can compare against the server-side ``created_at``.  Connectors
        whose source platform doesn't carry a timestamp omit this kwarg.
        """
        if self._write_stream is None:
            raise RuntimeError(
                "emit_inbound called before the MCP server started; "
                "call from setup() / live operation, not __init__"
            )
        # Block until the accounts snapshot + spool replay have been
        # pushed.  Otherwise an eager :meth:`serve` could ship a fresh
        # inbound past a half-initialized supervisor — out of order
        # with the very entries the spool is supposed to deliver
        # exactly once.
        await self._initial_state_done.wait()

        event_id = str(ULID())
        params: dict[str, Any] = {
            "event_id": event_id,
            "account": account,
            "chat_id": chat_id,
            "sender": dict(sender),
            "content": content,
        }
        if attachments:
            params["attachments"] = [att.as_params() for att in attachments]
        if metadata:
            params["metadata"] = dict(metadata)
        if timestamp is not None:
            params["timestamp"] = timestamp

        payload = json.dumps(params).encode("utf-8")
        self._spool.add(event_id, payload, created_at=time.time())
        self._maybe_warn_spool_growth()

        await self._send_aios_notification(_INBOUND_METHOD, params)
        return event_id

    # ── Run loop ──────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main entry point.  Subclass entry-point factories call this.

        Runs ``setup()`` → ``discover_accounts()`` BEFORE the MCP server
        starts so a stuck setup trips the supervisor's bounded init
        handshake (30s) rather than parking the supervisor forever.

        ``setup()`` runs INSIDE the try/finally so a partial-setup
        failure (e.g., signal-cli's ``__aenter__`` succeeded then
        ``discover_bot_uuid`` raised) still triggers ``teardown()`` —
        otherwise the daemon subprocess leaks and the supervisor's
        respawn would race the previous instance's socket lock.
        """
        try:
            await self.setup()
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
        """Dispatch a tool: inject focal kwargs, resolve sandbox paths, call."""
        kwargs = dict(arguments)
        if descriptor.focal_required:
            focal = self._focal_from_request_meta()
            if focal is None:
                raise ValueError(
                    f"tool {descriptor.name!r} requires a focal channel — "
                    "aios should inject _meta.aios.focal_channel_path"
                )
            account, _, chat_id = focal.partition("/")
            if "account" in descriptor.focal_kwargs:
                kwargs["account"] = account
            if "chat_id" in descriptor.focal_kwargs:
                kwargs["chat_id"] = chat_id

        if descriptor.sandbox_params:
            self._resolve_sandbox_kwargs(descriptor, kwargs)

        result = await descriptor.fn(self, **kwargs)
        if isinstance(result, list) and all(isinstance(x, TextContent) for x in result):
            return result
        if isinstance(result, str):
            return [TextContent(type="text", text=result)]
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

    def _resolve_sandbox_kwargs(self, descriptor: ToolDescriptor, kwargs: dict[str, Any]) -> None:
        """Replace each :data:`SandboxPath` / ``list[SandboxPath]`` argument
        with a resolved host :class:`Path`, mutating ``kwargs`` in place.

        Empty/absent values pass through; ``session_id`` is fetched lazily
        so text-only calls don't require ``_meta.aios.session_id``.
        """
        workspace_root = Path(os.environ.get("AIOS_WORKSPACE_ROOT", _DEFAULT_WORKSPACE_ROOT))
        session_id: str | None = None

        for param_name, kind in descriptor.sandbox_params.items():
            if param_name not in kwargs:
                continue
            value = kwargs[param_name]
            if value is None:
                continue
            if kind == "list":
                if not isinstance(value, list):
                    raise TypeError(
                        f"tool {descriptor.name!r} parameter {param_name!r} expects a "
                        f"list of in-sandbox path strings; got {type(value).__name__}"
                    )
                if not value:
                    continue
                if session_id is None:
                    session_id = self._require_session_id(descriptor.name)
                kwargs[param_name] = [
                    _resolve_one_sandbox_path(
                        sandbox_path=str(item),
                        session_id=session_id,
                        workspace_root=workspace_root,
                        tool_name=descriptor.name,
                    )
                    for item in value
                ]
            else:  # "scalar"
                if session_id is None:
                    session_id = self._require_session_id(descriptor.name)
                kwargs[param_name] = _resolve_one_sandbox_path(
                    sandbox_path=str(value),
                    session_id=session_id,
                    workspace_root=workspace_root,
                    tool_name=descriptor.name,
                )

    def _require_session_id(self, tool_name: str) -> str:
        session_id = self.current_session_id()
        if session_id is None:
            raise RuntimeError(
                f"{tool_name} with sandbox-path arguments requires "
                "_meta.aios.session_id; the harness should always inject "
                "this for tool dispatches"
            )
        return session_id

    def _focal_from_request_meta(self) -> str | None:
        """Read the focal-channel suffix from the active request, or ``None``."""
        return self._read_meta_str(_FOCAL_CHANNEL_META_KEY)

    def current_session_id(self) -> str | None:
        """Read ``_meta.aios.session_id`` from the active tool-call request.

        Request-scoped: returns ``None`` outside an active tool dispatch
        (e.g., when called from :meth:`serve` or an inbound pump).  Use
        only inside ``@tool``-decorated methods.
        """
        return self._read_meta_str(_SESSION_ID_META_KEY)

    def _read_meta_str(self, key: str) -> str | None:
        """Fetch a string-typed key from ``_meta`` of the active tool-call
        request, or ``None`` when absent / empty / wrong type."""
        from mcp.server.lowlevel.server import request_ctx

        try:
            ctx = request_ctx.get()
        except LookupError:
            return None
        meta: RequestParams.Meta | None = ctx.meta if ctx else None
        if meta is None:
            return None
        extra = getattr(meta, "model_extra", None) or {}
        value: Any = extra.get(key)
        if not isinstance(value, str) or not value:
            return None
        return value

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
        Sets ``_initial_state_done`` at the tail so ``emit_inbound``
        callers (live :meth:`serve` work) can resume.
        """
        try:
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
        finally:
            self._initial_state_done.set()

    def _maybe_warn_spool_growth(self) -> None:
        """Stat the spool every Nth emit; warn-once when it exceeds the soft threshold.

        ``size_bytes`` stats main-DB + WAL — a syscall pair we don't
        need on every inbound message.  Sampling every 64 emits trades
        a bounded delay (a runaway connector might emit ~64 messages
        past the threshold before warning) for skipping ~98% of stats
        on the hot path.  ``_warned`` latches so one outage doesn't
        flood the log.
        """
        self._emit_count += 1
        if self._warned or self._emit_count & 63 != 0:
            return
        size = self._spool.size_bytes()
        if size <= SOFT_WARN_BYTES:
            return
        self._warned = True
        log.warning(
            "spool.growth",
            path=str(self._spool.path),
            size_bytes=size,
            threshold_bytes=SOFT_WARN_BYTES,
            hint="aios is likely down — operator action recommended",
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
