"""Base class connector authors subclass.

Subclass :class:`HttpConnector`, decorate methods with :func:`tool`,
implement any platform-specific lifecycle by overriding :meth:`setup`,
:meth:`teardown`, or :meth:`serve`.  Then::

    if __name__ == "__main__":
        import asyncio
        asyncio.run(MyConnector().run())

The runner reads ``AIOS_URL`` and ``AIOS_CONNECTOR_TOKEN`` from env,
spins an :class:`AiosClient`, and runs an SSE-tail loop that dispatches
incoming tool-call events to your decorated methods.  Inbound is
emitted by your code calling :meth:`emit_inbound`.

The runner tracks answered ``tool_call_id``s in memory so SSE
reconnects (which replay the backfill) don't double-execute.  For
durability across container restart, override :meth:`load_answered` /
:meth:`save_answered` (a small SQLite spool ships at ``spool.py``).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import types
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import structlog
from ulid import ULID

from .client import AiosClient
from .sandbox import _SandboxPathMarker, resolve_sandbox_path
from .schema import derive_tool_spec

ToolFn = Callable[..., Awaitable[Any]]

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_TOOL_ATTR = "__aios_http_tool__"


def tool(
    *,
    name: str | None = None,
) -> Callable[[ToolFn], ToolFn]:
    """Decorate a method as a connector tool.

    The method's name becomes the published tool name (or the explicit
    ``name=`` override).  Arguments come from the model's tool_call
    arguments, parsed as JSON and passed as kwargs.  The method's
    return value is the tool result string the session log stores.

    The connector container must POST a matching ``ConnectionSetTools``
    body to ``PUT /v1/connections/{id}/tools`` so the model's tool list
    includes this tool — the SDK doesn't auto-publish schemas in v1.
    """

    def _wrap(f: ToolFn) -> ToolFn:
        setattr(f, _TOOL_ATTR, name or f.__name__)
        return f

    return _wrap


@dataclass(frozen=True, slots=True)
class _ToolMeta:
    """Per-tool reflection cache populated once at construction.

    Holds the bound method plus the precomputed bits dispatch needs:
    which params accept focal-channel injection, and which params
    carry :data:`SandboxPath` annotations (and whether scalar or list).
    Storing these as a frozen dataclass on ``self._tools`` keeps the
    per-call hot path free of ``inspect`` and ``get_type_hints`` calls.
    """

    fn: ToolFn
    focal_params: frozenset[str]
    sandbox_params: tuple[tuple[str, str], ...]  # (param_name, "scalar" | "list")


class HttpConnector:
    """Base class for HTTP-client connectors.

    Subclass and decorate methods with :func:`tool`.  Override the
    lifecycle hooks if you need them.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
    ) -> None:
        self._base_url = base_url or os.environ["AIOS_URL"]
        self._token = token or os.environ["AIOS_CONNECTOR_TOKEN"]
        self._client: AiosClient | None = None
        self._connection_id: str | None = None
        self._tools: dict[str, _ToolMeta] = self._collect_tools()
        self._answered: set[str] = set()
        self._secrets_cache: dict[str, str] | None = None

    # ─── lifecycle hooks (override as needed) ────────────────────────

    async def setup(self) -> None:
        """Override: open platform connections, start polling, etc."""

    async def teardown(self) -> None:
        """Override: close platform connections cleanly."""

    async def serve(self) -> None:
        """Override: long-running platform task that emits inbounds.

        Default is a no-op — connectors that only respond to outbound
        tool calls (no inbound) leave this empty.
        """
        await asyncio.Event().wait()  # block forever

    async def load_answered(self) -> set[str]:
        """Override: persisted tool_call_ids from a previous lifetime."""
        return set()

    async def save_answered(self, tool_call_id: str) -> None:
        """Override: persist a freshly-answered tool_call_id."""

    # ─── connector → aios glue ───────────────────────────────────────

    async def secrets(self) -> dict[str, str]:
        """Decrypted platform secrets for the caller's connection.

        Cached for the connector's lifetime: rotating secrets via the
        management API requires restarting the connector container to
        pick up new values.  Authors who need fresh-on-every-call
        semantics should call ``self._client.get_secrets()`` directly.
        """
        if self._secrets_cache is None:
            assert self._client is not None
            self._secrets_cache = await self._client.get_secrets()
        return self._secrets_cache

    async def emit_inbound(
        self,
        *,
        chat_id: str,
        sender: dict[str, Any],
        content: str,
        attachments: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: str | None = None,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Forward a platform inbound to aios.  Idempotent on event_id."""
        assert self._client is not None
        eid = event_id or str(ULID())
        log.info(
            "connector.inbound",
            connection_id=self._connection_id,
            chat_id=chat_id,
            sender=sender,
            content_len=len(content),
            attachment_count=len(attachments or []),
            event_id=eid,
        )
        return await self._client.post_inbound(
            event_id=eid,
            chat_id=chat_id,
            sender=sender,
            content=content,
            attachments=attachments,
            metadata=metadata,
            timestamp=timestamp,
        )

    # ─── runner ──────────────────────────────────────────────────────

    async def run(self) -> None:
        """Open the client, run setup → serve + tool-loop, then teardown."""
        async with AiosClient(self._base_url, self._token) as client:
            self._client = client
            self._connection_id = await client.whoami()
            self._answered = await self.load_answered()
            await self.setup()
            await self._publish_tool_schemas()
            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._tool_loop(), name="aios-tool-loop")
                    tg.create_task(self.serve(), name="aios-platform-serve")
            finally:
                await self.teardown()

    async def _publish_tool_schemas(self) -> None:
        """Derive a ToolSpec from each ``@tool`` method and publish it.

        Replaces the connection's tools wholesale on every startup, so
        the model's tool list always matches the running container.
        Operator-side hand-written ``tools.json`` is deliberately
        overwritten — the Python source is canonical.
        """
        assert self._client is not None
        specs = [derive_tool_spec(name, meta.fn) for name, meta in self._tools.items()]
        await self._client.set_connection_tools(specs)
        log.info(
            "connector.tools.published",
            connection_id=self._connection_id,
            tool_count=len(specs),
            tool_names=sorted(self._tools.keys()),
        )

    async def _tool_loop(self) -> None:
        """Tail the calls SSE stream and dispatch each call.

        Wraps the stream in exponential-backoff reconnect: a transient
        network blip drops the SSE iterator, but doesn't kill the
        connector.  The next reconnect's backfill replays any pending
        calls, and the answered-id set deduplicates so we never
        double-execute.
        """
        assert self._client is not None
        backoff = 1.0
        while True:
            try:
                async for call in self._client.stream_calls():
                    backoff = 1.0  # reset on first successful event
                    tool_call_id = call.get("tool_call_id", "")
                    if not tool_call_id or tool_call_id in self._answered:
                        continue
                    await self.dispatch_call(call)
                    self._answered.add(tool_call_id)
                    await self.save_answered(tool_call_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Stream dropped; backoff and reconnect.
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            # Stream returned cleanly (server closed without error) — also
            # reconnect, but with a small delay to avoid a tight loop.
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    async def dispatch_call(self, call: dict[str, Any]) -> None:
        """Run the tool method for ``call`` and POST the result.

        If the tool method's signature accepts ``account`` and/or
        ``chat_id`` kwargs, the runner parses them out of the call's
        ``focal_channel`` (shaped ``<connector>/<account>/<chat_id>``)
        and injects them.  ``SandboxPath``-annotated params get their
        in-sandbox path strings resolved to host :class:`Path`s before
        the tool body runs.

        Public surface so tests can drive dispatch without going
        through SSE — the production path is :meth:`_tool_loop`.
        """
        assert self._client is not None
        name = call.get("name", "")
        tool_call_id = call.get("tool_call_id", "")
        session_id = call.get("session_id", "")
        log.info(
            "connector.tool_call.dispatched",
            name=name,
            tool_call_id=tool_call_id,
            session_id=session_id,
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
            await self._client.post_tool_result(
                session_id=session_id,
                tool_call_id=tool_call_id,
                content=json.dumps({"error": f"unknown tool {name!r}"}),
                is_error=True,
            )
            return
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        focal_channel = call.get("focal_channel") or ""
        args = _inject_focal_kwargs(meta, args, focal_channel)
        try:
            args = _resolve_sandbox_paths(meta, args, session_id=session_id)
        except SandboxPathError as exc:
            log.warning(
                "connector.tool_call.failed",
                name=name,
                tool_call_id=tool_call_id,
                session_id=session_id,
                reason="sandbox_path",
                error=str(exc),
            )
            await self._client.post_tool_result(
                session_id=session_id,
                tool_call_id=tool_call_id,
                content=json.dumps({"error": str(exc)}),
                is_error=True,
            )
            return
        try:
            result = await meta.fn(**args)
        except Exception as exc:
            log.warning(
                "connector.tool_call.failed",
                name=name,
                tool_call_id=tool_call_id,
                session_id=session_id,
                reason="tool_exception",
                error=str(exc),
            )
            await self._client.post_tool_result(
                session_id=session_id,
                tool_call_id=tool_call_id,
                content=json.dumps({"error": str(exc)}),
                is_error=True,
            )
            return
        # Tool-result schema accepts ``str`` or a multimodal content array
        # (``list[dict]``).  Ints, dicts, etc. JSON-serialize to a string —
        # convenient for tool methods that return rich Python values.
        if not isinstance(result, str | list):
            result = json.dumps(result)
        await self._client.post_tool_result(
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

    def _collect_tools(self) -> dict[str, _ToolMeta]:
        """Walk ``self``, pick up every method tagged with @tool, and
        precompute the reflection bits ``dispatch_call`` needs."""
        out: dict[str, _ToolMeta] = {}
        for _, member in inspect.getmembers(self):
            if not (callable(member) and hasattr(member, _TOOL_ATTR)):
                continue
            tool_name: str = getattr(member, _TOOL_ATTR)
            out[tool_name] = _build_tool_meta(member)
        return out


_FOCAL_INJECTABLE: frozenset[str] = frozenset({"account", "chat_id"})


def _build_tool_meta(fn: ToolFn) -> _ToolMeta:
    """Inspect ``fn`` once and freeze the dispatch-time metadata."""
    sig = inspect.signature(fn)
    focal = frozenset(set(sig.parameters) & _FOCAL_INJECTABLE)
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}
    sandbox: list[tuple[str, str]] = []
    for param_name, hint in hints.items():
        kind = _sandbox_path_kind(hint)
        if kind is not None:
            sandbox.append((param_name, kind))
    return _ToolMeta(fn=fn, focal_params=focal, sandbox_params=tuple(sandbox))


class SandboxPathError(ValueError):
    """Raised by the dispatcher when a SandboxPath argument escapes
    ``/workspace/`` or ``/mnt/attachments/``.

    Surfaces to the model as a tool error result so it can self-correct
    on the next turn (e.g. retry with a path inside ``/workspace/``).
    """


def _resolve_sandbox_paths(
    meta: _ToolMeta, args: dict[str, Any], *, session_id: str
) -> dict[str, Any]:
    """Translate ``SandboxPath`` argument strings into host :class:`Path`s.

    For each :data:`~aios_connector_http.SandboxPath` parameter, maps the
    model-supplied in-sandbox path string against
    ``<workspace_root>/<session_id>``.  Out-of-tree or unmapped paths
    raise :class:`SandboxPathError` so the dispatcher can return an
    error result instead of silently dropping the value.
    """
    if not meta.sandbox_params or not session_id:
        return args
    out = dict(args)
    for param_name, kind in meta.sandbox_params:
        if param_name not in out:
            continue
        value = out[param_name]
        if kind == "scalar":
            out[param_name] = _resolve_one(value, session_id=session_id, name=param_name)
        else:
            if not isinstance(value, list):
                raise SandboxPathError(f"{param_name!r} must be a list of in-sandbox paths")
            out[param_name] = [
                _resolve_one(v, session_id=session_id, name=param_name) for v in value
            ]
    return out


def _resolve_one(value: Any, *, session_id: str, name: str) -> Path:
    if not isinstance(value, str):
        raise SandboxPathError(f"{name!r} must be an in-sandbox path string; got {value!r}")
    resolved = resolve_sandbox_path(session_id=session_id, sandbox_path=value)
    if resolved is None:
        raise SandboxPathError(
            f"{name!r}: path {value!r} is outside /workspace/ and /mnt/attachments/"
        )
    return resolved


def _sandbox_path_kind(hint: Any) -> str | None:
    """Return ``'scalar'`` / ``'list'`` / ``None`` for a type hint.

    Recognises ``SandboxPath``, ``list[SandboxPath]``, and either with
    ``| None`` (handy for optional attachment params).  Anything else
    returns ``None``.
    """
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
    """``T | None`` → ``T``; everything else passes through unchanged.

    Only collapses ``Union``/PEP-604 ``X | Y`` types — leaves
    parameterised generics like ``list[T]`` alone.
    """
    origin = get_origin(hint)
    if origin not in (Union, types.UnionType):
        return hint
    args = [a for a in get_args(hint) if a is not type(None)]
    return args[0] if len(args) == 1 else hint


def _is_sandbox_path(hint: Any) -> bool:
    """Detect ``Annotated[Path, SANDBOX_PATH_MARKER]`` regardless of nesting."""
    return any(isinstance(meta, _SandboxPathMarker) for meta in getattr(hint, "__metadata__", ()))


def _inject_focal_kwargs(
    meta: _ToolMeta, args: dict[str, Any], focal_channel: str
) -> dict[str, Any]:
    """Inject ``account`` / ``chat_id`` kwargs from ``focal_channel``.

    Returns ``args`` augmented with whichever of ``account`` and
    ``chat_id`` (a) the tool's signature accepts and (b) the caller
    didn't already pass explicitly.  ``focal_channel`` is the
    ``<connector>/<account>/<chat_id>`` form the server stamps onto
    each call event.
    """
    if not focal_channel or not meta.focal_params:
        return args
    parts = focal_channel.split("/", 2)
    if len(parts) < 3:
        return args
    _connector, account, chat_id = parts
    out = dict(args)
    if "account" in meta.focal_params and "account" not in out:
        out["account"] = account
    if "chat_id" in meta.focal_params and "chat_id" not in out:
        out["chat_id"] = chat_id
    return out
