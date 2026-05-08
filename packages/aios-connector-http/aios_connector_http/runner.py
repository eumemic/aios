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
from collections.abc import Awaitable, Callable
from typing import Any

from ulid import ULID

from .client import AiosClient

ToolFn = Callable[..., Awaitable[Any]]

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
        self._tools = self._collect_tools()
        self._answered: set[str] = set()

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
            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._tool_loop(), name="aios-tool-loop")
                    tg.create_task(self.serve(), name="aios-platform-serve")
            finally:
                await self.teardown()

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
                    await self._dispatch_call(call)
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

    async def _dispatch_call(self, call: dict[str, Any]) -> None:
        """Run the right tool method for ``call`` and POST the result.

        If the tool method's signature accepts ``account`` and/or
        ``chat_id`` kwargs, the runner parses them out of the call's
        ``focal_channel`` (shaped ``<connector>/<account>/<chat_id>``)
        and injects them.  Mirrors the legacy SDK's ``@focal_required``
        without a separate decorator — declare what you need in the
        signature and the runner threads it through.
        """
        assert self._client is not None
        name = call.get("name", "")
        tool_call_id = call.get("tool_call_id", "")
        session_id = call.get("session_id", "")
        fn = self._tools.get(name)
        if fn is None:
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
        args = _inject_focal_kwargs(fn, args, focal_channel)
        try:
            result = await fn(**args)
        except Exception as exc:
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

    def _collect_tools(self) -> dict[str, ToolFn]:
        """Walk ``self`` and pick up every method tagged with @tool."""
        out: dict[str, ToolFn] = {}
        for _, member in inspect.getmembers(self):
            if callable(member) and hasattr(member, _TOOL_ATTR):
                tool_name: str = getattr(member, _TOOL_ATTR)
                out[tool_name] = member
        return out


_FOCAL_INJECTABLE: frozenset[str] = frozenset({"account", "chat_id"})


def _inject_focal_kwargs(
    fn: ToolFn, args: dict[str, Any], focal_channel: str
) -> dict[str, Any]:
    """Inject ``account`` / ``chat_id`` kwargs from ``focal_channel``.

    Returns ``args`` augmented with whichever of ``account`` and
    ``chat_id`` (a) the tool's signature accepts and (b) the caller
    didn't already pass explicitly.  ``focal_channel`` is the
    ``<connector>/<account>/<chat_id>`` form the server stamps onto
    each call event.

    Mirrors the legacy SDK's ``@focal_required`` injection without
    requiring a separate decorator: tool authors declare ``account``
    and/or ``chat_id`` in their signature and the runner threads them
    through automatically.
    """
    if not focal_channel:
        return args
    parts = focal_channel.split("/", 2)
    if len(parts) < 3:
        return args
    _connector, account, chat_id = parts
    sig = inspect.signature(fn)
    accepted = set(sig.parameters) & _FOCAL_INJECTABLE
    out = dict(args)
    if "account" in accepted and "account" not in out:
        out["account"] = account
    if "chat_id" in accepted and "chat_id" not in out:
        out["chat_id"] = chat_id
    return out
