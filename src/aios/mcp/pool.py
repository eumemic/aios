"""Worker-scoped MCP session pool.

Holds a persistent ``ClientSession`` per ``(url, headers_hash)`` key so
tool discovery and invocation can reuse an already-initialized MCP
connection instead of opening a fresh one on every call.

Pool lifecycle:

- Created at worker startup, stashed on :mod:`aios.harness.runtime`.
- :meth:`get_or_connect` lazily opens and initializes a session on first
  use. A per-key ``asyncio.Lock`` prevents thundering-herd
  double-initialization.
- Callers evict on first failure; the next :meth:`get_or_connect`
  re-opens. Eviction drops the reference without attempting to close the
  broken session — closing a half-dead stack may hang.
- :meth:`close_all` is called from ``worker_main``'s ``finally`` at
  shutdown to tear everything down.
"""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import AsyncExitStack
from typing import Any

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import InitializeResult

from aios.logging import get_logger

log = get_logger("aios.mcp.pool")

# Mirrors the per-call bound used by :mod:`aios.mcp.client`. The pool's
# pooled sessions share a connection-level timeout so a stalled MCP server
# can't keep the worker on a dead socket indefinitely.
_MCP_HTTPX_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)

type _PoolKey = tuple[str, str]


def _headers_hash(headers: dict[str, str]) -> str:
    """Stable SHA-256 hash of a headers dict, sorted for determinism.

    Using the hash (rather than the raw key material) as part of the pool
    key avoids keeping bearer tokens in in-memory dict keys.
    """
    blob = "&".join(f"{k}={v}" for k, v in sorted(headers.items()))
    return hashlib.sha256(blob.encode()).hexdigest()


class _Entry:
    """A single pooled MCP session with its associated resource stack."""

    def __init__(
        self,
        session: ClientSession,
        init_result: InitializeResult,
        stack: AsyncExitStack,
    ) -> None:
        self.session = session
        self.init_result = init_result
        self._stack = stack

    async def close(self) -> None:
        """Tear down this entry's session and all its async contexts."""
        await self._stack.aclose()


class McpSessionPool:
    """Worker-scoped pool of persistent MCP ``ClientSession`` instances.

    Single-event-loop — no thread-safety concerns. Concurrent async tasks
    calling the same key serialise through a per-key ``asyncio.Lock``.
    """

    def __init__(self) -> None:
        self._entries: dict[_PoolKey, _Entry] = {}
        self._locks: dict[_PoolKey, asyncio.Lock] = {}

    def _lock_for(self, key: _PoolKey) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def _open_entry(self, url: str, headers: dict[str, str]) -> _Entry:
        """Open a fresh session. Stack must outlive this frame (entry owns it)."""
        stack: AsyncExitStack = AsyncExitStack()
        await stack.__aenter__()
        try:
            http_client: Any = await stack.enter_async_context(
                httpx.AsyncClient(headers=headers, timeout=_MCP_HTTPX_TIMEOUT)
            )
            read_stream, write_stream, _ = await stack.enter_async_context(
                streamable_http_client(url, http_client=http_client)
            )
            session: ClientSession = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            init_result: InitializeResult = await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        return _Entry(session, init_result, stack)

    async def get_or_connect(
        self, url: str, headers: dict[str, str]
    ) -> tuple[ClientSession, InitializeResult]:
        """Return a cached session, opening a fresh one if none exists.

        Double-checked locking: the fast unsynchronised check avoids lock
        contention on the common (already-connected) path; the slow path
        acquires the per-key lock and re-checks before opening.
        """
        key: _PoolKey = (url, _headers_hash(headers))

        entry = self._entries.get(key)
        if entry is not None:
            return entry.session, entry.init_result

        async with self._lock_for(key):
            entry = self._entries.get(key)
            if entry is not None:
                return entry.session, entry.init_result

            log.info("mcp_pool.connecting", url=url)
            entry = await self._open_entry(url, headers)
            self._entries[key] = entry
            log.info("mcp_pool.connected", url=url)

        return entry.session, entry.init_result

    def evict(self, url: str, headers: dict[str, str]) -> None:
        """Drop a cache entry without closing it.

        Called when a cached session has been found to be broken. Closing
        a broken session may hang or raise in unexpected ways, so we
        simply drop the reference and let GC handle it; the next
        :meth:`get_or_connect` for this key will open a fresh session.
        """
        key: _PoolKey = (url, _headers_hash(headers))
        self._entries.pop(key, None)
        log.info("mcp_pool.evicted", url=url)

    async def close_all(self) -> None:
        """Tear down all pooled sessions. Called at worker shutdown."""
        entries = list(self._entries.values())
        self._entries.clear()
        self._locks.clear()
        if not entries:
            return
        log.info("mcp_pool.close_all", count=len(entries))
        for entry in entries:
            try:
                await entry.close()
            except Exception:
                log.warning("mcp_pool.close_entry_failed", exc_info=True)
