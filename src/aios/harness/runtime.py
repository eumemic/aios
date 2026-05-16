"""Worker-process global state.

Procrastinate registers task functions at import time, which means task bodies
can't be closures over locally-bound state from ``worker_main``. The standard
workaround is a small module that holds module-level globals — set once at
worker startup, read by every task invocation.

The values are ``None`` between import and ``worker_main`` initialization, so
task bodies use the ``require_*`` family to fail loudly if a task fires
before the worker has finished setting up. The api process never sets these;
attempting to read them from inside an api request handler will raise.

This is intentionally NOT a global mutable singleton in the OO sense — it's a
deliberate per-process state holder for procrastinate's import-time task
registration model. Don't add new globals here without thinking about it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg

    from aios.crypto.vault import CryptoBox
    from aios.harness.task_registry import TaskRegistry
    from aios.mcp.pool import McpSessionPool
    from aios.models.memory_stores import MemoryStoreResourceEcho
    from aios.sandbox.mcp_proxy import McpBroker
    from aios.sandbox.registry import SandboxRegistry
    from aios.tools.providers import ToolProvider


pool: asyncpg.Pool[Any] | None = None
crypto_box: CryptoBox | None = None
worker_id: str | None = None
sandbox_registry: SandboxRegistry | None = None
task_registry: TaskRegistry | None = None
mcp_session_pool: McpSessionPool | None = None
mcp_broker: McpBroker | None = None
tool_provider: ToolProvider | None = None

# Per-session memory-mount cache. Populated at the top of every step (in
# ``loop._run_session_step_body``) and consumed by ``tools.memory_intercept``
# when a file tool resolves a path under ``/mnt/memory/``. The cache is
# purely a performance optimization — without it, every tool call would
# re-query the same row set.
_session_memory_mounts: dict[str, list[MemoryStoreResourceEcho]] = {}


def set_session_memory_mounts(session_id: str, echoes: list[MemoryStoreResourceEcho]) -> None:
    """Record the attached memory stores for ``session_id``."""
    _session_memory_mounts[session_id] = list(echoes)


def get_session_memory_mounts(session_id: str) -> list[MemoryStoreResourceEcho]:
    """Return the attached memory stores for ``session_id``, or an empty list.

    Empty list (rather than ``None``) means "no memory stores attached" —
    the absence of an entry is observationally equivalent to an empty list,
    which is what callers want.
    """
    return _session_memory_mounts.get(session_id, [])


def clear_session_memory_mounts(session_id: str) -> None:
    """Drop the cached mounts for ``session_id`` (e.g. after session unload)."""
    _session_memory_mounts.pop(session_id, None)


# Per-session "last sha read by tool" cache: read tool stamps; write tool
# gates updates on it. Mismatch surfaces as a typed precondition error so
# the model re-reads and retries — the optimistic-locking analog of an
# ESTALE error from a kernel-level shared filesystem.
_session_read_shas: dict[str, dict[tuple[str, str], str]] = {}


def set_read_sha(session_id: str, store_id: str, store_path: str, sha: str) -> None:
    """Stamp the sha the read tool just observed for ``(store_id, store_path)``."""
    _session_read_shas.setdefault(session_id, {})[(store_id, store_path)] = sha


def get_read_sha(session_id: str, store_id: str, store_path: str) -> str | None:
    """Return the cached read sha for ``(store_id, store_path)``, or ``None``.

    A miss means the model never read this path in this session — so the
    write tool treats it as a fresh write (no precondition).
    """
    return _session_read_shas.get(session_id, {}).get((store_id, store_path))


def clear_session_read_shas(session_id: str) -> None:
    """Drop the cached read shas for ``session_id`` (e.g. after session unload)."""
    _session_read_shas.pop(session_id, None)


def _require[T](name: str, value: T | None) -> T:
    if value is None:
        raise RuntimeError(
            f"aios.harness.runtime.{name} is not initialized; "
            f"this code is running outside a worker_main context"
        )
    return value


def require_pool() -> asyncpg.Pool[Any]:
    return _require("pool", pool)


def require_crypto_box() -> CryptoBox:
    return _require("crypto_box", crypto_box)


def require_sandbox_registry() -> SandboxRegistry:
    return _require("sandbox_registry", sandbox_registry)


def require_task_registry() -> TaskRegistry:
    return _require("task_registry", task_registry)


def require_mcp_broker() -> McpBroker:
    return _require("mcp_broker", mcp_broker)


def require_tool_provider() -> ToolProvider:
    return _require("tool_provider", tool_provider)
