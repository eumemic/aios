"""Worker-process global state.

Procrastinate registers task functions at import time, which means task bodies
can't be closures over locally-bound state from ``worker_main``. The standard
workaround is a small module that holds module-level globals — set once at
worker startup, read by every task invocation.

The values are ``None`` between import and ``worker_main`` initialization, so
task bodies must check / call :func:`require` to fail loudly if a task fires
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

    from aios.crypto.vault import Vault
    from aios.harness.task_registry import TaskRegistry
    from aios.sandbox.registry import SandboxRegistry


pool: asyncpg.Pool[Any] | None = None
vault: Vault | None = None
worker_id: str | None = None
sandbox_registry: SandboxRegistry | None = None
task_registry: TaskRegistry | None = None


def require_pool() -> asyncpg.Pool[Any]:
    if pool is None:
        raise RuntimeError(
            "aios.harness.runtime.pool is not initialized; "
            "this code is running outside a worker_main context"
        )
    return pool


def require_vault() -> Vault:
    if vault is None:
        raise RuntimeError(
            "aios.harness.runtime.vault is not initialized; "
            "this code is running outside a worker_main context"
        )
    return vault


def require_worker_id() -> str:
    if worker_id is None:
        raise RuntimeError(
            "aios.harness.runtime.worker_id is not initialized; "
            "this code is running outside a worker_main context"
        )
    return worker_id


def require_sandbox_registry() -> SandboxRegistry:
    if sandbox_registry is None:
        raise RuntimeError(
            "aios.harness.runtime.sandbox_registry is not initialized; "
            "this code is running outside a worker_main context"
        )
    return sandbox_registry


def require_task_registry() -> TaskRegistry:
    if task_registry is None:
        raise RuntimeError(
            "aios.harness.runtime.task_registry is not initialized; "
            "this code is running outside a worker_main context"
        )
    return task_registry
