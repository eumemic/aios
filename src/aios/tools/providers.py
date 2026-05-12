"""Pluggable per-session tool source for the harness prelude.

The per-step prelude in ``aios.harness.step_context`` needs to merge
session-scoped custom tools into the model's tool list every turn.
Defining the source as a Protocol keeps core code from importing the
subsystem that provides those tools: core calls against
``ToolProvider``, the subsystem registers an implementation at worker
startup via ``aios.harness.runtime.tool_provider``. The dependency
arrow only ever points core → interface ← subsystem.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import asyncpg


@runtime_checkable
class ToolProvider(Protocol):
    """Source of session-scoped custom tools for the per-step prelude."""

    async def list_tools_for_session(
        self, pool: asyncpg.Pool[Any], session_id: str
    ) -> list[dict[str, Any]]:
        """Return the JSON-schema tool dicts available to ``session_id``.

        Each dict is a ToolSpec ready for ``ToolSpec.model_validate`` —
        the same shape ``services.connections.list_tools_for_session``
        returns today.
        """
        ...
