"""Pluggable per-session tool source for the harness prelude.

The per-step prelude (``aios.harness.step_context.compute_step_prelude``)
needs to merge session-scoped custom tools into the model's tool list
every turn. Today those tools come from the connector-area
``services.connections.list_tools_for_session`` query directly. After
#328 PR 4 they'll come from the new ``aios_connectors`` subsystem
instead.

The point of this Protocol is to keep core code from importing
subsystem code: core calls against ``ToolProvider``; the subsystem
registers an implementation at worker startup via
``aios.harness.runtime.tool_provider``. The arrow only ever points
core → interface ← subsystem.

PR 3 just lands the contract. The slot in ``runtime`` is reserved but
nothing registers against it yet; the rewire of ``step_context.py`` is
PR 4's job, atomically with the subsystem impl.
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
