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

from aios.models.connectors import ConnectorCapabilities


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

    async def list_capabilities_for_session(
        self, pool: asyncpg.Pool[Any], session_id: str
    ) -> dict[str, ConnectorCapabilities]:
        """Return the typed capability descriptors for ``session_id``,
        keyed by connector type.

        The capability sibling to :meth:`list_tools_for_session`.  Shared
        rendering code branches on a declared KIND
        (``caps.draft_streaming is not None``) instead of a
        ``connector == '<type>'`` identity shim.  The only consumer (the #1335
        outbound delta renderer) is not yet in-tree; this is the seam it plugs
        into when it lands.
        """
        ...
