"""``ToolProvider`` implementation for the connector subsystem.

PR 4: delegates to ``aios.services.connections.list_tools_for_session``
so behavior is identical to pre-#328. PR 7 swaps the data source to
read directly from ``bindings`` joined to ``connectors.tools_schema``
(the new subsystem tables landed in PR 2). The Protocol indirection
is what's load-bearing here — once the harness calls against
``ToolProvider`` rather than importing core services directly, the
core→subsystem-data import is broken regardless of which underlying
SQL the impl reads.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.models.connectors import ConnectorCapabilities
from aios.services import connections as connections_service
from aios.services import sessions as sessions_service


class SubsystemToolProvider:
    """Default subsystem ``ToolProvider`` impl, registered at worker startup.

    Concrete class (not just the Protocol) so the worker's import is
    explicit; ``aios.tools.providers.ToolProvider`` only checks
    structural conformance, but a named class keeps stack traces and
    log lines self-documenting.
    """

    async def list_tools_for_session(
        self, pool: asyncpg.Pool[Any], session_id: str
    ) -> list[dict[str, Any]]:
        account_id = await sessions_service.load_session_account_id(pool, session_id)
        return await connections_service.list_tools_for_session(
            pool, session_id, account_id=account_id
        )

    async def list_capabilities_for_session(
        self, pool: asyncpg.Pool[Any], session_id: str
    ) -> dict[str, ConnectorCapabilities]:
        account_id = await sessions_service.load_session_account_id(pool, session_id)
        return await connections_service.list_capabilities_for_session(
            pool, session_id, account_id=account_id
        )
