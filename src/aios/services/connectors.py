"""Business logic for connector-type configuration.

Connectors are root-owned: the connector *type* (e.g. ``"telegram"``,
``"signal"``, ``"echo-http"``) is configured once by the root account
and shared across every tenant.  Tenants add their own *connections*
(per-account instances of a connector), but they don't get to add
their own connectors.  This module enforces that boundary on the
publication surface — the route handler is a thin shim.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db import queries
from aios.errors import ForbiddenError


async def update_tools_schema(
    pool: asyncpg.Pool[Any],
    *,
    connector: str,
    account_id: str,
    tools_schema: list[dict[str, Any]],
) -> None:
    """Publish ``connector``'s tool catalog.  Root-only.

    The ``connectors`` table is keyed on connector type only — one
    row per type, shared across every tenant — by design (migration
    0033's "type IS the identity").  A child tenant publishing the
    schema would overwrite the global row, and every other tenant
    whose sessions are bound to a connection of that connector type
    would see the new schema in its model's prelude.  A malicious
    schema (tool names, descriptions, parameter docs) is a
    cross-tenant prompt-injection vector.

    Restricting publication to the root account preserves the
    "connectors configured by root, connections added by tenants"
    architectural invariant while closing the injection surface.
    """
    async with pool.acquire() as conn:
        account = await queries.get_account(conn, account_id)
        if account is None or account.parent_account_id is not None:
            raise ForbiddenError(
                "publishing a connector's tools_schema is reserved for the root account",
                detail={"connector": connector},
            )
        await queries.update_connector_tools_schema(
            conn, connector, account_id=account_id, tools_schema=tools_schema
        )
