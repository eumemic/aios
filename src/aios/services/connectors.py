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
from pydantic import ValidationError as PydanticValidationError

from aios.db import queries
from aios.errors import ForbiddenError, ValidationError
from aios.models.agents import ToolSpec
from aios.models.connectors import ConnectorCapabilities


def _validate_tools_schema(connector: str, tools_schema: list[dict[str, Any]]) -> None:
    """Fail-fast validation of a connector ``tools_schema`` at the PUT edge.

    Each published entry is read back at step time via
    ``ToolSpec.model_validate`` in ``compute_step_prelude`` (step_context.py).
    A malformed entry raises there — lazily, at step time — crashing the
    prelude for EVERY session that holds a binding to this connector type
    (the ``tools_schema`` row is connector-type-wide). That is a fail-late
    DoS gated on operator/author-controlled input (#1652).

    Validate here, at the authoring boundary, so a bad schema is rejected
    at the operator/author edge (a 422) instead of wedging live sessions.
    The check mirrors the exact model and site the prelude uses, so what
    passes here is what the prelude will accept.
    """
    for index, entry in enumerate(tools_schema):
        try:
            ToolSpec.model_validate(entry)
        except PydanticValidationError as exc:
            raise ValidationError(
                "connector tools_schema entry is not a valid ToolSpec; "
                "publishing it would crash the step prelude for every session "
                "bound to this connector",
                detail={
                    "connector": connector,
                    "index": index,
                    "errors": exc.errors(include_url=False),
                },
            ) from exc


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
        # Fail-fast at the authoring edge (#1652): reject a malformed schema
        # here rather than letting it wedge every bound session's prelude at
        # step time. Runs after the root-auth gate so a child tenant still
        # sees Forbidden (not a validation error that leaks the write path).
        _validate_tools_schema(connector, tools_schema)
        await queries.update_connector_tools_schema(
            conn, connector, account_id=account_id, tools_schema=tools_schema
        )


async def update_capabilities(
    pool: asyncpg.Pool[Any],
    *,
    connector: str,
    account_id: str,
    capabilities: ConnectorCapabilities,
) -> None:
    """Publish ``connector``'s typed capability descriptor.  Root-only.

    The capability sibling to :func:`update_tools_schema`, carrying the
    byte-identical root-gate.  ``capabilities`` is a property of the connector
    *type* (one row per type, shared across every tenant), so a child tenant
    publishing it would overwrite the global row and change how every other
    tenant's session bound to a connection of that type is rendered.  The
    cross-tenant rationale that motivates the ``tools_schema`` root-gate applies
    identically — capabilities are render metadata at the same altitude.

    Restricting publication to the root account preserves the "connectors
    configured by root, connections added by tenants" architectural invariant.
    """
    async with pool.acquire() as conn:
        account = await queries.get_account(conn, account_id)
        if account is None or account.parent_account_id is not None:
            raise ForbiddenError(
                "publishing a connector's capabilities is reserved for the root account",
                detail={"connector": connector},
            )
        await queries.update_connector_capabilities(
            conn, connector, account_id=account_id, capabilities=capabilities.model_dump()
        )
