"""Resolve the *effective* inbound-admission policy for a connection.

The policy is stored on the ``connections.inbound_policy`` jsonb column and is
already parsed onto the loaded :class:`~aios.models.connections.Connection` row
by ``_row_to_connection`` (because ``_CONNECTION_COLUMNS`` selects ``c.*``).
This module is therefore I/O-free in the common path — it reads the field off
the already-``account_id``-scoped row and applies the **server default**
(``DenyAll`` / fail-closed) when the column is NULL.

**Connection-only.** There is no account-wide default and no parent-account
walk in v1 (deliberately deferred). The ``account_id`` argument is carried for
signature symmetry with the rest of ``db/queries`` and to leave the door open
for that deferred default; it is intentionally unused today. Cross-tenant
isolation is correct-by-construction: ``get_connection`` filters the token's
``account_id`` before this row is loaded, so two tenants holding the same
external ``chat_id`` resolve independent policies.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.models.connections import Connection
from aios.models.inbound_policy import DenyAll, InboundPolicy


async def resolve_effective_inbound_policy(
    pool: asyncpg.Pool[Any],
    *,
    connection: Connection,
    account_id: str,
) -> InboundPolicy:
    """Return the connection's effective inbound policy.

    NULL ``inbound_policy`` ⇒ the hardcoded server default ``DenyAll``
    (fail-closed). Otherwise the already-validated union member carried on
    the loaded ``connection`` row. No extra round-trips; ``pool`` and
    ``account_id`` are accepted for signature symmetry (see module docstring).
    """
    if connection.inbound_policy is None:
        return DenyAll()
    return connection.inbound_policy
