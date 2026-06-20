"""Connection, binding, chat-session, routing-rule, and connector-RPC queries.

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, NamedTuple, NoReturn

import asyncpg

from aios.crypto.vault import EncryptedBlob
from aios.db.queries import (
    _escape_like,
    parse_jsonb,
)
from aios.errors import (
    ConflictError,
    NotFoundError,
)
from aios.ids import (
    BINDING,
    CONNECTION,
    make_id,
)
from aios.models.connections import BindingMode, Connection, ConnectionMode
from aios.models.connectors import ConnectorCapabilities

# ─── bindings (#328 PR 7 — unit of curation, succeeded the in-place
#                          ``connections.session_id`` / ``session_template_id``
#                          columns) ────────────────────────────────────────


def _session_bound_to_connection_predicate(
    *,
    connection_alias: str,
    session_param_index: int,
    account_id_param_index: int,
) -> str:
    """SQL fragment for "this session is bound to ``<connection_alias>``."

    Used by every query that walks the connection→session lineage:
    ``is_session_bound_to_connection`` (existence check),
    ``_list_bound_connection_ids`` (filter), ``list_connection_tools_for_session``
    (filter), ``list_pending_calls_for_connector`` (join predicate).

    Two lineage paths after #328 PR 7:

    * an active ``single_session`` binding whose ``session_id`` matches; or
    * a row in ``chat_sessions`` for ``(connection_id, session_id)``.

    Both ``bindings`` and ``chat_sessions`` are account-scoped tables, so
    the predicate filters on ``account_id`` defensively even when the
    outer connections query already filtered on the same account. The
    redundancy is cheap (covered by the existing indexes) and gives the
    SQL layer the same tenant-isolation invariant the function signatures
    promise.
    """
    return f"""(
        EXISTS (SELECT 1 FROM bindings b
                 WHERE b.connection_id = {connection_alias}.id
                   AND b.archived_at IS NULL
                   AND b.session_id = ${session_param_index}
                   AND b.account_id = ${account_id_param_index})
        OR EXISTS (SELECT 1 FROM chat_sessions cs
                    WHERE cs.connection_id = {connection_alias}.id
                      AND cs.session_id = ${session_param_index}
                      AND cs.account_id = ${account_id_param_index})
    )"""


class ActiveBinding(NamedTuple):
    """Read view of a connection's single active binding."""

    id: str
    connection_id: str
    mode: BindingMode
    session_id: str | None
    session_template_id: str | None


def _row_to_active_binding(row: asyncpg.Record) -> ActiveBinding:
    return ActiveBinding(
        id=row["id"],
        connection_id=row["connection_id"],
        mode=row["mode"],
        session_id=row["session_id"],
        session_template_id=row["session_template_id"],
    )


async def get_active_binding(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    *,
    account_id: str,
) -> ActiveBinding | None:
    """Return the connection's active binding, if one exists.

    Returns ``None`` for detached connections. The
    ``bindings_connection_active_uniq`` partial-unique index enforces
    "at most one active binding per connection," so the result is
    unambiguous.
    """
    row = await conn.fetchrow(
        """
        SELECT id, connection_id, mode, session_id, session_template_id
          FROM bindings
         WHERE connection_id = $1 AND archived_at IS NULL
           AND account_id = $2
        """,
        connection_id,
        account_id,
    )
    if row is None:
        return None
    return _row_to_active_binding(row)


async def insert_binding(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    connection_id: str,
    mode: BindingMode,
    session_id: str | None = None,
    session_template_id: str | None = None,
) -> ActiveBinding:
    """Insert a new active binding for ``connection_id``.

    Race-safe via the partial-unique index ``bindings_connection_active_uniq``:
    a concurrent attempt to bind the same connection surfaces as
    :class:`ConflictError`. Missing or archived connection / session /
    template surfaces as :class:`NotFoundError` (we resolve which by
    a follow-up read to keep the error specific).
    """
    new_id = make_id(BINDING)
    try:
        await conn.execute(
            """
            INSERT INTO bindings (id, connection_id, mode,
                                  session_id, session_template_id, account_id)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            new_id,
            connection_id,
            mode,
            session_id,
            session_template_id,
            account_id,
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"connection {connection_id} is already bound; detach or unconfigure first",
            detail={"id": connection_id},
        ) from exc
    except asyncpg.ForeignKeyViolationError as exc:
        await _raise_for_failed_binding_insert(
            conn,
            connection_id=connection_id,
            session_id=session_id,
            session_template_id=session_template_id,
        )
        raise exc  # pragma: no cover — _raise_for_failed_binding_insert is NoReturn
    return ActiveBinding(
        id=new_id,
        connection_id=connection_id,
        mode=mode,
        session_id=session_id,
        session_template_id=session_template_id,
    )


async def archive_active_binding(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    *,
    account_id: str,
    expected_mode: BindingMode | None = None,
) -> ActiveBinding | None:
    """Soft-archive the connection's active binding.

    Returns the now-archived :class:`ActiveBinding`, or ``None`` if no
    matching binding existed.  When ``expected_mode`` is set, the
    archive is guarded by ``bindings.mode = expected_mode`` — a binding
    in the *other* mode is left intact and the call returns ``None``;
    callers diagnose via a follow-up read.
    """
    where = "connection_id = $1 AND archived_at IS NULL AND account_id = $2"
    args: tuple[Any, ...] = (connection_id, account_id)
    if expected_mode is not None:
        where += " AND mode = $3"
        args = (connection_id, account_id, expected_mode)
    row = await conn.fetchrow(
        f"""
        UPDATE bindings
           SET archived_at = now()
         WHERE {where}
        RETURNING id, connection_id, mode, session_id, session_template_id
        """,
        *args,
    )
    if row is None:
        return None
    return _row_to_active_binding(row)


async def _raise_for_failed_binding_insert(
    conn: asyncpg.Connection[Any],
    *,
    connection_id: str,
    session_id: str | None,
    session_template_id: str | None,
) -> NoReturn:
    """Translate an FK violation on bindings into a specific 4xx."""
    existing = await conn.fetchrow(
        "SELECT archived_at FROM connections WHERE id = $1",
        connection_id,
    )
    if existing is None:
        raise NotFoundError(
            f"connection {connection_id} not found",
            detail={"id": connection_id},
        )
    if existing["archived_at"] is not None:
        raise ConflictError(
            f"connection {connection_id} is archived",
            detail={"id": connection_id},
        )
    if session_id is not None:
        raise NotFoundError(
            f"session {session_id} not found",
            detail={"session_id": session_id},
        )
    if session_template_id is not None:
        raise NotFoundError(
            f"session template {session_template_id} not found",
            detail={"session_template_id": session_template_id},
        )
    raise ConflictError(
        f"failed to insert binding for connection {connection_id}",
        detail={"id": connection_id},
    )


# ─── connections ────────────────────────────────────────────────────────────
#
# Three valid mode views (derived from the active binding row in ``bindings``):
#
#   detached       — no active binding row
#   single_session — active binding row with mode='single_session'
#   per_chat       — active binding row with mode='per_chat'
#
# ``Connection.session_id`` / ``session_template_id`` / ``attached_at`` are
# projected from the binding via a LEFT JOIN — there is no per-connection
# session column.

_CONNECTION_COLUMNS = """
    c.*,
    b.session_id           AS binding_session_id,
    b.session_template_id  AS binding_session_template_id,
    b.created_at           AS binding_created_at
""".strip()

_CONNECTION_FROM = """
    connections c
    LEFT JOIN bindings b
           ON b.connection_id = c.id AND b.archived_at IS NULL
""".strip()

# Trailing JOIN for ``UPDATE connections ... RETURNING *`` CTEs that need
# to re-shape the row through ``_row_to_connection``: read the updated
# row's binding via the same LEFT JOIN as a plain SELECT would. The
# input alias ``u`` is the CTE's RETURNING table.
_CONNECTION_UPDATE_CTE_TAIL = """
    SELECT u.*,
           b.session_id           AS binding_session_id,
           b.session_template_id  AS binding_session_template_id,
           b.created_at           AS binding_created_at
      FROM updated u
      LEFT JOIN bindings b
             ON b.connection_id = u.id AND b.archived_at IS NULL
""".strip()


def _row_to_connection(row: asyncpg.Record) -> Connection:
    return Connection(
        id=row["id"],
        connector=row["connector"],
        external_account_id=row["external_account_id"],
        session_id=row["binding_session_id"],
        session_template_id=row["binding_session_template_id"],
        metadata=parse_jsonb(row["metadata"]),
        secrets_set=row["secrets_ciphertext"] is not None,
        created_at=row["created_at"],
        attached_at=row["binding_created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def insert_connection(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    connector: str,
    external_account_id: str,
    metadata: dict[str, Any],
    secrets_blob: EncryptedBlob | None = None,
) -> Connection:
    """Insert a detached connection, idempotent on the active uniqueness key.

    Per plan decision #5, both the explicit ``POST /v1/connections`` and
    the supervisor's auto-create-on-first-inbound path race-safely
    converge on a single row via ``INSERT ... ON CONFLICT DO NOTHING
    RETURNING``. The unique index
    ``(account_id, connector, external_account_id) WHERE archived_at IS
    NULL`` is per-account (relaxed from global by migration 0060 to
    support the reparent primitive #694): two accounts may hold the
    same external identity simultaneously, but a single account may
    not. On same-tenant conflict, re-read; a miss after re-read is an
    archive race (loop once).

    ``secrets_blob`` carries the encrypted credential dict.  ``None``
    leaves both secret columns NULL; the schema's
    ``connections_secrets_pair_ck`` keeps the pair-or-neither invariant
    intact at the storage boundary.

    Use ``attach_connection`` or ``configure_per_chat_connection`` to bind
    a routing mode after creation.
    """
    ciphertext = secrets_blob.ciphertext if secrets_blob is not None else None
    nonce = secrets_blob.nonce if secrets_blob is not None else None
    # Upsert into the connectors catalog so the runtime_tokens /
    # runtimes FK to ``connectors(connector)`` resolves for this type.
    # Migration 0033 backfilled rows for types active at migration time;
    # creating a connection of a fresh type after migration needs this
    # path (#328 PR 5).
    await conn.execute(
        "INSERT INTO connectors (connector) VALUES ($1) ON CONFLICT DO NOTHING",
        connector,
    )
    # Two attempts: the second only fires on the archive-race path
    # (same-tenant row archived between our INSERT and re-read).
    for _ in range(2):
        row = await conn.fetchrow(
            """
            WITH inserted AS (
                INSERT INTO connections (
                    id, connector, external_account_id, metadata,
                    secrets_ciphertext, secrets_nonce, account_id
                )
                VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
                ON CONFLICT (account_id, connector, external_account_id)
                    WHERE archived_at IS NULL DO NOTHING
                RETURNING *
            )
            SELECT i.*,
                   NULL::text        AS binding_session_id,
                   NULL::text        AS binding_session_template_id,
                   NULL::timestamptz AS binding_created_at
              FROM inserted i
            """,
            make_id(CONNECTION),
            connector,
            external_account_id,
            json.dumps(metadata),
            ciphertext,
            nonce,
            account_id,
        )
        if row is not None:
            return _row_to_connection(row)
        existing = await get_connection_for_account(
            conn,
            connector=connector,
            external_account_id=external_account_id,
            account_id=account_id,
        )
        if existing is not None:
            return existing
    # The archive race converges within two iterations under any realistic
    # contention pattern; if a third attempt would still be needed, the
    # system is in a hot insert/archive cycle that no retry resolves.
    raise RuntimeError(
        f"insert_connection({connector=}, {external_account_id=}) exhausted archive-race retries"
    )


async def get_connection(
    conn: asyncpg.Connection[Any], connection_id: str, *, account_id: str
) -> Connection:
    row = await conn.fetchrow(
        f"SELECT {_CONNECTION_COLUMNS} FROM {_CONNECTION_FROM} WHERE c.id = $1 AND c.account_id = $2",
        connection_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(
            f"connection {connection_id} not found",
            detail={"id": connection_id},
        )
    return _row_to_connection(row)


async def set_connection_secrets(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    *,
    account_id: str,
    secrets_blob: EncryptedBlob | None,
) -> Connection:
    """Replace a connection's encrypted secret blob.  Bumps ``updated_at``.

    ``None`` clears the columns; an :class:`EncryptedBlob` writes its
    paired ciphertext + nonce.  The schema's
    ``connections_secrets_pair_ck`` enforces pair-or-neither at the
    storage boundary.

    Refuses on archived rows.
    """
    ciphertext = secrets_blob.ciphertext if secrets_blob is not None else None
    nonce = secrets_blob.nonce if secrets_blob is not None else None
    row = await conn.fetchrow(
        f"""
        WITH updated AS (
            UPDATE connections
               SET secrets_ciphertext = $2,
                   secrets_nonce      = $3,
                   updated_at         = now()
             WHERE id = $1 AND archived_at IS NULL AND account_id = $4
            RETURNING *
        )
        {_CONNECTION_UPDATE_CTE_TAIL}
        """,
        connection_id,
        ciphertext,
        nonce,
        account_id,
    )
    if row is None:
        raise NotFoundError(
            f"connection {connection_id} not found or archived",
            detail={"id": connection_id},
        )
    return _row_to_connection(row)


async def get_connection_secret_blob(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    *,
    account_id: str,
) -> EncryptedBlob | None:
    """Read the encrypted secrets blob for a connection.

    Returns ``None`` if the connection has no secrets configured.  Raises
    :class:`NotFoundError` if the connection itself is missing or archived
    — connector containers should not see "secrets fetch returned empty"
    when the underlying connection is gone.
    """
    row = await conn.fetchrow(
        """
        SELECT secrets_ciphertext, secrets_nonce
          FROM connections
         WHERE id = $1 AND archived_at IS NULL AND account_id = $2
        """,
        connection_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(
            f"connection {connection_id} not found or archived",
            detail={"id": connection_id},
        )
    ciphertext = row["secrets_ciphertext"]
    nonce = row["secrets_nonce"]
    if ciphertext is None or nonce is None:
        return None
    return EncryptedBlob(ciphertext=ciphertext, nonce=nonce)


async def list_connection_tools_for_session(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> list[dict[str, Any]]:
    """Custom tool specs from every active connection bound to ``session_id``.

    Walks the two lineage paths enumerated in
    :func:`is_session_bound_to_connection` (single_session binding,
    per-chat ledger entry) to find the active connections bound to
    this session, then JOINs to ``connectors.tools_schema`` — the
    runtime container is the source of truth for what tools its
    connector type serves (PR 5).  The flattened tool-spec list is
    ready to feed through :func:`tools.registry.to_openai_tools_custom`.
    """
    rows = await conn.fetch(
        f"""
        SELECT cat.tools_schema AS tools
          FROM connectors cat
         WHERE cat.connector IN (
                SELECT DISTINCT c.connector
                  FROM connections c
                 WHERE c.archived_at IS NULL
                   AND c.account_id = $2
                   AND {
            _session_bound_to_connection_predicate(
                connection_alias="c", session_param_index=1, account_id_param_index=2
            )
        }
            )
        """,
        session_id,
        account_id,
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        out.extend(parse_jsonb(row["tools"]))
    return out


async def list_connection_capabilities_for_session(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> dict[str, ConnectorCapabilities]:
    """Typed capability descriptors for every connector type whose active
    connection is bound to ``session_id``, keyed by connector type.

    The capability sibling to :func:`list_connection_tools_for_session`: walks
    the same two lineage paths to the active connections bound to this session,
    then JOINs to ``connectors.capabilities`` — the runtime container is the
    source of truth for what richer renderings its connector type supports.

    A connector row carrying the empty ``'{}'`` row-default validates to the
    empty-floor :class:`ConnectorCapabilities` (every sub-descriptor absent),
    so the caller never has to special-case "no declared capabilities".

    This is the read seam #1335's outbound delta renderer plugs into: it reads
    the declared KIND (``caps.draft_streaming``/``caps.native_buttons``) rather
    than a ``connector == 'slack'`` identity branch.  No in-tree consumer calls
    it yet (the prelude read site is deferred to #1335) — a deliberate seam,
    exercised by the read-symmetry integration test.
    """
    rows = await conn.fetch(
        f"""
        SELECT cat.connector AS connector, cat.capabilities AS capabilities
          FROM connectors cat
         WHERE cat.connector IN (
                SELECT DISTINCT c.connector
                  FROM connections c
                 WHERE c.archived_at IS NULL
                   AND c.account_id = $2
                   AND {
            _session_bound_to_connection_predicate(
                connection_alias="c", session_param_index=1, account_id_param_index=2
            )
        }
            )
        """,
        session_id,
        account_id,
    )
    return {
        row["connector"]: ConnectorCapabilities.model_validate(
            parse_jsonb(row["capabilities"]) or {}
        )
        for row in rows
    }


async def get_connection_for_account(
    conn: asyncpg.Connection[Any],
    connector: str,
    external_account_id: str,
    *,
    account_id: str,
) -> Connection | None:
    """Active connection for ``(connector, external_account_id)`` within
    the caller's tenant, or ``None``."""
    row = await conn.fetchrow(
        f"""
        SELECT {_CONNECTION_COLUMNS}
          FROM {_CONNECTION_FROM}
         WHERE c.connector = $1 AND c.external_account_id = $2 AND c.archived_at IS NULL
           AND c.account_id = $3
        """,
        connector,
        external_account_id,
        account_id,
    )
    if row is None:
        return None
    return _row_to_connection(row)


async def list_connections(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    connector: str | None = None,
    session_id: str | None = None,
    mode: ConnectionMode | None = None,
    limit: int = 50,
    after: str | None = None,
) -> list[Connection]:
    """List active connections.  Optional filters narrow by connector type,
    attached session id, or routing mode (``detached`` / ``single_session``
    / ``per_chat``).

    ``session_id`` filters on the active binding's ``session_id`` —
    only single_session bindings match (per_chat bindings carry a
    ``session_template_id`` instead).  ``mode`` filters on the active
    binding's mode or its absence (detached).
    """
    args: list[Any] = [account_id]
    clauses: list[str] = ["c.archived_at IS NULL", "c.account_id = $1"]
    if connector is not None:
        args.append(connector)
        clauses.append(f"c.connector = ${len(args)}")
    if session_id is not None:
        args.append(session_id)
        clauses.append(f"b.session_id = ${len(args)}")
    if mode is not None:
        clauses.append(_MODE_PREDICATES[mode])
    if after is not None:
        args.append(after)
        clauses.append(f"c.id < ${len(args)}")
    args.append(limit)
    sql = (
        f"SELECT {_CONNECTION_COLUMNS} FROM {_CONNECTION_FROM} "
        f"WHERE {' AND '.join(clauses)} "
        f"ORDER BY c.id DESC LIMIT ${len(args)}"
    )
    rows = await conn.fetch(sql, *args)
    return [_row_to_connection(r) for r in rows]


# Mode predicates filter on the active-binding row, not the legacy
# in-place columns: a connection is detached iff no active binding
# exists; single_session / per_chat iff the active binding carries
# that mode value.
_MODE_PREDICATES: dict[ConnectionMode, str] = {
    "detached": "b.id IS NULL",
    "single_session": "b.mode = 'single_session'",
    "per_chat": "b.mode = 'per_chat'",
}


async def reparent_connection(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    *,
    destination_account_id: str,
    secrets_blob: EncryptedBlob | None = None,
) -> Connection:
    """Transfer an active connection to a different account.

    Updates ``account_id`` in place, preserving the ``connection.id`` so
    any dependent connector-daemon state (signal-cli's ``account.dat``,
    whatsmeow's ``sqlstore.db``, telegram webhook config, etc. — all
    keyed by ``connection.id``) carries over without churn.

    Also rewrites ``account_id`` on every account-scoped child row whose
    routing fate is tied to this connection: the active ``bindings`` row,
    every ``chat_sessions`` row, and every ``routing_rules`` row reachable
    through ``bindings.id``. All three resolver tiers
    (:mod:`aios_connectors.resolver`) filter on ``account_id``: leaving
    these children on the source account would make
    ``get_active_binding`` / ``list_routing_rules_for_connection`` /
    ``lookup_chat_session`` return ``None`` under the destination's scope,
    silently DETACH-dropping every inbound until an operator manually
    re-bound the connection. Doing all four UPDATEs in one statement
    keeps the carry-over atomic — there is no observable mid-reparent
    state where some children point at the source and others at the
    destination.

    Refuses on archived connection rows (the WHERE clause filters them
    out, so a zero-row UPDATE → :class:`NotFoundError`). Archived
    ``bindings`` are still moved (they're inert but stay tenant-coherent
    in case an operator un-archives one).

    Catches the per-account partial-unique index violation
    (``connections_active_account_external_uniq``) and surfaces it as
    :class:`ConflictError` — that's the destination-collision case
    documented in the operator-facing 409 response.

    ``secrets_blob`` rewrites the encrypted secret columns inside the
    same UPDATE. Secrets are keyed to the owning ``account_id`` via
    :meth:`CryptoBox.derive_account_subkey`, so the source-keyed
    ciphertext is unreadable under the destination's derived key.
    Service-layer callers decrypt with the source subkey and re-encrypt
    with the destination subkey before passing the new blob; passing
    ``None`` is reserved for connections that had no secrets to begin
    with (the schema's ``connections_secrets_pair_ck`` keeps the
    pair-or-neither invariant intact when both columns flip to NULL).

    The caller is responsible for the root-operator gate; this query
    deliberately doesn't scope by source ``account_id`` because the
    semantic of reparent is "cross-account move," and the service
    layer is what decides whether the caller is allowed to do that.
    """
    ciphertext = secrets_blob.ciphertext if secrets_blob is not None else None
    nonce = secrets_blob.nonce if secrets_blob is not None else None
    try:
        # Single CTE: move the connection row, then carry every
        # account-scoped child (bindings, chat_sessions, routing_rules)
        # across in the same statement. The child UPDATEs gate on
        # ``EXISTS (SELECT 1 FROM updated)`` so an archived/missing
        # source connection (the ``updated`` CTE returns zero rows)
        # leaves the children alone — the service layer's SELECT FOR
        # UPDATE already rejects archived before calling this query,
        # but the EXISTS gate keeps the query correct in isolation too.
        # routing_rules joins through bindings on connection_id (the
        # binding's account_id was just rewritten; we still match on
        # ``b.connection_id`` because that column is connection-keyed
        # and tenant-neutral).
        row = await conn.fetchrow(
            f"""
            WITH updated AS (
                UPDATE connections
                   SET account_id         = $2,
                       secrets_ciphertext = $3,
                       secrets_nonce      = $4,
                       updated_at         = now()
                 WHERE id = $1 AND archived_at IS NULL
                RETURNING *
            ),
            b_updated AS (
                UPDATE bindings
                   SET account_id = $2
                 WHERE connection_id = $1
                   AND EXISTS (SELECT 1 FROM updated)
                RETURNING id
            ),
            c_updated AS (
                UPDATE chat_sessions
                   SET account_id = $2
                 WHERE connection_id = $1
                   AND EXISTS (SELECT 1 FROM updated)
                RETURNING connection_id
            ),
            r_updated AS (
                UPDATE routing_rules
                   SET account_id = $2
                  FROM bindings b
                 WHERE routing_rules.binding_id = b.id
                   AND b.connection_id = $1
                   AND EXISTS (SELECT 1 FROM updated)
                RETURNING routing_rules.id
            )
            {_CONNECTION_UPDATE_CTE_TAIL}
            """,
            connection_id,
            destination_account_id,
            ciphertext,
            nonce,
        )
    except asyncpg.UniqueViolationError as exc:
        # The destination account already has an active connection for
        # the same ``(connector, external_account_id)`` — operator must
        # archive the existing destination row before reparenting.
        raise ConflictError(
            f"destination account {destination_account_id} already has an active "
            "connection for this (connector, external_account_id)",
            detail={"destination_account_id": destination_account_id},
        ) from exc
    if row is None:
        raise NotFoundError(
            f"connection {connection_id} not found or archived",
            detail={"id": connection_id},
        )
    return _row_to_connection(row)


async def archive_connection(
    conn: asyncpg.Connection[Any], connection_id: str, *, account_id: str
) -> Connection:
    """Soft-archive a connection AND scrub its encrypted secrets.

    Setting ``secrets_ciphertext = NULL`` / ``secrets_nonce = NULL``
    on archive matches the property documented on
    :mod:`aios.crypto.vault` (archived rows do not retain decryptable
    secrets) so a later DB dump or read on an archived row can't
    recover platform credentials.  The pair-or-neither check
    constraint is satisfied because both columns flip together.
    """
    row = await conn.fetchrow(
        f"""
        WITH updated AS (
            UPDATE connections
               SET archived_at        = now(),
                   updated_at         = now(),
                   secrets_ciphertext = NULL,
                   secrets_nonce      = NULL
             WHERE id = $1 AND archived_at IS NULL AND account_id = $2
            RETURNING *
        )
        {_CONNECTION_UPDATE_CTE_TAIL}
        """,
        connection_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(
            f"connection {connection_id} not found or already archived",
            detail={"id": connection_id},
        )
    return _row_to_connection(row)


# ─── chat_sessions (per_chat ledger, #328 PR 7) ─────────────────────────────


async def lookup_chat_session(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    chat_id: str,
    *,
    account_id: str,
) -> str | None:
    """Existing session_id for ``(connection_id, chat_id)``, else ``None``."""
    val: str | None = await conn.fetchval(
        "SELECT session_id FROM chat_sessions WHERE connection_id = $1 AND chat_id = $2 AND account_id = $3",
        connection_id,
        chat_id,
        account_id,
    )
    return val


async def insert_chat_session(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    connection_id: str,
    chat_id: str,
    session_id: str,
) -> str:
    """Race-safe insert: returns the session_id stored after the call.

    On conflict (a concurrent inbound for the same chat already wrote the
    row) returns the *existing* session_id; the caller is then on the
    hook to discard the just-created session as an orphan.
    """
    row = await conn.fetchrow(
        """
        INSERT INTO chat_sessions (connection_id, chat_id, session_id, account_id)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (connection_id, chat_id) DO NOTHING
        RETURNING session_id
        """,
        connection_id,
        chat_id,
        session_id,
        account_id,
    )
    if row is not None:
        return str(row["session_id"])
    existing = await lookup_chat_session(conn, connection_id, chat_id, account_id=account_id)
    if existing is None:
        # CONFLICT means the row existed at INSERT time; if it's gone now
        # the chat session was hard-deleted between the two queries.
        raise NotFoundError(
            f"chat session for ({connection_id}, {chat_id}) vanished after CONFLICT",
            detail={"connection_id": connection_id, "chat_id": chat_id},
        )
    return existing


async def delete_chat_session(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    chat_id: str,
    *,
    account_id: str,
) -> bool:
    """Remove a ``chat_sessions`` row.  Returns ``True`` iff a row was
    actually deleted.

    Used by the operator-bound chat unbind endpoint.  Hard delete (no
    soft-archive): the row is just an operator-curated route, deleting
    it returns the chat to the connection's mode-default fallback.
    """
    result = await conn.execute(
        "DELETE FROM chat_sessions WHERE connection_id = $1 AND chat_id = $2 AND account_id = $3",
        connection_id,
        chat_id,
        account_id,
    )
    return bool(result.endswith(" 1"))


async def get_chat_session_row(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    chat_id: str,
    *,
    account_id: str,
) -> tuple[str, str, datetime] | None:
    """Return ``(chat_id, session_id, created_at)`` for one row, or ``None``.

    Used after :func:`insert_chat_session` to materialise the just-bound
    row's ``created_at`` for the API response without re-listing the
    full per-connection set.
    """
    row = await conn.fetchrow(
        """
        SELECT chat_id, session_id, created_at
          FROM chat_sessions
         WHERE connection_id = $1 AND chat_id = $2 AND account_id = $3
        """,
        connection_id,
        chat_id,
        account_id,
    )
    if row is None:
        return None
    return row["chat_id"], row["session_id"], row["created_at"]


async def list_chat_sessions_for_connection(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    *,
    account_id: str,
) -> list[tuple[str, str, datetime]]:
    """List ``(chat_id, session_id, created_at)`` rows in chat_id order.

    Operator-bound and per-chat-spawned rows are returned together —
    the table doesn't tag the writer, and the union is what an operator
    wants to see when answering "where does each chat on this account
    route?".
    """
    rows = await conn.fetch(
        """
        SELECT chat_id, session_id, created_at
          FROM chat_sessions
         WHERE connection_id = $1 AND account_id = $2
         ORDER BY chat_id
        """,
        connection_id,
        account_id,
    )
    return [(r["chat_id"], r["session_id"], r["created_at"]) for r in rows]


async def list_session_ids_for_connection(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    *,
    account_id: str,
) -> list[str]:
    """List every distinct session_id bound to ``connection_id`` via
    either lineage path:

    * Active single_session binding on this connection.
    * Per-chat-spawned rows in ``chat_sessions`` for this connection.

    Same union :func:`is_session_bound_to_connection` checks for, but
    enumerated — used by the runtime/lifecycle endpoint to broadcast
    a connection-broken event onto every bound session at once.
    """
    rows = await conn.fetch(
        """
        SELECT session_id FROM bindings
         WHERE connection_id = $1
           AND account_id = $2
           AND mode = 'single_session'
           AND archived_at IS NULL
           AND session_id IS NOT NULL
        UNION
        SELECT session_id FROM chat_sessions
         WHERE connection_id = $1
           AND account_id = $2
         ORDER BY session_id
        """,
        connection_id,
        account_id,
    )
    return [r["session_id"] for r in rows]


# ─── routing_rules (#328 PR 2/4 — per-binding prefix demux) ─────────────────


async def list_routing_rules_for_connection(
    conn: asyncpg.Connection[Any],
    connection_id: str,
    *,
    account_id: str,
) -> list[tuple[str, str, str]]:
    """Return ``(prefix, target_type, target_id)`` rules for the active binding.

    Walks ``bindings`` → ``routing_rules`` for the given connection,
    filtered to the one active binding (``WHERE archived_at IS NULL``).
    Empty list if no binding or no rules. The resolver iterates these
    in arbitrary order — at v1 scale operators are expected to keep
    prefix sets disjoint per binding; first-match-wins.
    """
    rows = await conn.fetch(
        """
        SELECT rr.prefix, rr.target_type, rr.target_id
          FROM routing_rules rr
          JOIN bindings b ON b.id = rr.binding_id
         WHERE b.connection_id = $1
           AND b.archived_at IS NULL
           AND b.account_id = $2
        """,
        connection_id,
        account_id,
    )
    return [(row["prefix"], row["target_type"], row["target_id"]) for row in rows]


async def list_recent_chat_ids(
    conn: asyncpg.Connection[Any],
    connector: str,
    external_account_id: str,
    *,
    account_id: str,
    limit: int,
) -> list[tuple[str, datetime]]:
    """Distinct ``(chat_id, last_seen_at)`` for inbound user events
    matching the ``<connector>/<external_account_id>/<chat_id>`` channel prefix.

    Used by the operator's "what chats has this external identity
    produced inbound on?" helper — the input to ``aios connections
    bind-chat`` when the operator doesn't know the chat_id off the top
    of their head.

    The chat_id is the third path segment of the derived
    ``events.channel`` column; events arriving on a different
    ``focal_channel_at_arrival`` still have ``orig_channel`` set to
    their inbound channel, but ``channel`` (derived) collapses them
    correctly.  We filter on user role to skip assistant / tool rows
    that share the channel.
    """
    # Escape LIKE metacharacters in operator-supplied ``connector`` and
    # ``external_account_id``: ``_`` and ``%`` would otherwise act as
    # wildcards against the stored channel, e.g. an operator looking up
    # identity ``bot_a`` would see chats from ``botXa`` too. Mirrors the
    # ``_escape_like`` usage at the memory-prefix query below.
    prefix = f"{_escape_like(connector)}/{_escape_like(external_account_id)}/"
    rows = await conn.fetch(
        """
        SELECT
          split_part(channel, '/', 3) AS chat_id,
          MAX(created_at) AS last_seen_at
        FROM events
        WHERE channel LIKE $1
          AND account_id = $3
          AND kind = 'message'
          AND data->>'role' = 'user'
        GROUP BY chat_id
        ORDER BY last_seen_at DESC
        LIMIT $2
        """,
        prefix + "%",
        limit,
        account_id,
    )
    return [(r["chat_id"], r["last_seen_at"]) for r in rows if r["chat_id"]]


# ─── connector_inbound_acks (dedup ledger) ──────────────────────────────────


async def try_record_inbound_ack(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    connector: str,
    external_account_id: str,
    event_id: str,
    appended_seq: int,
) -> bool:
    """Insert a dedup-ledger row, returning ``True`` iff it actually inserted.

    Called from the worker's inbound handler in the same transaction as
    :func:`append_event`.  The PK
    ``(account_id, connector, external_account_id, event_id)`` enforces
    at-most-once event append: a duplicate inbound (same ULID re-emitted on
    connector reconnect because the previous worker crashed before acking)
    hits ``ON CONFLICT DO NOTHING`` and the caller rolls back the txn so no
    second event lands.

    ``account_id`` is part of the key because two tenants can independently
    hold the same external identity (migration 0060 relaxed connections
    uniqueness to per-account for the #694 reparent primitive) and
    ``event_id`` is a deterministic function of the chat namespace
    (``telegram-{chat}-{msg}``), not a global ULID — without tenant scoping
    one account's ack would swallow another's first-ever delivery of the
    same event_id.

    The ``appended_seq`` is the gapless seq the in-flight ``append_event``
    just allocated; it makes the ledger row queryable for the operator
    debugging "did this message land?".
    """
    row = await conn.fetchrow(
        """
        INSERT INTO connector_inbound_acks (
            account_id, connector, external_account_id, event_id, appended_seq
        )
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT DO NOTHING
        RETURNING 1
        """,
        account_id,
        connector,
        external_account_id,
        event_id,
        appended_seq,
    )
    return row is not None


# ─── connectors (type catalog) ───────────────────────────────────────────────


async def notify_connection_change(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    connector: str,
    connection_id: str,
    external_account_id: str,
    event: str,
) -> None:
    """Emit a ``connections_<connector>`` NOTIFY for discovery SSE consumers.

    Payload: ``"<event>|<connection_id>|<account_id>|<external_account_id>"``
    — the SSE generator parses this into an ``added``/``removed`` event
    and uses ``account_id`` (tenant) to filter cross-tenant events.
    Caller runs this on a pool-acquired (autocommit) connection OUTSIDE
    any transaction so subscribers never see a payload for an
    uncommitted row.
    """
    await conn.execute(
        "SELECT pg_notify($1, $2)",
        f"connections_{connector}",
        f"{event}|{connection_id}|{account_id}|{external_account_id}",
    )


async def update_connector_tools_schema(
    conn: asyncpg.Connection[Any],
    connector: str,
    *,
    account_id: str,
    tools_schema: list[dict[str, Any]],
) -> None:
    """Upsert ``connectors.tools_schema`` for ``connector`` wholesale.

    The runtime container (one per connector type) publishes its full
    tool catalog at startup via ``PUT /v1/connectors/{connector}/tools_schema``.
    A brand-new connector type — one not present at migration 0033's
    backfill time and not yet referenced by any ``insert_connection``
    upsert — can publish its schema before the operator creates its
    first connection.
    """
    await conn.execute(
        """
        INSERT INTO connectors (connector, tools_schema, created_at, updated_at)
        VALUES ($1, $2::jsonb, now(), now())
        ON CONFLICT (connector) DO UPDATE
           SET tools_schema = EXCLUDED.tools_schema,
               updated_at   = now()
        """,
        connector,
        json.dumps(tools_schema),
    )


async def update_connector_capabilities(
    conn: asyncpg.Connection[Any],
    connector: str,
    *,
    account_id: str,
    capabilities: dict[str, Any],
) -> None:
    """Upsert ``connectors.capabilities`` for ``connector`` wholesale.

    The capability sibling to :func:`update_connector_tools_schema`, on the
    same single-row-per-connector-type catalog.  The runtime container
    publishes its typed richness descriptor via
    ``PUT /v1/connectors/{connector}/capabilities``; a brand-new connector type
    can publish its capabilities before its first connection exists.  The
    ``ON CONFLICT (connector)`` upsert keeps one row per type.
    """
    await conn.execute(
        """
        INSERT INTO connectors (connector, capabilities, created_at, updated_at)
        VALUES ($1, $2::jsonb, now(), now())
        ON CONFLICT (connector) DO UPDATE
           SET capabilities = EXCLUDED.capabilities,
               updated_at   = now()
        """,
        connector,
        json.dumps(capabilities),
    )


# ─── pending management calls (operator→connector RPC plane) ──────────


async def insert_management_call(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    call_id: str,
    connector: str,
    method: str,
    params: dict[str, Any],
    expires_at: datetime,
) -> None:
    """Insert a fresh ``pending`` row for ``call_id``."""
    await conn.execute(
        """
        INSERT INTO pending_management_calls
            (id, connector, method, params, expires_at, account_id)
        VALUES ($1, $2, $3, $4::jsonb, $5, $6)
        """,
        call_id,
        connector,
        method,
        json.dumps(params),
        expires_at,
        account_id,
    )


async def list_pending_management_calls_for_connector(
    conn: asyncpg.Connection[Any],
    connector: str,
    *,
    account_id: str,
) -> list[dict[str, Any]]:
    """Pending, unexpired management calls for ``connector`` scoped to ``account_id``.

    Used by the runtime SSE backfill on connector reconnect.  Output dict
    shape::

        {"call_id": "mgmt_...", "method": "register", "params": {...}}

    Filtered by ``account_id`` so a runtime container authenticated for
    one tenant never sees another tenant's pending calls. The partial
    index ``pending_management_calls_connector_account_pending_idx``
    (migration 0049) backs this query directly.
    """
    rows = await conn.fetch(
        """
        SELECT id, method, params
          FROM pending_management_calls
         WHERE connector = $1
           AND account_id = $2
           AND status = 'pending'
           AND expires_at > now()
         ORDER BY created_at ASC
        """,
        connector,
        account_id,
    )
    return [
        {
            "call_id": row["id"],
            "method": row["method"],
            "params": parse_jsonb(row["params"]),
        }
        for row in rows
    ]


async def get_management_call(
    conn: asyncpg.Connection[Any], call_id: str, *, account_id: str
) -> dict[str, Any] | None:
    """Fetch one management call by id, or ``None`` if missing.

    Used by both the runtime SSE NOTIFY tail (to assemble the emit
    payload from the freshly-inserted row), the runtime result-intake
    route (to authorise the caller's bearer scope before the conditional
    UPDATE), and the operator-side wake to fetch the resolved row.
    """
    row = await conn.fetchrow(
        """
        SELECT id, connector, method, params, status, result, is_error
          FROM pending_management_calls
         WHERE id = $1 AND account_id = $2
        """,
        call_id,
        account_id,
    )
    if row is None:
        return None
    return {
        "id": row["id"],
        "connector": row["connector"],
        "method": row["method"],
        "params": parse_jsonb(row["params"]),
        "status": row["status"],
        "result": parse_jsonb(row["result"]) if row["result"] is not None else None,
        "is_error": row["is_error"],
    }


async def mark_management_call_resolved(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    call_id: str,
    result: Any,
    is_error: bool,
) -> bool:
    """Conditional UPDATE: only resolves a still-``pending`` row.

    Returns ``True`` iff this call moved the row from ``pending`` to a
    terminal state.  A second POST from a race / retry gets ``False`` —
    the caller no-ops the NOTIFY so the operator never sees a double wake.
    """
    new_status = "failed" if is_error else "succeeded"
    row = await conn.fetchrow(
        """
        UPDATE pending_management_calls
           SET status      = $2,
               result      = $3::jsonb,
               is_error    = $4,
               resolved_at = now()
         WHERE id = $1
           AND status = 'pending'
           AND account_id = $5
         RETURNING id
        """,
        call_id,
        new_status,
        json.dumps(result),
        is_error,
        account_id,
    )
    return row is not None


async def notify_management_call_dispatch(
    conn: asyncpg.Connection[Any],
    *,
    connector: str,
    call_id: str,
) -> None:
    """NOTIFY the per-connector dispatch channel after inserting a pending row.

    Payload is just ``call_id`` so subscribers re-fetch full details from
    the row; keeps the NOTIFY well under Postgres' 8000-byte cap and
    means an in-flight payload can't desync from a later UPDATE.

    Carries no tenancy info — subscribers fetch the row via
    :func:`get_management_call`, which enforces ``WHERE account_id = $N``.
    """
    await conn.execute(
        "SELECT pg_notify($1, $2)",
        f"connector_management_calls_{connector}",
        call_id,
    )


async def notify_management_call_result(
    conn: asyncpg.Connection[Any],
    *,
    call_id: str,
) -> None:
    """NOTIFY the per-call result channel after resolving the row.

    Payload is empty — listeners re-fetch the resolved row via
    :func:`get_management_call`, mirroring the dispatch-side convention
    (which also lets the fetch enforce tenancy).
    """
    await conn.execute(
        "SELECT pg_notify($1, $2)",
        f"connector_result_{call_id}",
        "",
    )
