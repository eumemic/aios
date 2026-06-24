"""Per-connection inbound-admission policy column + one-time known-chats backfill.

Part of #1500 (the inbound-admission epic, "the spine"). The connector inbound
path was implicitly fail-open; this migration carries the *data* half of the
fail-open→fail-closed flip:

1. ``ALTER TABLE connections ADD COLUMN inbound_policy jsonb`` — nullable, no
   DB default. NULL resolves to the server default ``DenyAll`` at read time
   (``aios.db.queries.inbound_policy.resolve_effective_inbound_policy``). The
   shape is validated on the write path by the pydantic discriminated union,
   not a DB CHECK — same posture as the triggers ``source`` / ``action`` jsonb.

2. A one-time **known-chats backfill** so the flip does not lock the install
   base out of its own live agents. Per connection, ``inbound_policy`` is set to
   ``AllowList(chat_ids=<known set>)`` where the known set is the **union** of:
     a. ``chat_id`` (verbatim) from this connection's ``chat_sessions`` ledger
        rows, and
     b. ``chat_id`` parsed from this connection's historical ``role='user'``
        events — taking the **whole remainder** after the
        ``'{connector}/{external_account_id}/'`` prefix.
   A connection with no rows in either source backfills to ``DenyAll``.

   **Slash-correctness (load-bearing):** the events scan does NOT reuse
   ``list_recent_chat_ids``'s ``split_part(channel, '/', 3)`` — that truncates a
   slash-bearing ``chat_id`` (e.g. a Signal group id) and would lock the person
   out. We strip the prefix and keep the whole remainder via ``substring``.

   **Prefix-scoping:** a session can carry events from multiple connections, so
   the events scan is scoped to *this* connection's own
   ``'{connector}/{external_account_id}/'`` prefix, with LIKE metacharacters in
   ``connector`` / ``external_account_id`` escaped (mirrors ``_escape_like``).

Additive only — no expand/contract, no row rewrites of existing columns; it
deploys cleanly in the new-code/old-schema window. ``downgrade()`` drops the
column.

NOTE: no ``@dataclass`` in this module — alembic loads migration modules under a
synthetic module name and ``@dataclass`` crashes there.

Revision ID: 0121
Revises: 0120
Create Date: 2026-06-24
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0121"
down_revision: str = "0120"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE connections ADD COLUMN inbound_policy jsonb")

    # One-time known-chats backfill. The known set per connection is the UNION
    # of the chat_sessions ledger and the slash-safe events scan. The events
    # table carries its own ``account_id`` and a derived ``channel`` column
    # (the same column ``list_recent_chat_ids`` reads), so no join to sessions
    # is needed.
    #
    # ``substring(e.channel FROM length(c.connector) + length(c.external_account_id) + 3)``
    # is the WHOLE remainder past the ``"{connector}/{external_account_id}/"``
    # prefix: two segment lengths + three '/' (one between the two segments,
    # one after the second, and the +1 of SQL's 1-based ``substring`` start) —
    # i.e. start offset = len(connector) + len(external_account_id) + 3. This
    # deliberately does NOT use ``split_part(channel,'/',3)``, which would
    # truncate a slash-bearing chat_id.
    #
    # LIKE metacharacters ('%' and '_') in ``connector`` / ``external_account_id``
    # are escaped so an identity like ``bot_a`` doesn't accidentally match
    # ``botXa``'s channels.
    op.execute(
        r"""
        WITH known AS (
            SELECT c.id AS connection_id, cs.chat_id
              FROM connections c
              JOIN chat_sessions cs ON cs.connection_id = c.id
          UNION
            SELECT c.id AS connection_id,
                   substring(
                       e.channel
                       FROM (length(c.connector) + length(c.external_account_id) + 3)
                   ) AS chat_id
              FROM connections c
              JOIN events e
                ON e.account_id = c.account_id
               AND e.kind = 'message'
               AND e.data->>'role' = 'user'
               AND e.channel LIKE replace(replace(c.connector, '%', '\%'), '_', '\_')
                               || '/'
                               || replace(replace(c.external_account_id, '%', '\%'), '_', '\_')
                               || '/%'
        )
        UPDATE connections c SET inbound_policy =
            CASE
                WHEN k.chat_ids IS NULL OR array_length(k.chat_ids, 1) = 0
                    THEN '{"kind":"deny_all"}'::jsonb
                ELSE jsonb_build_object(
                    'kind', 'allow_list',
                    'chat_ids', to_jsonb(k.chat_ids)
                )
            END
        FROM (
            SELECT connection_id, array_agg(DISTINCT chat_id) AS chat_ids
              FROM known
             WHERE chat_id IS NOT NULL AND chat_id <> ''
             GROUP BY connection_id
        ) k
        WHERE k.connection_id = c.id
        """
    )
    # Connections matching no ``known`` row keep ``inbound_policy`` NULL, which
    # resolves to the server default ``DenyAll`` (a zero-history connection
    # backfills fail-closed). Equivalently we could write ``deny_all``
    # explicitly; NULL→DenyAll makes that write optional.


def downgrade() -> None:
    op.execute("ALTER TABLE connections DROP COLUMN IF EXISTS inbound_policy")
