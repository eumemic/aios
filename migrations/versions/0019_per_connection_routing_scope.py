"""Per-connection routing scope: rules and bindings owned by a connection.

Moves ``routing_rules`` and ``channel_bindings`` from globally-addressable
to per-connection-scoped:

* ``routing_rules`` gains ``connection_id`` (FK, CASCADE).  ``prefix`` is
  reinterpreted as the path portion only — the ``{connector}/{account}``
  segments are implicit from the owning connection.  Empty string ``""``
  is the per-connection catch-all.
* ``channel_bindings`` gains ``connection_id`` (FK, CASCADE) plus a
  ``path`` column.  ``address`` is dropped — read-path reconstructs the
  full address by joining to ``connections``.

Data migration: for each existing rule/binding, parse the first two
``/``-segments as ``(connector, account)``, look up the matching
connection, and populate the new columns (rewriting ``prefix`` / ``path``
to the remainder).  Rows referencing an unregistered connection are
hard-deleted with a ``WARNING`` — they were dead weight (no inbound
route could ever fire them).

Revision ID: 0019
Revises: 0018
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0019"
down_revision: str = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── routing_rules ────────────────────────────────────────────────────
    #
    # Add connection_id nullable first so the UPDATE/DELETE can populate or
    # prune before we tighten to NOT NULL.
    op.execute("""
        ALTER TABLE routing_rules
            ADD COLUMN connection_id text
                REFERENCES connections(id) ON DELETE CASCADE
    """)

    # Strip the ``{connector}/{account}/`` prefix from ``prefix``.  The
    # CASE collapses the exact-match case ("rule for the whole connection")
    # to ``''`` so the catch-all convention kicks in cleanly; otherwise
    # we drop the first two segments.
    op.execute("""
        UPDATE routing_rules rr
           SET connection_id = c.id,
               prefix = CASE
                         WHEN rr.prefix = c.connector || '/' || c.account THEN ''
                         ELSE substring(
                             rr.prefix from
                             length(c.connector) + length(c.account) + 3
                         )
                       END
          FROM connections c
         WHERE split_part(rr.prefix, '/', 1) = c.connector
           AND split_part(rr.prefix, '/', 2) = c.account
    """)

    # Log + drop orphans (rules whose connector/account pair has no
    # registered connection — they could never have matched an inbound
    # message routed through a connection).
    op.execute("""
        DO $$
        DECLARE r RECORD;
        BEGIN
          FOR r IN SELECT id, prefix FROM routing_rules WHERE connection_id IS NULL LOOP
            RAISE WARNING
              'migration 0019: dropping orphan routing rule % (prefix=%) — no registered connection',
              r.id, r.prefix;
          END LOOP;
        END $$
    """)
    op.execute("DELETE FROM routing_rules WHERE connection_id IS NULL")

    op.execute("ALTER TABLE routing_rules ALTER COLUMN connection_id SET NOT NULL")
    op.execute("DROP INDEX routing_rules_prefix_uniq")
    op.execute("""
        CREATE UNIQUE INDEX routing_rules_conn_prefix_uniq
            ON routing_rules (connection_id, prefix) WHERE archived_at IS NULL
    """)

    # ── channel_bindings ─────────────────────────────────────────────────
    op.execute("""
        ALTER TABLE channel_bindings
            ADD COLUMN connection_id text
                REFERENCES connections(id) ON DELETE CASCADE,
            ADD COLUMN path text
    """)

    # Populate both new columns from the address string.  ``path`` is
    # the substring after ``{connector}/{account}/`` — always present
    # since addresses require at least three segments (enforced by the
    # inbound-message validator).
    op.execute("""
        UPDATE channel_bindings cb
           SET connection_id = c.id,
               path = substring(
                   cb.address from
                   length(c.connector) + length(c.account) + 3
               )
          FROM connections c
         WHERE split_part(cb.address, '/', 1) = c.connector
           AND split_part(cb.address, '/', 2) = c.account
    """)

    op.execute("""
        DO $$
        DECLARE r RECORD;
        BEGIN
          FOR r IN SELECT id, address FROM channel_bindings WHERE connection_id IS NULL LOOP
            RAISE WARNING
              'migration 0019: dropping orphan channel binding % (address=%) — no registered connection',
              r.id, r.address;
          END LOOP;
        END $$
    """)
    op.execute("DELETE FROM channel_bindings WHERE connection_id IS NULL")

    op.execute("""
        ALTER TABLE channel_bindings
            ALTER COLUMN connection_id SET NOT NULL,
            ALTER COLUMN path SET NOT NULL
    """)
    op.execute("DROP INDEX channel_bindings_address_uniq")
    op.execute("ALTER TABLE channel_bindings DROP COLUMN address")
    op.execute("""
        CREATE UNIQUE INDEX channel_bindings_conn_path_uniq
            ON channel_bindings (connection_id, path) WHERE archived_at IS NULL
    """)


def downgrade() -> None:
    # Restoring the pre-0019 shape is structurally possible but data-lossy
    # (orphans are gone, and the per-connection ``prefix``/``path`` values
    # can't be reconstituted into full addresses without connections still
    # existing).  We reverse the schema ops best-effort and leave the
    # content of the tables as-is where possible.
    op.execute("DROP INDEX IF EXISTS channel_bindings_conn_path_uniq")
    op.execute("ALTER TABLE channel_bindings ADD COLUMN address text")
    op.execute("""
        UPDATE channel_bindings cb
           SET address = c.connector || '/' || c.account || '/' || cb.path
          FROM connections c
         WHERE c.id = cb.connection_id
    """)
    op.execute("ALTER TABLE channel_bindings ALTER COLUMN address SET NOT NULL")
    op.execute("""
        CREATE UNIQUE INDEX channel_bindings_address_uniq
            ON channel_bindings (address) WHERE archived_at IS NULL
    """)
    op.execute("ALTER TABLE channel_bindings DROP COLUMN path")
    op.execute("ALTER TABLE channel_bindings DROP COLUMN connection_id")

    op.execute("DROP INDEX IF EXISTS routing_rules_conn_prefix_uniq")
    # prefix already holds the path-only value; rebuilding the full prefix
    # requires a join.  CASE covers the catch-all ('') case → plain
    # connector/account with no trailing slash.
    op.execute("""
        UPDATE routing_rules rr
           SET prefix = CASE
                          WHEN rr.prefix = '' THEN c.connector || '/' || c.account
                          ELSE c.connector || '/' || c.account || '/' || rr.prefix
                        END
          FROM connections c
         WHERE c.id = rr.connection_id
    """)
    op.execute("""
        CREATE UNIQUE INDEX routing_rules_prefix_uniq
            ON routing_rules (prefix) WHERE archived_at IS NULL
    """)
    op.execute("ALTER TABLE routing_rules DROP COLUMN connection_id")
