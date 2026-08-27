"""Browser control plane: takeover grants + worker-consumed calls (jarbot#106).

``browser_grants`` is the control-plane record of a human takeover of an
account's shared browser — worker-written after the driver acks the epoch
barrier, API-read with standard scoped queries. The partial unique index
makes "at most one open grant per computer" true by construction. ``handback``
persists the post-human snapshot/signed-in delta so the product layer can
fetch it after a TTL expiry (the synchronous close returns it inline);
screenshot bytes stay on the host plane — only the path crosses.

``browser_calls`` is the worker-consumed RPC row family (LISTEN-before-INSERT
+ NOTIFY, mirroring ``pending_management_calls`` 0041/0049 minus the
``connector`` column): its consumer is the WORKER's browser-call listener,
never a connector runtime, so it gets its own table + channel family rather
than a ``connector`` value another runtime token could subscribe to.

Revision ID: 0175
Revises: 0174
"""

from alembic import op

revision = "0175"
down_revision = "0174"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE browser_grants (
            id            text PRIMARY KEY,
            account_id    text NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            -- The requesting agent session. No FK: the grant audit must
            -- outlive event pruning / session archival (the trun precedent).
            session_id    text NOT NULL,
            status        text NOT NULL DEFAULT 'open'
                          CHECK (status IN ('open', 'closed', 'expired')),
            reason        text NOT NULL DEFAULT '',
            -- Driver boot ULID + epoch issued at open: the viewer pins these
            -- and refuses frames/input that do not match (trusted chrome).
            boot          text NOT NULL,
            epoch         bigint NOT NULL,
            target        jsonb NOT NULL DEFAULT '{}'::jsonb,
            ttl_seconds   integer NOT NULL,
            outcome       text,
            handback      jsonb,
            created_at    timestamptz NOT NULL DEFAULT now(),
            heartbeat_at  timestamptz NOT NULL DEFAULT now(),
            closed_at     timestamptz
        );
        CREATE INDEX browser_grants_account_idx
            ON browser_grants (account_id, created_at);
        CREATE UNIQUE INDEX browser_grants_one_open_per_account
            ON browser_grants (account_id) WHERE status = 'open';
        CREATE INDEX browser_grants_open_heartbeat_idx
            ON browser_grants (heartbeat_at) WHERE status = 'open';
        """
    )
    op.execute(
        """
        CREATE TABLE browser_calls (
            id           text PRIMARY KEY,
            account_id   text NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            method       text NOT NULL
                         CHECK (method IN ('open', 'close', 'status',
                                           'revoke_site', 'clear_state')),
            params       jsonb NOT NULL,
            status       text NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending', 'succeeded', 'failed')),
            result       jsonb,
            is_error     boolean NOT NULL DEFAULT false,
            created_at   timestamptz NOT NULL DEFAULT now(),
            expires_at   timestamptz NOT NULL,
            resolved_at  timestamptz
        );
        -- The listener's on-(re)connect redrive sweep (lost-NOTIFY recovery).
        CREATE INDEX browser_calls_pending_idx
            ON browser_calls (created_at) WHERE status = 'pending';
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE browser_calls")
    op.execute("DROP TABLE browser_grants")
