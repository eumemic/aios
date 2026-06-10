"""Add the ``environment_variable`` vault credential kind.

The existing credential kinds (``bearer_header`` / ``oauth2_refresh`` /
``basic`` / ``custom_header``) are keyed by an immutable ``target_url`` and
consumed worker-side as outbound auth headers. The new
``environment_variable`` kind has no ``target_url``: it is materialized into
the sandbox as an env var named ``secret_name`` and carries an
``allowed_hosts`` egress scope (bare hostnames, optionally with a
path-prefix) instead. The secret value still lives in the encrypted blob.

Schema changes:
  - ``target_url`` becomes nullable (NULL for ``environment_variable`` rows).
  - new ``secret_name text`` and ``allowed_hosts text[]`` columns.
  - the ``auth_type`` CHECK is widened to include ``environment_variable``.
  - a new ``vault_credentials_shape_check`` makes the two kinds disjoint and
    self-consistent: ``environment_variable`` rows carry
    ``secret_name``/``allowed_hosts`` (non-empty) and no ``target_url``;
    every other kind carries ``target_url`` and neither new column. The DB
    owns this invariant because later epic stages (#874 materialization,
    #876 swap proxy) read the row directly, bypassing the pydantic layer.
    Element-level validity of ``allowed_hosts`` (hostname grammar, path
    charset) is model-layer only, matching the split in
    ``models/environments.py``.
  - a partial unique index on ``(vault_id, secret_name)`` among active rows,
    the ``environment_variable`` analog of ``vault_credentials_url_uniq``.

Part of #871. Concretizes #171.

Downgrade narrows everything back and is fail-loud (see ``downgrade``).

Revision ID: 0081
Revises: 0080
Create Date: 2026-06-10
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0081"
down_revision: str = "0080"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE vault_credentials ALTER COLUMN target_url DROP NOT NULL")
    op.execute("ALTER TABLE vault_credentials ADD COLUMN secret_name text")
    op.execute("ALTER TABLE vault_credentials ADD COLUMN allowed_hosts text[]")

    op.execute("ALTER TABLE vault_credentials DROP CONSTRAINT vault_credentials_auth_type_check")
    op.execute(
        "ALTER TABLE vault_credentials "
        "ADD CONSTRAINT vault_credentials_auth_type_check "
        "CHECK (auth_type IN "
        "('bearer_header', 'oauth2_refresh', 'basic', 'custom_header', 'environment_variable'))"
    )

    op.execute(
        "ALTER TABLE vault_credentials ADD CONSTRAINT vault_credentials_shape_check CHECK ("
        "(auth_type = 'environment_variable' "
        "AND target_url IS NULL AND secret_name IS NOT NULL "
        "AND allowed_hosts IS NOT NULL AND cardinality(allowed_hosts) > 0) "
        "OR (auth_type <> 'environment_variable' "
        "AND target_url IS NOT NULL AND secret_name IS NULL AND allowed_hosts IS NULL))"
    )

    op.execute(
        "CREATE UNIQUE INDEX vault_credentials_secret_name_uniq "
        "ON vault_credentials (vault_id, secret_name) WHERE archived_at IS NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX vault_credentials_secret_name_uniq")
    op.execute("ALTER TABLE vault_credentials DROP CONSTRAINT vault_credentials_shape_check")
    op.execute("ALTER TABLE vault_credentials DROP CONSTRAINT vault_credentials_auth_type_check")
    # Re-add the narrowed CHECK BEFORE dropping the columns, so it fails loud
    # (aborting the whole migration) if any ``environment_variable`` row
    # exists — INCLUDING archived husks (archive zeroes the secret blob but
    # keeps ``auth_type``, and archived rows are filtered out of API list
    # output, so they are invisible). The ``target_url SET NOT NULL`` restore
    # below is a second fail-loud gate on the same rows. Remedy:
    #   DELETE FROM vault_credentials WHERE auth_type = 'environment_variable';
    op.execute(
        "ALTER TABLE vault_credentials "
        "ADD CONSTRAINT vault_credentials_auth_type_check "
        "CHECK (auth_type IN ('bearer_header', 'oauth2_refresh', 'basic', 'custom_header'))"
    )
    op.execute("ALTER TABLE vault_credentials DROP COLUMN allowed_hosts")
    op.execute("ALTER TABLE vault_credentials DROP COLUMN secret_name")
    op.execute("ALTER TABLE vault_credentials ALTER COLUMN target_url SET NOT NULL")
