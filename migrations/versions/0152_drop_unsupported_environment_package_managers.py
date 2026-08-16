"""Drop unsupported package-manager keys from persisted environment configs.

Revision ID: 0152
Revises: 0151
"""

from __future__ import annotations

from alembic import op

revision = "0152"
down_revision = "0151"
branch_labels = None
depends_on = None


_SUPPORTED_PACKAGE_MANAGERS = "ARRAY['apt', 'pip', 'npm', 'cargo', 'gem', 'go']"


def upgrade() -> None:
    op.execute(
        f"""
        UPDATE environments AS environment
        SET config = jsonb_set(
            environment.config,
            '{{packages}}',
            COALESCE(
                (SELECT jsonb_object_agg(package.key, package.value)
                 FROM jsonb_each(environment.config->'packages') AS package
                 WHERE package.key = ANY ({_SUPPORTED_PACKAGE_MANAGERS})),
                '{{}}'::jsonb
            )
        )
        WHERE environment.config ? 'packages'
          AND jsonb_typeof(environment.config->'packages') = 'object'
          AND EXISTS (
              SELECT 1 FROM jsonb_object_keys(environment.config->'packages') AS manager
              WHERE manager <> ALL ({_SUPPORTED_PACKAGE_MANAGERS})
          )
        """
    )


def downgrade() -> None:
    pass
