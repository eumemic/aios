"""Account-scoped LiteLLM defaults and root Claude posture.

Revision ID: 0167
Revises: 0166
"""

from alembic import op

revision = "0167"
down_revision = "0166"
branch_labels = None
depends_on = None

_DEFAULTS = '\'{"thinking": {"type": "adaptive", "display": "summarized"}}\'::jsonb'


def upgrade() -> None:
    op.execute(
        "ALTER TABLE model_providers ADD COLUMN litellm_defaults jsonb NOT NULL DEFAULT '{}'::jsonb"
    )
    op.execute(
        "INSERT INTO model_providers "
        "(id, account_id, provider, api_base, ciphertext, nonce, litellm_defaults) "
        "SELECT 'mpr_root_anthropic_defaults', id, 'anthropic', NULL, ''::bytea, ''::bytea, "
        f"{_DEFAULTS} FROM accounts WHERE parent_account_id IS NULL AND archived_at IS NULL "
        "ON CONFLICT (account_id, provider) WHERE archived_at IS NULL "
        f"DO UPDATE SET litellm_defaults = {_DEFAULTS}, updated_at = now()"
    )


def downgrade() -> None:
    op.execute(
        "DELETE FROM model_providers WHERE id = 'mpr_root_anthropic_defaults' AND ciphertext = ''::bytea"
    )
    op.execute("ALTER TABLE model_providers DROP COLUMN litellm_defaults")
