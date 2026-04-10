"""Alembic environment for aios.

Uses SQLAlchemy purely as the migration runner — every migration script writes
raw SQL via ``op.execute(...)``. No declarative models, no autogeneration.
The database URL is loaded from ``AIOS_DB_URL`` at runtime so we don't need
to hard-code it in alembic.ini.

Postgres-only.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

# Pull the connection URL from the environment instead of alembic.ini.
db_url = os.environ.get("AIOS_DB_URL")
if not db_url:
    raise RuntimeError("AIOS_DB_URL must be set to run migrations")

# Alembic uses sync SQLAlchemy under the hood for migration application.
# Strip any +async driver suffix that aios might use elsewhere.
if db_url.startswith("postgresql+asyncpg://"):
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
elif db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

config.set_main_option("sqlalchemy.url", db_url)

# We don't use SQLAlchemy ORM models, so no metadata target.
target_metadata = None


def run_migrations_offline() -> None:
    """Generate SQL without connecting (`alembic upgrade --sql`)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the DB and apply migrations."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
