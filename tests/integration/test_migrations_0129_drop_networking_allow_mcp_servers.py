"""Migration 0129 strips the retired ``allow_mcp_servers`` key from persisted ``environments.config``.

#1477 collapses the dead ``NetworkPolicy`` value-object family and deletes the
API-honored-but-runtime-ignored ``allow_mcp_servers`` boolean from
``LimitedNetworking``. ``LimitedNetworking`` / ``EnvironmentConfig`` carry
``extra="forbid"``, so a persisted ``environments`` row whose
``config.networking`` still lists ``allow_mcp_servers`` would 500 on read after
the code change ships. Per the chairman clean-break directive there is NO
read-tolerance shim; migration 0129 rewrites the offending rows instead. This
exercises the set-based ``#-`` path removal:

- a ``limited`` row carrying ``allow_mcp_servers`` has exactly that key stripped
  (``allowed_hosts`` / ``allow_package_managers`` preserved verbatim);
- an ``unrestricted`` row (no ``allow_mcp_servers``) is left byte-untouched by the
  WHERE gate;
- a row whose networking is a ``limited`` block WITHOUT the key is untouched too.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any, cast

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

# acc → three environments:
#   env_mcp   — limited networking still carrying the dropped allow_mcp_servers key.
#   env_clean — limited networking WITHOUT the key (untouched by the EXISTS gate).
#   env_unres — unrestricted networking (no key; untouched).
_SEED_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_mig', NULL, TRUE, 'mig-test');
INSERT INTO environments (id, name, config, account_id)
VALUES
  ('env_mcp', 'mcp-env',
   '{"networking": {"type": "limited", "allowed_hosts": ["api.example.com"],
     "allow_package_managers": true, "allow_mcp_servers": true}}'::jsonb,
   'acc_mig'),
  ('env_clean', 'clean-env',
   '{"networking": {"type": "limited", "allowed_hosts": ["cdn.example.com"]}}'::jsonb,
   'acc_mig'),
  ('env_unres', 'unres-env',
   '{"networking": {"type": "unrestricted"}}'::jsonb,
   'acc_mig');
"""


@pytest.fixture
def postgres() -> Iterator[object]:
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _fetch_config(db_url: str, env_id: str) -> dict[str, Any]:
    conn = await asyncpg.connect(db_url)
    try:
        raw = await conn.fetchval("SELECT config FROM environments WHERE id = $1", env_id)
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return cast("dict[str, Any]", parsed)
    finally:
        await conn.close()


async def _execute(db_url: str, sql: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_allow_mcp_servers_key_stripped(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    # Seed AFTER 0128 (the pre-0129 head) so the offending row is pristine.
    up = _run_alembic(["upgrade", "0128"], db_url)
    assert up.returncode == 0, f"upgrade to 0128 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))

    up = _run_alembic(["upgrade", "0129"], db_url)
    assert up.returncode == 0, f"upgrade to 0129 failed:\n{up.stderr}\n{up.stdout}"

    # The key is gone; every sibling field is preserved verbatim.
    mcp = asyncio.run(_fetch_config(db_url, "env_mcp"))
    assert "allow_mcp_servers" not in mcp["networking"]
    assert mcp["networking"] == {
        "type": "limited",
        "allowed_hosts": ["api.example.com"],
        "allow_package_managers": True,
    }

    # A limited row without the key is left untouched by the EXISTS/IS NOT NULL gate.
    clean = asyncio.run(_fetch_config(db_url, "env_clean"))
    assert clean["networking"] == {"type": "limited", "allowed_hosts": ["cdn.example.com"]}

    # An unrestricted row never carried the key → untouched.
    unres = asyncio.run(_fetch_config(db_url, "env_unres"))
    assert unres["networking"] == {"type": "unrestricted"}
