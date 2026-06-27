"""Integration test: the boot-admission gate against a real migrated DB (#1575).

The unit tests pin the gate's branch logic with a fake connection; this test
proves the gate runs its real SQL — the alembic-head read and the per-surface
residue aggregate — against an actual Postgres migrated to head with the seeded
registry's surface predicates. It covers:

* a clean migrated DB (alembic at head ≥ every contract_rev, zero residue) is
  admissible, and
* a retired token planted on a registered surface trips the live-residue
  refusal + algedonic alert.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
import structlog

from aios.db.pool import create_pool
from aios.retirements.boot_gate import (
    LiveResidueDetected,
    assert_retirements_admissible,
)
from aios.services import agents as agents_service

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_boot_gate"


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    p = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        yield p
    finally:
        await p.close()


async def test_clean_migrated_db_is_admissible(pool: asyncpg.Pool[Any]) -> None:
    """A DB migrated to head with no retired-token residue passes the gate.

    The conftest migrates to head (≥ every seeded ``contract_rev``) and the
    truncate-reset leaves no rows carrying a retired token, so the gate must
    return cleanly — readiness may flip green.
    """
    await assert_retirements_admissible(pool)


async def test_residue_on_real_surface_refuses(pool: asyncpg.Pool[Any]) -> None:
    """A retired token planted in ``agents.tools`` trips the live-residue refusal.

    ``complete_goal`` is a seeded ``drop`` retirement whose contract migration
    (0122) ran in the migrate-to-head; planting it back onto the real
    ``agents.tools`` JSONB surface is exactly the "new code meets old rows"
    residue the gate exists to refuse — cleanly, with an algedonic alert,
    instead of the #1525 per-row crash-loop.
    """
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ($1, NULL, TRUE, 'boot-gate-test')",
            _ACCOUNT,
        )
    # Create a valid agent via the real service, then UPDATE its tools JSONB to
    # plant a retired builtin directly — modelling an old row written before the
    # token was retired (the model layer would reject it on construction today).
    agent = await agents_service.create_agent(
        pool,
        account_id=_ACCOUNT,
        name="residue-agent",
        model="openrouter/test",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE agents SET tools = $2::jsonb WHERE id = $1",
            agent.id,
            '[{"type": "complete_goal"}]',
        )

    with structlog.testing.capture_logs() as logs, pytest.raises(LiveResidueDetected):
        await assert_retirements_admissible(pool)

    alerts = [
        r
        for r in logs
        if r.get("event") == "boot_gate.live_residue" and r.get("algedonic") is True
    ]
    assert alerts, logs
    # The breach names the real surface it found the token on.
    breaches = alerts[0]["breaches"]
    assert any(b["table"] == "agents" and b["token"] == "complete_goal" for b in breaches)
