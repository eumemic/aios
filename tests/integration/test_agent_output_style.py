"""Round-trip of the versioned ``output_style`` agent field through the DB layer.

Pins the preserve-when-omitted contract of ``PUT /v1/agents/{id}``'s merge
(an update without ``output_style`` keeps the prior value; passing it flips
it and creates a new version) and that ``agent_versions`` history records
the field per version.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.services import agents as agents_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_concise', NULL, TRUE, 'tenant-concise')
                """
            )
        yield pool
    finally:
        await pool.close()


async def _create(pool: asyncpg.Pool[Any], name: str, **kwargs: Any) -> Any:
    return await agents_service.create_agent(
        pool,
        account_id="acc_concise",
        name=name,
        model="openrouter/test",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
        **kwargs,
    )


async def test_output_style_round_trip_and_preserve_when_omitted(pool: asyncpg.Pool[Any]) -> None:
    agent = await _create(pool, "concise-rt", output_style="concise")
    assert agent.output_style == "concise"
    assert agent.version == 1

    fetched = await agents_service.get_agent(pool, agent.id, account_id="acc_concise")
    assert fetched.output_style == "concise"

    # Update WITHOUT the field → prior value preserved, new version created.
    v2 = await agents_service.update_agent(
        pool,
        agent.id,
        account_id="acc_concise",
        expected_version=1,
        system="updated",
    )
    assert v2.version == 2
    assert v2.output_style == "concise"

    # Update WITH output_style="default" → flips, new version.
    v3 = await agents_service.update_agent(
        pool,
        agent.id,
        account_id="acc_concise",
        expected_version=2,
        output_style="default",
    )
    assert v3.version == 3
    assert v3.output_style == "default"

    # agent_versions history records the field per version.
    for version, expected in ((1, "concise"), (2, "concise"), (3, "default")):
        snap = await agents_service.get_agent_version(
            pool, agent.id, version, account_id="acc_concise"
        )
        assert snap.output_style == expected


async def test_output_style_only_update_is_versioned_and_noop_detected(
    pool: asyncpg.Pool[Any],
) -> None:
    agent = await _create(pool, "concise-noop")
    assert agent.output_style == "default"  # the field defaults off

    # Flipping ONLY output_style creates a new version (it is a config field).
    v2 = await agents_service.update_agent(
        pool,
        agent.id,
        account_id="acc_concise",
        expected_version=1,
        output_style="concise",
    )
    assert v2.version == 2
    assert v2.output_style == "concise"

    # Re-sending the current value is a no-op: no new version.
    same = await agents_service.update_agent(
        pool,
        agent.id,
        account_id="acc_concise",
        expected_version=2,
        output_style="concise",
    )
    assert same.version == 2
