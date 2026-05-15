"""Integration tests: agent window bounds are validated on both writers.

Pre-fix: ``services.create_agent`` validated ``window_min < window_max``
at the service entrance, but the matching check was missing from the
update path. ``services.update_agent`` forwarded ``window_min`` /
``window_max`` to ``queries.update_agent``, which does partial-merge
(omitted = current state). A one-sided ``PUT`` setting only
``window_max`` to a value at-or-below the current ``window_min``
produced an agent version with ``window_min == window_max``. The next
session step then called ``read_windowed_events`` →
``harness.tokens.tokens_to_drop``, computing
``chunk = window_max - window_min`` (= 0) and raising
``ZeroDivisionError`` — the session burned its retry budget and landed
in terminal ``errored``.

The fix moves both checks to the queries layer (``insert_agent`` and
``update_agent``), so any caller — service, future direct caller,
batch backfill — gets the same invariant in one place. The update-side
check fires on the *resolved* (post-merge) values, which is the only
altitude at which the partial-merge case can be caught.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.errors import ValidationError
from aios.models.agents import ToolSpec
from aios.services import agents as agents_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_and_account(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    """Yield a pool + seeded account id."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_test', NULL, TRUE, 'tenant-test')
                """
            )
        yield pool, "acc_test"
    finally:
        await pool.close()


async def _seed_valid_agent(pool: asyncpg.Pool[Any], account_id: str, name: str) -> Any:
    return await agents_service.create_agent(
        pool,
        account_id=account_id,
        name=name,
        model="openrouter/test",
        system="",
        tools=[ToolSpec(type="bash")],
        description=None,
        metadata={},
        window_min=5_000,
        window_max=10_000,
    )


class TestUpdateAgentWindowValidation:
    async def test_partial_put_setting_window_max_to_current_min_is_rejected(
        self, pool_and_account: tuple[asyncpg.Pool[Any], str]
    ) -> None:
        """PUT setting only ``window_max`` equal to current ``window_min`` must raise.

        After the queries-level merge, this produces
        ``new_wmin == new_wmax`` — a downstream session step would
        ``ZeroDivisionError`` in ``tokens_to_drop`` without this guard.
        """
        pool, account_id = pool_and_account
        agent = await _seed_valid_agent(pool, account_id, "bound-test")

        with pytest.raises(ValidationError) as exc_info:
            await agents_service.update_agent(
                pool,
                agent.id,
                account_id=account_id,
                expected_version=agent.version,
                window_max=5_000,
            )

        assert exc_info.value.detail == {"window_min": 5_000, "window_max": 5_000}

    async def test_partial_put_setting_window_min_to_current_max_is_rejected(
        self, pool_and_account: tuple[asyncpg.Pool[Any], str]
    ) -> None:
        """Symmetric merge direction: bumping ``window_min`` up to the
        current max also produces an invalid resolved pair."""
        pool, account_id = pool_and_account
        agent = await _seed_valid_agent(pool, account_id, "bound-test-reverse")

        with pytest.raises(ValidationError) as exc_info:
            await agents_service.update_agent(
                pool,
                agent.id,
                account_id=account_id,
                expected_version=agent.version,
                window_min=10_000,
            )

        assert exc_info.value.detail == {"window_min": 10_000, "window_max": 10_000}

    async def test_simultaneous_both_invalid_pair_is_rejected(
        self, pool_and_account: tuple[asyncpg.Pool[Any], str]
    ) -> None:
        """PUT supplying both ``window_min`` and ``window_max`` as an
        invalid pair must raise — the queries-layer guard fires on
        resolved values, so the both-provided path is covered too.
        """
        pool, account_id = pool_and_account
        agent = await _seed_valid_agent(pool, account_id, "bound-test-both")

        with pytest.raises(ValidationError) as exc_info:
            await agents_service.update_agent(
                pool,
                agent.id,
                account_id=account_id,
                expected_version=agent.version,
                window_min=8_000,
                window_max=8_000,
            )

        assert exc_info.value.detail == {"window_min": 8_000, "window_max": 8_000}


class TestCreateAgentWindowValidation:
    async def test_create_rejects_invalid_pair(
        self, pool_and_account: tuple[asyncpg.Pool[Any], str]
    ) -> None:
        """Create still rejects invalid pairs — the symmetric guard was
        moved from ``services.create_agent`` down into
        ``queries.insert_agent`` so both writers enforce the same
        invariant in one place."""
        pool, account_id = pool_and_account

        with pytest.raises(ValidationError) as exc_info:
            await agents_service.create_agent(
                pool,
                account_id=account_id,
                name="invalid-bounds",
                model="openrouter/test",
                system="",
                tools=[ToolSpec(type="bash")],
                description=None,
                metadata={},
                window_min=10_000,
                window_max=5_000,
            )

        assert exc_info.value.detail == {"window_min": 10_000, "window_max": 5_000}
