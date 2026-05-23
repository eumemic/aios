"""Integration test: ``archive_connection`` and ``attach_connection``
must not race into an invariant-violating end state (archived
connection row + active binding row).

Pre-fix, ``services.archive_connection`` ran three autocommit
statements on a single pool connection — ``get_connection``,
``get_active_binding``, ``queries.archive_connection`` — with no
transaction or row lock. Concurrent ``attach_connection`` (also
autocommit) could ``insert_binding`` between archive's
``get_active_binding`` (returning ``None``) and the
``archive_connection`` UPDATE. End state:
``connections.archived_at IS NOT NULL`` AND an active row in
``bindings``. The ``bindings.connection_id`` FK accepts archived rows
(no partial ``WHERE archived_at IS NULL``), so the invariant violation
persists silently until the resolver's DETACH safety net (#526/#541)
fires on the next inbound.

The test forces the race deterministically by patching
``queries.get_active_binding`` with a wrapper that sleeps after the
real query returns. Pre-fix the sleep happens between autocommit
statements; concurrent ``attach_connection`` commits during the sleep
window and the archive UPDATE then runs on stale state. Post-fix the
sleep happens inside archive's transaction while it holds ``SELECT
FOR UPDATE`` on the ``connections`` row, so the concurrent attach
blocks behind the lock until archive commits, then re-reads
``archived_at`` and raises :class:`ConflictError` — invariant
preserved.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError
from aios.services import agents as agents_service
from aios.services import connections as connections_service
from aios.services import environments as environments_service

pytestmark = pytest.mark.integration

# Sleep duration inside the patched ``get_active_binding``. Long enough
# that a concurrent autocommit ``insert_binding`` from
# ``attach_connection`` reliably commits during the window pre-fix.
# Post-fix the sleep is held inside archive's locked transaction and
# attach blocks behind the row lock for the duration.
_RACE_WINDOW_S = 0.2


@pytest.fixture
async def pool_acc_a_with_connection_and_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, connection_id, session_id)`` for acc_a."""
    pool = await create_pool(migrated_db_url, min_size=2, max_size=8)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_root', NULL,       TRUE,  'tenant-root'),
                       ('acc_a',    'acc_root', FALSE, 'tenant-a')
                """
            )

        agent_a = await agents_service.create_agent(
            pool,
            account_id="acc_a",
            name="a-agent",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env_a = await environments_service.create_environment(
            pool, account_id="acc_a", name="a-env"
        )
        async with pool.acquire() as conn:
            session = await queries.insert_session(
                conn,
                account_id="acc_a",
                agent_id=agent_a.id,
                environment_id=env_a.id,
                agent_version=agent_a.version,
                title=None,
                metadata={},
            )
            connection = await queries.insert_connection(
                conn,
                account_id="acc_a",
                connector="signal",
                external_account_id="+15550001",
                metadata={},
            )
        yield pool, connection.id, session.id
    finally:
        await pool.close()


class TestArchiveConnectionAttachRace:
    async def test_concurrent_archive_and_attach_preserves_invariant(
        self,
        pool_acc_a_with_connection_and_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """Concurrent ``archive_connection`` + ``attach_connection`` must
        end in an invariant-preserving state.

        Acceptable end states:

        * attach wins → connection NOT archived, active binding exists;
          archive raised ``ConflictError`` (saw the binding under lock).
        * archive wins → connection archived, NO active binding;
          attach raised ``ConflictError`` (saw archived under lock).

        Both ``archived AND active binding`` is the bug — silent
        invariant violation, reachable pre-fix because archive's reads
        and write were unlocked autocommit statements.
        """
        pool, connection_id, session_id = pool_acc_a_with_connection_and_session

        original_get_active_binding = queries.get_active_binding

        async def slow_get_active_binding(
            conn: asyncpg.Connection[Any], connection_id_arg: str, *, account_id: str
        ) -> Any:
            """Pause after the real query returns to widen archive's
            race window. Pre-fix this just sleeps between autocommit
            statements; post-fix it sleeps inside the locked txn."""
            result = await original_get_active_binding(
                conn, connection_id_arg, account_id=account_id
            )
            await asyncio.sleep(_RACE_WINDOW_S)
            return result

        archive_result: BaseException | None = None
        attach_result: BaseException | None = None

        async def _archive() -> None:
            nonlocal archive_result
            try:
                await connections_service.archive_connection(
                    pool, connection_id, account_id="acc_a"
                )
            except BaseException as exc:
                archive_result = exc

        async def _attach() -> None:
            nonlocal attach_result
            try:
                await connections_service.attach_connection(
                    pool, connection_id, account_id="acc_a", session_id=session_id
                )
            except BaseException as exc:
                attach_result = exc

        with patch.object(queries, "get_active_binding", slow_get_active_binding):
            archive_task = asyncio.create_task(_archive())
            # Give archive time to reach the slow get_active_binding
            # — pre-fix it's into the race window; post-fix it has
            # already acquired the FOR UPDATE lock.
            await asyncio.sleep(_RACE_WINDOW_S / 4)
            attach_task = asyncio.create_task(_attach())
            await asyncio.wait_for(
                asyncio.gather(archive_task, attach_task, return_exceptions=True),
                timeout=_RACE_WINDOW_S * 10,
            )

        async with pool.acquire() as conn:
            connection = await queries.get_connection(conn, connection_id, account_id="acc_a")
            binding = await queries.get_active_binding(conn, connection_id, account_id="acc_a")

        invariant_violated = connection.archived_at is not None and binding is not None
        assert not invariant_violated, (
            f"archive_connection + attach_connection race produced "
            f"archived={connection.archived_at is not None} AND "
            f"active_binding={binding is not None} — silent invariant "
            f"violation. archive_result={archive_result!r} "
            f"attach_result={attach_result!r}"
        )
        # Both operations must have either succeeded cleanly or raised
        # ConflictError. Any other exception indicates the fix is
        # malformed (e.g., deadlock, FK violation, etc.).
        for label, result in (("archive", archive_result), ("attach", attach_result)):
            assert result is None or isinstance(result, ConflictError), (
                f"{label} raised an unexpected exception: {result!r}"
            )
