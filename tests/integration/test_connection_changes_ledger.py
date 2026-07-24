"""Concurrency + retention regressions for the connection_changes ledger (#1905).

Pins the two properties the adversarial review (PR #1938) proved the
first cut lacked, against a real Postgres:

1. **Commit order == seq order within a stream.**  Identity ``seq`` is
   allocated at INSERT, not commit, so without serialization an
   out-of-order commit (X allocates 100 and stalls; Y commits 101; a
   reader advances to 101; X commits 100) permanently skips a committed
   change.  ``insert_connection_change`` closes this by taking a
   transaction-scoped advisory lock on the ``(account, connector)``
   stream *before* allocating — the second writer blocks until the first
   commits, so a reader can never observe seq N while any seq < N in the
   stream is still uncommitted.

2. **The pruning watermark fails closed on an empty ledger.**  The
   durable ``connection_change_horizons`` row survives the DELETE it
   describes, unlike the old derived ``MIN(seq)`` floor (NULL once
   retention empties the table → every stale cursor accepted).
"""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg
import pytest

from aios.db import queries

pytestmark = pytest.mark.integration


async def _insert(
    conn: asyncpg.Connection[Any], *, connector: str = "matrix", connection_id: str
) -> int:
    return await queries.insert_connection_change(
        conn,
        account_id="acc_a",
        connector=connector,
        kind="added",
        connection_id=connection_id,
        external_account_id="ext",
    )


class TestCommitOrderSerialization:
    async def test_out_of_order_commit_cannot_skip_a_change(
        self, conn_two_accounts: asyncpg.Connection[Any], migrated_db_url: str
    ) -> None:
        """The exact loss scenario from the review, live: writer X inserts
        and stalls uncommitted; writer Y (same stream) must BLOCK on the
        advisory lock rather than allocate-and-commit a higher seq.  A
        reader polling MAX(seq) therefore never advances past an
        uncommitted change — no committed row is ever skipped."""
        conn_x = await asyncpg.connect(migrated_db_url)
        conn_y = await asyncpg.connect(migrated_db_url)
        try:
            tx_x = conn_x.transaction()
            await tx_x.start()
            seq_x = await _insert(conn_x, connection_id="con_x")

            async def _y_insert_committed() -> int:
                async with conn_y.transaction():
                    return await _insert(conn_y, connection_id="con_y")

            y_task = asyncio.create_task(_y_insert_committed())
            await asyncio.sleep(0.3)
            # Y is stuck behind X's advisory lock — it has not even
            # allocated a seq yet, let alone committed one.
            assert not y_task.done()

            # Reader's view while X is in flight: the stream high-water is
            # 0/previous — critically NOT some seq above X's.
            high_water = await queries.get_connection_change_high_water(
                conn_two_accounts, account_id="acc_a", connector="matrix"
            )
            assert high_water < seq_x

            await tx_x.commit()
            seq_y = await asyncio.wait_for(y_task, timeout=5.0)
            assert seq_y > seq_x

            # After both commit, replay from any observed watermark is
            # complete: everything above high_water is exactly {X, Y}.
            rows = await queries.list_connection_changes(
                conn_two_accounts, account_id="acc_a", connector="matrix", after_seq=high_water
            )
            assert [int(r["seq"]) for r in rows] == [seq_x, seq_y]
        finally:
            await conn_x.close()
            await conn_y.close()

    async def test_unrelated_streams_do_not_contend(
        self, conn_two_accounts: asyncpg.Connection[Any], migrated_db_url: str
    ) -> None:
        """Serialization is per-(account, connector): a different connector's
        writer sails past an open transaction in another stream."""
        conn_x = await asyncpg.connect(migrated_db_url)
        conn_y = await asyncpg.connect(migrated_db_url)
        try:
            tx_x = conn_x.transaction()
            await tx_x.start()
            await _insert(conn_x, connector="matrix", connection_id="con_x")

            async def _other_stream() -> int:
                async with conn_y.transaction():
                    return await _insert(conn_y, connector="signal", connection_id="con_s")

            # Completes while X's matrix-stream transaction is still open.
            seq_s = await asyncio.wait_for(_other_stream(), timeout=5.0)
            assert seq_s > 0
            await tx_x.rollback()
        finally:
            await conn_x.close()
            await conn_y.close()


class TestPruningWatermark:
    async def test_watermark_survives_emptying_the_ledger(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        """Retention deletes every row; the durable horizon still reports
        the pruned-through seq, so a stale tail cursor is still caught
        (the old MIN(seq) floor returned NULL here and failed open)."""
        conn = conn_two_accounts
        seqs = [await _insert(conn, connection_id=f"con_{i}") for i in range(3)]
        await queries.set_connection_change_pruned_through(
            conn, account_id="acc_a", connector="matrix", pruned_through_seq=seqs[-1]
        )
        await conn.execute("DELETE FROM connection_changes WHERE account_id = 'acc_a'")

        assert (
            await queries.get_connection_change_pruned_through(
                conn, account_id="acc_a", connector="matrix"
            )
            == seqs[-1]
        )

    async def test_watermark_is_monotonic_and_per_stream(
        self, conn_two_accounts: asyncpg.Connection[Any]
    ) -> None:
        conn = conn_two_accounts
        await queries.set_connection_change_pruned_through(
            conn, account_id="acc_a", connector="matrix", pruned_through_seq=10
        )
        # A stale pruner run must not move it backwards.
        await queries.set_connection_change_pruned_through(
            conn, account_id="acc_a", connector="matrix", pruned_through_seq=5
        )
        assert (
            await queries.get_connection_change_pruned_through(
                conn, account_id="acc_a", connector="matrix"
            )
            == 10
        )
        # Never-pruned streams report 0.
        assert (
            await queries.get_connection_change_pruned_through(
                conn, account_id="acc_a", connector="signal"
            )
            == 0
        )
