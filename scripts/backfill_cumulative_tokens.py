#!/usr/bin/env python
"""Backfill ``cumulative_tokens`` for existing message events.

Run after migration 0012.  Idempotent: re-running recomputes all values
from scratch using the canonical :func:`approx_tokens` estimator.

Usage::

    set -a && source .env && set +a
    uv run python scripts/backfill_cumulative_tokens.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# Ensure the project root is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aios.db.pool import create_pool
from aios.harness.tokens import approx_tokens


async def backfill(db_url: str) -> None:
    pool = await create_pool(db_url, min_size=1, max_size=4)

    async with pool.acquire() as conn:
        session_rows = await conn.fetch(
            "SELECT DISTINCT session_id FROM events WHERE kind = 'message'"
        )

    total = len(session_rows)
    print(f"Backfilling {total} sessions")

    for i, row in enumerate(session_rows, 1):
        sid = row["session_id"]
        async with pool.acquire() as conn:
            events = await conn.fetch(
                "SELECT id, data FROM events "
                "WHERE session_id = $1 AND kind = 'message' "
                "ORDER BY seq ASC",
                sid,
            )
            running = 0
            updates: list[tuple[int, str]] = []
            for evt in events:
                data = json.loads(evt["data"]) if isinstance(evt["data"], str) else evt["data"]
                running += approx_tokens([data])
                updates.append((running, evt["id"]))

            if updates:
                await conn.executemany(
                    "UPDATE events SET cumulative_tokens = $1 WHERE id = $2",
                    updates,
                )

        if i % 100 == 0 or i == total:
            print(f"  {i}/{total} sessions done")

    await pool.close()
    print("Backfill complete")


def main() -> None:
    db_url = os.environ.get("AIOS_DB_URL")
    if not db_url:
        print("ERROR: AIOS_DB_URL not set", file=sys.stderr)
        sys.exit(1)
    asyncio.run(backfill(db_url))


if __name__ == "__main__":
    main()
