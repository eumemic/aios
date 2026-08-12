#!/usr/bin/env python
"""Race-free, idempotent activation of image-aware token baseline v2."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aios.db.pool import create_pool
from aios.db.queries import parse_jsonb
from aios.db.queries.events import _MESSAGE_CONTENT_CLASSES, _message_content_class
from aios.harness.tokens import approx_tokens, approx_tokens_by_class


async def backfill(db_url: str) -> None:
    pool = await create_pool(db_url, min_size=1, max_size=4)
    async with pool.acquire() as conn:
        session_rows = await conn.fetch(
            "SELECT id FROM sessions WHERE token_baseline_v < 2 ORDER BY id"
        )

    total = len(session_rows)
    print(f"Backfilling {total} sessions")
    for index, row in enumerate(session_rows, 1):
        session_id = row["id"]
        async with pool.acquire() as conn, conn.transaction():
            # The same session-row lock append_event uses to allocate ordinals
            # fences appends for the complete replay and marker flip.
            marker = await conn.fetchval(
                "SELECT token_baseline_v FROM sessions WHERE id = $1 FOR UPDATE",
                session_id,
            )
            if marker == 2:
                continue
            events = await conn.fetch(
                "SELECT id, seq, role, data FROM events "
                "WHERE session_id = $1 AND kind = 'message' ORDER BY seq",
                session_id,
            )
            running = 0
            running_messages = 0
            running_mass = {c: 0 for c in _MESSAGE_CONTENT_CLASSES}
            for event in events:
                data = parse_jsonb(event["data"])
                delta = approx_tokens([data])
                by_class = approx_tokens_by_class([data])
                dominant = _message_content_class(event["role"], data)
                by_class[dominant] += delta - sum(by_class[c] for c in _MESSAGE_CONTENT_CLASSES)
                running += delta
                running_messages += int(event["role"] in ("user", "assistant"))
                for content_class in _MESSAGE_CONTENT_CLASSES:
                    running_mass[content_class] += by_class[content_class]
                await conn.execute(
                    "UPDATE events SET cumulative_tokens=$1, cumulative_messages=$2, "
                    "cumulative_text_mass=$3, cumulative_tool_result_mass=$4, "
                    "cumulative_thinking_mass=$5, cumulative_tool_use_mass=$6, "
                    "cumulative_image_mass=$7, token_baseline_v=2 WHERE id=$8",
                    running,
                    running_messages,
                    running_mass["text"],
                    running_mass["tool_result"],
                    running_mass["thinking"],
                    running_mass["tool_use"],
                    running_mass["image"],
                    event["id"],
                )
            await conn.execute(
                "UPDATE events SET token_baseline_v=2 WHERE session_id=$1 AND kind <> 'message'",
                session_id,
            )
            await conn.execute("UPDATE sessions SET token_baseline_v=2 WHERE id=$1", session_id)
        if index % 100 == 0 or index == total:
            print(f"  {index}/{total} sessions done")
    await pool.close()
    print("Backfill complete")


def main() -> None:
    db_url = os.environ.get("AIOS_DB_URL")
    if not db_url:
        print("ERROR: AIOS_DB_URL not set", file=sys.stderr)
        raise SystemExit(1)
    asyncio.run(backfill(db_url))


if __name__ == "__main__":
    main()
