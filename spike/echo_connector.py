"""Spike: a connector built as a pure HTTP client of the aios management API.

Proves that the architectural recommendation in
``/Users/tom/.claude/plans/snappy-dreaming-wirth.md`` is feasible.  This script
is a complete connector — same conceptual surface as
``connectors/telegram/`` (642 lines) or ``connectors/signal/`` (636 lines)
plus the ``aios_connector`` base SDK (1345 lines) — but it speaks ONLY HTTP
to the management API.  No MCP, no stdio, no subprocess.

The "platform" is stdin/stdout: lines typed at the terminal become inbound
user messages; the model's ``echo_send`` tool calls become printed lines.
That's a faithful structural stand-in for any real chat platform — Telegram,
Signal, Discord — the difference is only WHICH protocol library replaces
``input()``/``print()``.

Run::

    export AIOS_URL=http://localhost:8090 AIOS_API_KEY=...
    python spike/echo_connector.py --session-id <id>

Lines you type are POSTed as user messages.  When the model calls
``echo_send(text=...)``, this script prints ``[OUTBOUND] <text>`` and POSTs
the result, resuming the session.

What this spike validates:

1. The custom-tool flow works end-to-end as a connector substrate.
2. A connector can subscribe to its session via SSE, execute custom tool
   calls, and POST results — entirely as an HTTP client.
3. The connector code is ~150 lines vs ~3000 for the MCP-stdio path.

What this spike does NOT yet validate (separately scoped):

- Multi-session routing (a production connector handles many chats; this
  is a single-session demo).  Solved by adding a per-connector calls
  stream endpoint — flagged in the plan.
- Multi-account routing (one container, one account here).  Solved the
  same way today's MCP path solves it: env-scoped credentials per container.
- Attachment uploads (text-only round-trip here).  Solved by extending
  ``POST /sessions/:id/tool-results`` for multipart, or using object
  storage URLs in the result content.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from typing import Any

import httpx


async def stream_events(
    client: httpx.AsyncClient, session_id: str, after_seq: int
) -> AsyncIterator[dict[str, Any]]:
    """Yield events from the session's SSE stream past ``after_seq``."""
    url = f"/v1/sessions/{session_id}/stream"
    async with client.stream("GET", url, params={"after_seq": after_seq}) as r:
        r.raise_for_status()
        buf = ""
        async for chunk in r.aiter_text():
            buf += chunk
            while "\n\n" in buf:
                frame, buf = buf.split("\n\n", 1)
                data_lines = [ln[5:].lstrip() for ln in frame.splitlines() if ln.startswith("data:")]
                if not data_lines:
                    continue
                payload = "\n".join(data_lines)
                if payload == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue


async def execute_echo_send(args: dict[str, Any]) -> str:
    """The 'platform' work: print the outbound message.

    A Telegram connector would replace this with ``application.bot.send_message``;
    a Signal connector with an RPC to signal-cli.  The shape is identical:
    take args, do platform-specific work, return an ack string.
    """
    text = args.get("text", "")
    print(f"[OUTBOUND] {text}", flush=True)
    return "ok"


OUR_TOOLS = {"echo_send": execute_echo_send}


async def outbound_loop(client: httpx.AsyncClient, session_id: str) -> None:
    """Watch assistant events for tool calls we own; execute and POST results.

    Tracks answered call_ids so SSE reconnects (which replay from
    ``after_seq=0``) don't double-execute.  In production each connector
    persists this set per-connection — same role today's SQLite spool plays.
    """
    answered: set[str] = set()
    async for event in stream_events(client, session_id, after_seq=0):
        if event.get("kind") != "message":
            continue
        data = event.get("data") or {}
        if data.get("role") != "assistant":
            continue
        for tc in data.get("tool_calls") or []:
            call_id = tc.get("id")
            name = tc.get("function", {}).get("name")
            if not call_id or name not in OUR_TOOLS or call_id in answered:
                continue
            args = json.loads(tc["function"].get("arguments") or "{}")
            try:
                content = await OUR_TOOLS[name](args)
                is_error = False
            except Exception as exc:
                content = json.dumps({"error": str(exc)})
                is_error = True
            await client.post(
                f"/v1/sessions/{session_id}/tool-results",
                json={"tool_call_id": call_id, "content": content, "is_error": is_error},
            )
            answered.add(call_id)


async def inbound_loop(client: httpx.AsyncClient, session_id: str) -> None:
    """Read stdin lines and POST each as an inbound user message."""
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return
        line = line.rstrip("\n")
        if not line:
            continue
        r = await client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"content": line},
        )
        r.raise_for_status()
        print(f"[INBOUND]  {line}", flush=True)


async def amain() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--url", default=os.environ.get("AIOS_URL", "http://localhost:8090"))
    parser.add_argument("--api-key", default=os.environ.get("AIOS_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("AIOS_API_KEY required (env or --api-key)")

    headers = {"Authorization": f"Bearer {args.api_key}"}
    async with (
        httpx.AsyncClient(base_url=args.url, headers=headers, timeout=None) as client,
        asyncio.TaskGroup() as tg,
    ):
        tg.create_task(outbound_loop(client, args.session_id))
        tg.create_task(inbound_loop(client, args.session_id))


if __name__ == "__main__":
    asyncio.run(amain())
