"""Channel helpers: prompt augmentation, monologue prefix, and the
bindings → connections translation.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db import queries
from aios.models.channel_bindings import ChannelBinding
from aios.models.connections import CONNECTION_SERVER_NAME_PREFIX, Connection

MONOLOGUE_PREFIX = "INTERNAL_MONOLOGUE: "


def connection_server_name(c: Connection) -> str:
    return f"{CONNECTION_SERVER_NAME_PREFIX}{c.id}"


async def list_bindings_and_connections(
    pool: asyncpg.Pool[Any], session_id: str
) -> tuple[list[ChannelBinding], list[Connection]]:
    """Load the session's bindings and the distinct connections they
    reference in a single pool acquisition.
    """
    async with pool.acquire() as conn:
        bindings = await queries.list_session_bindings(conn, session_id)
        pairs = {
            (parts[0], parts[1]) for b in bindings if len(parts := b.address.split("/", 2)) >= 2
        }
        connections = await queries.get_connections_by_pairs(conn, list(pairs)) if pairs else []
    return bindings, connections


def build_channels_system_block(bindings: list[ChannelBinding]) -> str:
    if not bindings:
        return ""
    lines = ["You are bound to the following channels:"]
    for b in bindings:
        lines.append(f"  - {b.address}")
    lines.append("")
    lines.append(
        "Use the appropriate connector tool to respond to each channel. "
        "Bare assistant text is not sent to any channel — it is internal "
        f"thinking and will be prefixed with {MONOLOGUE_PREFIX.strip()!r} in your "
        "conversation history as a reminder that it is not visible to users."
    )
    return "\n".join(lines)


def augment_with_channels(base_system: str, bindings: list[ChannelBinding]) -> str:
    block = build_channels_system_block(bindings)
    if not block:
        return base_system
    if base_system:
        return base_system + "\n\n" + block
    return block


def _prefix_text(s: str) -> str:
    return s if s.startswith(MONOLOGUE_PREFIX) else MONOLOGUE_PREFIX + s


def apply_monologue_prefix(assistant_msg: dict[str, Any]) -> dict[str, Any]:
    """Prefix every text segment of an assistant message's content.

    The prefix is persisted in the event log — the model seeing its
    own prefix on every subsequent step IS the teaching mechanism.
    Do not strip on replay.
    """
    content = assistant_msg.get("content")
    if not content:
        return assistant_msg
    if isinstance(content, str):
        return {**assistant_msg, "content": _prefix_text(content)}
    if isinstance(content, list):
        new_blocks: list[Any] = [
            {**b, "text": _prefix_text(b.get("text", ""))}
            if isinstance(b, dict) and b.get("type") == "text"
            else b
            for b in content
        ]
        return {**assistant_msg, "content": new_blocks}
    return assistant_msg
