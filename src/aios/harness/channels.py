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
    # c.id is "conn_<ULID>" — the ids.CONNECTION prefix is already the
    # reserved namespace marker, so use it directly instead of stuttering.
    assert c.id.startswith(CONNECTION_SERVER_NAME_PREFIX)
    return c.id


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
    """Generic, connector-agnostic prose introducing the channels paradigm.

    Per-platform specifics (Signal markdown subset, mention syntax,
    response idioms) live in each connector and travel through the MCP
    ``InitializeResult.instructions`` field — see
    :func:`build_connector_instructions_block`.
    """
    if not bindings:
        return ""
    lines = ["You are bound to the following channels:"]
    for b in bindings:
        lines.append(f"  - {b.address}")
    lines.append("")
    lines.append(
        "A channel is a conversation reachable through a connector "
        "(Signal, Slack, etc.). Each address is path-shaped: "
        "connector/account/chat-id."
    )
    lines.append("")
    lines.append(
        "To respond to a channel you must call the connector's response "
        "tool; each connector describes its own tools in the "
        "per-connector sections below. Bare assistant text is NOT "
        "delivered to any channel; it is internal thinking and will be "
        f"prefixed with {MONOLOGUE_PREFIX.strip()!r} in your "
        "conversation history as a reminder that no human will see it."
    )
    lines.append("")
    lines.append(
        "You may take any number of tool calls before responding (web "
        "fetches, file edits, sandbox commands). Tools run "
        "asynchronously — new user messages can arrive while a tool is "
        "in flight, and you will see them on your next step. There is "
        "no obligation to respond on every step; silence is the right "
        "choice when there is nothing new requiring a reply."
    )
    return "\n".join(lines)


def augment_with_channels(base_system: str, bindings: list[ChannelBinding]) -> str:
    block = build_channels_system_block(bindings)
    if not block:
        return base_system
    if base_system:
        return base_system + "\n\n" + block
    return block


def build_connector_instructions_block(
    instructions_by_server: dict[str, str],
    connections: list[Connection],
) -> str:
    """Render per-connector affordance prose grouped by connection.

    ``instructions_by_server`` maps server_name (which for connection-
    provided MCP servers equals ``connection_server_name(c)``) to the
    server's ``InitializeResult.instructions`` string.  Connections are
    iterated in the caller-supplied order so the prompt is stable
    across steps (cache friendly).

    Connections without an entry in the dict are skipped — a connector
    that supplies no instructions contributes no block.
    """
    sections: list[str] = []
    for c in connections:
        name = connection_server_name(c)
        text = instructions_by_server.get(name)
        if not text:
            continue
        sections.append(f"## Connector: {c.connector}/{c.account}\n\n{text}")
    return "\n\n".join(sections)


def augment_with_connector_instructions(
    base_system: str,
    instructions_by_server: dict[str, str],
    connections: list[Connection],
) -> str:
    block = build_connector_instructions_block(instructions_by_server, connections)
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
