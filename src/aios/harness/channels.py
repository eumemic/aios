"""Phase 2 (#31) channel helpers: prompt augmentation, monologue prefix,
and the bindings → connections translation.

Kept deliberately small; the step function in :mod:`aios.harness.loop`
composes these into the discovery + augmentation + append pipeline.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db import queries
from aios.models.channel_bindings import ChannelBinding
from aios.models.connections import Connection

_MONOLOGUE_PREFIX = "INTERNAL_MONOLOGUE: "


# ── connection_server_name ─────────────────────────────────────────────────


def connection_server_name(c: Connection) -> str:
    """Deterministic, collision-free MCP server name for a connection.

    Uses the connection id directly — globally unique by construction,
    no sanitization needed.  The tool name the model sees
    (``mcp__conn_conn_01HQ…__send``) is less readable than
    ``conn_signal_alice`` would be, but tool names are for the model,
    not humans, and eliminating the connector/account string pair as a
    collision source removes an entire class of bug by construction.
    """
    return f"conn_{c.id}"


# ── bindings → connections ─────────────────────────────────────────────────


async def connections_for_bindings(
    pool: asyncpg.Pool[Any], bindings: list[ChannelBinding]
) -> list[Connection]:
    """Look up the distinct active connections referenced by a session's
    bindings.

    A binding address is ``{connector}/{account}/{path}`` — we extract
    the leading ``(connector, account)`` pair, dedupe, and batch-look
    them up in the connections table.  Bindings with malformed
    addresses (fewer than two segments) are silently ignored; this is
    defensive against bindings created out-of-band.
    """
    pairs: set[tuple[str, str]] = set()
    for b in bindings:
        parts = b.address.split("/", 2)
        if len(parts) >= 2:
            pairs.add((parts[0], parts[1]))
    if not pairs:
        return []
    async with pool.acquire() as conn:
        return await queries.get_connections_by_pairs(conn, list(pairs))


# ── system-prompt augmentation ─────────────────────────────────────────────


def build_channels_system_block(bindings: list[ChannelBinding]) -> str:
    """Build a natural-language block describing this session's bound
    channels.  Returns an empty string when the session has no bindings
    (so ``augment_with_channels`` is a no-op in that case).
    """
    if not bindings:
        return ""
    lines = ["You are bound to the following channels:"]
    for b in bindings:
        lines.append(f"  - {b.address}")
    lines.append("")
    lines.append(
        "Use the appropriate connector tool to respond to each channel. "
        "Bare assistant text is not sent to any channel — it is internal "
        "thinking and will be prefixed with 'INTERNAL_MONOLOGUE:' in your "
        "conversation history as a reminder that it is not visible to users."
    )
    return "\n".join(lines)


def augment_with_channels(base_system: str, bindings: list[ChannelBinding]) -> str:
    """Append the bound-channels block to the agent's system prompt.

    No-op when ``bindings`` is empty — the Phase-2 invariant is
    "session with no bindings behaves identically to pre-Phase-2".
    """
    block = build_channels_system_block(bindings)
    if not block:
        return base_system
    if base_system:
        return base_system + "\n\n" + block
    return block


# ── INTERNAL_MONOLOGUE prefix ──────────────────────────────────────────────


def _prefix_text(s: str) -> str:
    return s if s.startswith(_MONOLOGUE_PREFIX) else _MONOLOGUE_PREFIX + s


def apply_monologue_prefix(assistant_msg: dict[str, Any]) -> dict[str, Any]:
    """Prefix every text segment of an assistant message's ``content``
    with ``INTERNAL_MONOLOGUE:``.

    No-op when ``content`` is missing, ``None``, or empty.  Idempotent
    on a per-segment basis — already-prefixed text is left as-is.
    Returns a new dict; the input is not mutated.

    Design intent: the prefix is persisted in the event log and shown
    to the model on every subsequent step's context build — that IS
    the teaching mechanism.  Do not strip on replay.
    """
    if "content" not in assistant_msg:
        return assistant_msg
    content = assistant_msg["content"]
    if not content:
        return assistant_msg
    if isinstance(content, str):
        return {**assistant_msg, "content": _prefix_text(content)}
    if isinstance(content, list):
        new_blocks: list[Any] = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                new_blocks.append({**b, "text": _prefix_text(b.get("text", ""))})
            else:
                new_blocks.append(b)
        return {**assistant_msg, "content": new_blocks}
    return assistant_msg
