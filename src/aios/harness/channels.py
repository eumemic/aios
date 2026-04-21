"""Channel helpers: prompt augmentation, monologue prefix, bindings →
connections translation, and focal-channel unread derivation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import asyncpg

from aios.db import queries
from aios.models.channel_bindings import ChannelBinding
from aios.models.connections import CONNECTION_SERVER_NAME_PREFIX, Connection
from aios.models.events import Event

MONOLOGUE_PREFIX = "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: "

# Key under a switch_channel tool_result's ``data["metadata"]`` that
# records the target and outcome — ``{"target": str | None, "success": bool}``.
# :func:`derive_last_seen` / :func:`derive_unread_counts` anchor the
# per-channel ``last_seen`` watermark off successful switches.
SWITCH_CHANNEL_METADATA_KEY = "switch_channel"

# Top-level key inside the ``_meta`` field sent on JSON-RPC tool-call
# requests to connection-provided MCP servers.  The value is the
# focal-channel suffix (the focal channel address with its first two
# ``<connector>/<account>`` segments stripped, since the connector
# already knows its own connection identity).  The connector splits
# this on ``/`` to recover its own per-chat identifiers.
FOCAL_CHANNEL_META_KEY = "aios.focal_channel_path"


def focal_channel_path(focal: str | None) -> str | None:
    """Return the connector-specific suffix of a focal address.

    The ``<connector>/<account>`` prefix is information the MCP server
    already has (it was invoked *by* that connection), so sending it
    would be redundant; we strip it.  For a 3-segment address like
    ``signal/<bot>/<chat>`` the suffix is just ``<chat>``.  For
    ``telegram/<bot>/<chat>/<thread>`` it's ``<chat>/<thread>``.

    Returns ``None`` if ``focal`` is ``None`` or malformed (fewer than
    three segments) — neither should reach the dispatch path, but
    degrading gracefully avoids leaking garbled metadata to connectors.
    """
    if not focal:
        return None
    parts = focal.split("/", 2)
    if len(parts) < 3 or not parts[2]:
        return None
    return parts[2]


def connection_server_name(c: Connection) -> str:
    # c.id is "conn_<ULID>" — the ids.CONNECTION prefix is already the
    # reserved namespace marker, so use it directly instead of stuttering.
    assert c.id.startswith(CONNECTION_SERVER_NAME_PREFIX)
    return c.id


async def list_bindings_and_connections(
    pool: asyncpg.Pool[Any], session_id: str
) -> tuple[list[ChannelBinding], list[Connection]]:
    """Load the session's bindings and the distinct connections they reference."""
    async with pool.acquire() as conn:
        bindings = await queries.list_session_bindings(conn, session_id)
        conn_ids = sorted({b.connection_id for b in bindings})
        connections = await queries.list_connections_by_ids(conn, conn_ids) if conn_ids else []
    return bindings, connections


def build_focal_paradigm_block(bindings: list[ChannelBinding]) -> str:
    """Generic, connector-agnostic prose introducing the focal-channel paradigm.

    Cache-stable: the block's text does not vary across steps, so the
    prompt prefix stays hot.  Per-channel state (unread counts, recent
    previews) lives in the ephemeral tail block — see
    :func:`build_channels_tail_block` — which is rebuilt each step and
    appended AFTER ``build_messages`` so its mutations don't bust the
    prefix cache.

    Per-platform specifics (Signal markdown subset, mention syntax,
    response idioms) live in each connector and travel through the MCP
    ``InitializeResult.instructions`` field — see
    :func:`build_connector_instructions_block`.
    """
    if not bindings:
        return ""
    return (
        "## Channels & focal attention\n"
        "\n"
        "You operate across one or more connector channels (Signal, "
        "Slack, etc.). A channel address is path-shaped: "
        "`connector/account/chat-id`.\n"
        "\n"
        "At any moment you have exactly one focal channel, or none "
        '("phone down"). Inbound messages on your focal channel '
        "render in full in your context; inbound on other bound "
        "channels render as short notification markers (🔔 ...). "
        "The listing at the tail of your context shows the current "
        "state:\n"
        "\n"
        "* ▸ — your focal channel.\n"
        "* ○ — another bound channel, with unread count + preview.\n"
        "* ◌ — a muted bound channel (counts only, no preview).\n"
        "\n"
        "### Shifting focus\n"
        "\n"
        "Call `switch_channel(target=<address>)` to focus on a "
        "different bound channel. Its result is a re-orient block "
        "quoting recent messages on that channel so you can pick up "
        "the conversation in context. Call `switch_channel(target=null)` "
        "to put your phone down — every inbound renders as a "
        "notification, connector response tools disappear from your "
        "tool list. Switching repeatedly is cheap but not free: each "
        "switch's re-orient block appends tokens to your context.\n"
        "\n"
        "### Responding\n"
        "\n"
        "When focused on a channel, the connector's response tools "
        "(e.g. `signal_send`, `signal_react`) operate on your focal "
        "channel implicitly — no channel/chat-id argument required. "
        "Bare assistant text is NOT delivered to any channel — it is "
        "private thinking no human sees. Prefix any such thinking with "
        f"{MONOLOGUE_PREFIX.strip()!r} so it is unambiguous in your "
        "history that you understood it was internal.\n"
        "\n"
        "### Timing\n"
        "\n"
        "Tools run asynchronously — new user messages can arrive while "
        "a tool is in flight, and you will see them on your next step. "
        "There is no obligation to respond on every step; silence is "
        "the right choice when there is nothing new requiring a reply."
    )


def augment_with_focal_paradigm(base_system: str, bindings: list[ChannelBinding]) -> str:
    block = build_focal_paradigm_block(bindings)
    if not block:
        return base_system
    if base_system:
        return base_system + "\n\n" + block
    return block


def build_channels_tail_block(
    bindings: list[ChannelBinding],
    events: list[Event],
    focal_channel: str | None,
) -> dict[str, Any] | None:
    """Ephemeral per-step listing of bound channels with unread counts.

    Rebuilt at each step from the monotonic event log; appended after
    :func:`~aios.harness.context.build_messages` as the last user-role
    message so per-step mutations don't bust the prompt prefix cache.
    Pure data — the paradigm prose (what the symbols mean, how
    switch_channel works) lives in the cache-stable
    :func:`build_focal_paradigm_block`.

    Returns ``None`` when the session has no active bindings (no
    listing to render and the paradigm block is also omitted).
    """
    if not bindings:
        return None

    addresses = [b.address for b in bindings]
    unread = derive_unread_counts(events, addresses)

    # Index last inbound per channel for the preview clause.
    last_content: dict[str, str] = {}
    for e in events:
        if e.kind != "message" or e.data.get("role") != "user":
            continue
        orig = e.orig_channel
        if not isinstance(orig, str):
            continue
        content = e.data.get("content") or ""
        if isinstance(content, str):
            last_content[orig] = content

    lines = ["━━━ Channels ━━━"]
    for b in bindings:
        addr = b.address
        muted = b.notification_mode == "silent"
        if addr == focal_channel:
            lines.append(f"▸ {addr} (focal)")
            continue
        count = unread.get(addr, 0)
        if muted:
            lines.append(f"◌ {addr} (muted) — {count} unread")
            continue
        if count > 0:
            preview = last_content.get(addr, "")
            preview = preview.replace("\n", " ").strip()
            if len(preview) > 60:
                preview = preview[:60] + "…"
            preview_clause = f': "{preview}"' if preview else ""
            lines.append(f"○ {addr} — {count} unread{preview_clause}")
        else:
            lines.append(f"○ {addr} — 0 unread")
    return {"role": "user", "content": "\n".join(lines)}


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


def _switch_marker(e: Event) -> dict[str, Any] | None:
    """Return the switch_channel marker on a tool_result event, if present.

    Shape: ``{"target": str | None, "success": bool}``.  Any deviation
    (missing keys, wrong types) returns None so malformed markers are
    ignored by downstream derivation.
    """
    if e.kind != "message":
        return None
    data = e.data
    if data.get("role") != "tool":
        return None
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return None
    marker = metadata.get(SWITCH_CHANNEL_METADATA_KEY)
    if not isinstance(marker, dict):
        return None
    if not isinstance(marker.get("success"), bool):
        return None
    target = marker.get("target")
    if target is not None and not isinstance(target, str):
        return None
    return marker


def derive_last_seen(events: Iterable[Event], channel: str) -> int:
    """Compute ``last_seen_in_X`` — the max seq where the agent consumed
    peer content on ``channel``.

    Consumption happens via two signals:

    1. A peer event whose body rendered full-content in the agent's
       context: ``orig_channel == channel`` AND
       ``focal_channel_at_arrival == channel``.  (A peer event on
       ``channel`` arriving while focal is elsewhere renders only as a
       notification marker — heads-up, not body — so it does not
       anchor.)
    2. A successful ``switch_channel(target=channel)`` tool_result
       marker: the recap quotes recent peer content on ``channel`` so
       the switch itself counts as consumption.

    Agent emissions (assistant/tool events) don't anchor — they're not
    peer content.  Failed switches and ``switch_channel(target=None)``
    don't anchor.  Returns ``0`` when no consumption has happened.

    Output is identical to an earlier rule that anchored on *any* event
    with ``focal_at_arrival == channel``: focal transitions always go
    through ``switch_channel``, whose own marker already anchors past
    any prior peer-on-channel events, so the extra focal-based anchoring
    only moved ``last_seen`` forward within a span already covered by
    peer/switch anchors.  The new form is equivalent but easier to read.
    """
    last = 0
    for e in events:
        if e.orig_channel == channel and e.focal_channel_at_arrival == channel and e.seq > last:
            last = e.seq
        marker = _switch_marker(e)
        if (
            marker is not None
            and marker["success"]
            and marker["target"] == channel
            and e.seq > last
        ):
            last = e.seq
    return last


def derive_unread_counts(events: Iterable[Event], channels: Iterable[str]) -> dict[str, int]:
    """Compute per-channel unread counts.

    ``unread_in_channel = count of events where orig_channel == channel
    AND seq > last_seen_in_channel`` — i.e. peer events on the channel
    whose body the agent hasn't yet consumed (see :func:`derive_last_seen`
    for the consumption definition).

    Single pass: build every channel's ``last_seen`` watermark and
    collect candidate events in one walk, then count candidates whose
    seq exceeds their channel's watermark.  O(N + C) where N is events
    and C is candidates — versus the naïve per-channel derivation
    which is O(K*N).
    """
    channel_set = set(channels)
    last_seen = dict.fromkeys(channel_set, 0)
    candidates: list[tuple[str, int]] = []
    for e in events:
        orig = e.orig_channel
        if isinstance(orig, str) and orig in last_seen:
            if e.focal_channel_at_arrival == orig and e.seq > last_seen[orig]:
                last_seen[orig] = e.seq
            candidates.append((orig, e.seq))
        marker = _switch_marker(e)
        if marker is not None and marker["success"]:
            target = marker["target"]
            if target in last_seen and e.seq > last_seen[target]:
                last_seen[target] = e.seq
    counts = dict.fromkeys(channel_set, 0)
    for orig, seq in candidates:
        if seq > last_seen[orig]:
            counts[orig] += 1
    return counts


def _prefix_text(s: str) -> str:
    return s if s.startswith(MONOLOGUE_PREFIX) else MONOLOGUE_PREFIX + s


def apply_monologue_prefix(assistant_msg: dict[str, Any]) -> dict[str, Any]:
    """Prefix the *start* of an assistant message's text content.

    Safety net: the paradigm prose instructs the model to open its bare
    text with the prefix; this fills in the prefix when it forgets, so
    the log is uniform on replay. Idempotent — see :func:`_prefix_text`.

    For list-shaped content (providers that emit a reasoning block first
    or interleave text with tool_use blocks), the prefix is stamped on
    the *first* text block only — the message is one logical turn, and
    stamping every text segment produced double/triple prefixes in the
    log (observed on Gemma, which emits a ``thought\\n...`` text block
    followed by the actual response).
    """
    content = assistant_msg.get("content")
    if not content:
        return assistant_msg
    if isinstance(content, str):
        return {**assistant_msg, "content": _prefix_text(content)}
    if isinstance(content, list):
        new_blocks: list[Any] = []
        prefixed = False
        for block in content:
            if not prefixed and isinstance(block, dict) and block.get("type") == "text":
                new_blocks.append({**block, "text": _prefix_text(block.get("text", ""))})
                prefixed = True
            else:
                new_blocks.append(block)
        return {**assistant_msg, "content": new_blocks}
    return assistant_msg
