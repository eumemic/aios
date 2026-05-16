"""Inbound message handling for connector containers (#301, #318, #328).

The ``POST /v1/connectors/inbound`` endpoint calls
:func:`handle_inbound`, which wraps the dedup / attachment-staging /
session-resolution logic for inbound user messages from connector
runtimes. Session resolution is delegated to
:func:`aios_connectors.resolver.resolve_target_session` (the three-tier
resolver introduced in #328 PR 4). Built on
:func:`aios.db.queries.try_record_inbound_ack`,
:func:`aios.services.sessions.append_event`, and
:func:`aios.services.attachment_staging.stage_inbound_attachments`.
"""

from __future__ import annotations

import contextlib
from enum import StrEnum
from typing import Any, NamedTuple

import asyncpg

from aios.db import queries
from aios.errors import NotFoundError
from aios.models.sessions import MAX_USER_MESSAGE_CHARS
from aios.services.attachment_staging import (
    AttachmentStagingError,
    InboundAttachment,
    stage_inbound_attachments,
)
from aios.services.wake import defer_wake

# Metadata keys the trusted server-side path writes from request / state.
# Stripped from connector-supplied ``metadata`` before the merge so a
# crafted payload can't plant fields the renderer / logging trusts —
# most acutely ``attachments``, which the harness's vision renderer
# resolves through ``resolve_to_host_path`` and reads from disk.
_RESERVED_METADATA_KEYS = frozenset({"channel", "sender", "attachments", "platform_timestamp"})


class _DedupRollback(Exception):
    """Internal: signals the dedup-ledger conflict path inside the txn."""


class InboundDrop(StrEnum):
    """Why an inbound was not delivered to a session.

    Each maps to an HTTP response in the router: PAYLOAD_TOO_LARGE → 413;
    DETACHED and ARCHIVED_TEMPLATE → 422 (operator config issue);
    ATTACHMENT_STAGING_FAILED and SESSION_MISSING → 500.  Replays of an
    already-processed ``event_id`` are NOT a drop — they return 200
    with ``deduped=True``.
    """

    PAYLOAD_TOO_LARGE = "payload_too_large"
    DETACHED = "detached"
    ARCHIVED_TEMPLATE = "archived_template"
    ATTACHMENT_STAGING_FAILED = "attachment_staging_failed"
    SESSION_MISSING = "session_missing"


class InboundResult(NamedTuple):
    appended_event_id: str | None
    session_id: str | None
    drop_reason: InboundDrop | None
    deduped: bool


async def handle_inbound(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    connection_id: str,
    event_id: str,
    chat_id: str,
    sender: dict[str, Any],
    content: str,
    attachments: list[InboundAttachment] | None = None,
    connector_metadata: dict[str, Any] | None = None,
    platform_timestamp: str | None = None,
) -> InboundResult:
    """Resolve target session, stage attachments, dedup-append, defer wake.

    Idempotent on ``event_id``: a replay returns ``deduped=True`` and
    re-defers the wake (procrastinate's queueing_lock coalesces).

    ``attachments`` is a list of :class:`InboundAttachment` — streamable
    multipart bodies plus their metadata. The router constructs them
    from FastAPI's :class:`UploadFile` instances.
    """
    if len(content) > MAX_USER_MESSAGE_CHARS:
        return InboundResult(None, None, InboundDrop.PAYLOAD_TOO_LARGE, False)

    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, connection_id, account_id=account_id)

    # Refuse to route inbounds for archived connections. ``get_connection``
    # doesn't filter ``archived_at IS NULL`` (other callers — listing,
    # archive itself, audit views — legitimately need archived rows), and
    # ``archive_connection`` doesn't cascade-delete the per-chat
    # ``chat_sessions`` ledger rows, so the resolver's tier-1
    # ``lookup_chat_session`` would short-circuit on the stale ledger entry
    # and land the message in the session the operator believes is
    # decommissioned. Surface ``DETACHED`` so the router returns the same
    # 422 it would for any other "no live routing target" case — connector
    # containers cache state and the NOTIFY-on-removal is best-effort, so
    # post-archive inbounds during the propagation window are routine
    # rather than exceptional.
    if connection.archived_at is not None:
        return InboundResult(None, None, InboundDrop.DETACHED, False)

    # Session resolution: three-tier resolver in the connector subsystem
    # (chat_sessions → routing_rules → bindings.mode). Imported inside the
    # function so the core service module stays import-clean against the
    # subsystem at top level — the registration boundary is enforced by
    # convention (see ``aios_connectors/__init__.py``).
    from aios_connectors.resolver import ResolveDrop, resolve_target_session

    resolution = await resolve_target_session(
        pool, connection=connection, chat_id=chat_id, account_id=account_id
    )
    if resolution.drop is not None:
        drop = (
            InboundDrop.DETACHED
            if resolution.drop is ResolveDrop.DETACHED
            else InboundDrop.ARCHIVED_TEMPLATE
        )
        return InboundResult(None, None, drop, False)
    target_session_id = resolution.session_id
    assert target_session_id is not None

    try:
        staged_attachments, newly_staged_paths = await stage_inbound_attachments(
            session_id=target_session_id,
            connector_name=connection.connector,
            event_id=event_id,
            attachments=attachments or [],
        )
    except AttachmentStagingError:
        return InboundResult(None, target_session_id, InboundDrop.ATTACHMENT_STAGING_FAILED, False)

    try:
        appended = await _append_with_dedup(
            pool,
            connector=connection.connector,
            external_account_id=connection.external_account_id,
            event_id=event_id,
            session_id=target_session_id,
            chat_id=chat_id,
            sender=sender,
            content=content,
            attachments=staged_attachments,
            connector_metadata=connector_metadata,
            platform_timestamp=platform_timestamp,
            account_id=account_id,
        )
    except NotFoundError:
        # Session vanished between resolution and append: clean up freshly
        # materialized attachment bytes (replay-skipped paths are excluded
        # by stage_inbound_attachments).
        for path in newly_staged_paths:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        return InboundResult(None, target_session_id, InboundDrop.SESSION_MISSING, False)

    # Defer wake unconditionally — both first-append and dedup paths
    # heal the case where a prior attempt committed but failed to wake.
    await defer_wake(pool, target_session_id, cause="inbound", account_id=account_id)

    return InboundResult(
        appended_event_id=event_id,
        session_id=target_session_id,
        drop_reason=None,
        deduped=not appended,
    )


async def _append_with_dedup(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    connector: str,
    external_account_id: str,
    event_id: str,
    session_id: str,
    chat_id: str,
    sender: dict[str, Any],
    content: str,
    attachments: Any,
    connector_metadata: dict[str, Any] | None,
    platform_timestamp: str | None,
) -> bool:
    """Append the user-message event AND record the dedup ledger row.

    Both writes run in one transaction so a replayed event_id rolls the
    txn back via :class:`_DedupRollback`.  Returns True on first-append,
    False on dedup hit.
    """
    channel = f"{connector}/{external_account_id}/{chat_id}"
    sender_name = sender.get("display_name")
    metadata: dict[str, Any] = {}
    if connector_metadata is not None:
        # Strip keys the trusted server-side path also writes. Most acutely
        # ``attachments``: a connector-supplied record with
        # ``in_sandbox_path="/workspace/..."`` would otherwise be picked up
        # by the harness's vision renderer, which resolves the path through
        # ``resolve_to_host_path`` (accepts ``/workspace`` + ``/mnt/attachments``),
        # ``read_bytes`` it, and inlines as a base64 ``image_url`` part to
        # the model provider — exfiltration of arbitrary /workspace files
        # via the model's vision capability and reply. ``sender`` and
        # ``platform_timestamp`` are similarly server-side state the
        # connector has no business overriding; ``channel`` is already
        # unconditionally overwritten below, but list it here for clarity.
        metadata.update(
            {k: v for k, v in connector_metadata.items() if k not in _RESERVED_METADATA_KEYS}
        )
    metadata["channel"] = channel
    if isinstance(sender_name, str):
        metadata["sender"] = sender_name
    if isinstance(attachments, list) and attachments:
        metadata["attachments"] = attachments
    if platform_timestamp is not None:
        metadata["platform_timestamp"] = platform_timestamp
    data: dict[str, Any] = {"role": "user", "content": content, "metadata": metadata}

    try:
        async with pool.acquire() as conn, conn.transaction():
            event = await queries.append_event(
                conn,
                session_id=session_id,
                kind="message",
                data=data,
                orig_channel=channel,
                account_id=account_id,
            )
            inserted = await queries.try_record_inbound_ack(
                conn,
                connector=connector,
                external_account_id=external_account_id,
                event_id=event_id,
                appended_seq=event.seq,
                account_id=account_id,
            )
            if not inserted:
                raise _DedupRollback()
    except _DedupRollback:
        return False
    return True
