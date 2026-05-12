"""Inbound message handling for connector containers (#301, #318).

The ``POST /v1/connectors/inbound`` endpoint calls
:func:`handle_inbound`, which wraps the dedup / attachment-staging /
session-resolution logic shared with the peer HTTP-client connector
architecture introduced in #318. Built on
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
from aios.models.connections import Connection
from aios.models.sessions import MAX_USER_MESSAGE_CHARS
from aios.services import sessions as sessions_service
from aios.services.attachment_staging import (
    AttachmentStagingError,
    stage_inbound_attachments,
)
from aios.services.wake import defer_wake


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


class _PerChatResolution(NamedTuple):
    session_id: str | None
    drop: InboundDrop | None


async def handle_inbound(
    pool: asyncpg.Pool[Any],
    *,
    connection_id: str,
    event_id: str,
    chat_id: str,
    sender: dict[str, Any],
    content: str,
    attachments: list[Any] | None = None,
    connector_metadata: dict[str, Any] | None = None,
    platform_timestamp: str | None = None,
) -> InboundResult:
    """Resolve target session, stage attachments, dedup-append, defer wake.

    Idempotent on ``event_id``: a replay returns ``deduped=True`` and
    re-defers the wake (procrastinate's queueing_lock coalesces).
    """
    if len(content) > MAX_USER_MESSAGE_CHARS:
        return InboundResult(None, None, InboundDrop.PAYLOAD_TOO_LARGE, False)

    async with pool.acquire() as conn:
        connection = await queries.get_connection(conn, connection_id)
        existing_session_id = await queries.lookup_chat_session(conn, connection.id, chat_id)

    target_session_id = existing_session_id
    if target_session_id is None:
        if connection.session_id is not None:
            target_session_id = connection.session_id
        elif connection.session_template_id is not None:
            resolution = await _resolve_per_chat(pool, connection=connection, chat_id=chat_id)
            if resolution.drop is not None:
                return InboundResult(None, None, resolution.drop, False)
            target_session_id = resolution.session_id
        else:
            return InboundResult(None, None, InboundDrop.DETACHED, False)

    assert target_session_id is not None  # both branches above either set it or returned

    try:
        staged_attachments, newly_staged_paths = stage_inbound_attachments(
            session_id=target_session_id,
            connector_name=connection.connector,
            event_id=event_id,
            raw_attachments=attachments,
        )
    except AttachmentStagingError:
        return InboundResult(None, target_session_id, InboundDrop.ATTACHMENT_STAGING_FAILED, False)

    try:
        appended = await _append_with_dedup(
            pool,
            connector=connection.connector,
            account=connection.account,
            event_id=event_id,
            session_id=target_session_id,
            chat_id=chat_id,
            sender=sender,
            content=content,
            attachments=staged_attachments,
            connector_metadata=connector_metadata,
            platform_timestamp=platform_timestamp,
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
    await defer_wake(pool, target_session_id, cause="inbound")

    return InboundResult(
        appended_event_id=event_id,
        session_id=target_session_id,
        drop_reason=None,
        deduped=not appended,
    )


async def _resolve_per_chat(
    pool: asyncpg.Pool[Any], *, connection: Connection, chat_id: str
) -> _PerChatResolution:
    """Spawn (or reuse) a session for ``(connection.id, chat_id)`` per-chat mode."""
    assert connection.session_template_id is not None
    async with pool.acquire() as conn:
        template = await queries.get_session_template(conn, connection.session_template_id)
    if template.archived_at is not None:
        return _PerChatResolution(session_id=None, drop=InboundDrop.ARCHIVED_TEMPLATE)

    focal_channel = f"{connection.connector}/{connection.account}/{chat_id}"
    session = await sessions_service.create_session(
        pool,
        agent_id=template.agent_id,
        environment_id=template.environment_id,
        agent_version=template.agent_version,
        title=None,
        metadata={},
        vault_ids=template.vault_ids or None,
        focal_channel=focal_channel,
        spawned_from_connection_id=connection.id,
    )

    # Race-safe register: insert_chat_session returns the existing
    # session_id if another writer beat us, and the just-spawned session
    # is intentionally orphaned (operator can archive later).
    async with pool.acquire() as conn:
        registered = await queries.insert_chat_session(
            conn,
            connection_id=connection.id,
            chat_id=chat_id,
            session_id=session.id,
        )
    return _PerChatResolution(session_id=registered, drop=None)


async def _append_with_dedup(
    pool: asyncpg.Pool[Any],
    *,
    connector: str,
    account: str,
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
    channel = f"{connector}/{account}/{chat_id}"
    sender_name = sender.get("display_name")
    metadata: dict[str, Any] = {}
    if connector_metadata is not None:
        metadata.update(connector_metadata)
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
            )
            inserted = await queries.try_record_inbound_ack(
                conn,
                connector=connector,
                account=account,
                event_id=event_id,
                appended_seq=event.seq,
            )
            if not inserted:
                raise _DedupRollback()
            await queries.flip_idle_to_pending(conn, session_id)
    except _DedupRollback:
        return False
    return True
