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

from aios.config import get_settings
from aios.db import queries
from aios.errors import NotFoundError
from aios.models.inbound_policy import AllowAll, AllowList, DenyAll, InboundPolicy
from aios.models.sessions import MAX_USER_MESSAGE_CHARS
from aios.services.attachment_staging import (
    AttachmentStagingError,
    InboundAttachment,
    stage_inbound_attachments,
)
from aios.services.inbound_budget import check_inbound_budget
from aios.services.wake import defer_wake

# Metadata keys the trusted server-side path writes from request / state.
# Stripped from connector-supplied ``metadata`` before the merge so a
# crafted payload can't plant fields the renderer / logging trusts —
# most acutely ``attachments``, which the harness's vision renderer
# resolves through ``resolve_to_host_path`` and reads from disk.
#
# ``sender_name`` is the key the renderer + ``events_search.sender_name``
# derivation actually consume; reserving the historical ``sender`` key
# instead left the security boundary off the consumed slot — a connector
# could forge an arbitrary ``sender_name`` and the renderer would surface
# it as the trusted ``from=`` clause.
_RESERVED_METADATA_KEYS = frozenset({"channel", "sender_name", "attachments", "platform_timestamp"})


class _DedupRollback(Exception):
    """Internal: signals the dedup-ledger conflict path inside the txn."""


class InboundDrop(StrEnum):
    """Why an inbound was not delivered to a session.

    Each maps to an HTTP response in the router: PAYLOAD_TOO_LARGE → 413;
    DETACHED, ARCHIVED_TEMPLATE and DENIED_BY_POLICY → 422 (operator config
    / admission issue); RATE_LIMITED → 429 (per-counterparty budget exceeded);
    ATTACHMENT_STAGING_FAILED and SESSION_MISSING → 500.
    Replays of an already-processed ``event_id`` are NOT a drop — they return
    200 with ``deduped=True``.

    DENIED_BY_POLICY and RATE_LIMITED MUST map to a non-fatal status (422 or
    429), never 401/403/5xx: ``_is_fatal_inbound_status`` in the connector-http
    runner treats 401/403/5xx as fatal (crash-restarts the connector container,
    killing every connection it serves), so a denied stranger — or one
    over-budget counterparty — must not be able to take the container down.
    ``_is_fatal_inbound_status(429) is False`` (see #1504).
    """

    PAYLOAD_TOO_LARGE = "payload_too_large"
    DETACHED = "detached"
    ARCHIVED_TEMPLATE = "archived_template"
    DENIED_BY_POLICY = "denied_by_policy"
    RATE_LIMITED = "rate_limited"
    ATTACHMENT_STAGING_FAILED = "attachment_staging_failed"
    SESSION_MISSING = "session_missing"


class InboundResult(NamedTuple):
    appended_event_id: str | None
    session_id: str | None
    drop_reason: InboundDrop | None
    deduped: bool


def _admits(policy: InboundPolicy, chat_id: str) -> bool:
    """Pure admission predicate — no I/O.

    ``AllowAll`` → always; ``AllowList`` → membership in its set; ``DenyAll``
    → never. The ``match`` is exhaustive over the discriminated union.
    """
    match policy:
        case AllowAll():
            return True
        case AllowList():
            return chat_id in set(policy.chat_ids)
        case DenyAll():
            return False


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

    # Inbound-admission gate (#1500). Resolve the connection's effective
    # inbound policy and drop an unadmitted sender *before any side effect* —
    # no attachment staging, no session spawn, no ``append_event``, no
    # ``defer_wake``. The server default (NULL policy) is ``DenyAll``
    # (fail-closed). The denial maps to HTTP 422 in the router so a denied
    # stranger drops one envelope rather than crash-restarting the connector
    # (which 403/5xx would trigger via ``_is_fatal_inbound_status``).
    policy = await queries.resolve_effective_inbound_policy(
        pool, connection=connection, account_id=account_id
    )
    if not _admits(policy, chat_id):
        return InboundResult(None, None, InboundDrop.DENIED_BY_POLICY, False)

    # Per-counterparty inbound rate/cost budget (#1504). Admission decides
    # *whether* this sender may talk; the budget bounds *how much*. Enforce it
    # here — after the admission gate, before ``resolve_target_session`` and
    # before any append+wake — where ``chat_id`` and the loaded ``connection``
    # (hence ``connector`` + ``external_account_id``) are both in scope. An
    # over-budget counterparty drops with RATE_LIMITED *before any side effect*:
    # zero session spawn, zero event row, zero ``defer_wake``. The cheap-drop
    # path is one already-cheap aggregate read over the rolling window. The
    # budget is disabled by default (threshold 0 ⇒ short-circuit admit, no
    # query — byte-identical to pre-feature behavior). The drop maps to a
    # non-fatal 429 in the router so a throttle never crash-restarts the
    # connector (which 401/403/5xx would, via ``_is_fatal_inbound_status``).
    within_budget = await check_inbound_budget(
        pool,
        account_id=account_id,
        connector=connection.connector,
        external_account_id=connection.external_account_id,
        chat_id=chat_id,
    )
    if not within_budget:
        return InboundResult(None, None, InboundDrop.RATE_LIMITED, False)

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

    if not appended:
        # Dedup hit: an earlier delivery of this same ``event_id``
        # already wrote the canonical event row.  Any files THIS call
        # materialized — e.g. a freshly-encoded inline sibling on the
        # post-feature-deploy replay of a pre-feature event whose
        # metadata.attachments lacks an ``inline`` sub-record — are
        # orphans.  The persisted record references whatever was on
        # disk at first-delivery time, not what we just wrote.  Without
        # this cleanup they'd sit inside the GC's 300s recent-file
        # protection window until the next worker-restart sweep reaps
        # them, accumulating during the deploy migration window in
        # proportion to webhook-retry rate times pre-feature-event count.
        for path in newly_staged_paths:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)

    # Defer wake unconditionally — both first-append and dedup paths
    # heal the case where a prior attempt committed but failed to wake.
    #
    # Inbound debounce (#799): when configured (> 0), schedule the first
    # wake of an idle session ``inbound_debounce_seconds`` out instead of
    # immediately. ``defer_wake``'s ``queueing_lock=session_id`` then
    # collapses any follow-on inbounds that arrive inside the window into
    # this same single wake — one turn for a bursty sender's whole burst.
    # Only the inbound path is debounced; the harness retry (cause=
    # 'reschedule') and tool-completion (cause='connector_tool_result')
    # paths pass their own delay semantics and are untouched. When the
    # knob is 0.0 (off), omit ``delay_seconds`` entirely so the call is
    # byte-identical to pre-feature behavior (and so the test assertion
    # for the off case can require its absence).
    debounce_seconds = get_settings().inbound_debounce_seconds
    wake_kwargs: dict[str, Any] = {"cause": "inbound", "account_id": account_id}
    if debounce_seconds > 0:
        wake_kwargs["delay_seconds"] = debounce_seconds
    await defer_wake(pool, target_session_id, **wake_kwargs)

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
        # See ``_RESERVED_METADATA_KEYS`` for why this writes under
        # ``sender_name`` (the key the renderer + events_search column
        # consume) rather than the pre-fix ``sender`` (a dead key).
        metadata["sender_name"] = sender_name
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
