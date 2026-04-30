"""MCP inbound message ingestion and channel-state application."""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db import queries
from aios.errors import ValidationError
from aios.harness.wake import defer_wake
from aios.models.events import Event
from aios.models.inbound import InboundSubscriptionSpec
from aios.models.session_channels import SessionChannel


def parse_account_relative_channel(channel: str, *, account_id: str) -> str:
    """Return the path part of ``<account_id>/<path>`` after validation."""
    account, sep, path = channel.partition("/")
    if not sep or not account or not path:
        raise ValidationError(
            "inbound channel must be '<account>/<path>'",
            detail={"channel": channel},
        )
    if account != account_id:
        raise ValidationError(
            "inbound channel account does not match credential account",
            detail={"channel": channel, "account_id": account_id},
        )
    return path


def _display_name_from_metadata(metadata: dict[str, Any]) -> str | None:
    for key in ("chat_name", "sender_name", "display_name", "name"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


async def ingest_inbound_message(
    pool: asyncpg.Pool[Any],
    spec: InboundSubscriptionSpec,
    *,
    event_id: str,
    channel: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> Event | None:
    """Persist one inbound MCP message and wake the owning session.

    Returns ``None`` when the connector event id has already been processed
    for this session/subscription.
    """
    if not event_id:
        raise ValidationError("inbound message missing event_id")
    if not isinstance(content, str):
        raise ValidationError("inbound message content must be a string")

    path = parse_account_relative_channel(channel, account_id=spec.account_id)
    address = f"{spec.mcp_server_name}/{channel}"
    clean_metadata = dict(metadata or {})
    clean_metadata["channel"] = address
    clean_metadata["inbound_event_id"] = event_id
    clean_metadata["mcp_server_name"] = spec.mcp_server_name
    clean_metadata["mcp_server_url"] = spec.mcp_server_url
    clean_metadata["account_id"] = spec.account_id

    async with pool.acquire() as conn, conn.transaction():
        await queries.upsert_session_channel(
            conn,
            session_id=spec.session_id,
            mcp_server_name=spec.mcp_server_name,
            mcp_server_url=spec.mcp_server_url,
            account_id=spec.account_id,
            path=path,
            display_name=_display_name_from_metadata(clean_metadata),
            metadata={
                k: v
                for k, v in clean_metadata.items()
                if k
                in {
                    "chat_name",
                    "sender_name",
                    "chat_type",
                    "display_name",
                    "name",
                }
            },
        )
        inserted = await queries.insert_inbound_receipt(
            conn,
            session_id=spec.session_id,
            mcp_server_name=spec.mcp_server_name,
            account_id=spec.account_id,
            event_id=event_id,
        )
        if not inserted:
            return None
        event = await queries.append_event(
            conn,
            session_id=spec.session_id,
            kind="message",
            data={"role": "user", "content": content, "metadata": clean_metadata},
            orig_channel=address,
        )
        await queries.set_inbound_receipt_event(
            conn,
            session_id=spec.session_id,
            mcp_server_name=spec.mcp_server_name,
            account_id=spec.account_id,
            event_id=event_id,
            event_row_id=event.id,
        )
        await queries.set_inbound_cursor(
            conn,
            session_id=spec.session_id,
            mcp_server_name=spec.mcp_server_name,
            mcp_server_url=spec.mcp_server_url,
            vault_credential_id=spec.vault_credential_id,
            account_id=spec.account_id,
            last_event_id=event_id,
        )
        await conn.execute(
            "UPDATE sessions SET status = 'pending', updated_at = now() "
            "WHERE id = $1 AND status = 'idle'",
            spec.session_id,
        )

    await defer_wake(pool, spec.session_id, cause="inbound_message")
    return event


async def apply_channels_snapshot(
    pool: asyncpg.Pool[Any],
    spec: InboundSubscriptionSpec,
    channels: list[Any],
) -> list[SessionChannel]:
    """Replace-ish channel state from a connector snapshot.

    The coexistence implementation upserts channels from the snapshot but does
    not archive omitted rows; connectors may send partial snapshots.
    """
    out: list[SessionChannel] = []
    async with pool.acquire() as conn:
        for entry in channels:
            parsed = _parse_channel_entry(entry, account_id=spec.account_id)
            if parsed is None:
                continue
            path, display_name, notification_mode, metadata = parsed
            out.append(
                await queries.upsert_session_channel(
                    conn,
                    session_id=spec.session_id,
                    mcp_server_name=spec.mcp_server_name,
                    mcp_server_url=spec.mcp_server_url,
                    account_id=spec.account_id,
                    path=path,
                    display_name=display_name,
                    notification_mode=notification_mode,
                    metadata=metadata,
                )
            )
    return out


async def apply_channels_delta(
    pool: asyncpg.Pool[Any],
    spec: InboundSubscriptionSpec,
    *,
    upserts: list[Any],
    archives: list[Any],
) -> None:
    async with pool.acquire() as conn:
        for entry in upserts:
            parsed = _parse_channel_entry(entry, account_id=spec.account_id)
            if parsed is None:
                continue
            path, display_name, notification_mode, metadata = parsed
            await queries.upsert_session_channel(
                conn,
                session_id=spec.session_id,
                mcp_server_name=spec.mcp_server_name,
                mcp_server_url=spec.mcp_server_url,
                account_id=spec.account_id,
                path=path,
                display_name=display_name,
                notification_mode=notification_mode,
                metadata=metadata,
            )
        for entry in archives:
            channel = entry.get("channel") if isinstance(entry, dict) else entry
            if not isinstance(channel, str):
                continue
            path = parse_account_relative_channel(channel, account_id=spec.account_id)
            await queries.archive_session_channel(
                conn,
                session_id=spec.session_id,
                mcp_server_name=spec.mcp_server_name,
                account_id=spec.account_id,
                path=path,
            )


async def record_replay_lost(
    pool: asyncpg.Pool[Any],
    spec: InboundSubscriptionSpec,
    params: dict[str, Any],
) -> None:
    async with pool.acquire() as conn:
        await queries.append_event(
            conn,
            session_id=spec.session_id,
            kind="span",
            data={
                "event": "mcp_inbound_replay_lost",
                "mcp_server_name": spec.mcp_server_name,
                "account_id": spec.account_id,
                "params": params,
            },
        )
        await queries.set_inbound_cursor(
            conn,
            session_id=spec.session_id,
            mcp_server_name=spec.mcp_server_name,
            mcp_server_url=spec.mcp_server_url,
            vault_credential_id=spec.vault_credential_id,
            account_id=spec.account_id,
            last_event_id=None,
        )


def _parse_channel_entry(
    entry: Any, *, account_id: str
) -> tuple[str, str | None, str, dict[str, Any]] | None:
    if isinstance(entry, str):
        return (
            parse_account_relative_channel(entry, account_id=account_id),
            None,
            "focal_candidate",
            {},
        )
    if not isinstance(entry, dict):
        return None
    channel = entry.get("channel")
    if not isinstance(channel, str):
        return None
    path = parse_account_relative_channel(channel, account_id=account_id)
    display_name = entry.get("display_name") or entry.get("name")
    if not isinstance(display_name, str):
        display_name = None
    notification_mode = entry.get("notification_mode")
    if notification_mode not in ("focal_candidate", "silent"):
        notification_mode = "focal_candidate"
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return path, display_name, notification_mode, metadata
