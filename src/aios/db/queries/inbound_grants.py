"""Tenant-scoped queries for the system-owned inbound approval ledger."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import asyncpg

from aios.errors import ConflictError, NotFoundError
from aios.models.inbound_grants import InboundGrant


def _grant(row: asyncpg.Record) -> InboundGrant:
    return InboundGrant.model_validate(dict(row))


async def upsert_pending_inbound_grant(
    conn: asyncpg.Connection[Any], *, account_id: str, connection_id: str, chat_id: str
) -> None:
    await conn.execute(
        """INSERT INTO inbound_grants (account_id, connection_id, chat_id, status)
           SELECT $1, id, $3, 'pending' FROM connections
            WHERE id = $2 AND account_id = $1 AND archived_at IS NULL
           ON CONFLICT (connection_id, chat_id) WHERE status <> 'revoked' DO NOTHING""",
        account_id,
        connection_id,
        chat_id,
    )


async def list_pending_inbound_grants(
    conn: asyncpg.Connection[Any], connection_id: str, *, account_id: str
) -> list[InboundGrant]:
    rows = await conn.fetch(
        """SELECT g.* FROM inbound_grants g JOIN connections c ON c.id = g.connection_id
            WHERE g.account_id = $2 AND g.connection_id = $1 AND g.status = 'pending'
              AND c.account_id = $2 AND c.archived_at IS NULL
            ORDER BY g.created_at""",
        connection_id,
        account_id,
    )
    return [_grant(row) for row in rows]


async def approve_inbound_grant(
    conn: asyncpg.Connection[Any], connection_id: str, chat_id: str, *, account_id: str
) -> InboundGrant:
    row = await conn.fetchrow(
        """WITH locked AS (
               SELECT id, inbound_policy FROM connections
                WHERE id = $1 AND account_id = $3 AND archived_at IS NULL FOR UPDATE
           ), pending AS (
               SELECT id FROM inbound_grants
                WHERE connection_id = $1 AND chat_id = $2 AND account_id = $3
                  AND status = 'pending'
                ORDER BY created_at DESC LIMIT 1 FOR UPDATE
           ), promoted AS (
               UPDATE inbound_grants SET status = 'active', approved_by = $3,
                      approved_at = now(), approved_via_channel = 'operator_api', updated_at = now()
                WHERE id = (SELECT id FROM pending)
                  AND (SELECT inbound_policy->>'kind' FROM locked) = 'require_approval'
               RETURNING *
           ), inserted AS (
               INSERT INTO inbound_grants (
                   account_id, connection_id, chat_id, status,
                   approved_by, approved_at, approved_via_channel
               )
               SELECT $3, $1, $2, 'active', $3, now(), 'operator_api'
                WHERE NOT EXISTS (SELECT 1 FROM pending)
                  AND (SELECT inbound_policy->>'kind' FROM locked) = 'require_approval'
                  AND EXISTS (
                      SELECT 1 FROM inbound_grants
                       WHERE connection_id = $1 AND chat_id = $2 AND account_id = $3
                         AND status = 'revoked'
                  )
               RETURNING *
           ), granted AS (
               SELECT * FROM promoted UNION ALL SELECT * FROM inserted
           ), policy AS (
               UPDATE connections SET inbound_policy = jsonb_set(
                   inbound_policy, '{approved}',
                   COALESCE(inbound_policy->'approved', '[]'::jsonb) || to_jsonb($2::text), true),
                   updated_at = now()
                WHERE id = (SELECT id FROM locked) AND EXISTS (SELECT 1 FROM granted)
                  AND NOT COALESCE(inbound_policy->'approved', '[]'::jsonb) ? $2
           ), deleted AS (
               DELETE FROM chat_sessions WHERE connection_id = $1 AND chat_id = $2 AND account_id = $3
                 AND EXISTS (SELECT 1 FROM granted)
           ) SELECT * FROM granted""",
        connection_id,
        chat_id,
        account_id,
    )
    if row is None:
        raise ConflictError("grant is not pending/revoked or policy does not require approval")
    return _grant(row)


async def revoke_inbound_grant(
    conn: asyncpg.Connection[Any], connection_id: str, chat_id: str, *, account_id: str
) -> InboundGrant:
    row = await conn.fetchrow(
        """WITH locked AS (
               SELECT id, inbound_policy FROM connections
                WHERE id = $1 AND account_id = $3 AND archived_at IS NULL FOR UPDATE
           ), updated AS (
               UPDATE inbound_grants SET status = 'revoked', updated_at = now()
                WHERE account_id = $3 AND connection_id = $1 AND chat_id = $2
                  AND status IN ('pending', 'active') AND EXISTS (SELECT 1 FROM locked)
               RETURNING *
           ), policy AS (
               UPDATE connections SET inbound_policy = jsonb_set(
                   inbound_policy, '{approved}',
                   COALESCE((SELECT jsonb_agg(value) FROM jsonb_array_elements_text(
                     COALESCE(inbound_policy->'approved', '[]'::jsonb)) AS approved(value)
                     WHERE value <> $2), '[]'::jsonb), true),
                   updated_at = now()
                WHERE id = (SELECT id FROM locked) AND inbound_policy->>'kind' = 'require_approval'
                  AND EXISTS (SELECT 1 FROM updated)
           ), deleted AS (
               DELETE FROM chat_sessions WHERE connection_id = $1 AND chat_id = $2 AND account_id = $3
                 AND EXISTS (SELECT 1 FROM updated)
           ) SELECT * FROM updated""",
        connection_id,
        chat_id,
        account_id,
    )
    if row is None:
        raise NotFoundError("live inbound grant not found")
    return _grant(row)


async def reap_pending_inbound_grants(conn: asyncpg.Connection[Any], *, ttl: timedelta) -> int:
    result = await conn.execute(
        "DELETE FROM inbound_grants WHERE status = 'pending' AND created_at < now() - $1::interval",
        ttl,
    )
    return int(result.rsplit(" ", 1)[-1])
