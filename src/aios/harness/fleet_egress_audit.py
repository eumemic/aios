"""Fleet-wide audit of sandbox egress provisioning outcomes.

The durable ``events`` table is the authoritative source.  This audit deliberately
reads it directly (including archived sessions) rather than relying on worker-local
sandbox state, which can disappear with a worker or container restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg

from aios.logging import get_logger

log = get_logger("aios.harness.fleet_egress_audit")


@dataclass(frozen=True)
class EgressAuditFinding:
    event_id: str
    session_id: str
    account_id: str
    created_at: datetime
    event: str
    data: dict[str, Any]


@dataclass(frozen=True)
class EgressAuditResult:
    events_examined: int
    findings: tuple[EgressAuditFinding, ...]


# Keep the time predicate in Postgres.  Worker clocks are not authoritative for
# event timestamps, and passing a Python cutoff can introduce clock-skew gaps.
_FLEET_EGRESS_FINDINGS_SQL = """
SELECT id, session_id, account_id, created_at, data
  FROM events
 WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
   AND kind = 'lifecycle'
   AND (
       data->>'event' = 'egress_provision_failed'
       OR (
           data->>'event' = 'egress_provisioned'
           AND jsonb_typeof(data->'hosts_skipped') = 'array'
           AND jsonb_array_length(data->'hosts_skipped') > 0
       )
   )
 ORDER BY created_at, session_id, seq
"""


async def run_fleet_egress_audit(pool: asyncpg.Pool[Any]) -> EgressAuditResult:
    """Read and alert on every adverse fleet egress outcome in the last 24h.

    Read errors intentionally propagate.  In particular, this function never
    emits the completion/health log unless the authoritative event query has
    succeeded; the scheduling loop records a distinct ``tick_failed`` alert and
    retries on its next tick.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(_FLEET_EGRESS_FINDINGS_SQL)

    findings = tuple(
        EgressAuditFinding(
            event_id=row["id"],
            session_id=row["session_id"],
            account_id=row["account_id"],
            created_at=row["created_at"],
            event=row["data"]["event"],
            data=dict(row["data"]),
        )
        for row in rows
    )
    for finding in findings:
        log.warning(
            "fleet_egress_audit.finding",
            event_id=finding.event_id,
            session_id=finding.session_id,
            account_id=finding.account_id,
            created_at=finding.created_at.isoformat(),
            outcome_event=finding.event,
            reason=finding.data.get("reason"),
            hosts_skipped=finding.data.get("hosts_skipped"),
        )

    log.info("fleet_egress_audit.swept", findings=len(findings), window_hours=24)
    return EgressAuditResult(events_examined=len(rows), findings=findings)
