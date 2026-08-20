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
    healthy_events_observed: int
    findings: tuple[EgressAuditFinding, ...]


class EgressLifecycleWriterSilentError(RuntimeError):
    """No fully healthy provision proves the lifecycle writer is live."""


# Keep the time predicate in Postgres. Worker clocks are not authoritative for
# event timestamps, and passing a Python cutoff can introduce clock-skew gaps.
#
# This intentionally reads *all* provision outcomes rather than only findings.
# Healthy provisions are the audit's positive dead-man signal: without one, a
# successful empty query is indistinguishable from a silent lifecycle writer.
_FLEET_EGRESS_EVENTS_SQL = """
SELECT id, session_id, account_id, created_at, data
  FROM events
 WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
   AND kind = 'lifecycle'
   AND data->>'event' IN ('egress_provisioned', 'egress_provision_failed')
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
        rows = await conn.fetch(_FLEET_EGRESS_EVENTS_SQL)

    adverse_rows = [
        row
        for row in rows
        if row["data"]["event"] == "egress_provision_failed"
        or bool(row["data"].get("hosts_skipped"))
    ]
    healthy_events_observed = sum(
        row["data"]["event"] == "egress_provisioned" and row["data"].get("hosts_skipped") == []
        for row in rows
    )
    findings = tuple(
        EgressAuditFinding(
            event_id=row["id"],
            session_id=row["session_id"],
            account_id=row["account_id"],
            created_at=row["created_at"],
            event=row["data"]["event"],
            data=dict(row["data"]),
        )
        for row in adverse_rows
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

    if not healthy_events_observed:
        raise EgressLifecycleWriterSilentError(
            "no healthy egress_provisioned event observed in the last 24 hours"
        )

    log.info(
        "fleet_egress_audit.swept",
        findings=len(findings),
        healthy_events_observed=healthy_events_observed,
        window_hours=24,
    )
    return EgressAuditResult(
        events_examined=len(rows),
        healthy_events_observed=healthy_events_observed,
        findings=findings,
    )
